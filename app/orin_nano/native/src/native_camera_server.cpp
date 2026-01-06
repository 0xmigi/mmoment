/**
 * native_camera_server.cpp - Native C++ camera server with GPU-accelerated inference
 *
 * Combines:
 * - GStreamer camera capture with NVMM (GPU memory)
 * - TensorRT YOLOv8-pose + InsightFace inference
 * - Unix socket server for Docker communication
 *
 * Key optimization: Frames stay on GPU from capture through inference.
 * Only copied to CPU when client requests a frame.
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaEGL.h>

// EGL for NVMM-CUDA interop
#include <EGL/egl.h>
#include <EGL/eglext.h>

// NVIDIA buffer surface API for NVMM memory access
extern "C" {
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
}

#include "tensorrt_engine.h"
#include "insightface_engine.h"
#include "reid_engine.h"
#include "native_pipeline_api.h"
#include "byte_tracker.h"

using namespace mmoment;

// =============================================================================
// Configuration
// =============================================================================

static const char* SOCKET_PATH = "/tmp/native_inference.sock";
static const int FRAME_WIDTH = 1280;
static const int FRAME_HEIGHT = 720;
static const int FRAME_FPS = 30;
static const int RING_BUFFER_SIZE = 4;
static const bool ENABLE_FACE_DETECTION = true;   // Enable InsightFace
static const int FACE_DETECTION_INTERVAL = 5;     // Run face detection every N frames
static const bool ENABLE_REID = true;             // Enable OSNet ReID for body appearance
static const int REID_UPDATE_INTERVAL = 10;       // Update ReID embedding every N frames

// GStreamer pipeline - NVMM output keeps frames on GPU
// Memory type 4 (SURFACE_ARRAY) requires NvBufSurfaceTransform for CUDA interop
static const char* CAMERA_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "nvv4l2decoder mjpeg=1 ! "
    "nvvidconv ! "
    "video/x-raw(memory:NVMM), format=BGRx ! "
    "appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true";

// Preprocessing kernel declarations
extern "C" void launchPreprocessBGRxToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

// Rotation preprocessing for portrait camera orientation
extern "C" void launchRotateBGRx90CCW(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    cudaStream_t stream
);

extern "C" void launchPreprocessBGRxToRGBFloatRotate90CCW(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    int* rotatedW, int* rotatedH,
    cudaStream_t stream
);

// GPU transpose kernel - [56, 8400] -> [8400, 56]
extern "C" void launchTransposeYoloOutput(
    const float* src, float* dst,
    int rows, int cols,
    cudaStream_t stream
);

// GPU BGRx to BGR conversion
extern "C" void launchConvertBGRxToBGR(
    const void* src, void* dst,
    int width, int height,
    cudaStream_t stream
);

// GPU BGR to BGRx conversion (for process_image - adds 4th channel)
extern "C" void launchConvertBGRToBGRx(
    const void* src, void* dst,
    int width, int height,
    cudaStream_t stream
);

// =============================================================================
// Global State
// =============================================================================

static std::atomic<bool> g_running{true};
static std::mutex g_resultMutex;
static std::condition_variable g_resultCond;

// Inference engines
static TensorRTEngine* g_yoloEngine = nullptr;
static InsightFacePipeline* g_insightFace = nullptr;
static ReIDEngine* g_reidEngine = nullptr;
static YoloPosePostProcessor* g_postProcessor = nullptr;
static ByteTracker* g_tracker = nullptr;

// GPU buffers
static void* g_gpuFrame = nullptr;          // Current frame on GPU
static void* g_gpuYoloInput = nullptr;      // 640x640 preprocessed for YOLO
static float* g_gpuYoloTransposed = nullptr; // Transposed YOLO output [8400, 56]
static void* g_gpuBGRFrame = nullptr;       // BGR frame for output (landscape - unused now)
static void* g_gpuRotatedFrame = nullptr;   // 720x1280 BGRx rotated frame for InsightFace
static void* g_gpuRotatedBGR = nullptr;     // 720x1280 BGR rotated frame for Python output
static cudaStream_t g_stream = nullptr;

// Rotated frame dimensions (after 90° CCW rotation)
static const int ROTATED_WIDTH = FRAME_HEIGHT;   // 720
static const int ROTATED_HEIGHT = FRAME_WIDTH;   // 1280

// Process image buffers (for phone selfie registration)
static void* g_gpuProcessBGR = nullptr;     // BGR input from client
static void* g_gpuProcessBGRx = nullptr;    // Converted to BGRx for InsightFace
static const int MAX_PROCESS_WIDTH = 1920;
static const int MAX_PROCESS_HEIGHT = 1920;
static std::mutex g_processImageMutex;       // Protect process_image resources

// Latest result (protected by mutex)
static NativeFrameResult g_latestResult = {};
static std::vector<uint8_t> g_latestFrameCPU;  // Frame copied to CPU for clients
static bool g_resultReady = false;

// Cached face detection results (updated every N frames)
static std::vector<FaceRecognition> g_cachedFaces;
static int g_faceDetectionCounter = 0;
static int g_reidUpdateCounter = 0;

// Stats
static std::atomic<int> g_frameCount{0};
static std::atomic<float> g_avgFps{0.0f};
static std::atomic<float> g_avgInferenceMs{0.0f};

// =============================================================================
// Signal Handler
// =============================================================================

void signalHandler(int sig) {
    std::cout << "\n[Server] Received signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

// =============================================================================
// JSON Helpers
// =============================================================================

std::string buildResultJson(const NativeFrameResult& result, int width, int height) {
    std::string json = "{";

    // Frame dimensions
    json += "\"width\":" + std::to_string(width) + ",";
    json += "\"height\":" + std::to_string(height) + ",";
    json += "\"channels\":3,";

    // Timing
    json += "\"timing\":{";
    json += "\"preprocess_ms\":" + std::to_string(result.preprocess_ms) + ",";
    json += "\"yolo_ms\":" + std::to_string(result.yolo_ms) + ",";
    json += "\"face_ms\":" + std::to_string(result.face_ms) + ",";
    json += "\"total_ms\":" + std::to_string(result.total_ms);
    json += "},";

    // Persons array
    json += "\"persons\":[";
    for (int i = 0; i < result.num_persons; i++) {
        if (i > 0) json += ",";
        const auto& p = result.persons[i];
        json += "{";
        json += "\"bbox\":[" + std::to_string(p.x1) + "," + std::to_string(p.y1) + ","
                             + std::to_string(p.x2) + "," + std::to_string(p.y2) + "],";
        json += "\"confidence\":" + std::to_string(p.confidence) + ",";
        json += "\"track_id\":" + std::to_string(p.track_id) + ",";
        json += "\"identity_label\":\"" + std::string(p.identity_label) + "\",";
        json += "\"identity_confidence\":" + std::to_string(p.identity_confidence) + ",";
        json += "\"has_reid_embedding\":" + std::to_string(p.has_reid_embedding) + ",";
        if (p.has_reid_embedding) {
            json += "\"reid_embedding\":[";
            for (int e = 0; e < 512; e++) {
                if (e > 0) json += ",";
                json += std::to_string(p.reid_embedding[e]);
            }
            json += "],";
        }
        json += "\"keypoints\":[";
        for (int k = 0; k < 17; k++) {
            if (k > 0) json += ",";
            json += "[" + std::to_string(p.keypoints[k][0]) + ","
                       + std::to_string(p.keypoints[k][1]) + ","
                       + std::to_string(p.keypoints[k][2]) + "]";
        }
        json += "]}";
    }
    json += "],";

    // Faces array
    json += "\"faces\":[";
    for (int i = 0; i < result.num_faces; i++) {
        if (i > 0) json += ",";
        const auto& f = result.faces[i];
        json += "{";
        json += "\"bbox\":[" + std::to_string(f.face.x1) + "," + std::to_string(f.face.y1) + ","
                             + std::to_string(f.face.x2) + "," + std::to_string(f.face.y2) + "],";
        json += "\"confidence\":" + std::to_string(f.face.confidence) + ",";
        json += "\"landmarks\":[";
        for (int l = 0; l < 5; l++) {
            if (l > 0) json += ",";
            json += "[" + std::to_string(f.face.landmarks[l][0]) + ","
                       + std::to_string(f.face.landmarks[l][1]) + "]";
        }
        json += "],";
        json += "\"embedding\":[";
        for (int e = 0; e < 512; e++) {
            if (e > 0) json += ",";
            json += std::to_string(f.embedding[e]);
        }
        json += "],";
        json += "\"quality\":" + std::to_string(f.quality) + ",";
        json += "\"person_track_id\":" + std::to_string(f.person_track_id) + ",";
        json += "\"identity_label\":\"" + std::string(f.identity_label) + "\",";
        json += "\"identity_confidence\":" + std::to_string(f.identity_confidence);
        json += "}";
    }
    json += "]";

    json += "}";
    return json;
}

// =============================================================================
// Socket Helpers
// =============================================================================

bool sendAll(int sock, const void* data, size_t len) {
    const char* ptr = (const char*)data;
    while (len > 0) {
        ssize_t sent = send(sock, ptr, len, MSG_NOSIGNAL);
        if (sent <= 0) return false;
        ptr += sent;
        len -= sent;
    }
    return true;
}

bool recvAll(int sock, void* data, size_t len) {
    char* ptr = (char*)data;
    while (len > 0) {
        ssize_t received = recv(sock, ptr, len, 0);
        if (received <= 0) return false;
        ptr += received;
        len -= received;
    }
    return true;
}

// =============================================================================
// Client Handler
// =============================================================================

void handleClient(int clientSock) {
    std::cout << "[Server] Client connected" << std::endl;

    char buffer[4096];

    while (g_running) {
        // Read message length (4 bytes, big-endian)
        uint32_t msgLen;
        if (!recvAll(clientSock, &msgLen, 4)) break;
        msgLen = ntohl(msgLen);

        if (msgLen > sizeof(buffer) - 1) {
            std::cerr << "[Server] Message too large: " << msgLen << std::endl;
            break;
        }

        // Read message
        if (!recvAll(clientSock, buffer, msgLen)) break;
        buffer[msgLen] = '\0';

        // Parse command (simple JSON parsing)
        std::string msg(buffer);

        if (msg.find("\"get_frame\"") != std::string::npos) {
            // Get latest frame and result
            std::unique_lock<std::mutex> lock(g_resultMutex);

            // Wait for a result to be ready (with timeout)
            if (!g_resultReady) {
                g_resultCond.wait_for(lock, std::chrono::milliseconds(100));
            }

            if (!g_resultReady || g_latestFrameCPU.empty()) {
                // Send error response
                std::string errorJson = "{\"error\":\"No frame available\"}";
                uint32_t len = htonl(errorJson.size());
                sendAll(clientSock, &len, 4);
                sendAll(clientSock, errorJson.c_str(), errorJson.size());
                continue;
            }

            // Build response JSON
            // Send ROTATED dimensions (720x1280 portrait) - frame is already rotated on GPU
            std::string responseJson = buildResultJson(g_latestResult, ROTATED_WIDTH, ROTATED_HEIGHT);

            // Send response header + JSON
            uint32_t jsonLen = htonl(responseJson.size());
            if (!sendAll(clientSock, &jsonLen, 4)) break;
            if (!sendAll(clientSock, responseJson.c_str(), responseJson.size())) break;

            // Send ROTATED frame data (BGR, portrait orientation)
            // Frame is already rotated 90° CCW on GPU before copying to CPU
            size_t frameSize = ROTATED_WIDTH * ROTATED_HEIGHT * 3;
            uint32_t frameLenNet = htonl(frameSize);
            if (!sendAll(clientSock, &frameLenNet, 4)) break;
            if (!sendAll(clientSock, g_latestFrameCPU.data(), frameSize)) break;

        } else if (msg.find("\"ping\"") != std::string::npos) {
            std::string response = "{\"status\":\"ok\",\"version\":\"1.0.0-native-cpp\"}";
            uint32_t len = htonl(response.size());
            sendAll(clientSock, &len, 4);
            sendAll(clientSock, response.c_str(), response.size());

        } else if (msg.find("\"stats\"") != std::string::npos) {
            std::string response = "{\"total_frames\":" + std::to_string(g_frameCount.load()) +
                                   ",\"avg_fps\":" + std::to_string(g_avgFps.load()) +
                                   ",\"avg_inference_ms\":" + std::to_string(g_avgInferenceMs.load()) + "}";
            uint32_t len = htonl(response.size());
            sendAll(clientSock, &len, 4);
            sendAll(clientSock, response.c_str(), response.size());

        } else if (msg.find("\"process_image\"") != std::string::npos) {
            // =================================================================
            // Process arbitrary image through InsightFace for face registration
            // This ensures phone selfies use the SAME ArcFace model as runtime
            // =================================================================

            // Parse width, height from JSON (simple parsing)
            int width = 0, height = 0, channels = 3;
            size_t wPos = msg.find("\"width\":");
            size_t hPos = msg.find("\"height\":");
            size_t cPos = msg.find("\"channels\":");

            if (wPos != std::string::npos) {
                width = std::stoi(msg.substr(wPos + 8));
            }
            if (hPos != std::string::npos) {
                height = std::stoi(msg.substr(hPos + 9));
            }
            if (cPos != std::string::npos) {
                channels = std::stoi(msg.substr(cPos + 11));
            }

            if (width <= 0 || height <= 0 || width > MAX_PROCESS_WIDTH || height > MAX_PROCESS_HEIGHT) {
                std::string response = "{\"error\":\"Invalid image dimensions\"}";
                uint32_t len = htonl(response.size());
                sendAll(clientSock, &len, 4);
                sendAll(clientSock, response.c_str(), response.size());
                continue;
            }

            // Receive image data
            size_t imageSize = width * height * channels;
            std::vector<uint8_t> imageData(imageSize);
            if (!recvAll(clientSock, imageData.data(), imageSize)) {
                std::cerr << "[Server] Failed to receive image data" << std::endl;
                break;
            }

            std::cout << "[Server] Processing image " << width << "x" << height << " for face extraction" << std::endl;

            // Lock process_image resources (in case multiple clients)
            std::lock_guard<std::mutex> lock(g_processImageMutex);

            // Copy image to GPU and convert BGR->BGRx
            cudaMemcpy(g_gpuProcessBGR, imageData.data(), imageSize, cudaMemcpyHostToDevice);
            launchConvertBGRToBGRx(g_gpuProcessBGR, g_gpuProcessBGRx, width, height, g_stream);
            cudaStreamSynchronize(g_stream);

            // Run InsightFace on full image (no person bbox, use full frame)
            // pitch = width * 4 for BGRx
            auto faces = g_insightFace->process(
                g_gpuProcessBGRx,
                width, height, width * 4,
                0, 0, width, height,  // Full image bounds
                0.5f
            );

            // Build JSON response with faces and embeddings
            std::string response = "{\"success\":true,\"faces\":[";
            for (size_t i = 0; i < faces.size(); i++) {
                if (i > 0) response += ",";
                const auto& f = faces[i];
                response += "{";
                response += "\"bbox\":[" + std::to_string(f.face.x1) + "," + std::to_string(f.face.y1) + ","
                                         + std::to_string(f.face.x2) + "," + std::to_string(f.face.y2) + "],";
                response += "\"confidence\":" + std::to_string(f.face.confidence) + ",";
                response += "\"landmarks\":[";
                for (int l = 0; l < 5; l++) {
                    if (l > 0) response += ",";
                    response += "[" + std::to_string(f.face.landmarks[l][0]) + ","
                                    + std::to_string(f.face.landmarks[l][1]) + "]";
                }
                response += "],";
                response += "\"embedding\":[";
                for (int e = 0; e < 512; e++) {
                    if (e > 0) response += ",";
                    response += std::to_string(f.embedding[e]);
                }
                response += "],";
                response += "\"quality\":" + std::to_string(f.quality);
                response += "}";
            }
            response += "],\"inference_ms\":0}";

            std::cout << "[Server] Extracted " << faces.size() << " faces from image" << std::endl;

            uint32_t len = htonl(response.size());
            sendAll(clientSock, &len, 4);
            sendAll(clientSock, response.c_str(), response.size());

        } else {
            std::string response = "{\"error\":\"Unknown command\"}";
            uint32_t len = htonl(response.size());
            sendAll(clientSock, &len, 4);
            sendAll(clientSock, response.c_str(), response.size());
        }
    }

    close(clientSock);
    std::cout << "[Server] Client disconnected" << std::endl;
}

// =============================================================================
// Socket Server Thread
// =============================================================================

void socketServerThread() {
    // Remove existing socket
    unlink(SOCKET_PATH);

    // Create socket
    int serverSock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (serverSock < 0) {
        std::cerr << "[Server] Failed to create socket" << std::endl;
        return;
    }

    // Bind to path
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(serverSock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[Server] Failed to bind socket" << std::endl;
        close(serverSock);
        return;
    }

    // Set permissions for Docker access
    chmod(SOCKET_PATH, 0777);

    // Listen
    listen(serverSock, 5);
    std::cout << "[Server] Listening on " << SOCKET_PATH << std::endl;

    // Accept loop
    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(serverSock, &fds);

        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int ret = select(serverSock + 1, &fds, nullptr, nullptr, &tv);
        if (ret > 0) {
            int clientSock = accept(serverSock, nullptr, nullptr);
            if (clientSock >= 0) {
                // Handle client in new thread
                std::thread(handleClient, clientSock).detach();
            }
        }
    }

    close(serverSock);
    unlink(SOCKET_PATH);
    std::cout << "[Server] Socket server stopped" << std::endl;
}

// =============================================================================
// Main Processing Loop
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "=== MMOMENT Native Camera Server ===" << std::endl;
    std::cout << "GPU-accelerated inference with minimal memory copies" << std::endl;
    std::cout << std::endl;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // ==========================================================================
    // Load TensorRT engines
    // ==========================================================================
    std::cout << "[Init] Loading YOLOv8-pose engine..." << std::endl;
    g_yoloEngine = new TensorRTEngine();
    if (!g_yoloEngine->loadEngine("yolov8n-pose-native.engine")) {
        std::cerr << "[Init] Failed to load YOLO engine" << std::endl;
        return 1;
    }

    std::cout << "[Init] Loading InsightFace engines..." << std::endl;
    g_insightFace = new InsightFacePipeline();
    if (!g_insightFace->initialize("retinaface.engine", "arcface_r50.engine")) {
        std::cerr << "[Init] Failed to load InsightFace engines" << std::endl;
        return 1;
    }

    if (ENABLE_REID) {
        std::cout << "[Init] Loading OSNet ReID engine..." << std::endl;
        g_reidEngine = new ReIDEngine();
        if (!g_reidEngine->loadEngine("osnet_x0_25.engine")) {
            std::cerr << "[Init] Failed to load ReID engine (optional, continuing without)" << std::endl;
            delete g_reidEngine;
            g_reidEngine = nullptr;
        }
    }

    g_postProcessor = new YoloPosePostProcessor(0.15f, 0.45f);  // Lower confidence threshold for detection

    // Initialize ByteTracker - tuned for stable track IDs
    // - maxAge=60: Keep lost tracks for 2s at 30fps
    // - highThresh=0.15: Lower threshold to catch more detections
    // - lowThresh=0.1: Low confidence detections for second-pass matching
    // - matchThresh=0.3: LOWER IoU threshold = MORE lenient matching = stable IDs
    g_tracker = new ByteTracker(60, 0.15f, 0.1f, 0.3f);

    // ==========================================================================
    // Allocate GPU buffers
    // ==========================================================================
    std::cout << "[Init] Initializing CUDA driver API..." << std::endl;
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) {
        std::cerr << "[Init] cuInit failed: " << cuErr << std::endl;
        return 1;
    }

    std::cout << "[Init] Allocating GPU buffers..." << std::endl;
    cudaMalloc(&g_gpuFrame, FRAME_WIDTH * FRAME_HEIGHT * 4);  // BGRx CUDA buffer
    cudaMalloc(&g_gpuYoloInput, 640 * 640 * 3 * sizeof(float));
    cudaMalloc(&g_gpuYoloTransposed, 8400 * 56 * sizeof(float));  // Transposed YOLO output
    cudaMalloc(&g_gpuBGRFrame, FRAME_WIDTH * FRAME_HEIGHT * 3);   // BGR output frame (landscape)
    cudaMalloc(&g_gpuRotatedFrame, ROTATED_WIDTH * ROTATED_HEIGHT * 4);  // 720x1280 BGRx rotated for InsightFace
    cudaMalloc(&g_gpuRotatedBGR, ROTATED_WIDTH * ROTATED_HEIGHT * 3);    // 720x1280 BGR rotated for Python
    cudaStreamCreate(&g_stream);
    std::cout << "[Init] Allocated rotated frame buffer (" << ROTATED_WIDTH << "x" << ROTATED_HEIGHT << ")" << std::endl;

    // Allocate buffers for process_image (phone selfie face registration)
    cudaMalloc(&g_gpuProcessBGR, MAX_PROCESS_WIDTH * MAX_PROCESS_HEIGHT * 3);   // BGR input
    cudaMalloc(&g_gpuProcessBGRx, MAX_PROCESS_WIDTH * MAX_PROCESS_HEIGHT * 4);  // BGRx for InsightFace
    std::cout << "[Init] Allocated process_image buffers (max " << MAX_PROCESS_WIDTH << "x" << MAX_PROCESS_HEIGHT << ")" << std::endl;

    // Allocate CPU buffer for ROTATED frame output (BGR, 3 channels, portrait orientation)
    g_latestFrameCPU.resize(ROTATED_WIDTH * ROTATED_HEIGHT * 3);

    // ==========================================================================
    // Create GStreamer pipeline
    // ==========================================================================
    std::cout << "[Init] Starting camera pipeline..." << std::endl;

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(CAMERA_PIPELINE, &error);
    if (error) {
        std::cerr << "[Init] Pipeline failed: " << error->message << std::endl;
        g_error_free(error);
        return 1;
    }
    std::cout << "[Init] Using NVMM pipeline (GPU memory)" << std::endl;

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // ==========================================================================
    // Start socket server thread
    // ==========================================================================
    std::thread serverThread(socketServerThread);

    // ==========================================================================
    // Main processing loop
    // ==========================================================================
    std::cout << "[Server] Processing frames..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    float totalInferenceTime = 0;
    int localFrameCount = 0;

    while (g_running) {
        // Pull frame from camera
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 100 * GST_MSECOND);
        if (!sample) continue;

        auto frameStart = std::chrono::high_resolution_clock::now();

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }

        // Get source NvBufSurface from NVMM
        NvBufSurface* srcSurface = (NvBufSurface*)map.data;
        if (!srcSurface || srcSurface->numFilled == 0) {
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            continue;
        }

        // Map NVMM surface to EGLImage for CUDA access
        if (NvBufSurfaceMapEglImage(srcSurface, 0) != 0) {
            std::cerr << "[Error] NvBufSurfaceMapEglImage failed" << std::endl;
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            continue;
        }

        EGLImageKHR eglImage = srcSurface->surfaceList[0].mappedAddr.eglImage;

        // Register EGLImage with CUDA
        CUgraphicsResource cudaResource;
        CUresult cuErr = cuGraphicsEGLRegisterImage(&cudaResource, eglImage,
                                                     CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
        if (cuErr != CUDA_SUCCESS) {
            std::cerr << "[Error] cuGraphicsEGLRegisterImage failed: " << cuErr << std::endl;
            NvBufSurfaceUnMapEglImage(srcSurface, 0);
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            continue;
        }

        // Get the CUDA array from the registered resource
        CUeglFrame eglFrame;
        cuErr = cuGraphicsResourceGetMappedEglFrame(&eglFrame, cudaResource, 0, 0);
        if (cuErr != CUDA_SUCCESS) {
            std::cerr << "[Error] cuGraphicsResourceGetMappedEglFrame failed: " << cuErr << std::endl;
            cuGraphicsUnregisterResource(cudaResource);
            NvBufSurfaceUnMapEglImage(srcSurface, 0);
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            continue;
        }

        // eglFrame.frame.pPitch[0] is the CUDA-accessible GPU pointer
        void* gpuFramePtr = (void*)eglFrame.frame.pPitch[0];
        int pitch = eglFrame.pitch;

        auto preprocessStart = std::chrono::high_resolution_clock::now();

        // =======================================================================
        // YOLOv8-pose inference WITH ROTATION
        // Camera is in portrait orientation but captures landscape 1280x720
        // We rotate 90° CCW so YOLO sees an upright person (720x1280 portrait)
        // =======================================================================
        float scaleRatio;
        int rotatedW, rotatedH;
        launchPreprocessBGRxToRGBFloatRotate90CCW(
            gpuFramePtr, g_gpuYoloInput,
            FRAME_WIDTH, FRAME_HEIGHT, pitch,
            640, 640,
            &scaleRatio,
            &rotatedW, &rotatedH,  // 720x1280 after rotation
            g_stream
        );
        cudaStreamSynchronize(g_stream);

        auto yoloStart = std::chrono::high_resolution_clock::now();

        // Run inference - output stays on GPU
        float* gpuYoloOutput;
        size_t outputElements;
        g_yoloEngine->inferGPU(g_gpuYoloInput, 640, 640, 3, &gpuYoloOutput, &outputElements);

        // Transpose on GPU: [1, 56, 8400] -> [8400, 56]
        const int NUM_DETS = 8400;
        launchTransposeYoloOutput(gpuYoloOutput, g_gpuYoloTransposed, 56, NUM_DETS, g_stream);

        // Copy transposed result to CPU for post-processing
        std::vector<float> transposed(NUM_DETS * 56);
        cudaMemcpyAsync(transposed.data(), g_gpuYoloTransposed,
                        NUM_DETS * 56 * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
        cudaStreamSynchronize(g_stream);

        // Post-process with ROTATED dimensions (720x1280 portrait)
        // This gives us coordinates in portrait space
        auto detections = g_postProcessor->process(transposed, NUM_DETS, scaleRatio,
                                                    rotatedW, rotatedH);

        // Transform coordinates from portrait (720x1280) back to landscape (1280x720)
        // 90° CW rotation: portrait(x, y) -> landscape(rotatedH - 1 - y, x)
        for (auto& det : detections) {
            // Transform bbox
            float newX1 = rotatedH - 1 - det.y2;  // portrait y2 -> landscape x1
            float newY1 = det.x1;                  // portrait x1 -> landscape y1
            float newX2 = rotatedH - 1 - det.y1;  // portrait y1 -> landscape x2
            float newY2 = det.x2;                  // portrait x2 -> landscape y2
            det.x1 = newX1;
            det.y1 = newY1;
            det.x2 = newX2;
            det.y2 = newY2;

            // Transform keypoints
            for (int k = 0; k < 17; k++) {
                float portX = det.keypoints[k][0];
                float portY = det.keypoints[k][1];
                det.keypoints[k][0] = rotatedH - 1 - portY;  // new X
                det.keypoints[k][1] = portX;                  // new Y
            }
        }

        // =======================================================================
        // ByteTrack for smooth multi-object tracking
        // =======================================================================
        // Convert PoseDetection to Detection for ByteTracker
        std::vector<Detection> trackerInput;
        for (const auto& det : detections) {
            Detection d;
            d.x1 = det.x1;
            d.y1 = det.y1;
            d.x2 = det.x2;
            d.y2 = det.y2;
            d.score = det.confidence;
            std::memcpy(d.keypoints, det.keypoints, sizeof(d.keypoints));
            trackerInput.push_back(d);
        }

        // Update tracker and get smoothed tracks
        std::vector<STrack> trackedTracks = g_tracker->update(trackerInput, localFrameCount);

        auto yoloEnd = std::chrono::high_resolution_clock::now();

        // =======================================================================
        // InsightFace for each person (runs every N frames to save CPU)
        // IMPORTANT: Camera is rotated 90° CCW, so we must:
        // 1. Rotate frame to portrait (720x1280) so faces appear upright
        // 2. Convert landscape bbox to portrait coordinates
        // =======================================================================
        auto faceStart = std::chrono::high_resolution_clock::now();

        std::vector<FaceRecognition> allFaces;
        if (ENABLE_FACE_DETECTION) {
            g_faceDetectionCounter++;

            // Only run face detection every N frames
            if (g_faceDetectionCounter >= FACE_DETECTION_INTERVAL) {
                g_faceDetectionCounter = 0;
                g_cachedFaces.clear();

                // Rotate frame to portrait for InsightFace (people appear upright)
                launchRotateBGRx90CCW(
                    gpuFramePtr, g_gpuRotatedFrame,
                    FRAME_WIDTH, FRAME_HEIGHT, pitch,
                    g_stream
                );
                cudaStreamSynchronize(g_stream);

                // Use tracked persons for face detection (only mature tracks)
                for (const auto& track : trackedTracks) {
                    // Only detect faces for confirmed tracks with good confidence
                    if (track.score > 0.5f && track.hits >= 3) {
                        // Get smoothed bounding box from Kalman filter (landscape coords)
                        float tx1, ty1, tx2, ty2;
                        track.getSmoothedBox(tx1, ty1, tx2, ty2);

                        // Convert landscape bbox to portrait coordinates
                        // For 90° CCW rotation: portrait(x,y) = (landY, srcW - 1 - landX)
                        // where srcW = FRAME_WIDTH = 1280
                        float px1 = ty1;                          // portX1 = landY1
                        float py1 = FRAME_WIDTH - 1 - tx2;        // portY1 = W-1-landX2
                        float px2 = ty2;                          // portX2 = landY2
                        float py2 = FRAME_WIDTH - 1 - tx1;        // portY2 = W-1-landX1

                        auto faces = g_insightFace->process(
                            g_gpuRotatedFrame,
                            ROTATED_WIDTH, ROTATED_HEIGHT, ROTATED_WIDTH * 4,  // 720x1280, pitch=720*4
                            px1, py1, px2, py2,
                            0.5f
                        );

                        // Convert face coords back to landscape for display
                        // Inverse: landscape(x,y) = (srcW - 1 - portY, portX)
                        for (auto& face : faces) {
                            // Convert face bbox back to landscape
                            float landX1 = FRAME_WIDTH - 1 - face.face.y2;
                            float landY1 = face.face.x1;
                            float landX2 = FRAME_WIDTH - 1 - face.face.y1;
                            float landY2 = face.face.x2;
                            face.face.x1 = landX1;
                            face.face.y1 = landY1;
                            face.face.x2 = landX2;
                            face.face.y2 = landY2;

                            // Convert landmarks back to landscape
                            for (int l = 0; l < 5; l++) {
                                float portLmX = face.face.landmarks[l][0];
                                float portLmY = face.face.landmarks[l][1];
                                face.face.landmarks[l][0] = FRAME_WIDTH - 1 - portLmY;
                                face.face.landmarks[l][1] = portLmX;
                            }

                            face.personTrackId = track.trackId;
                            g_cachedFaces.push_back(face);
                        }
                    }
                }
            }

            // Use cached results
            allFaces = g_cachedFaces;
        }

        auto faceEnd = std::chrono::high_resolution_clock::now();

        // =======================================================================
        // OSNet ReID for each person (runs every N frames)
        // Updates the track's ReID embedding for identity matching
        // =======================================================================
        if (ENABLE_REID && g_reidEngine != nullptr) {
            g_reidUpdateCounter++;

            // Update ReID embeddings periodically
            if (g_reidUpdateCounter >= REID_UPDATE_INTERVAL) {
                g_reidUpdateCounter = 0;

                // Compute ReID for each mature track
                for (auto& track : trackedTracks) {
                    if (track.score > 0.5f && track.hits >= 3) {
                        float tx1, ty1, tx2, ty2;
                        track.getSmoothedBox(tx1, ty1, tx2, ty2);

                        float embedding[512];
                        bool success = g_reidEngine->getEmbeddingFromCrop(
                            gpuFramePtr,
                            FRAME_WIDTH, FRAME_HEIGHT, pitch,
                            tx1, ty1, tx2, ty2,
                            embedding
                        );

                        if (success) {
                            track.updateReidEmbedding(embedding, localFrameCount);
                        }
                    }
                }
            }
        }

        // =======================================================================
        // Build result and copy frame to CPU for clients
        // =======================================================================
        {
            std::lock_guard<std::mutex> lock(g_resultMutex);

            // Free previous result
            if (g_latestResult.persons) delete[] g_latestResult.persons;
            if (g_latestResult.faces) delete[] g_latestResult.faces;

            // Timing
            g_latestResult.preprocess_ms = std::chrono::duration<float, std::milli>(
                yoloStart - preprocessStart).count();
            g_latestResult.yolo_ms = std::chrono::duration<float, std::milli>(
                yoloEnd - yoloStart).count();
            g_latestResult.face_ms = std::chrono::duration<float, std::milli>(
                faceEnd - faceStart).count();
            g_latestResult.total_ms = std::chrono::duration<float, std::milli>(
                faceEnd - frameStart).count();

            // Copy tracked persons with smoothed bounding boxes and keypoints
            // Only include tracks with minimum hit count to prevent flickering
            const int MIN_HITS_TO_DISPLAY = 3;  // Track must be confirmed for 3 frames

            // First count how many mature tracks we have
            int matureTrackCount = 0;
            for (const auto& track : trackedTracks) {
                if (track.hits >= MIN_HITS_TO_DISPLAY) {
                    matureTrackCount++;
                }
            }

            g_latestResult.num_persons = matureTrackCount;
            if (g_latestResult.num_persons > 0) {
                g_latestResult.persons = new NativePoseDetection[g_latestResult.num_persons];
                int idx = 0;
                for (const auto& src : trackedTracks) {
                    // Skip tracks that haven't been confirmed yet
                    if (src.hits < MIN_HITS_TO_DISPLAY) {
                        continue;
                    }

                    auto& dst = g_latestResult.persons[idx++];

                    // Use smoothed bounding box from Kalman filter
                    src.getSmoothedBox(dst.x1, dst.y1, dst.x2, dst.y2);
                    dst.confidence = src.score;
                    dst.track_id = src.trackId;

                    // Use RAW keypoints from YOLO detection (no smoothing)
                    // This matches Python/cyrusbehr approach - let confidence threshold filter bad detections
                    for (int k = 0; k < 17; k++) {
                        dst.keypoints[k][0] = src.keypoints[k][0];
                        dst.keypoints[k][1] = src.keypoints[k][1];
                        dst.keypoints[k][2] = src.keypoints[k][2];
                    }

                    // Identity matching is done by Python layer
                    // Initialize with empty identity (Python will fill this in based on face matching)
                    dst.identity_label[0] = '\0';
                    dst.identity_confidence = 0.0f;

                    // Copy ReID embedding if available
                    if (src.hasReidEmbedding) {
                        memcpy(dst.reid_embedding, src.reidEmbedding, 512 * sizeof(float));
                        dst.has_reid_embedding = 1;
                    } else {
                        dst.has_reid_embedding = 0;
                    }
                }
            } else {
                g_latestResult.persons = nullptr;
            }

            // Copy faces
            g_latestResult.num_faces = allFaces.size();
            if (g_latestResult.num_faces > 0) {
                g_latestResult.faces = new NativeFaceRecognition[g_latestResult.num_faces];
                for (int i = 0; i < g_latestResult.num_faces; i++) {
                    auto& dst = g_latestResult.faces[i];
                    auto& src = allFaces[i];
                    dst.face.x1 = src.face.x1;
                    dst.face.y1 = src.face.y1;
                    dst.face.x2 = src.face.x2;
                    dst.face.y2 = src.face.y2;
                    dst.face.confidence = src.face.confidence;
                    for (int l = 0; l < 5; l++) {
                        dst.face.landmarks[l][0] = src.face.landmarks[l][0];
                        dst.face.landmarks[l][1] = src.face.landmarks[l][1];
                    }
                    memcpy(dst.embedding, src.embedding, 512 * sizeof(float));
                    dst.quality = src.quality;

                    // Link face to its person track
                    dst.person_track_id = src.personTrackId;

                    // Identity matching is done by Python layer
                    // Initialize with empty identity (Python will fill this in)
                    dst.identity_label[0] = '\0';
                    dst.identity_confidence = 0.0f;
                }
            } else {
                g_latestResult.faces = nullptr;
            }

            // Rotate frame 90° CCW on GPU, then convert to BGR and copy to CPU
            // This sends portrait-oriented frames to Python, eliminating CPU rotation
            launchRotateBGRx90CCW(gpuFramePtr, g_gpuRotatedFrame,
                                  FRAME_WIDTH, FRAME_HEIGHT, pitch, g_stream);
            launchConvertBGRxToBGR(g_gpuRotatedFrame, g_gpuRotatedBGR,
                                    ROTATED_WIDTH, ROTATED_HEIGHT, g_stream);
            cudaMemcpyAsync(g_latestFrameCPU.data(), g_gpuRotatedBGR,
                           ROTATED_WIDTH * ROTATED_HEIGHT * 3, cudaMemcpyDeviceToHost, g_stream);
            cudaStreamSynchronize(g_stream);

            g_resultReady = true;
        }
        g_resultCond.notify_all();

        // =======================================================================
        // Stats
        // =======================================================================
        localFrameCount++;
        g_frameCount++;
        totalInferenceTime += g_latestResult.total_ms;

        if (localFrameCount % 30 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float fps = localFrameCount / elapsed;
            float avgMs = totalInferenceTime / localFrameCount;

            g_avgFps = fps;
            g_avgInferenceMs = avgMs;

            std::cout << "[Stats] FPS: " << fps
                      << " | YOLO: " << g_latestResult.yolo_ms << "ms"
                      << " | Face: " << g_latestResult.face_ms << "ms"
                      << " | Tracked: " << trackedTracks.size()
                      << " | Faces: " << allFaces.size()
                      << " | GPU-native: YES"
                      << std::endl;
        }

        // Cleanup EGL-CUDA resources for this frame
        cuGraphicsUnregisterResource(cudaResource);
        NvBufSurfaceUnMapEglImage(srcSurface, 0);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
    }

    // ==========================================================================
    // Cleanup
    // ==========================================================================
    std::cout << "[Server] Shutting down..." << std::endl;

    // Stop GStreamer
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);

    // Wait for server thread
    serverThread.join();

    // Free GPU resources
    cudaFree(g_gpuFrame);
    cudaFree(g_gpuYoloInput);
    cudaFree(g_gpuYoloTransposed);
    cudaFree(g_gpuBGRFrame);
    cudaFree(g_gpuRotatedFrame);
    cudaFree(g_gpuRotatedBGR);
    cudaFree(g_gpuProcessBGR);
    cudaFree(g_gpuProcessBGRx);
    cudaStreamDestroy(g_stream);

    // Free engines
    delete g_yoloEngine;
    delete g_insightFace;
    if (g_reidEngine) delete g_reidEngine;
    delete g_postProcessor;
    delete g_tracker;

    // Free result
    if (g_latestResult.persons) delete[] g_latestResult.persons;
    if (g_latestResult.faces) delete[] g_latestResult.faces;

    std::cout << "[Server] Goodbye!" << std::endl;
    return 0;
}
