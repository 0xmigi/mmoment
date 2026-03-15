/**
 * test_inference.cpp - Camera capture + TensorRT YOLOv8-pose inference
 *
 * Combines:
 * 1. GStreamer camera capture with GPU MJPEG decode
 * 2. CUDA preprocessing (BGRx → RGB float, resize, pad)
 * 3. TensorRT inference (YOLOv8-pose)
 * 4. Post-processing (NMS, coordinate scaling)
 *
 * Note: Uses system memory output from nvvidconv with cudaMemcpy to GPU
 * (NVMM direct access requires EGL interop which adds complexity)
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>

#include "tensorrt_engine.h"

// External CUDA preprocessing function
extern "C" void launchPreprocessBGRxToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

// GStreamer pipeline for USB camera with GPU decode → system memory output
// GPU decode (nvv4l2decoder) → GPU color convert (nvvidconv) → system memory
const char* CAMERA_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "nvv4l2decoder mjpeg=1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "  // System memory (not NVMM)
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

class InferencePipeline {
public:
    InferencePipeline() : m_pipeline(nullptr), m_sink(nullptr), m_stream(nullptr),
                          m_preprocessBuffer(nullptr), m_cameraGpuBuffer(nullptr) {
        cudaStreamCreate(&m_stream);
    }

    ~InferencePipeline() {
        stop();
        if (m_preprocessBuffer) cudaFree(m_preprocessBuffer);
        if (m_cameraGpuBuffer) cudaFree(m_cameraGpuBuffer);
        if (m_stream) cudaStreamDestroy(m_stream);
    }

    bool init(const std::string& enginePath) {
        // Load TensorRT engine
        if (!m_engine.loadEngine(enginePath)) {
            std::cerr << "Failed to load TensorRT engine" << std::endl;
            return false;
        }

        // Get model input dimensions
        auto inputDims = m_engine.getInputDims();
        m_modelWidth = inputDims.d[3];   // NCHW format
        m_modelHeight = inputDims.d[2];
        m_modelChannels = inputDims.d[1];

        std::cout << "[Pipeline] Model input: " << m_modelChannels << "x"
                  << m_modelHeight << "x" << m_modelWidth << std::endl;

        // Allocate preprocessing buffer (RGB float output)
        size_t bufferSize = m_modelChannels * m_modelHeight * m_modelWidth * sizeof(float);
        cudaMalloc(&m_preprocessBuffer, bufferSize);
        std::cout << "[Pipeline] Allocated preprocess buffer: " << bufferSize / 1024 << " KB" << std::endl;

        // Allocate camera frame GPU buffer (BGRx from camera: 1280x720x4)
        m_camWidth = 1280;
        m_camHeight = 720;
        m_camChannels = 4;  // BGRx
        size_t camBufferSize = m_camWidth * m_camHeight * m_camChannels;
        cudaMalloc(&m_cameraGpuBuffer, camBufferSize);
        std::cout << "[Pipeline] Allocated camera GPU buffer: " << camBufferSize / 1024 << " KB" << std::endl;

        // Initialize GStreamer
        GError* error = nullptr;
        m_pipeline = gst_parse_launch(CAMERA_PIPELINE, &error);
        if (error) {
            std::cerr << "GStreamer error: " << error->message << std::endl;
            g_error_free(error);
            return false;
        }

        m_sink = gst_bin_get_by_name(GST_BIN(m_pipeline), "sink");
        g_object_set(m_sink, "emit-signals", TRUE, nullptr);

        return true;
    }

    bool start() {
        GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start GStreamer pipeline" << std::endl;
            return false;
        }
        std::cout << "[Pipeline] Camera started" << std::endl;
        return true;
    }

    void stop() {
        if (m_pipeline) {
            gst_element_set_state(m_pipeline, GST_STATE_NULL);
            gst_object_unref(m_pipeline);
            m_pipeline = nullptr;
        }
        if (m_sink) {
            gst_object_unref(m_sink);
            m_sink = nullptr;
        }
    }

    // Run one frame through the pipeline
    bool processFrame(std::vector<mmoment::PoseDetection>& detections,
                      double& captureMs, double& uploadMs, double& preprocessMs,
                      double& inferMs, double& postMs) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Pull frame from camera
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(m_sink), GST_SECOND);
        if (!sample) return false;

        auto t1 = std::chrono::high_resolution_clock::now();

        // Get buffer and map to CPU
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }

        // Upload to GPU (async)
        cudaMemcpyAsync(m_cameraGpuBuffer, map.data, map.size, cudaMemcpyHostToDevice, m_stream);

        auto t2 = std::chrono::high_resolution_clock::now();

        // Preprocess: BGRx → RGB float tensor
        float scaleRatio;
        int srcPitch = m_camWidth * m_camChannels;  // 1280 * 4 = 5120 bytes
        launchPreprocessBGRxToRGBFloatNN(
            m_cameraGpuBuffer, m_preprocessBuffer,
            m_camWidth, m_camHeight, srcPitch,
            m_modelWidth, m_modelHeight,
            &scaleRatio,
            m_stream
        );
        cudaStreamSynchronize(m_stream);

        auto t3 = std::chrono::high_resolution_clock::now();

        // Release GStreamer buffer early
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        // Run TensorRT inference
        std::vector<float> outputData;
        if (!m_engine.infer(m_preprocessBuffer, m_modelWidth, m_modelHeight, m_modelChannels, outputData)) {
            return false;
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        // Post-process: decode detections
        auto outputDims = m_engine.getOutputDims();
        int numDetections = outputDims.d[2];  // YOLOv8 output: [1, 56, num_dets]

        // Transpose output from [1, 56, N] to [N, 56] for easier processing
        std::vector<float> transposed(numDetections * 56);
        for (int i = 0; i < numDetections; i++) {
            for (int j = 0; j < 56; j++) {
                transposed[i * 56 + j] = outputData[j * numDetections + i];
            }
        }

        mmoment::YoloPosePostProcessor postProc(0.25f, 0.45f);
        detections = postProc.process(transposed, numDetections, scaleRatio, m_camWidth, m_camHeight);

        auto t5 = std::chrono::high_resolution_clock::now();

        // Calculate timings
        captureMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uploadMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        preprocessMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        inferMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
        postMs = std::chrono::duration<double, std::milli>(t5 - t4).count();

        return true;
    }

private:
    GstElement* m_pipeline;
    GstElement* m_sink;
    cudaStream_t m_stream;

    mmoment::TensorRTEngine m_engine;
    void* m_preprocessBuffer;
    void* m_cameraGpuBuffer;

    int m_modelWidth = 640;
    int m_modelHeight = 640;
    int m_modelChannels = 3;

    int m_camWidth = 1280;
    int m_camHeight = 720;
    int m_camChannels = 4;
};

void printDetections(const std::vector<mmoment::PoseDetection>& detections) {
    for (size_t i = 0; i < detections.size(); i++) {
        const auto& det = detections[i];
        std::cout << "  Person " << i << ": "
                  << "bbox=[" << (int)det.x1 << "," << (int)det.y1 << ","
                  << (int)det.x2 << "," << (int)det.y2 << "] "
                  << "conf=" << det.confidence << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== MMOMENT Native Inference Test ===" << std::endl;

    // Check for engine file argument
    std::string enginePath = "/mnt/nvme/mmoment/app/orin_nano/native/yolov8n-pose-native.engine";
    if (argc > 1) {
        enginePath = argv[1];
    }

    std::cout << "Using engine: " << enginePath << std::endl;

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Create pipeline
    InferencePipeline pipeline;
    if (!pipeline.init(enginePath)) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return 1;
    }

    if (!pipeline.start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Running inference (10 seconds)..." << std::endl;
    std::cout << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    double totalCapture = 0, totalUpload = 0, totalPreproc = 0, totalInfer = 0, totalPost = 0;

    while (true) {
        std::vector<mmoment::PoseDetection> detections;
        double captureMs, uploadMs, preprocessMs, inferMs, postMs;

        if (!pipeline.processFrame(detections, captureMs, uploadMs, preprocessMs, inferMs, postMs)) {
            std::cerr << "Frame processing failed" << std::endl;
            continue;
        }

        frameCount++;
        totalCapture += captureMs;
        totalUpload += uploadMs;
        totalPreproc += preprocessMs;
        totalInfer += inferMs;
        totalPost += postMs;

        // Print stats every 30 frames
        if (frameCount % 30 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();
            double fps = frameCount / elapsed;

            std::cout << "[Frame " << frameCount << "] "
                      << "FPS: " << fps << " | "
                      << "Cap: " << (totalCapture / frameCount) << "ms | "
                      << "Upload: " << (totalUpload / frameCount) << "ms | "
                      << "Prep: " << (totalPreproc / frameCount) << "ms | "
                      << "Infer: " << (totalInfer / frameCount) << "ms | "
                      << "Post: " << (totalPost / frameCount) << "ms | "
                      << "Dets: " << detections.size() << std::endl;

            if (!detections.empty()) {
                printDetections(detections);
            }
        }

        // Stop after 10 seconds
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - startTime).count() >= 10.0) break;
    }

    pipeline.stop();

    std::cout << std::endl;
    std::cout << "=== Final Stats ===" << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    std::cout << "Avg Capture: " << (totalCapture / frameCount) << " ms" << std::endl;
    std::cout << "Avg Upload: " << (totalUpload / frameCount) << " ms" << std::endl;
    std::cout << "Avg Preprocess: " << (totalPreproc / frameCount) << " ms" << std::endl;
    std::cout << "Avg Inference: " << (totalInfer / frameCount) << " ms" << std::endl;
    std::cout << "Avg Postprocess: " << (totalPost / frameCount) << " ms" << std::endl;
    double totalPerFrame = (totalCapture + totalUpload + totalPreproc + totalInfer + totalPost) / frameCount;
    std::cout << "Total per frame: " << totalPerFrame << " ms" << std::endl;
    std::cout << "Theoretical FPS: " << (1000.0 / totalPerFrame) << std::endl;

    return 0;
}
