/**
 * test_camera.cpp - GStreamer camera capture with NVIDIA acceleration
 *
 * This test validates:
 * 1. Camera capture via GStreamer
 * 2. GPU-accelerated format conversion via nvvidconv
 * 3. Zero-copy NVMM buffer handling
 * 4. Frame timing and throughput
 *
 * For NVMM buffers, we use NvBufSurface API to access GPU memory directly.
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>

// NVIDIA buffer surface API for NVMM memory access
extern "C" {
#include "nvbufsurface.h"
}

// Pipeline configurations
// USB Camera: v4l2src → nvvidconv (GPU scaling) → appsink
const char* USB_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw(memory:NVMM), format=NV12 ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

// Alternative: Convert to RGBA on GPU for easier processing
const char* USB_PIPELINE_RGBA =
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw(memory:NVMM), format=RGBA, width=640, height=640 ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

// GPU-accelerated MJPEG decode + GPU color conversion
// This is the optimal path for USB cameras on Jetson
const char* GPU_MJPEG_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "nvv4l2decoder mjpeg=1 ! "                    // GPU JPEG decode → NV12 in NVMM
    "nvvidconv ! "                                 // GPU color convert
    "video/x-raw(memory:NVMM), format=BGRx ! "    // Keep in GPU memory
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

// Fallback: CPU JPEG decode (if GPU decode fails)
const char* SIMPLE_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "jpegdec ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

class GStreamerCapture {
public:
    GStreamerCapture() : m_pipeline(nullptr), m_sink(nullptr), m_frameCount(0) {}

    ~GStreamerCapture() {
        stop();
    }

    bool start(const char* pipelineStr) {
        GError* error = nullptr;

        std::cout << "[GStreamer] Creating pipeline..." << std::endl;
        std::cout << "[GStreamer] " << pipelineStr << std::endl;

        m_pipeline = gst_parse_launch(pipelineStr, &error);
        if (error) {
            std::cerr << "[GStreamer] Pipeline error: " << error->message << std::endl;
            g_error_free(error);
            return false;
        }

        // Get appsink element
        m_sink = gst_bin_get_by_name(GST_BIN(m_pipeline), "sink");
        if (!m_sink) {
            std::cerr << "[GStreamer] Failed to get appsink" << std::endl;
            return false;
        }

        // Configure appsink
        g_object_set(m_sink, "emit-signals", TRUE, nullptr);

        // Start pipeline
        GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[GStreamer] Failed to start pipeline" << std::endl;
            return false;
        }

        std::cout << "[GStreamer] Pipeline started successfully" << std::endl;
        return true;
    }

    // Pull a frame and return timing info
    // For NVMM buffers, extracts NvBufSurface pointer for zero-copy GPU access
    bool pullFrame(double& captureMs, size_t& dataSize, bool& isNvmm, void** gpuPtr = nullptr) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Pull sample with timeout
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(m_sink), GST_SECOND);
        if (!sample) {
            return false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        // Get buffer info
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);

        // Map buffer to check memory type
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            // Check if this is NVMM memory by looking at the buffer contents
            // For NVMM, the mapped data contains a pointer to NvBufSurface
            NvBufSurface* nvbuf = (NvBufSurface*)map.data;

            // Validate it looks like a real NvBufSurface
            if (map.size >= sizeof(NvBufSurface) && nvbuf->numFilled > 0) {
                isNvmm = true;
                dataSize = nvbuf->surfaceList[0].dataSize;

                if (gpuPtr) {
                    *gpuPtr = nvbuf->surfaceList[0].dataPtr;
                }

                // Print detailed info on first frame
                if (m_frameCount == 0) {
                    std::cout << "[NVMM] Surface info:" << std::endl;
                    std::cout << "  - Size: " << nvbuf->surfaceList[0].width
                              << "x" << nvbuf->surfaceList[0].height << std::endl;
                    std::cout << "  - Pitch: " << nvbuf->surfaceList[0].pitch << std::endl;
                    std::cout << "  - Color format: " << nvbuf->surfaceList[0].colorFormat << std::endl;
                    std::cout << "  - Data size: " << nvbuf->surfaceList[0].dataSize << " bytes" << std::endl;
                    std::cout << "  - GPU ptr: " << nvbuf->surfaceList[0].dataPtr << std::endl;
                    std::cout << "  - Memory type: " << nvbuf->memType << std::endl;
                }
            } else {
                // Regular CPU buffer
                isNvmm = false;
                dataSize = map.size;
            }

            gst_buffer_unmap(buffer, &map);
        } else {
            isNvmm = false;
            dataSize = gst_buffer_get_size(buffer);
        }

        // Get caps info on first frame
        if (caps && m_frameCount == 0) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            const gchar* format = gst_structure_get_string(structure, "format");
            gint width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);

            std::cout << "[GStreamer] Caps: " << (format ? format : "unknown")
                      << ", " << width << "x" << height << std::endl;
        }

        gst_sample_unref(sample);
        m_frameCount++;

        captureMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
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

    int getFrameCount() const { return m_frameCount; }

private:
    GstElement* m_pipeline;
    GstElement* m_sink;
    int m_frameCount;
};

void printCudaInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "[CUDA] No devices found" << std::endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "[CUDA] Device: " << prop.name << std::endl;
    std::cout << "[CUDA] Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "[CUDA] Compute: " << prop.major << "." << prop.minor << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== MMOMENT Native Camera Test ===" << std::endl;
    std::cout << std::endl;

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Print CUDA info
    printCudaInfo();
    std::cout << std::endl;

    // Try GPU pipeline first, fallback to CPU
    const char* pipelines[] = {
        GPU_MJPEG_PIPELINE,  // GPU decode + GPU conversion (optimal)
        SIMPLE_PIPELINE,     // CPU fallback
    };

    GStreamerCapture capture;

    for (const char* pipeline : pipelines) {
        std::cout << "=== Testing Pipeline ===" << std::endl;

        if (!capture.start(pipeline)) {
            std::cout << "Pipeline failed to start, trying next..." << std::endl;
            capture.stop();
            continue;
        }

        std::cout << std::endl;
        std::cout << "Capturing frames (10 seconds)..." << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();
        double totalCaptureMs = 0;
        int frames = 0;
        int failedPulls = 0;

        while (true) {
            double captureMs;
            size_t dataSize;
            bool isNvmm;

            if (!capture.pullFrame(captureMs, dataSize, isNvmm)) {
                failedPulls++;
                if (failedPulls > 5) {
                    std::cerr << "Too many failed pulls, trying next pipeline..." << std::endl;
                    break;
                }
                continue;
            }

            failedPulls = 0;  // Reset on success
            totalCaptureMs += captureMs;
            frames++;

            // Print stats every second
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();

            if (frames % 30 == 0) {
                double fps = frames / elapsed;
                double avgCapture = totalCaptureMs / frames;
                std::cout << "[Stats] Frames: " << frames
                          << ", FPS: " << fps
                          << ", Avg capture: " << avgCapture << "ms"
                          << ", NVMM: " << (isNvmm ? "YES" : "NO")
                          << std::endl;
            }

            // Stop after 10 seconds
            if (elapsed >= 10.0) break;
        }

        capture.stop();

        if (frames > 0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startTime).count();
            std::cout << std::endl;
            std::cout << "Results: " << frames << " frames in " << elapsed << "s"
                      << " = " << frames/elapsed << " FPS" << std::endl;
            std::cout << std::endl;
            break;  // Success - stop trying pipelines
        }
    }

    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}
