/**
 * test_full_pipeline.cpp - Full native pipeline test
 *
 * Camera -> YOLOv8-pose (person detection) -> InsightFace (face recognition)
 *
 * This demonstrates the complete native C++ inference pipeline with
 * minimal CPU<->GPU memory transfers.
 */

#include <iostream>
#include <chrono>
#include <signal.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>

#include "tensorrt_engine.h"
#include "insightface_engine.h"

using namespace mmoment;

static volatile bool g_running = true;

void signalHandler(int sig) {
    std::cout << "\nStopping..." << std::endl;
    g_running = false;
}

// GStreamer pipeline - system memory output for maximum compatibility
const char* CAMERA_PIPELINE =
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1280, height=720, framerate=30/1 ! "
    "nvv4l2decoder mjpeg=1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true";

// Preprocessing kernel declaration (single scale ratio output)
extern "C" void launchPreprocessBGRxToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

int main(int argc, char* argv[]) {
    std::cout << "=== MMOMENT Full Native Pipeline Test ===" << std::endl;
    std::cout << "Camera + YOLOv8-pose + InsightFace" << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;

    signal(SIGINT, signalHandler);

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Load YOLOv8-pose engine
    TensorRTEngine yoloEngine;
    std::cout << "Loading YOLOv8-pose engine..." << std::endl;
    if (!yoloEngine.loadEngine("yolov8n-pose-native.engine")) {
        std::cerr << "Failed to load YOLO engine" << std::endl;
        return 1;
    }

    // Initialize InsightFace pipeline
    InsightFacePipeline insightFace;
    std::cout << "Loading InsightFace engines..." << std::endl;
    if (!insightFace.initialize("retinaface.engine", "arcface_r50.engine")) {
        std::cerr << "Failed to initialize InsightFace" << std::endl;
        return 1;
    }

    // Create GStreamer pipeline
    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(CAMERA_PIPELINE, &error);
    if (error) {
        std::cerr << "Pipeline error: " << error->message << std::endl;
        g_error_free(error);
        return 1;
    }

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Allocate GPU buffers
    const int FRAME_W = 1280;
    const int FRAME_H = 720;
    const int YOLO_INPUT_SIZE = 640;

    void* gpuFrame = nullptr;
    void* gpuYoloInput = nullptr;

    cudaMalloc(&gpuFrame, FRAME_W * FRAME_H * 4);  // BGRx
    cudaMalloc(&gpuYoloInput, YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3 * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create YOLO post-processor
    YoloPosePostProcessor postProcessor(0.5f, 0.45f);  // conf threshold, NMS threshold

    // Stats
    int frameCount = 0;
    int detectionCount = 0;
    int faceCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    float totalYoloTime = 0;
    float totalFaceTime = 0;

    std::cout << "Pipeline running...\n" << std::endl;

    while (g_running) {
        // Pull frame from camera
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 100 * GST_MSECOND);
        if (!sample) continue;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_READ);

        // Upload frame to GPU
        cudaMemcpyAsync(gpuFrame, map.data, FRAME_W * FRAME_H * 4, cudaMemcpyHostToDevice, stream);

        // =========================================================================
        // Stage 1: YOLOv8-pose person detection
        // =========================================================================
        auto yoloStart = std::chrono::high_resolution_clock::now();

        // Preprocess for YOLO
        float scaleRatio;
        launchPreprocessBGRxToRGBFloatNN(
            gpuFrame, gpuYoloInput,
            FRAME_W, FRAME_H, FRAME_W * 4,
            YOLO_INPUT_SIZE, YOLO_INPUT_SIZE,
            &scaleRatio,
            stream
        );
        cudaStreamSynchronize(stream);

        // Run YOLO inference
        std::vector<float> yoloOutput;
        yoloEngine.infer(gpuYoloInput, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3, yoloOutput);

        // Post-process YOLO output - need to transpose [1, 56, 8400] -> [8400, 56]
        const int NUM_DETS = 8400;
        std::vector<float> transposed(NUM_DETS * 56);
        for (int i = 0; i < NUM_DETS; i++) {
            for (int j = 0; j < 56; j++) {
                transposed[i * 56 + j] = yoloOutput[j * NUM_DETS + i];
            }
        }
        auto detections = postProcessor.process(transposed, NUM_DETS, scaleRatio, FRAME_W, FRAME_H);

        auto yoloEnd = std::chrono::high_resolution_clock::now();
        float yoloMs = std::chrono::duration<float, std::milli>(yoloEnd - yoloStart).count();
        totalYoloTime += yoloMs;

        // =========================================================================
        // Stage 2: InsightFace for each person detection
        // (Currently disabled - RetinaFace postprocessing not implemented yet)
        // =========================================================================
        auto faceStart = std::chrono::high_resolution_clock::now();

        std::vector<FaceRecognition> allFaces;

        // Process each person detection with InsightFace
        for (const auto& det : detections) {
            // Only process person detections (class 0) with high confidence
            if (det.classId == 0 && det.confidence > 0.5f) {
                // Run InsightFace on person bbox
                auto faces = insightFace.process(
                    gpuFrame,
                    FRAME_W, FRAME_H, FRAME_W * 4,
                    det.x1, det.y1, det.x2, det.y2,
                    0.5f  // Face detection threshold
                );

                for (auto& face : faces) {
                    allFaces.push_back(face);
                }
            }
        }

        auto faceEnd = std::chrono::high_resolution_clock::now();
        float faceMs = std::chrono::duration<float, std::milli>(faceEnd - faceStart).count();
        if (!detections.empty()) {
            totalFaceTime += faceMs;
        }

        // =========================================================================
        // Stats and output
        // =========================================================================
        frameCount++;
        detectionCount += detections.size();
        faceCount += allFaces.size();

        // Print periodic stats
        if (frameCount % 30 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float fps = frameCount / elapsed;
            float avgYolo = totalYoloTime / frameCount;
            float avgFace = detectionCount > 0 ? totalFaceTime / frameCount : 0;

            std::cout << "FPS: " << fps
                      << " | YOLO: " << avgYolo << "ms"
                      << " | Face: " << avgFace << "ms"
                      << " | Persons: " << detections.size()
                      << " | Faces: " << allFaces.size();

            // Print embedding preview for first face
            if (!allFaces.empty()) {
                std::cout << " | Emb[0:3]: ["
                          << allFaces[0].embedding[0] << ", "
                          << allFaces[0].embedding[1] << ", "
                          << allFaces[0].embedding[2] << "]";
            }

            std::cout << std::endl;
        }

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
    }

    // Final stats
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float>(endTime - startTime).count();

    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    std::cout << "Total time: " << totalTime << "s" << std::endl;
    std::cout << "Average FPS: " << frameCount / totalTime << std::endl;
    std::cout << "Total persons detected: " << detectionCount << std::endl;
    std::cout << "Total faces recognized: " << faceCount << std::endl;
    std::cout << "Avg YOLO time: " << totalYoloTime / frameCount << "ms" << std::endl;
    if (detectionCount > 0) {
        std::cout << "Avg Face time: " << totalFaceTime / frameCount << "ms" << std::endl;
    }

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);

    cudaFree(gpuFrame);
    cudaFree(gpuYoloInput);
    cudaStreamDestroy(stream);

    std::cout << "Done." << std::endl;
    return 0;
}
