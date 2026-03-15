/**
 * native_pipeline_api.cpp - C API implementation for native pipeline
 */

#include "native_pipeline_api.h"
#include "tensorrt_engine.h"
#include "insightface_engine.h"

#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <mutex>

// Global state
static mmoment::TensorRTEngine* g_yoloEngine = nullptr;
static mmoment::InsightFacePipeline* g_insightFace = nullptr;
static mmoment::YoloPosePostProcessor* g_postProcessor = nullptr;

static void* g_gpuFrame = nullptr;
static void* g_gpuYoloInput = nullptr;
static cudaStream_t g_stream = nullptr;

static float g_personConfThreshold = 0.5f;
static float g_faceConfThreshold = 0.5f;
static bool g_faceRecognitionEnabled = true;
static bool g_initialized = false;
static std::mutex g_mutex;

static const char* VERSION = "1.0.0-native";

// External CUDA preprocessing function
extern "C" void launchPreprocessBGRToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

extern "C" void launchPreprocessBGRxToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

// ============================================================================
// Lifecycle
// ============================================================================

int native_pipeline_init(
    const char* yolo_engine_path,
    const char* retinaface_engine_path,
    const char* arcface_engine_path
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized) {
        std::cerr << "[Native] Pipeline already initialized" << std::endl;
        return 0;
    }

    std::cout << "[Native] Initializing pipeline..." << std::endl;

    // Create CUDA stream
    cudaStreamCreate(&g_stream);

    // Load YOLO engine
    g_yoloEngine = new mmoment::TensorRTEngine();
    if (!g_yoloEngine->loadEngine(yolo_engine_path)) {
        std::cerr << "[Native] Failed to load YOLO engine: " << yolo_engine_path << std::endl;
        return -1;
    }
    std::cout << "[Native] YOLO engine loaded" << std::endl;

    // Load InsightFace
    g_insightFace = new mmoment::InsightFacePipeline();
    if (!g_insightFace->initialize(retinaface_engine_path, arcface_engine_path)) {
        std::cerr << "[Native] Failed to load InsightFace engines" << std::endl;
        return -2;
    }
    std::cout << "[Native] InsightFace engines loaded" << std::endl;

    // Create post-processor
    g_postProcessor = new mmoment::YoloPosePostProcessor(g_personConfThreshold, 0.45f);

    // Allocate GPU buffers (max 1920x1080 input)
    const int MAX_WIDTH = 1920;
    const int MAX_HEIGHT = 1080;
    const int YOLO_SIZE = 640;

    cudaMalloc(&g_gpuFrame, MAX_WIDTH * MAX_HEIGHT * 4);  // BGRA
    cudaMalloc(&g_gpuYoloInput, YOLO_SIZE * YOLO_SIZE * 3 * sizeof(float));

    g_initialized = true;
    std::cout << "[Native] Pipeline initialized successfully" << std::endl;

    return 0;
}

int native_pipeline_is_ready(void) {
    return g_initialized ? 1 : 0;
}

void native_pipeline_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) return;

    std::cout << "[Native] Shutting down pipeline..." << std::endl;

    delete g_yoloEngine;
    delete g_insightFace;
    delete g_postProcessor;

    g_yoloEngine = nullptr;
    g_insightFace = nullptr;
    g_postProcessor = nullptr;

    if (g_gpuFrame) cudaFree(g_gpuFrame);
    if (g_gpuYoloInput) cudaFree(g_gpuYoloInput);
    if (g_stream) cudaStreamDestroy(g_stream);

    g_gpuFrame = nullptr;
    g_gpuYoloInput = nullptr;
    g_stream = nullptr;

    g_initialized = false;
    std::cout << "[Native] Pipeline shutdown complete" << std::endl;
}

// ============================================================================
// Frame Processing
// ============================================================================

static int process_frame_internal(
    const uint8_t* frame_data,
    int width,
    int height,
    int stride,
    bool is_bgra,
    NativeFrameResult* result
) {
    if (!g_initialized || !result) return -1;

    std::lock_guard<std::mutex> lock(g_mutex);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Upload frame to GPU
    size_t frame_size = stride * height;
    cudaMemcpyAsync(g_gpuFrame, frame_data, frame_size, cudaMemcpyHostToDevice, g_stream);

    auto t_upload = std::chrono::high_resolution_clock::now();

    // Preprocess for YOLO
    const int YOLO_SIZE = 640;
    float scaleRatio;

    if (is_bgra) {
        launchPreprocessBGRxToRGBFloatNN(
            g_gpuFrame, g_gpuYoloInput,
            width, height, stride,
            YOLO_SIZE, YOLO_SIZE,
            &scaleRatio,
            g_stream
        );
    } else {
        launchPreprocessBGRToRGBFloatNN(
            g_gpuFrame, g_gpuYoloInput,
            width, height, stride,
            YOLO_SIZE, YOLO_SIZE,
            &scaleRatio,
            g_stream
        );
    }
    cudaStreamSynchronize(g_stream);

    auto t_preprocess = std::chrono::high_resolution_clock::now();

    // YOLO inference
    std::vector<float> yoloOutput;
    g_yoloEngine->infer(g_gpuYoloInput, YOLO_SIZE, YOLO_SIZE, 3, yoloOutput);

    // Transpose output [1, 56, 8400] -> [8400, 56]
    const int NUM_DETS = 8400;
    std::vector<float> transposed(NUM_DETS * 56);
    for (int i = 0; i < NUM_DETS; i++) {
        for (int j = 0; j < 56; j++) {
            transposed[i * 56 + j] = yoloOutput[j * NUM_DETS + i];
        }
    }

    // Post-process
    auto detections = g_postProcessor->process(transposed, NUM_DETS, scaleRatio, width, height);

    auto t_yolo = std::chrono::high_resolution_clock::now();

    // Filter by confidence and allocate result
    std::vector<mmoment::PoseDetection> persons;
    for (const auto& det : detections) {
        if (det.classId == 0 && det.confidence >= g_personConfThreshold) {
            persons.push_back(det);
        }
    }

    result->num_persons = persons.size();
    if (result->num_persons > 0) {
        result->persons = new NativePoseDetection[result->num_persons];
        for (int i = 0; i < result->num_persons; i++) {
            result->persons[i].x1 = persons[i].x1;
            result->persons[i].y1 = persons[i].y1;
            result->persons[i].x2 = persons[i].x2;
            result->persons[i].y2 = persons[i].y2;
            result->persons[i].confidence = persons[i].confidence;
            result->persons[i].track_id = -1;  // Not implemented yet

            // Copy keypoints
            for (int k = 0; k < 17; k++) {
                result->persons[i].keypoints[k][0] = persons[i].keypoints[k][0];
                result->persons[i].keypoints[k][1] = persons[i].keypoints[k][1];
                result->persons[i].keypoints[k][2] = persons[i].keypoints[k][2];
            }
        }
    } else {
        result->persons = nullptr;
    }

    // Face recognition (if enabled)
    std::vector<mmoment::FaceRecognition> allFaces;

    if (g_faceRecognitionEnabled && result->num_persons > 0) {
        for (const auto& person : persons) {
            auto faces = g_insightFace->process(
                g_gpuFrame,
                width, height, stride,
                person.x1, person.y1, person.x2, person.y2,
                g_faceConfThreshold
            );

            for (auto& face : faces) {
                allFaces.push_back(face);
            }
        }
    }

    auto t_face = std::chrono::high_resolution_clock::now();

    // Copy face results
    result->num_faces = allFaces.size();
    if (result->num_faces > 0) {
        result->faces = new NativeFaceRecognition[result->num_faces];
        for (int i = 0; i < result->num_faces; i++) {
            result->faces[i].face.x1 = allFaces[i].face.x1;
            result->faces[i].face.y1 = allFaces[i].face.y1;
            result->faces[i].face.x2 = allFaces[i].face.x2;
            result->faces[i].face.y2 = allFaces[i].face.y2;
            result->faces[i].face.confidence = allFaces[i].face.confidence;

            for (int j = 0; j < 5; j++) {
                result->faces[i].face.landmarks[j][0] = allFaces[i].face.landmarks[j][0];
                result->faces[i].face.landmarks[j][1] = allFaces[i].face.landmarks[j][1];
            }

            memcpy(result->faces[i].embedding, allFaces[i].embedding, 512 * sizeof(float));
            result->faces[i].quality = allFaces[i].quality;
        }
    } else {
        result->faces = nullptr;
    }

    // Timing
    auto t_end = std::chrono::high_resolution_clock::now();

    result->preprocess_ms = std::chrono::duration<float, std::milli>(t_preprocess - t_upload).count();
    result->yolo_ms = std::chrono::duration<float, std::milli>(t_yolo - t_preprocess).count();
    result->face_ms = std::chrono::duration<float, std::milli>(t_face - t_yolo).count();
    result->total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    return 0;
}

int native_pipeline_process_bgr(
    const uint8_t* frame_data,
    int width,
    int height,
    int stride,
    NativeFrameResult* result
) {
    return process_frame_internal(frame_data, width, height, stride, false, result);
}

int native_pipeline_process_bgra(
    const uint8_t* frame_data,
    int width,
    int height,
    int stride,
    NativeFrameResult* result
) {
    return process_frame_internal(frame_data, width, height, stride, true, result);
}

void native_pipeline_free_result(NativeFrameResult* result) {
    if (!result) return;

    if (result->persons) {
        delete[] result->persons;
        result->persons = nullptr;
    }

    if (result->faces) {
        delete[] result->faces;
        result->faces = nullptr;
    }

    result->num_persons = 0;
    result->num_faces = 0;
}

// ============================================================================
// Configuration
// ============================================================================

void native_pipeline_set_thresholds(float person_conf, float face_conf) {
    g_personConfThreshold = person_conf;
    g_faceConfThreshold = face_conf;

    if (g_postProcessor) {
        delete g_postProcessor;
        g_postProcessor = new mmoment::YoloPosePostProcessor(person_conf, 0.45f);
    }
}

void native_pipeline_enable_face_recognition(int enabled) {
    g_faceRecognitionEnabled = (enabled != 0);
}

// ============================================================================
// Utility
// ============================================================================

float native_compute_similarity(const float* emb1, const float* emb2) {
    return mmoment::InsightFacePipeline::cosineSimilarity(emb1, emb2);
}

const char* native_pipeline_version(void) {
    return VERSION;
}
