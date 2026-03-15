/**
 * insightface_engine.cpp - InsightFace TensorRT inference implementation
 */

#include "insightface_engine.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace mmoment {

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

// ============================================================================
// RetinaFace Engine Implementation
// ============================================================================

RetinaFaceEngine::RetinaFaceEngine() : m_logger(std::make_unique<TRTLogger>()) {
    cudaStreamCreate(&m_stream);
}

RetinaFaceEngine::~RetinaFaceEngine() {
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

bool RetinaFaceEngine::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    if (!m_runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Get input/output tensor info
    int numIO = m_engine->getNbIOTensors();
    std::cout << "RetinaFace engine has " << numIO << " I/O tensors:" << std::endl;

    for (int i = 0; i < numIO; i++) {
        const char* name = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(name);
        auto dims = m_engine->getTensorShape(name);

        std::cout << "  " << name << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "]: ";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            m_inputDims = dims;
        }
    }

    return allocateBuffers();
}

bool RetinaFaceEngine::allocateBuffers() {
    // Input: 1x3x640x640
    m_inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    cudaMalloc(&m_inputBuffer, m_inputSize);

    // Output: RetinaFace has multiple output tensors, we'll handle them in inference
    // For now allocate a large buffer for all outputs
    m_outputSize = 1024 * 1024 * sizeof(float);  // 4MB should be enough
    cudaMalloc(&m_outputBuffer, m_outputSize);

    return m_inputBuffer != nullptr && m_outputBuffer != nullptr;
}

void RetinaFaceEngine::freeBuffers() {
    if (m_inputBuffer) { cudaFree(m_inputBuffer); m_inputBuffer = nullptr; }
    if (m_outputBuffer) { cudaFree(m_outputBuffer); m_outputBuffer = nullptr; }
}

std::vector<FaceDetection> RetinaFaceEngine::detect(void* gpuInput, int inputWidth, int inputHeight,
                                                      float scaleRatio, float confThreshold) {
    // Copy preprocessed input to our buffer
    cudaMemcpyAsync(m_inputBuffer, gpuInput, m_inputSize, cudaMemcpyDeviceToDevice, m_stream);

    // Set input tensor address
    const char* inputName = m_engine->getIOTensorName(0);
    m_context->setTensorAddress(inputName, m_inputBuffer);

    // RetinaFace det_10g output structure (3 FPN levels with 2 anchors each):
    // Stride 8:  80x80x2 = 12800 anchors
    // Stride 16: 40x40x2 = 3200 anchors
    // Stride 32: 20x20x2 = 800 anchors
    // Outputs per level: scores (Nx1), bbox (Nx4), landmarks (Nx10)
    struct FPNLevel {
        int numAnchors;
        int stride;
        int featW, featH;
        std::vector<float> scores;
        std::vector<float> bbox;
        std::vector<float> landmarks;
    };

    FPNLevel levels[3] = {
        {12800, 8, 80, 80, {}, {}, {}},
        {3200, 16, 40, 40, {}, {}, {}},
        {800, 32, 20, 20, {}, {}, {}}
    };

    // Allocate output buffers and set tensor addresses
    // Output order from model: scores (3 levels), bbox (3 levels), landmarks (3 levels)
    int numIO = m_engine->getNbIOTensors();
    std::vector<void*> outputBuffers;
    std::vector<size_t> outputSizes;
    std::vector<std::string> outputNames;
    std::vector<nvinfer1::Dims> outputDims;

    for (int i = 0; i < numIO; i++) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = m_engine->getTensorShape(name);
            size_t size = 1;
            for (int d = 0; d < dims.nbDims; d++) {
                size *= dims.d[d];
            }
            size *= sizeof(float);

            void* buffer;
            cudaMalloc(&buffer, size);
            outputBuffers.push_back(buffer);
            outputSizes.push_back(size);
            outputNames.push_back(name);
            outputDims.push_back(dims);

            m_context->setTensorAddress(name, buffer);
        }
    }

    // Run inference
    bool success = m_context->enqueueV3(m_stream);
    cudaStreamSynchronize(m_stream);

    if (!success) {
        std::cerr << "RetinaFace inference failed" << std::endl;
        for (auto buf : outputBuffers) cudaFree(buf);
        return {};
    }

    // Copy outputs to host and organize by type
    // Based on output shapes: Nx1 = scores, Nx4 = bbox, Nx10 = landmarks
    for (size_t i = 0; i < outputBuffers.size(); i++) {
        int numElements = outputSizes[i] / sizeof(float);
        int numAnchors = outputDims[i].d[0];
        int channels = outputDims[i].nbDims > 1 ? outputDims[i].d[1] : 1;

        std::vector<float> hostData(numElements);
        cudaMemcpy(hostData.data(), outputBuffers[i], outputSizes[i], cudaMemcpyDeviceToHost);

        // Find matching FPN level
        for (int l = 0; l < 3; l++) {
            if (numAnchors == levels[l].numAnchors) {
                if (channels == 1) {
                    levels[l].scores = std::move(hostData);
                } else if (channels == 4) {
                    levels[l].bbox = std::move(hostData);
                } else if (channels == 10) {
                    levels[l].landmarks = std::move(hostData);
                }
                break;
            }
        }
    }

    // Free output buffers
    for (auto buf : outputBuffers) cudaFree(buf);

    // Decode anchors and apply NMS
    std::vector<FaceDetection> faces;

    // Anchor sizes for RetinaFace (2 anchors per position)
    const float anchorSizes[3][2] = {
        {16.0f, 32.0f},   // stride 8
        {64.0f, 128.0f},  // stride 16
        {256.0f, 512.0f}  // stride 32
    };

    for (int l = 0; l < 3; l++) {
        if (levels[l].scores.empty() || levels[l].bbox.empty()) continue;

        int stride = levels[l].stride;
        int featW = levels[l].featW;
        int featH = levels[l].featH;

        for (int y = 0; y < featH; y++) {
            for (int x = 0; x < featW; x++) {
                for (int a = 0; a < 2; a++) {  // 2 anchors per position
                    int idx = (y * featW + x) * 2 + a;

                    // Score (sigmoid already applied in model)
                    float score = levels[l].scores[idx];
                    if (score < confThreshold) continue;

                    // Anchor center
                    float anchorCx = (x + 0.5f) * stride;
                    float anchorCy = (y + 0.5f) * stride;
                    float anchorSize = anchorSizes[l][a];

                    // Decode bbox using DISTANCE format (det_10g style)
                    // The 4 values are distances from anchor center: left, top, right, bottom
                    float left = levels[l].bbox[idx * 4 + 0];
                    float top = levels[l].bbox[idx * 4 + 1];
                    float right = levels[l].bbox[idx * 4 + 2];
                    float bottom = levels[l].bbox[idx * 4 + 3];

                    // Convert distances to bbox coordinates
                    // Distances are in feature map space, multiply by stride
                    FaceDetection face;
                    face.x1 = (anchorCx - left * stride) * scaleRatio;
                    face.y1 = (anchorCy - top * stride) * scaleRatio;
                    face.x2 = (anchorCx + right * stride) * scaleRatio;
                    face.y2 = (anchorCy + bottom * stride) * scaleRatio;
                    face.confidence = score;

                    // Decode landmarks (5 points, each with dx, dy in stride units)
                    if (!levels[l].landmarks.empty()) {
                        for (int p = 0; p < 5; p++) {
                            float ldx = levels[l].landmarks[idx * 10 + p * 2 + 0];
                            float ldy = levels[l].landmarks[idx * 10 + p * 2 + 1];
                            // Landmarks are offsets from anchor center, scaled by stride
                            face.landmarks[p][0] = (anchorCx + ldx * stride) * scaleRatio;
                            face.landmarks[p][1] = (anchorCy + ldy * stride) * scaleRatio;
                        }
                    }

                    faces.push_back(face);
                }
            }
        }
    }

    // Apply NMS
    if (faces.size() > 1) {
        // Sort by confidence
        std::sort(faces.begin(), faces.end(),
            [](const FaceDetection& a, const FaceDetection& b) {
                return a.confidence > b.confidence;
            });

        std::vector<FaceDetection> nmsResult;
        std::vector<bool> suppressed(faces.size(), false);

        for (size_t i = 0; i < faces.size(); i++) {
            if (suppressed[i]) continue;
            nmsResult.push_back(faces[i]);

            for (size_t j = i + 1; j < faces.size(); j++) {
                if (suppressed[j]) continue;

                // Calculate IoU
                float x1 = std::max(faces[i].x1, faces[j].x1);
                float y1 = std::max(faces[i].y1, faces[j].y1);
                float x2 = std::min(faces[i].x2, faces[j].x2);
                float y2 = std::min(faces[i].y2, faces[j].y2);

                float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
                float area1 = (faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1);
                float area2 = (faces[j].x2 - faces[j].x1) * (faces[j].y2 - faces[j].y1);
                float iou = interArea / (area1 + area2 - interArea + 1e-6f);

                if (iou > 0.4f) {
                    suppressed[j] = true;
                }
            }
        }
        faces = std::move(nmsResult);
    }

    return faces;
}

std::vector<FaceDetection> RetinaFaceEngine::postProcess(const std::vector<float>& output,
                                                          int inputW, int inputH,
                                                          float scaleRatio, float confThreshold) {
    // This is a placeholder - actual implementation depends on RetinaFace output format
    // The det_10g model from InsightFace has specific anchor configurations
    std::vector<FaceDetection> faces;
    return faces;
}

// ============================================================================
// SCRFD Engine Implementation (better small face detection)
// ============================================================================

SCRFDEngine::SCRFDEngine() : m_logger(std::make_unique<TRTLogger>()) {
    cudaStreamCreate(&m_stream);
    initAnchors();
}

SCRFDEngine::~SCRFDEngine() {
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

void SCRFDEngine::initAnchors() {
    // SCRFD anchor sizes (2 anchors per position at each FPN level)
    // These are base sizes used to generate anchors
    m_anchorSizes[0][0] = 16.0f;   m_anchorSizes[0][1] = 32.0f;   // stride 8
    m_anchorSizes[1][0] = 64.0f;   m_anchorSizes[1][1] = 128.0f;  // stride 16
    m_anchorSizes[2][0] = 256.0f;  m_anchorSizes[2][1] = 512.0f;  // stride 32
}

bool SCRFDEngine::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open SCRFD engine file: " << enginePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    if (!m_runtime) {
        std::cerr << "Failed to create TensorRT runtime for SCRFD" << std::endl;
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        std::cerr << "Failed to deserialize SCRFD engine" << std::endl;
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "Failed to create SCRFD execution context" << std::endl;
        return false;
    }

    // Get I/O tensor info and verify expected bindings
    int numIO = m_engine->getNbIOTensors();
    std::cout << "SCRFD engine has " << numIO << " I/O tensors:" << std::endl;

    for (int i = 0; i < numIO; i++) {
        const char* name = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(name);
        auto dims = m_engine->getTensorShape(name);

        std::cout << "  " << name << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "]: ";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;
    }

    return allocateBuffers();
}

bool SCRFDEngine::allocateBuffers() {
    // Input: 1x3x640x640
    size_t inputSize = 1 * 3 * INPUT_W * INPUT_H * sizeof(float);
    cudaMalloc(&m_inputBuffer, inputSize);

    // Output sizes for SCRFD (stride 8, 16, 32)
    // score_8: 2 * 80 * 80 = 12800
    // bbox_8: 8 * 80 * 80 = 51200 (4 values * 2 anchors)
    // kps_8: 20 * 80 * 80 = 128000 (10 values * 2 anchors)
    size_t score8Size = 2 * FEAT_W_8 * FEAT_H_8 * sizeof(float);
    size_t bbox8Size = 8 * FEAT_W_8 * FEAT_H_8 * sizeof(float);
    size_t kps8Size = 20 * FEAT_W_8 * FEAT_H_8 * sizeof(float);

    size_t score16Size = 2 * FEAT_W_16 * FEAT_H_16 * sizeof(float);
    size_t bbox16Size = 8 * FEAT_W_16 * FEAT_H_16 * sizeof(float);
    size_t kps16Size = 20 * FEAT_W_16 * FEAT_H_16 * sizeof(float);

    size_t score32Size = 2 * FEAT_W_32 * FEAT_H_32 * sizeof(float);
    size_t bbox32Size = 8 * FEAT_W_32 * FEAT_H_32 * sizeof(float);
    size_t kps32Size = 20 * FEAT_W_32 * FEAT_H_32 * sizeof(float);

    cudaMalloc(&m_score8Buffer, score8Size);
    cudaMalloc(&m_bbox8Buffer, bbox8Size);
    cudaMalloc(&m_kps8Buffer, kps8Size);

    cudaMalloc(&m_score16Buffer, score16Size);
    cudaMalloc(&m_bbox16Buffer, bbox16Size);
    cudaMalloc(&m_kps16Buffer, kps16Size);

    cudaMalloc(&m_score32Buffer, score32Size);
    cudaMalloc(&m_bbox32Buffer, bbox32Size);
    cudaMalloc(&m_kps32Buffer, kps32Size);

    // Allocate host buffers
    m_score8Host.resize(2 * FEAT_W_8 * FEAT_H_8);
    m_bbox8Host.resize(8 * FEAT_W_8 * FEAT_H_8);
    m_kps8Host.resize(20 * FEAT_W_8 * FEAT_H_8);

    m_score16Host.resize(2 * FEAT_W_16 * FEAT_H_16);
    m_bbox16Host.resize(8 * FEAT_W_16 * FEAT_H_16);
    m_kps16Host.resize(20 * FEAT_W_16 * FEAT_H_16);

    m_score32Host.resize(2 * FEAT_W_32 * FEAT_H_32);
    m_bbox32Host.resize(8 * FEAT_W_32 * FEAT_H_32);
    m_kps32Host.resize(20 * FEAT_W_32 * FEAT_H_32);

    return m_inputBuffer != nullptr;
}

void SCRFDEngine::freeBuffers() {
    if (m_inputBuffer) { cudaFree(m_inputBuffer); m_inputBuffer = nullptr; }
    if (m_score8Buffer) { cudaFree(m_score8Buffer); m_score8Buffer = nullptr; }
    if (m_bbox8Buffer) { cudaFree(m_bbox8Buffer); m_bbox8Buffer = nullptr; }
    if (m_kps8Buffer) { cudaFree(m_kps8Buffer); m_kps8Buffer = nullptr; }
    if (m_score16Buffer) { cudaFree(m_score16Buffer); m_score16Buffer = nullptr; }
    if (m_bbox16Buffer) { cudaFree(m_bbox16Buffer); m_bbox16Buffer = nullptr; }
    if (m_kps16Buffer) { cudaFree(m_kps16Buffer); m_kps16Buffer = nullptr; }
    if (m_score32Buffer) { cudaFree(m_score32Buffer); m_score32Buffer = nullptr; }
    if (m_bbox32Buffer) { cudaFree(m_bbox32Buffer); m_bbox32Buffer = nullptr; }
    if (m_kps32Buffer) { cudaFree(m_kps32Buffer); m_kps32Buffer = nullptr; }
}

void SCRFDEngine::generateProposals(int stride, int featW, int featH, int numAnchors,
                                     const float* scoreData, const float* bboxData, const float* kpsData,
                                     float confThreshold, float scaleRatio, int wpad, int hpad,
                                     int origW, int origH,
                                     std::vector<FaceDetection>& proposals) {
    // SCRFD output format is [N, features] where N = featW * featH * numAnchors
    // - score: [N, 1] - one score per anchor
    // - bbox: [N, 4] - 4 distance values per anchor
    // - kps: [N, 10] - 5 landmarks * 2 coords per anchor

    for (int y = 0; y < featH; y++) {
        for (int x = 0; x < featW; x++) {
            for (int a = 0; a < numAnchors; a++) {
                // Flat anchor index: position * numAnchors + anchor
                int anchorIdx = (y * featW + x) * numAnchors + a;

                // Score is [N, 1] format
                float score = scoreData[anchorIdx];

                if (score < confThreshold) continue;

                // Anchor center (in 640x640 space)
                float anchorCx = (x + 0.5f) * stride;
                float anchorCy = (y + 0.5f) * stride;

                // Decode bbox - format is [N, 4] so index is anchorIdx * 4 + offset
                // SCRFD uses distance-based format: left, top, right, bottom distances from anchor
                float dl = bboxData[anchorIdx * 4 + 0];  // distance to left edge
                float dt = bboxData[anchorIdx * 4 + 1];  // distance to top edge
                float dr = bboxData[anchorIdx * 4 + 2];  // distance to right edge
                float db = bboxData[anchorIdx * 4 + 3];  // distance to bottom edge

                // Convert distances to bbox coordinates
                float x0 = anchorCx - dl * stride;
                float y0 = anchorCy - dt * stride;
                float x1 = anchorCx + dr * stride;
                float y1 = anchorCy + db * stride;

                // Scale back to original crop size
                // Image is at (0,0) with padding on RIGHT/BOTTOM, so no offset needed
                x0 = x0 / scaleRatio;
                y0 = y0 / scaleRatio;
                x1 = x1 / scaleRatio;
                y1 = y1 / scaleRatio;

                // Clamp to original image bounds
                x0 = std::max(0.0f, std::min(x0, (float)origW - 1));
                y0 = std::max(0.0f, std::min(y0, (float)origH - 1));
                x1 = std::max(0.0f, std::min(x1, (float)origW - 1));
                y1 = std::max(0.0f, std::min(y1, (float)origH - 1));

                FaceDetection face;
                face.x1 = x0;
                face.y1 = y0;
                face.x2 = x1;
                face.y2 = y1;
                face.confidence = score;

                // Decode landmarks - format is [N, 10] with interleaved x,y pairs
                for (int p = 0; p < 5; p++) {
                    float lx = anchorCx + kpsData[anchorIdx * 10 + p * 2] * stride;
                    float ly = anchorCy + kpsData[anchorIdx * 10 + p * 2 + 1] * stride;

                    // Scale back to original crop size
                    lx = lx / scaleRatio;
                    ly = ly / scaleRatio;

                    // Clamp
                    lx = std::max(0.0f, std::min(lx, (float)origW - 1));
                    ly = std::max(0.0f, std::min(ly, (float)origH - 1));

                    face.landmarks[p][0] = lx;
                    face.landmarks[p][1] = ly;
                }

                proposals.push_back(face);
            }
        }
    }
}

std::vector<FaceDetection> SCRFDEngine::detect(void* gpuInput, int cropW, int cropH,
                                                 float inputScaleRatio, float confThreshold,
                                                 float nmsThreshold) {
    // Note: inputScaleRatio is from the crop-to-640x640 resize
    // But SCRFD also pads the image to maintain aspect ratio

    // Calculate actual padding used during preprocessing
    // The crop was resized to fit within 640x640 while maintaining aspect ratio
    float scale = (float)INPUT_H / std::max(cropW, cropH);
    int resizedW = static_cast<int>(cropW * scale);
    int resizedH = static_cast<int>(cropH * scale);
    int wpad = INPUT_W - resizedW;
    int hpad = INPUT_H - resizedH;

    // Copy input
    cudaMemcpyAsync(m_inputBuffer, gpuInput,
                    3 * INPUT_W * INPUT_H * sizeof(float),
                    cudaMemcpyDeviceToDevice, m_stream);

    // Set tensor addresses by name
    int numIO = m_engine->getNbIOTensors();
    for (int i = 0; i < numIO; i++) {
        const char* name = m_engine->getIOTensorName(i);
        std::string nameStr(name);

        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_context->setTensorAddress(name, m_inputBuffer);
        } else {
            // Match output names
            if (nameStr == "score_8") m_context->setTensorAddress(name, m_score8Buffer);
            else if (nameStr == "bbox_8") m_context->setTensorAddress(name, m_bbox8Buffer);
            else if (nameStr == "kps_8") m_context->setTensorAddress(name, m_kps8Buffer);
            else if (nameStr == "score_16") m_context->setTensorAddress(name, m_score16Buffer);
            else if (nameStr == "bbox_16") m_context->setTensorAddress(name, m_bbox16Buffer);
            else if (nameStr == "kps_16") m_context->setTensorAddress(name, m_kps16Buffer);
            else if (nameStr == "score_32") m_context->setTensorAddress(name, m_score32Buffer);
            else if (nameStr == "bbox_32") m_context->setTensorAddress(name, m_bbox32Buffer);
            else if (nameStr == "kps_32") m_context->setTensorAddress(name, m_kps32Buffer);
            else {
                std::cerr << "Unknown SCRFD output tensor: " << name << std::endl;
            }
        }
    }

    // Run inference
    bool success = m_context->enqueueV3(m_stream);
    cudaStreamSynchronize(m_stream);

    if (!success) {
        std::cerr << "SCRFD inference failed" << std::endl;
        return {};
    }

    // Copy outputs to host
    cudaMemcpy(m_score8Host.data(), m_score8Buffer, m_score8Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_bbox8Host.data(), m_bbox8Buffer, m_bbox8Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_kps8Host.data(), m_kps8Buffer, m_kps8Host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_score16Host.data(), m_score16Buffer, m_score16Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_bbox16Host.data(), m_bbox16Buffer, m_bbox16Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_kps16Host.data(), m_kps16Buffer, m_kps16Host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(m_score32Host.data(), m_score32Buffer, m_score32Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_bbox32Host.data(), m_bbox32Buffer, m_bbox32Host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_kps32Host.data(), m_kps32Buffer, m_kps32Host.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Generate proposals from all FPN levels
    std::vector<FaceDetection> proposals;

    generateProposals(8, FEAT_W_8, FEAT_H_8, NUM_ANCHORS,
                      m_score8Host.data(), m_bbox8Host.data(), m_kps8Host.data(),
                      confThreshold, scale, wpad, hpad, cropW, cropH, proposals);

    generateProposals(16, FEAT_W_16, FEAT_H_16, NUM_ANCHORS,
                      m_score16Host.data(), m_bbox16Host.data(), m_kps16Host.data(),
                      confThreshold, scale, wpad, hpad, cropW, cropH, proposals);

    generateProposals(32, FEAT_W_32, FEAT_H_32, NUM_ANCHORS,
                      m_score32Host.data(), m_bbox32Host.data(), m_kps32Host.data(),
                      confThreshold, scale, wpad, hpad, cropW, cropH, proposals);

    // Apply NMS
    if (proposals.size() <= 1) {
        return proposals;
    }

    // Sort by confidence
    std::sort(proposals.begin(), proposals.end(),
        [](const FaceDetection& a, const FaceDetection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<FaceDetection> nmsResult;
    std::vector<bool> suppressed(proposals.size(), false);

    for (size_t i = 0; i < proposals.size(); i++) {
        if (suppressed[i]) continue;
        nmsResult.push_back(proposals[i]);

        for (size_t j = i + 1; j < proposals.size(); j++) {
            if (suppressed[j]) continue;

            // Calculate IoU
            float x1 = std::max(proposals[i].x1, proposals[j].x1);
            float y1 = std::max(proposals[i].y1, proposals[j].y1);
            float x2 = std::min(proposals[i].x2, proposals[j].x2);
            float y2 = std::min(proposals[i].y2, proposals[j].y2);

            float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = (proposals[i].x2 - proposals[i].x1) * (proposals[i].y2 - proposals[i].y1);
            float area2 = (proposals[j].x2 - proposals[j].x1) * (proposals[j].y2 - proposals[j].y1);
            float iou = interArea / (area1 + area2 - interArea + 1e-6f);

            if (iou > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return nmsResult;
}

// ============================================================================
// ArcFace Engine Implementation
// ============================================================================

ArcFaceEngine::ArcFaceEngine() : m_logger(std::make_unique<TRTLogger>()) {
    cudaStreamCreate(&m_stream);
}

ArcFaceEngine::~ArcFaceEngine() {
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

bool ArcFaceEngine::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    if (!m_runtime) return false;

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) return false;

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) return false;

    // Print tensor info
    int numIO = m_engine->getNbIOTensors();
    std::cout << "ArcFace engine has " << numIO << " I/O tensors:" << std::endl;

    for (int i = 0; i < numIO; i++) {
        const char* name = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(name);
        auto dims = m_engine->getTensorShape(name);

        std::cout << "  " << name << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "]: ";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            m_inputDims = dims;
        } else {
            m_outputDims = dims;
        }
    }

    return allocateBuffers();
}

bool ArcFaceEngine::allocateBuffers() {
    // Input: 1x3x112x112
    m_inputSize = 1 * 3 * 112 * 112 * sizeof(float);
    cudaMalloc(&m_inputBuffer, m_inputSize);

    // Output: 1x512
    m_outputSize = 512 * sizeof(float);
    cudaMalloc(&m_outputBuffer, m_outputSize);

    return m_inputBuffer != nullptr && m_outputBuffer != nullptr;
}

void ArcFaceEngine::freeBuffers() {
    if (m_inputBuffer) { cudaFree(m_inputBuffer); m_inputBuffer = nullptr; }
    if (m_outputBuffer) { cudaFree(m_outputBuffer); m_outputBuffer = nullptr; }
}

bool ArcFaceEngine::getEmbedding(void* gpuInput, float* embedding) {
    // Copy input
    cudaMemcpyAsync(m_inputBuffer, gpuInput, m_inputSize, cudaMemcpyDeviceToDevice, m_stream);

    // Set tensor addresses
    const char* inputName = m_engine->getIOTensorName(0);
    const char* outputName = m_engine->getIOTensorName(1);

    m_context->setTensorAddress(inputName, m_inputBuffer);
    m_context->setTensorAddress(outputName, m_outputBuffer);

    // Run inference
    bool success = m_context->enqueueV3(m_stream);
    cudaStreamSynchronize(m_stream);

    if (!success) {
        std::cerr << "ArcFace inference failed" << std::endl;
        return false;
    }

    // Copy output embedding
    cudaMemcpy(embedding, m_outputBuffer, m_outputSize, cudaMemcpyDeviceToHost);

    // L2 normalize the embedding
    float norm = 0.0f;
    for (int i = 0; i < 512; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm + 1e-10f);
    for (int i = 0; i < 512; i++) {
        embedding[i] /= norm;
    }

    return true;
}

// ============================================================================
// InsightFace Pipeline Implementation
// ============================================================================

InsightFacePipeline::InsightFacePipeline() {
    cudaStreamCreate(&m_stream);
}

InsightFacePipeline::~InsightFacePipeline() {
    freeIntermediateBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

bool InsightFacePipeline::initialize(const std::string& detectorPath, const std::string& arcFacePath) {
    // Determine which detector to use based on filename
    m_useSCRFD = (detectorPath.find("scrfd") != std::string::npos);

    if (m_useSCRFD) {
        std::cout << "Loading SCRFD face detector engine..." << std::endl;
        if (!m_scrfdDetector.loadEngine(detectorPath)) {
            std::cerr << "Failed to load SCRFD engine" << std::endl;
            return false;
        }
    } else {
        std::cout << "Loading RetinaFace engine..." << std::endl;
        if (!m_retinaDetector.loadEngine(detectorPath)) {
            std::cerr << "Failed to load RetinaFace engine" << std::endl;
            return false;
        }
    }

    std::cout << "Loading ArcFace/AdaFace engine..." << std::endl;
    if (!m_recognizer.loadEngine(arcFacePath)) {
        std::cerr << "Failed to load face recognition engine" << std::endl;
        return false;
    }

    return allocateIntermediateBuffers();
}

bool InsightFacePipeline::allocateIntermediateBuffers() {
    // Crop buffer: 640x640x3 float for RetinaFace input
    m_cropBufferSize = 640 * 640 * 3 * sizeof(float);
    cudaMalloc(&m_cropBuffer, m_cropBufferSize);

    // Face buffer: 112x112x3 float for ArcFace input
    m_faceBufferSize = 112 * 112 * 3 * sizeof(float);
    cudaMalloc(&m_faceBuffer, m_faceBufferSize);

    return m_cropBuffer != nullptr && m_faceBuffer != nullptr;
}

void InsightFacePipeline::freeIntermediateBuffers() {
    if (m_cropBuffer) { cudaFree(m_cropBuffer); m_cropBuffer = nullptr; }
    if (m_faceBuffer) { cudaFree(m_faceBuffer); m_faceBuffer = nullptr; }
}

std::vector<FaceRecognition> InsightFacePipeline::process(
    void* frameGpu, int frameW, int frameH, int framePitch,
    float personX1, float personY1, float personX2, float personY2,
    float detConfThreshold
) {
    std::vector<FaceRecognition> results;

    // Clamp person bbox to frame bounds
    int cropX = std::max(0, (int)personX1);
    int cropY = std::max(0, (int)personY1);
    int cropW = std::min(frameW - cropX, (int)(personX2 - personX1));
    int cropH = std::min(frameH - cropY, (int)(personY2 - personY1));

    if (cropW < 20 || cropH < 20) {
        return results;  // Person bbox too small
    }

    // Step 1: Crop person region and resize to 640x640 for RetinaFace
    float scaleRatio;
    launchCropBGRxToRGBFloat(
        frameGpu, m_cropBuffer,
        frameW, frameH, framePitch,
        cropX, cropY, cropW, cropH,
        640, 640,
        &scaleRatio,
        m_stream
    );
    cudaStreamSynchronize(m_stream);

    // Step 2: Run face detection (SCRFD or RetinaFace)
    std::vector<FaceDetection> faces;
    if (m_useSCRFD) {
        faces = m_scrfdDetector.detect(m_cropBuffer, cropW, cropH, scaleRatio, detConfThreshold);
    } else {
        faces = m_retinaDetector.detect(m_cropBuffer, 640, 640, scaleRatio, detConfThreshold);
    }

    // Step 3: For each detected face, align and get embedding
    for (const auto& face : faces) {
        // Note: face coordinates from detect() are already in crop coordinates
        // (detect() already applied scaleRatio to convert from 640x640 to crop size)
        // We just need to add the crop offset to get frame coordinates

        // Convert face landmarks from crop coordinates to frame coordinates
        float frameLandmarks[5][2];
        bool landmarksValid = true;
        for (int i = 0; i < 5; i++) {
            frameLandmarks[i][0] = cropX + face.landmarks[i][0];  // Already scaled, just add offset
            frameLandmarks[i][1] = cropY + face.landmarks[i][1];

            // Check if landmark is within CROP bounds (not just frame bounds)
            // Landmarks detected in padding regions of 640x640 buffer produce garbage
            // face.landmarks are in crop coordinates, so check against cropW/cropH
            if (face.landmarks[i][0] < 0 || face.landmarks[i][0] >= cropW ||
                face.landmarks[i][1] < 0 || face.landmarks[i][1] >= cropH) {
                landmarksValid = false;
            }
        }

        // Skip faces with out-of-bounds landmarks (detected in padding region)
        if (!landmarksValid) {
            continue;
        }

        // Validate face size - a real face shouldn't span more than 50% of person bbox width
        float faceWidth = face.x2 - face.x1;
        if (faceWidth > cropW * 0.5f) {
            continue;
        }

        // Step 3a: Warp face to 112x112 aligned image
        launchWarpFaceToArcFace(
            frameGpu, m_faceBuffer,
            frameW, frameH, framePitch,
            frameLandmarks,
            m_stream
        );
        cudaStreamSynchronize(m_stream);

        // Step 3b: Get embedding from ArcFace
        FaceRecognition result;
        result.face = face;

        // Convert face bbox to frame coordinates (just add crop offset)
        result.face.x1 = cropX + face.x1;  // Already in crop coordinates
        result.face.y1 = cropY + face.y1;
        result.face.x2 = cropX + face.x2;
        result.face.y2 = cropY + face.y2;

        // Copy landmarks to result
        for (int i = 0; i < 5; i++) {
            result.face.landmarks[i][0] = frameLandmarks[i][0];
            result.face.landmarks[i][1] = frameLandmarks[i][1];
        }

        if (m_recognizer.getEmbedding(m_faceBuffer, result.embedding)) {
            // Calculate quality score based on face size and detection confidence
            float faceW = result.face.x2 - result.face.x1;
            float faceH = result.face.y2 - result.face.y1;
            float sizeScore = std::min(1.0f, (faceW * faceH) / (112.0f * 112.0f));
            result.quality = result.face.confidence * 0.7f + sizeScore * 0.3f;

            results.push_back(result);
        }
    }

    return results;
}

float InsightFacePipeline::cosineSimilarity(const float* emb1, const float* emb2) {
    float dot = 0.0f;
    for (int i = 0; i < 512; i++) {
        dot += emb1[i] * emb2[i];
    }
    // Embeddings are already L2 normalized, so dot product = cosine similarity
    return dot;
}

bool InsightFacePipeline::getEmbeddingFromAligned(void* alignedFaceGpu, float* embedding) {
    // Input: 112x112 RGB float CHW, already normalized to [-1, 1]
    // Just run ArcFace directly - no detection or alignment needed
    return m_recognizer.getEmbedding(alignedFaceGpu, embedding);
}

} // namespace mmoment
