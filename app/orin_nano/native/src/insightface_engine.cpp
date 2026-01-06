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

bool InsightFacePipeline::initialize(const std::string& retinaFacePath, const std::string& arcFacePath) {
    std::cout << "Loading RetinaFace engine..." << std::endl;
    if (!m_detector.loadEngine(retinaFacePath)) {
        std::cerr << "Failed to load RetinaFace engine" << std::endl;
        return false;
    }

    std::cout << "Loading ArcFace engine..." << std::endl;
    if (!m_recognizer.loadEngine(arcFacePath)) {
        std::cerr << "Failed to load ArcFace engine" << std::endl;
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

    // Step 2: Run RetinaFace to detect faces
    auto faces = m_detector.detect(m_cropBuffer, 640, 640, scaleRatio, detConfThreshold);

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
        // (The crop is the full person body, face is just the head portion)
        float faceWidth = face.x2 - face.x1;
        float maxFaceRatio = 0.5f;

        if (faceWidth > cropW * maxFaceRatio) {
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

} // namespace mmoment
