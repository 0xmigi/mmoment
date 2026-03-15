/**
 * reid_engine.cpp - OSNet ReID TensorRT inference implementation
 */

#include "reid_engine.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace mmoment {

// ============================================================================
// ReID Engine Implementation
// ============================================================================

ReIDEngine::ReIDEngine() : m_logger(std::make_unique<TRTLoggerReID>()) {
    cudaStreamCreate(&m_stream);
}

ReIDEngine::~ReIDEngine() {
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

bool ReIDEngine::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[ReID] Failed to open engine file: " << enginePath << std::endl;
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
        std::cerr << "[ReID] Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        std::cerr << "[ReID] Failed to deserialize engine" << std::endl;
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[ReID] Failed to create execution context" << std::endl;
        return false;
    }

    // Print tensor info
    int numIO = m_engine->getNbIOTensors();
    std::cout << "[ReID] OSNet engine loaded with " << numIO << " I/O tensors:" << std::endl;

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

bool ReIDEngine::allocateBuffers() {
    // Input: 1x3x256x128 (batch x channels x height x width)
    // OSNet expects 256x128 (HxW) input
    m_inputSize = 1 * 3 * 256 * 128 * sizeof(float);
    cudaMalloc(&m_inputBuffer, m_inputSize);

    // Output: 1x512 embedding
    m_outputSize = 512 * sizeof(float);
    cudaMalloc(&m_outputBuffer, m_outputSize);

    // Crop buffer for preprocessing (reuse for different crops)
    // Max person crop size we'll handle
    size_t cropBufferSize = 512 * 1024 * 4;  // 512x1024 BGRx max
    cudaMalloc(&m_cropBuffer, cropBufferSize);

    bool success = m_inputBuffer != nullptr &&
                   m_outputBuffer != nullptr &&
                   m_cropBuffer != nullptr;

    if (success) {
        std::cout << "[ReID] Buffers allocated: input=" << m_inputSize
                  << " output=" << m_outputSize << std::endl;
    }

    return success;
}

void ReIDEngine::freeBuffers() {
    if (m_inputBuffer) { cudaFree(m_inputBuffer); m_inputBuffer = nullptr; }
    if (m_outputBuffer) { cudaFree(m_outputBuffer); m_outputBuffer = nullptr; }
    if (m_cropBuffer) { cudaFree(m_cropBuffer); m_cropBuffer = nullptr; }
}

bool ReIDEngine::getEmbedding(void* gpuInput, float* embedding) {
    if (!m_context) {
        std::cerr << "[ReID] Engine not initialized" << std::endl;
        return false;
    }

    // Copy preprocessed input to our buffer
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
        std::cerr << "[ReID] Inference failed" << std::endl;
        return false;
    }

    // Copy output embedding to CPU
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

bool ReIDEngine::getEmbeddingFromCrop(
    void* frameGpu, int frameW, int frameH, int framePitch,
    float x1, float y1, float x2, float y2,
    float* embedding
) {
    if (!m_context) {
        return false;
    }

    // Clamp bbox to frame bounds
    int cropX = std::max(0, (int)x1);
    int cropY = std::max(0, (int)y1);
    int cropW = std::min(frameW - cropX, (int)(x2 - x1));
    int cropH = std::min(frameH - cropY, (int)(y2 - y1));

    if (cropW < 32 || cropH < 64) {
        // Person too small for reliable ReID
        return false;
    }

    // Preprocess: crop, resize to 256x128, convert to RGB float CHW, normalize
    launchCropPersonToReID(
        frameGpu, m_inputBuffer,
        frameW, frameH, framePitch,
        cropX, cropY, cropW, cropH,
        m_stream
    );
    cudaStreamSynchronize(m_stream);

    // Run inference (input is already in m_inputBuffer)
    const char* inputName = m_engine->getIOTensorName(0);
    const char* outputName = m_engine->getIOTensorName(1);

    m_context->setTensorAddress(inputName, m_inputBuffer);
    m_context->setTensorAddress(outputName, m_outputBuffer);

    bool success = m_context->enqueueV3(m_stream);
    cudaStreamSynchronize(m_stream);

    if (!success) {
        return false;
    }

    // Copy and normalize embedding
    cudaMemcpy(embedding, m_outputBuffer, m_outputSize, cudaMemcpyDeviceToHost);

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

float ReIDEngine::cosineSimilarity(const float* emb1, const float* emb2) {
    float dot = 0.0f;
    for (int i = 0; i < 512; i++) {
        dot += emb1[i] * emb2[i];
    }
    // Embeddings are L2 normalized, so dot product = cosine similarity
    return dot;
}

} // namespace mmoment
