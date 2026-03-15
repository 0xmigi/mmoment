/**
 * tensorrt_engine.cpp - Minimal TensorRT inference implementation
 */

#include "tensorrt_engine.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <NvOnnxParser.h>

namespace mmoment {

// =============================================================================
// Logger
// =============================================================================

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
        std::cerr << "[TRT ERROR] " << msg << std::endl;
    } else if (m_verbose && severity == Severity::kWARNING) {
        std::cerr << "[TRT WARN] " << msg << std::endl;
    } else if (m_verbose && severity == Severity::kINFO) {
        std::cout << "[TRT INFO] " << msg << std::endl;
    }
}

// =============================================================================
// TensorRTEngine
// =============================================================================

TensorRTEngine::TensorRTEngine() {
    cudaStreamCreate(&m_stream);
}

TensorRTEngine::~TensorRTEngine() {
    freeBuffers();
    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }
}

bool TensorRTEngine::loadEngine(const std::string& enginePath) {
    std::cout << "[TensorRT] Loading engine: " << enginePath << std::endl;

    // Read engine file
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[TensorRT] Cannot open engine file: " << enginePath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    std::cout << "[TensorRT] Engine file size: " << size / 1024 / 1024 << " MB" << std::endl;

    // Create runtime and engine
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        std::cerr << "[TensorRT] Failed to create runtime" << std::endl;
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        std::cerr << "[TensorRT] Failed to deserialize engine" << std::endl;
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "[TensorRT] Failed to create execution context" << std::endl;
        return false;
    }

    // Get input/output info
    int numIOTensors = m_engine->getNbIOTensors();
    std::cout << "[TensorRT] IO Tensors: " << numIOTensors << std::endl;

    for (int i = 0; i < numIOTensors; i++) {
        const char* name = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(name);
        auto dims = m_engine->getTensorShape(name);

        std::cout << "[TensorRT]   " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " '" << name << "': [";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d];
            if (d < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            m_inputDims = dims;
        } else {
            m_outputDims = dims;
        }
    }

    // Allocate buffers
    if (!allocateBuffers()) {
        return false;
    }

    std::cout << "[TensorRT] Engine loaded successfully" << std::endl;
    return true;
}

bool TensorRTEngine::buildFromONNX(const std::string& onnxPath, const std::string& enginePath,
                                    const EngineConfig& config) {
    std::cout << "[TensorRT] Building engine from ONNX: " << onnxPath << std::endl;

    // Create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        std::cerr << "[TensorRT] Failed to create builder" << std::endl;
        return false;
    }

    // Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "[TensorRT] Failed to create network" << std::endl;
        return false;
    }

    // Parse ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, m_logger));
    if (!parser->parseFromFile(onnxPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TensorRT] Failed to parse ONNX" << std::endl;
        return false;
    }

    // Create config
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Set precision
    if (config.precision == Precision::FP16 && builder->platformHasFastFp16()) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[TensorRT] Using FP16 precision" << std::endl;
    }

    // Build engine
    std::cout << "[TensorRT] Building engine (this may take several minutes)..." << std::endl;
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builderConfig));

    if (!serializedEngine) {
        std::cerr << "[TensorRT] Failed to build engine" << std::endl;
        return false;
    }

    // Save to file
    std::ofstream file(enginePath, std::ios::binary);
    file.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    file.close();

    std::cout << "[TensorRT] Engine saved to: " << enginePath << std::endl;

    // Load the engine we just built
    return loadEngine(enginePath);
}

bool TensorRTEngine::allocateBuffers() {
    // Calculate input size
    m_inputSize = 1;
    for (int i = 0; i < m_inputDims.nbDims; i++) {
        m_inputSize *= m_inputDims.d[i];
    }
    m_inputSize *= sizeof(float);

    // Calculate output size
    m_outputSize = 1;
    for (int i = 0; i < m_outputDims.nbDims; i++) {
        m_outputSize *= m_outputDims.d[i];
    }
    m_outputSize *= sizeof(float);

    std::cout << "[TensorRT] Allocating buffers - Input: " << m_inputSize / 1024 << " KB, Output: "
              << m_outputSize / 1024 << " KB" << std::endl;

    // Allocate GPU memory
    cudaError_t err;
    err = cudaMalloc(&m_inputBuffer, m_inputSize);
    if (err != cudaSuccess) {
        std::cerr << "[TensorRT] Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    err = cudaMalloc(&m_outputBuffer, m_outputSize);
    if (err != cudaSuccess) {
        std::cerr << "[TensorRT] Failed to allocate output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(m_inputBuffer);
        m_inputBuffer = nullptr;
        return false;
    }

    // Set tensor addresses for TensorRT
    const char* inputName = m_engine->getIOTensorName(0);
    const char* outputName = m_engine->getIOTensorName(1);

    m_context->setTensorAddress(inputName, m_inputBuffer);
    m_context->setTensorAddress(outputName, m_outputBuffer);

    return true;
}

void TensorRTEngine::freeBuffers() {
    if (m_inputBuffer) {
        cudaFree(m_inputBuffer);
        m_inputBuffer = nullptr;
    }
    if (m_outputBuffer) {
        cudaFree(m_outputBuffer);
        m_outputBuffer = nullptr;
    }
}

bool TensorRTEngine::infer(void* gpuInput, int inputWidth, int inputHeight, int inputChannels,
                           std::vector<float>& outputData) {
    if (!m_context) {
        std::cerr << "[TensorRT] No context - engine not loaded" << std::endl;
        return false;
    }

    // Copy input to engine's input buffer (if not already there)
    if (gpuInput != m_inputBuffer) {
        cudaMemcpyAsync(m_inputBuffer, gpuInput, m_inputSize, cudaMemcpyDeviceToDevice, m_stream);
    }

    // Run inference
    bool success = m_context->enqueueV3(m_stream);
    if (!success) {
        std::cerr << "[TensorRT] Inference failed" << std::endl;
        return false;
    }

    // Copy output to CPU
    size_t outputElements = m_outputSize / sizeof(float);
    outputData.resize(outputElements);
    cudaMemcpyAsync(outputData.data(), m_outputBuffer, m_outputSize, cudaMemcpyDeviceToHost, m_stream);

    // Synchronize
    cudaStreamSynchronize(m_stream);

    return true;
}

bool TensorRTEngine::inferGPU(void* gpuInput, int inputWidth, int inputHeight, int inputChannels,
                               float** gpuOutputPtr, size_t* outputElements) {
    if (!m_context) {
        std::cerr << "[TensorRT] No context - engine not loaded" << std::endl;
        return false;
    }

    // Copy input to engine's input buffer (if not already there)
    if (gpuInput != m_inputBuffer) {
        cudaMemcpyAsync(m_inputBuffer, gpuInput, m_inputSize, cudaMemcpyDeviceToDevice, m_stream);
    }

    // Run inference
    bool success = m_context->enqueueV3(m_stream);
    if (!success) {
        std::cerr << "[TensorRT] Inference failed" << std::endl;
        return false;
    }

    // Return GPU output buffer pointer directly (no copy to CPU)
    *gpuOutputPtr = (float*)m_outputBuffer;
    *outputElements = m_outputSize / sizeof(float);

    // Synchronize to ensure inference is complete
    cudaStreamSynchronize(m_stream);

    return true;
}

nvinfer1::Dims TensorRTEngine::getInputDims() const {
    return m_inputDims;
}

nvinfer1::Dims TensorRTEngine::getOutputDims() const {
    return m_outputDims;
}

// =============================================================================
// YoloPosePostProcessor
// =============================================================================

YoloPosePostProcessor::YoloPosePostProcessor(float confThreshold, float nmsThreshold)
    : m_confThreshold(confThreshold), m_nmsThreshold(nmsThreshold) {}

std::vector<PoseDetection> YoloPosePostProcessor::process(
    const std::vector<float>& rawOutput,
    int numDetections,
    float scaleRatio,
    int originalWidth, int originalHeight) {

    std::vector<PoseDetection> detections;

    // YOLOv8-pose output format: [batch, 56, num_detections]
    // 56 = 4 (x,y,w,h) + 1 (conf) + 51 (17 keypoints * 3)
    // After transpose: [num_detections, 56]

    const int featuresPerDet = 56;

    for (int i = 0; i < numDetections; i++) {
        int offset = i * featuresPerDet;

        // Extract confidence
        float conf = rawOutput[offset + 4];
        if (conf < m_confThreshold) continue;

        PoseDetection det;
        det.confidence = conf;
        det.classId = 0; // Person class

        // Extract bbox (center x, y, w, h) and convert to corners
        float cx = rawOutput[offset + 0];
        float cy = rawOutput[offset + 1];
        float w = rawOutput[offset + 2];
        float h = rawOutput[offset + 3];

        // Scale back to original image coordinates
        // Since we pad right/bottom only, just multiply by ratio
        det.x1 = (cx - w / 2) * scaleRatio;
        det.y1 = (cy - h / 2) * scaleRatio;
        det.x2 = (cx + w / 2) * scaleRatio;
        det.y2 = (cy + h / 2) * scaleRatio;

        // Clamp to image bounds
        det.x1 = std::clamp(det.x1, 0.0f, (float)originalWidth);
        det.y1 = std::clamp(det.y1, 0.0f, (float)originalHeight);
        det.x2 = std::clamp(det.x2, 0.0f, (float)originalWidth);
        det.y2 = std::clamp(det.y2, 0.0f, (float)originalHeight);

        // Extract keypoints
        for (int k = 0; k < 17; k++) {
            int kpOffset = offset + 5 + k * 3;
            det.keypoints[k][0] = rawOutput[kpOffset + 0] * scaleRatio; // x
            det.keypoints[k][1] = rawOutput[kpOffset + 1] * scaleRatio; // y
            det.keypoints[k][2] = rawOutput[kpOffset + 2];              // conf

            // Clamp coordinates
            det.keypoints[k][0] = std::clamp(det.keypoints[k][0], 0.0f, (float)originalWidth);
            det.keypoints[k][1] = std::clamp(det.keypoints[k][1], 0.0f, (float)originalHeight);
        }

        detections.push_back(det);
    }

    // Apply NMS
    if (detections.size() > 1) {
        auto keepIndices = nms(detections);
        std::vector<PoseDetection> nmsDetections;
        for (int idx : keepIndices) {
            nmsDetections.push_back(detections[idx]);
        }
        return nmsDetections;
    }

    return detections;
}

std::vector<int> YoloPosePostProcessor::nms(const std::vector<PoseDetection>& detections) {
    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

    // Sort by confidence (descending)
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(detections.size(), false);

    for (int idx : indices) {
        if (suppressed[idx]) continue;

        keep.push_back(idx);
        const auto& det = detections[idx];

        // Suppress overlapping detections
        for (size_t j = 0; j < detections.size(); j++) {
            if (suppressed[j] || (int)j == idx) continue;

            const auto& other = detections[j];

            // Calculate IoU
            float x1 = std::max(det.x1, other.x1);
            float y1 = std::max(det.y1, other.y1);
            float x2 = std::min(det.x2, other.x2);
            float y2 = std::min(det.y2, other.y2);

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = (det.x2 - det.x1) * (det.y2 - det.y1);
            float area2 = (other.x2 - other.x1) * (other.y2 - other.y1);
            float unionArea = area1 + area2 - intersection;

            float iou = intersection / (unionArea + 1e-6f);

            if (iou > m_nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

} // namespace mmoment
