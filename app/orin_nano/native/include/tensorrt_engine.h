/**
 * tensorrt_engine.h - Minimal TensorRT inference engine
 *
 * Designed to work directly with NVMM GPU buffers from GStreamer.
 * No OpenCV CUDA dependency - uses raw CUDA pointers.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace mmoment {

// Precision modes
enum class Precision { FP32, FP16, INT8 };

// Engine configuration
struct EngineConfig {
    Precision precision = Precision::FP16;
    int deviceIndex = 0;
    int maxBatchSize = 1;
};

// Detection result for pose estimation
struct PoseDetection {
    float x1, y1, x2, y2;    // Bounding box
    float confidence;
    int classId;
    float keypoints[17][3]; // 17 COCO keypoints: x, y, confidence
};

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void setVerbose(bool verbose) { m_verbose = verbose; }
private:
    bool m_verbose = false;
};

// Main inference engine class
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    // Load a pre-built TensorRT engine file
    bool loadEngine(const std::string& enginePath);

    // Build engine from ONNX (if engine doesn't exist)
    bool buildFromONNX(const std::string& onnxPath, const std::string& enginePath,
                       const EngineConfig& config = EngineConfig());

    // Run inference on GPU buffer
    // Input: Raw GPU pointer (BGRx format from NVMM, or RGB float)
    // Output: Raw detection tensor
    bool infer(void* gpuInput, int inputWidth, int inputHeight, int inputChannels,
               std::vector<float>& outputData);

    // Run inference and keep output on GPU (zero-copy path)
    // Returns pointer to GPU output buffer (valid until next infer call)
    bool inferGPU(void* gpuInput, int inputWidth, int inputHeight, int inputChannels,
                  float** gpuOutputPtr, size_t* outputElements);

    // Get input/output dimensions
    nvinfer1::Dims getInputDims() const;
    nvinfer1::Dims getOutputDims() const;

    // Utility: Preprocess BGRx NVMM buffer to RGB float tensor
    // Handles resize + pad (right/bottom) + normalize
    bool preprocessNVMM(void* nvmmInput, int srcWidth, int srcHeight, int srcPitch,
                        void* gpuOutput, int dstWidth, int dstHeight,
                        float& scaleRatio);

private:
    bool allocateBuffers();
    void freeBuffers();

    TRTLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    // GPU buffers for input/output
    void* m_inputBuffer = nullptr;
    void* m_outputBuffer = nullptr;
    size_t m_inputSize = 0;
    size_t m_outputSize = 0;

    // Dimensions
    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;

    cudaStream_t m_stream = nullptr;
};

// Post-processing for YOLOv8-pose output
class YoloPosePostProcessor {
public:
    YoloPosePostProcessor(float confThreshold = 0.25f, float nmsThreshold = 0.45f);

    // Process raw YOLO output tensor into detections
    // rawOutput: [num_detections, 56] where 56 = 4 (bbox) + 1 (conf) + 51 (17*3 keypoints)
    std::vector<PoseDetection> process(const std::vector<float>& rawOutput,
                                        int numDetections,
                                        float scaleRatio,
                                        int originalWidth, int originalHeight);

private:
    float m_confThreshold;
    float m_nmsThreshold;

    // NMS implementation
    std::vector<int> nms(const std::vector<PoseDetection>& detections);
};

} // namespace mmoment
