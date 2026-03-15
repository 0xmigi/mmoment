/**
 * reid_engine.h - OSNet ReID TensorRT inference for person re-identification
 *
 * Uses OSNet x0.25 trained on Market1501 to generate 512-dim appearance embeddings.
 * These embeddings are used to maintain person identity when face is not visible.
 *
 * Input: Person crop resized to 256x128 (HxW)
 * Output: 512-dim L2-normalized embedding
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace mmoment {

// TensorRT logger for ReID engine
class TRTLoggerReID : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT-ReID] " << msg << std::endl;
        }
    }
};

/**
 * OSNet ReID Engine - generates person appearance embeddings
 *
 * Unlike face recognition which needs face detection + alignment,
 * ReID works directly on the person bounding box crop.
 */
class ReIDEngine {
public:
    ReIDEngine();
    ~ReIDEngine();

    /**
     * Load TensorRT engine from file
     * @param enginePath Path to .engine file (OSNet x0.25)
     * @return true if loaded successfully
     */
    bool loadEngine(const std::string& enginePath);

    /**
     * Check if engine is loaded
     */
    bool isReady() const { return m_context != nullptr; }

    /**
     * Get embedding for a person crop
     * @param gpuInput GPU buffer containing preprocessed person crop (256x128 RGB float CHW)
     * @param embedding Output 512-dim embedding (CPU memory)
     * @return true if successful
     */
    bool getEmbedding(void* gpuInput, float* embedding);

    /**
     * Preprocess person crop from BGRx frame and get embedding
     * This is a convenience method that handles cropping and preprocessing.
     *
     * @param frameGpu Full frame in GPU memory (BGRx)
     * @param frameW Frame width
     * @param frameH Frame height
     * @param framePitch Frame pitch (bytes per row)
     * @param x1, y1, x2, y2 Person bounding box
     * @param embedding Output 512-dim embedding (CPU memory)
     * @return true if successful
     */
    bool getEmbeddingFromCrop(
        void* frameGpu, int frameW, int frameH, int framePitch,
        float x1, float y1, float x2, float y2,
        float* embedding
    );

    /**
     * Compute cosine similarity between two ReID embeddings
     * @return Similarity in range [-1, 1], higher = more similar
     */
    static float cosineSimilarity(const float* emb1, const float* emb2);

    /**
     * Get input dimensions
     */
    int getInputHeight() const { return 256; }
    int getInputWidth() const { return 128; }
    int getEmbeddingDim() const { return 512; }

private:
    bool allocateBuffers();
    void freeBuffers();

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    // GPU buffers
    void* m_inputBuffer = nullptr;   // 1x3x256x128 float
    void* m_outputBuffer = nullptr;  // 1x512 float
    void* m_cropBuffer = nullptr;    // For preprocessing person crop

    size_t m_inputSize = 0;
    size_t m_outputSize = 0;

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;

    cudaStream_t m_stream = nullptr;
    std::unique_ptr<TRTLoggerReID> m_logger;
};

// CUDA kernel for preprocessing person crop to OSNet input
// Crops from BGRx frame, resizes to 256x128, converts to RGB float CHW, normalizes
extern "C" void launchCropPersonToReID(
    const void* src,           // Source frame (BGRx GPU buffer)
    void* dst,                 // Destination (256x128x3 float CHW)
    int srcW, int srcH,        // Source frame dimensions
    int srcPitch,              // Source pitch (bytes per row)
    int cropX, int cropY,      // Crop origin
    int cropW, int cropH,      // Crop dimensions
    cudaStream_t stream
);

} // namespace mmoment
