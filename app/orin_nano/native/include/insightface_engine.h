/**
 * insightface_engine.h - InsightFace TensorRT inference for face recognition
 *
 * Pipeline:
 * 1. Person bbox (from YOLOv8-pose) -> Crop from frame
 * 2. RetinaFace: Detect faces + 5 landmarks in person crop
 * 3. Face alignment: Warp to 112x112 using landmarks
 * 4. ArcFace: Generate 512-dim embedding
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace mmoment {

// Face detection result from RetinaFace
struct FaceDetection {
    float x1, y1, x2, y2;    // Face bounding box (in person crop coordinates)
    float confidence;
    float landmarks[5][2];   // 5 facial landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
};

// Face recognition result
struct FaceRecognition {
    FaceDetection face;
    float embedding[512];    // ArcFace 512-dim embedding
    float quality;           // Face quality score (based on detection confidence + face size)
    int personTrackId;       // Track ID of the person this face belongs to (-1 if none)
};

// Forward declaration
class TRTLogger;

/**
 * RetinaFace detector - detects faces and landmarks in an image
 */
class RetinaFaceEngine {
public:
    RetinaFaceEngine();
    ~RetinaFaceEngine();

    bool loadEngine(const std::string& enginePath);

    // Run face detection on a person crop (GPU buffer, RGB float CHW)
    // Returns detected faces with landmarks
    std::vector<FaceDetection> detect(void* gpuInput, int inputWidth, int inputHeight,
                                       float scaleRatio, float confThreshold = 0.5f);

private:
    bool allocateBuffers();
    void freeBuffers();

    // Post-process RetinaFace output to get faces
    std::vector<FaceDetection> postProcess(const std::vector<float>& output,
                                            int inputW, int inputH,
                                            float scaleRatio, float confThreshold);

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void* m_inputBuffer = nullptr;
    void* m_outputBuffer = nullptr;
    size_t m_inputSize = 0;
    size_t m_outputSize = 0;

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;

    cudaStream_t m_stream = nullptr;
    std::unique_ptr<TRTLogger> m_logger;
};

/**
 * ArcFace recognizer - generates face embeddings
 */
class ArcFaceEngine {
public:
    ArcFaceEngine();
    ~ArcFaceEngine();

    bool loadEngine(const std::string& enginePath);

    // Run face recognition on aligned face (GPU buffer, 112x112 RGB float CHW)
    // Returns 512-dim embedding
    bool getEmbedding(void* gpuInput, float* embedding);

private:
    bool allocateBuffers();
    void freeBuffers();

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void* m_inputBuffer = nullptr;
    void* m_outputBuffer = nullptr;
    size_t m_inputSize = 0;
    size_t m_outputSize = 0;

    nvinfer1::Dims m_inputDims;
    nvinfer1::Dims m_outputDims;

    cudaStream_t m_stream = nullptr;
    std::unique_ptr<TRTLogger> m_logger;
};

/**
 * Combined InsightFace pipeline
 * Handles full flow: person crop -> face detection -> alignment -> embedding
 */
class InsightFacePipeline {
public:
    InsightFacePipeline();
    ~InsightFacePipeline();

    // Load both engines
    bool initialize(const std::string& retinaFacePath, const std::string& arcFacePath);

    // Process a person crop and return face recognitions
    // Input: Full frame (BGRx GPU buffer) + person bbox
    // Output: Face recognitions with embeddings
    std::vector<FaceRecognition> process(
        void* frameGpu,           // Full frame in GPU memory (BGRx)
        int frameW, int frameH, int framePitch,
        float personX1, float personY1, float personX2, float personY2,  // Person bbox
        float detConfThreshold = 0.5f
    );

    // Compute cosine similarity between two embeddings
    static float cosineSimilarity(const float* emb1, const float* emb2);

private:
    RetinaFaceEngine m_detector;
    ArcFaceEngine m_recognizer;

    // GPU buffers for intermediate results
    void* m_cropBuffer = nullptr;      // Person crop resized for RetinaFace (640x640)
    void* m_faceBuffer = nullptr;      // Aligned face for ArcFace (112x112)
    size_t m_cropBufferSize = 0;
    size_t m_faceBufferSize = 0;

    cudaStream_t m_stream = nullptr;

    bool allocateIntermediateBuffers();
    void freeIntermediateBuffers();
};

// CUDA kernel declarations (implemented in preprocess.cu)
extern "C" void launchCropBGRxToRGBFloat(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int cropX, int cropY, int cropW, int cropH,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
);

extern "C" void launchWarpFaceToArcFace(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    const float landmarks[5][2],
    cudaStream_t stream
);

} // namespace mmoment
