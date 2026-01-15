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
 * SCRFD detector - InsightFace's improved face detector for small faces
 * Output: 9 tensors (score, bbox, kps at 3 FPN levels: stride 8, 16, 32)
 */
class SCRFDEngine {
public:
    SCRFDEngine();
    ~SCRFDEngine();

    bool loadEngine(const std::string& enginePath);

    // Run face detection on preprocessed input (GPU buffer, RGB float CHW, 640x640)
    // Returns detected faces with landmarks in the original crop coordinates
    std::vector<FaceDetection> detect(void* gpuInput, int cropW, int cropH,
                                       float scaleRatio, float confThreshold = 0.5f,
                                       float nmsThreshold = 0.4f);

private:
    bool allocateBuffers();
    void freeBuffers();

    // Anchor generation for SCRFD
    void initAnchors();

    // Generate proposals from a single FPN level
    void generateProposals(int stride, int featW, int featH, int numAnchors,
                          const float* scoreData, const float* bboxData, const float* kpsData,
                          float confThreshold, float scaleRatio, int wpad, int hpad,
                          int origW, int origH,
                          std::vector<FaceDetection>& proposals);

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    // Pre-allocated GPU buffers
    void* m_inputBuffer = nullptr;
    void* m_score8Buffer = nullptr;
    void* m_bbox8Buffer = nullptr;
    void* m_kps8Buffer = nullptr;
    void* m_score16Buffer = nullptr;
    void* m_bbox16Buffer = nullptr;
    void* m_kps16Buffer = nullptr;
    void* m_score32Buffer = nullptr;
    void* m_bbox32Buffer = nullptr;
    void* m_kps32Buffer = nullptr;

    // Host buffers for output data
    std::vector<float> m_score8Host, m_bbox8Host, m_kps8Host;
    std::vector<float> m_score16Host, m_bbox16Host, m_kps16Host;
    std::vector<float> m_score32Host, m_bbox32Host, m_kps32Host;

    // Anchor configuration (2 anchors per position, 3 strides)
    float m_anchorSizes[3][2];  // [stride_idx][anchor_idx]

    cudaStream_t m_stream = nullptr;
    std::unique_ptr<TRTLogger> m_logger;

    // Constants
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int FEAT_W_8 = 80;
    static constexpr int FEAT_H_8 = 80;
    static constexpr int FEAT_W_16 = 40;
    static constexpr int FEAT_H_16 = 40;
    static constexpr int FEAT_W_32 = 20;
    static constexpr int FEAT_H_32 = 20;
    static constexpr int NUM_ANCHORS = 2;
};

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
 * Now uses SCRFD for better small face detection
 */
class InsightFacePipeline {
public:
    InsightFacePipeline();
    ~InsightFacePipeline();

    // Load both engines (detector path can be SCRFD or RetinaFace)
    bool initialize(const std::string& detectorPath, const std::string& arcFacePath);

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

    // Get embedding from pre-aligned 112x112 face (already in GPU as RGB float CHW)
    // Input must be already converted to ArcFace format: 112x112 RGB float CHW, [-1,1] normalized
    bool getEmbeddingFromAligned(void* alignedFaceGpu, float* embedding);

private:
    SCRFDEngine m_scrfdDetector;       // SCRFD for better small face detection
    RetinaFaceEngine m_retinaDetector; // RetinaFace fallback (not used currently)
    ArcFaceEngine m_recognizer;
    bool m_useSCRFD = true;            // Use SCRFD by default

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
