/**
 * native_pipeline_api.h - C API for MMOMENT native inference pipeline
 *
 * This API allows Python (via ctypes) to call into the native C++ pipeline
 * for high-performance GPU-accelerated inference.
 */

#ifndef NATIVE_PIPELINE_API_H
#define NATIVE_PIPELINE_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Data Structures
// ============================================================================

// Person detection from YOLOv8-pose
typedef struct {
    float x1, y1, x2, y2;       // Bounding box
    float confidence;
    int track_id;               // Unique track ID from ByteTracker
    float keypoints[17][3];     // 17 COCO keypoints (x, y, conf)
    char identity_label[64];    // Matched identity label (display name, empty if not matched)
    float identity_confidence;  // Confidence of identity match (0-1)
    // ReID embedding (512-dim, from OSNet)
    float reid_embedding[512];  // Body appearance embedding for re-identification
    int has_reid_embedding;     // 1 if embedding is valid, 0 otherwise
} NativePoseDetection;

// Face detection with landmarks
typedef struct {
    float x1, y1, x2, y2;       // Face bounding box
    float confidence;
    float landmarks[5][2];      // 5 facial landmarks (x, y)
} NativeFaceDetection;

// Face recognition result
typedef struct {
    NativeFaceDetection face;
    float embedding[512];       // ArcFace 512-dim embedding
    float quality;              // Face quality score
    int person_track_id;        // Track ID of the person this face belongs to (-1 if none)
    char identity_label[64];    // Matched identity label (empty if not matched)
    float identity_confidence;  // Confidence of identity match (0-1)
} NativeFaceRecognition;

// Frame processing result
typedef struct {
    // Persons detected
    NativePoseDetection* persons;
    int num_persons;

    // Faces recognized
    NativeFaceRecognition* faces;
    int num_faces;

    // Timing info (ms)
    float preprocess_ms;
    float yolo_ms;
    float face_ms;
    float total_ms;
} NativeFrameResult;

// ============================================================================
// Pipeline Lifecycle
// ============================================================================

/**
 * Initialize the native pipeline
 * @param yolo_engine_path Path to YOLOv8-pose TensorRT engine
 * @param retinaface_engine_path Path to RetinaFace TensorRT engine
 * @param arcface_engine_path Path to ArcFace TensorRT engine
 * @return 0 on success, non-zero on failure
 */
int native_pipeline_init(
    const char* yolo_engine_path,
    const char* retinaface_engine_path,
    const char* arcface_engine_path
);

/**
 * Check if pipeline is initialized
 * @return 1 if initialized, 0 otherwise
 */
int native_pipeline_is_ready(void);

/**
 * Shutdown the native pipeline and free resources
 */
void native_pipeline_shutdown(void);

// ============================================================================
// Frame Processing
// ============================================================================

/**
 * Process a frame (BGR format, CPU memory)
 * @param frame_data Pointer to BGR frame data
 * @param width Frame width
 * @param height Frame height
 * @param stride Bytes per row (usually width * 3 for BGR)
 * @param result Output result structure (caller allocated)
 * @return 0 on success, non-zero on failure
 */
int native_pipeline_process_bgr(
    const uint8_t* frame_data,
    int width,
    int height,
    int stride,
    NativeFrameResult* result
);

/**
 * Process a frame (BGRA/BGRx format, CPU memory)
 * @param frame_data Pointer to BGRA frame data
 * @param width Frame width
 * @param height Frame height
 * @param stride Bytes per row (usually width * 4 for BGRA)
 * @param result Output result structure (caller allocated)
 * @return 0 on success, non-zero on failure
 */
int native_pipeline_process_bgra(
    const uint8_t* frame_data,
    int width,
    int height,
    int stride,
    NativeFrameResult* result
);

/**
 * Free memory allocated in result structure
 * @param result Result structure to free
 */
void native_pipeline_free_result(NativeFrameResult* result);

// ============================================================================
// Configuration
// ============================================================================

/**
 * Set detection thresholds
 * @param person_conf Confidence threshold for person detection (0.0-1.0)
 * @param face_conf Confidence threshold for face detection (0.0-1.0)
 */
void native_pipeline_set_thresholds(float person_conf, float face_conf);

/**
 * Enable/disable face recognition (just detection if disabled)
 * @param enabled 1 to enable, 0 to disable
 */
void native_pipeline_enable_face_recognition(int enabled);

// ============================================================================
// Utility
// ============================================================================

/**
 * Compute cosine similarity between two embeddings
 * @param emb1 First 512-dim embedding
 * @param emb2 Second 512-dim embedding
 * @return Cosine similarity (-1 to 1, higher = more similar)
 */
float native_compute_similarity(const float* emb1, const float* emb2);

/**
 * Get version string
 * @return Version string (do not free)
 */
const char* native_pipeline_version(void);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_PIPELINE_API_H
