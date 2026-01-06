/**
 * byte_tracker.h - Lightweight ByteTrack implementation for pose tracking
 *
 * Based on ByteTrack paper: https://arxiv.org/abs/2110.06864
 * Provides smooth tracking with Kalman filtering and IoU association.
 */

#pragma once

#include <vector>
#include <memory>
#include <cmath>

namespace mmoment {

// Kalman filter for bounding box tracking
// State: [x, y, w, h, vx, vy, vw, vh]
class KalmanFilter {
public:
    KalmanFilter();

    // Initialize with measurement [x, y, w, h]
    void init(float x, float y, float w, float h);

    // Predict next state
    void predict();

    // Update with measurement
    void update(float x, float y, float w, float h);

    // Get current state
    void getState(float& x, float& y, float& w, float& h) const;

    // Get predicted state (after predict, before update)
    void getPredicted(float& x, float& y, float& w, float& h) const;

private:
    float m_state[8];       // [x, y, w, h, vx, vy, vw, vh]
    float m_P[8][8];        // Covariance matrix
    bool m_initialized;
};

// Single tracked object
struct STrack {
    int trackId;
    int frameId;            // Last seen frame
    int startFrame;         // First seen frame
    int hits;               // Total detections
    int timeSinceUpdate;    // Frames since last detection

    KalmanFilter kalman;

    // Current state
    float x1, y1, x2, y2;   // Bounding box
    float score;

    // Smoothed keypoints (17 COCO keypoints)
    float keypoints[17][3]; // x, y, conf (raw from detection)
    float keypointSmooth[17][2]; // Final smoothed x, y (for output)
    float keypointOffsets[17][2]; // Smoothed offsets from bbox center

    // Track state
    enum State { New, Tracked, Lost, Removed };
    State state;

    // ReID embedding (512-dim, from OSNet)
    // Updated when person is visible, used for re-identification
    float reidEmbedding[512];
    bool hasReidEmbedding;          // True if embedding has been computed
    int reidUpdateFrame;            // Frame when embedding was last updated

    STrack();
    void activate(int frameId, int newId);
    void reActivate(const STrack& newTrack, int frameId);
    void update(const STrack& newTrack, int frameId);
    void markLost();
    void markRemoved();

    // Get smoothed bounding box
    void getSmoothedBox(float& x1, float& y1, float& x2, float& y2) const;

    // Update ReID embedding
    void updateReidEmbedding(const float* embedding, int frameId);
};

// Detection input for tracker
struct Detection {
    float x1, y1, x2, y2;
    float score;
    float keypoints[17][3];
};

// Main tracker class
class ByteTracker {
public:
    ByteTracker(int maxAge = 30, float highThresh = 0.6f, float lowThresh = 0.1f,
                float matchThresh = 0.8f);

    // Update tracker with new detections, returns tracked objects
    std::vector<STrack> update(const std::vector<Detection>& detections, int frameId);

    // Get all active tracks
    const std::vector<STrack>& getTrackedTracks() const { return m_trackedTracks; }

private:
    // Compute IoU between two boxes
    static float iou(float x1a, float y1a, float x2a, float y2a,
                     float x1b, float y1b, float x2b, float y2b);

    // Linear assignment using IoU matrix
    void linearAssignment(const std::vector<std::vector<float>>& costMatrix,
                         float thresh,
                         std::vector<std::pair<int, int>>& matches,
                         std::vector<int>& unmatchedA,
                         std::vector<int>& unmatchedB,
                         int numB);

    std::vector<STrack> m_trackedTracks;
    std::vector<STrack> m_lostTracks;

    int m_frameId;
    int m_trackIdCount;
    int m_maxAge;           // Max frames to keep lost track
    float m_highThresh;     // High confidence threshold
    float m_lowThresh;      // Low confidence threshold
    float m_matchThresh;    // IoU threshold for matching
};

} // namespace mmoment
