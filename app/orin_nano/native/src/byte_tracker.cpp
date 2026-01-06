/**
 * byte_tracker.cpp - Lightweight ByteTrack implementation
 */

#include "byte_tracker.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace mmoment {

// =============================================================================
// Kalman Filter
// =============================================================================

KalmanFilter::KalmanFilter() : m_initialized(false) {
    std::memset(m_state, 0, sizeof(m_state));
    std::memset(m_P, 0, sizeof(m_P));
}

void KalmanFilter::init(float x, float y, float w, float h) {
    // State: [cx, cy, w, h, vx, vy, vw, vh]
    m_state[0] = x + w / 2;  // center x
    m_state[1] = y + h / 2;  // center y
    m_state[2] = w;
    m_state[3] = h;
    m_state[4] = 0;  // velocity x
    m_state[5] = 0;  // velocity y
    m_state[6] = 0;  // velocity w
    m_state[7] = 0;  // velocity h

    // Initialize covariance with high uncertainty for velocities
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            m_P[i][j] = (i == j) ? (i < 4 ? 10.0f : 100.0f) : 0.0f;
        }
    }
    m_initialized = true;
}

void KalmanFilter::predict() {
    if (!m_initialized) return;

    // Simple constant velocity model: x_new = x + v * dt (dt = 1)
    // Apply velocity with slight damping to reduce oscillation
    const float velDamping = 0.95f;
    m_state[0] += m_state[4];
    m_state[1] += m_state[5];
    m_state[2] += m_state[6];
    m_state[3] += m_state[7];

    // Dampen velocities to prevent runaway predictions
    m_state[4] *= velDamping;
    m_state[5] *= velDamping;
    m_state[6] *= velDamping;
    m_state[7] *= velDamping;

    // Increase uncertainty (process noise) - reduced for smoother tracking
    for (int i = 0; i < 4; i++) {
        m_P[i][i] += 0.3f;  // Reduced from 1.0f
    }
    for (int i = 4; i < 8; i++) {
        m_P[i][i] += 0.15f;  // Reduced from 0.5f
    }
}

void KalmanFilter::update(float x, float y, float w, float h) {
    if (!m_initialized) {
        init(x, y, w, h);
        return;
    }

    // Measurement: [cx, cy, w, h]
    float z[4] = {x + w / 2, y + h / 2, w, h};

    // Innovation (measurement residual)
    float y_innov[4];
    for (int i = 0; i < 4; i++) {
        y_innov[i] = z[i] - m_state[i];
    }

    // Adaptive Kalman gain based on covariance
    // Lower gains = smoother but slower response
    // Position gains lower for stability, size gains even lower to prevent jitter
    float K[4] = {0.25f, 0.25f, 0.15f, 0.15f};  // Reduced from {0.5, 0.5, 0.3, 0.3}

    // State update
    for (int i = 0; i < 4; i++) {
        m_state[i] += K[i] * y_innov[i];
    }

    // Velocity update with heavy smoothing to prevent oscillation
    // Use exponential smoothing with low alpha for stable velocity estimation
    const float velAlpha = 0.15f;  // Reduced from 0.3f
    m_state[4] = (1.0f - velAlpha) * m_state[4] + velAlpha * y_innov[0];
    m_state[5] = (1.0f - velAlpha) * m_state[5] + velAlpha * y_innov[1];
    m_state[6] = (1.0f - velAlpha) * m_state[6] + velAlpha * y_innov[2];
    m_state[7] = (1.0f - velAlpha) * m_state[7] + velAlpha * y_innov[3];

    // Reduce covariance
    for (int i = 0; i < 8; i++) {
        m_P[i][i] *= (1.0f - K[i % 4]);
    }
}

void KalmanFilter::getState(float& x, float& y, float& w, float& h) const {
    w = std::max(1.0f, m_state[2]);
    h = std::max(1.0f, m_state[3]);
    x = m_state[0] - w / 2;
    y = m_state[1] - h / 2;
}

void KalmanFilter::getPredicted(float& x, float& y, float& w, float& h) const {
    getState(x, y, w, h);
}

// =============================================================================
// STrack
// =============================================================================

STrack::STrack()
    : trackId(-1), frameId(0), startFrame(0), hits(0), timeSinceUpdate(0),
      x1(0), y1(0), x2(0), y2(0), score(0), state(New),
      hasReidEmbedding(false), reidUpdateFrame(-1) {
    std::memset(keypoints, 0, sizeof(keypoints));
    std::memset(keypointSmooth, 0, sizeof(keypointSmooth));
    std::memset(keypointOffsets, 0, sizeof(keypointOffsets));
    std::memset(reidEmbedding, 0, sizeof(reidEmbedding));
}

void STrack::activate(int frame, int newId) {
    trackId = newId;
    frameId = frame;
    startFrame = frame;
    hits = 1;
    timeSinceUpdate = 0;
    state = Tracked;

    kalman.init(x1, y1, x2 - x1, y2 - y1);

    // Initialize keypoint offsets from bbox center
    float centerX = (x1 + x2) / 2.0f;
    float centerY = (y1 + y2) / 2.0f;

    for (int i = 0; i < 17; i++) {
        keypointOffsets[i][0] = keypoints[i][0] - centerX;
        keypointOffsets[i][1] = keypoints[i][1] - centerY;
        keypointSmooth[i][0] = keypoints[i][0];
        keypointSmooth[i][1] = keypoints[i][1];
    }
}

void STrack::reActivate(const STrack& newTrack, int frame) {
    x1 = newTrack.x1;
    y1 = newTrack.y1;
    x2 = newTrack.x2;
    y2 = newTrack.y2;
    score = newTrack.score;
    std::memcpy(keypoints, newTrack.keypoints, sizeof(keypoints));

    kalman.update(x1, y1, x2 - x1, y2 - y1);

    // Re-initialize offsets for reactivated track
    float centerX = (x1 + x2) / 2.0f;
    float centerY = (y1 + y2) / 2.0f;

    for (int i = 0; i < 17; i++) {
        if (keypoints[i][2] > 0.5f) {
            keypointOffsets[i][0] = keypoints[i][0] - centerX;
            keypointOffsets[i][1] = keypoints[i][1] - centerY;
            keypointSmooth[i][0] = keypoints[i][0];
            keypointSmooth[i][1] = keypoints[i][1];
        }
    }

    frameId = frame;
    hits++;
    timeSinceUpdate = 0;
    state = Tracked;
}

void STrack::update(const STrack& newTrack, int frame) {
    x1 = newTrack.x1;
    y1 = newTrack.y1;
    x2 = newTrack.x2;
    y2 = newTrack.y2;
    score = newTrack.score;
    std::memcpy(keypoints, newTrack.keypoints, sizeof(keypoints));

    kalman.update(x1, y1, x2 - x1, y2 - y1);

    // Get the SMOOTHED bbox center from Kalman filter
    float smoothX1, smoothY1, smoothX2, smoothY2;
    kalman.getState(smoothX1, smoothY1, smoothX2, smoothY2);
    smoothX2 += smoothX1; // getState returns x,y,w,h
    smoothY2 += smoothY1;
    float smoothCenterX = (smoothX1 + smoothX2) / 2.0f;
    float smoothCenterY = (smoothY1 + smoothY2) / 2.0f;

    // Detection bbox center (raw)
    float detCenterX = (x1 + x2) / 2.0f;
    float detCenterY = (y1 + y2) / 2.0f;

    // Smoothing parameters
    const float offsetAlpha = 0.4f;  // EMA for offset smoothing (lower = smoother)
    const float minConfidence = 0.5f;

    for (int i = 0; i < 17; i++) {
        if (keypoints[i][2] > minConfidence) {
            // Calculate offset from detection bbox center
            float newOffsetX = keypoints[i][0] - detCenterX;
            float newOffsetY = keypoints[i][1] - detCenterY;

            // Smooth the offsets with EMA
            keypointOffsets[i][0] = offsetAlpha * newOffsetX + (1.0f - offsetAlpha) * keypointOffsets[i][0];
            keypointOffsets[i][1] = offsetAlpha * newOffsetY + (1.0f - offsetAlpha) * keypointOffsets[i][1];

            // Apply smoothed offsets to SMOOTHED bbox center
            keypointSmooth[i][0] = smoothCenterX + keypointOffsets[i][0];
            keypointSmooth[i][1] = smoothCenterY + keypointOffsets[i][1];
        }
        // If below threshold, keep previous smoothed position
    }

    frameId = frame;
    hits++;
    timeSinceUpdate = 0;
}

void STrack::markLost() {
    state = Lost;
}

void STrack::markRemoved() {
    state = Removed;
}

void STrack::getSmoothedBox(float& sx1, float& sy1, float& sx2, float& sy2) const {
    float x, y, w, h;
    kalman.getState(x, y, w, h);
    sx1 = x;
    sy1 = y;
    sx2 = x + w;
    sy2 = y + h;
}

void STrack::updateReidEmbedding(const float* embedding, int frame) {
    std::memcpy(reidEmbedding, embedding, 512 * sizeof(float));
    hasReidEmbedding = true;
    reidUpdateFrame = frame;
}

// =============================================================================
// ByteTracker
// =============================================================================

ByteTracker::ByteTracker(int maxAge, float highThresh, float lowThresh, float matchThresh)
    : m_frameId(0), m_trackIdCount(0), m_maxAge(maxAge),
      m_highThresh(highThresh), m_lowThresh(lowThresh), m_matchThresh(matchThresh) {}

float ByteTracker::iou(float x1a, float y1a, float x2a, float y2a,
                       float x1b, float y1b, float x2b, float y2b) {
    float xx1 = std::max(x1a, x1b);
    float yy1 = std::max(y1a, y1b);
    float xx2 = std::min(x2a, x2b);
    float yy2 = std::min(y2a, y2b);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float intersection = w * h;

    float areaA = (x2a - x1a) * (y2a - y1a);
    float areaB = (x2b - x1b) * (y2b - y1b);
    float unionArea = areaA + areaB - intersection;

    return unionArea > 0 ? intersection / unionArea : 0.0f;
}

void ByteTracker::linearAssignment(const std::vector<std::vector<float>>& costMatrix,
                                   float thresh,
                                   std::vector<std::pair<int, int>>& matches,
                                   std::vector<int>& unmatchedA,
                                   std::vector<int>& unmatchedB,
                                   int numB) {
    matches.clear();
    unmatchedA.clear();
    unmatchedB.clear();

    if (costMatrix.empty()) {
        // No rows (no A items) - all B items are unmatched
        for (int i = 0; i < numB; i++) unmatchedB.push_back(i);
        return;
    }

    if (costMatrix[0].empty()) {
        // No columns (no B items) - all A items are unmatched
        for (size_t i = 0; i < costMatrix.size(); i++) unmatchedA.push_back(i);
        return;
    }

    int n = costMatrix.size();
    int m = costMatrix[0].size();

    std::vector<bool> usedA(n, false);
    std::vector<bool> usedB(m, false);

    // Greedy matching (simplified Hungarian)
    for (int iter = 0; iter < std::min(n, m); iter++) {
        float bestCost = thresh;
        int bestI = -1, bestJ = -1;

        for (int i = 0; i < n; i++) {
            if (usedA[i]) continue;
            for (int j = 0; j < m; j++) {
                if (usedB[j]) continue;
                // Cost matrix is 1 - IoU, so lower is better
                if (costMatrix[i][j] < bestCost) {
                    bestCost = costMatrix[i][j];
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI >= 0) {
            matches.push_back({bestI, bestJ});
            usedA[bestI] = true;
            usedB[bestJ] = true;
        } else {
            break;
        }
    }

    for (int i = 0; i < n; i++) {
        if (!usedA[i]) unmatchedA.push_back(i);
    }
    for (int j = 0; j < m; j++) {
        if (!usedB[j]) unmatchedB.push_back(j);
    }
}

std::vector<STrack> ByteTracker::update(const std::vector<Detection>& detections, int frameId) {
    m_frameId = frameId;

    // Separate high and low confidence detections
    std::vector<STrack> highDets, lowDets;
    for (const auto& det : detections) {
        STrack track;
        track.x1 = det.x1;
        track.y1 = det.y1;
        track.x2 = det.x2;
        track.y2 = det.y2;
        track.score = det.score;
        std::memcpy(track.keypoints, det.keypoints, sizeof(det.keypoints));

        if (det.score >= m_highThresh) {
            highDets.push_back(track);
        } else if (det.score >= m_lowThresh) {
            lowDets.push_back(track);
        }
    }

    // Predict existing tracks
    for (auto& track : m_trackedTracks) {
        track.kalman.predict();
        track.timeSinceUpdate++;
    }
    for (auto& track : m_lostTracks) {
        track.kalman.predict();
        track.timeSinceUpdate++;
    }

    // First association: tracked tracks with high confidence detections
    std::vector<std::vector<float>> costMatrix;
    for (const auto& track : m_trackedTracks) {
        std::vector<float> row;
        float tx1, ty1, tx2, ty2;
        track.getSmoothedBox(tx1, ty1, tx2, ty2);
        for (const auto& det : highDets) {
            float iouVal = iou(tx1, ty1, tx2, ty2, det.x1, det.y1, det.x2, det.y2);
            row.push_back(1.0f - iouVal);  // Cost = 1 - IoU
        }
        costMatrix.push_back(row);
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatchedTracks, unmatchedDets;
    linearAssignment(costMatrix, 1.0f - m_matchThresh, matches, unmatchedTracks, unmatchedDets, highDets.size());

    // Collect matched tracks from first association
    std::vector<STrack> matchedTracks;
    for (const auto& match : matches) {
        m_trackedTracks[match.first].update(highDets[match.second], frameId);
        matchedTracks.push_back(m_trackedTracks[match.first]);
    }

    // Second association: unmatched tracks with low confidence detections
    std::vector<STrack> remainTracks;
    for (int idx : unmatchedTracks) {
        remainTracks.push_back(m_trackedTracks[idx]);
    }

    // Clear m_trackedTracks - we'll rebuild it properly
    m_trackedTracks.clear();

    costMatrix.clear();
    for (const auto& track : remainTracks) {
        std::vector<float> row;
        float tx1, ty1, tx2, ty2;
        track.getSmoothedBox(tx1, ty1, tx2, ty2);
        for (const auto& det : lowDets) {
            float iouVal = iou(tx1, ty1, tx2, ty2, det.x1, det.y1, det.x2, det.y2);
            row.push_back(1.0f - iouVal);
        }
        costMatrix.push_back(row);
    }

    std::vector<std::pair<int, int>> matches2;
    std::vector<int> unmatchedTracks2, unmatchedDets2;
    linearAssignment(costMatrix, 1.0f - 0.5f, matches2, unmatchedTracks2, unmatchedDets2, lowDets.size());

    for (const auto& match : matches2) {
        remainTracks[match.first].update(lowDets[match.second], frameId);
    }

    // Mark unmatched tracks as lost
    for (int idx : unmatchedTracks2) {
        if (remainTracks[idx].state == STrack::Tracked) {
            remainTracks[idx].markLost();
            m_lostTracks.push_back(remainTracks[idx]);
        }
    }

    // Third association: lost tracks with remaining high confidence detections
    std::vector<STrack> remainDets;
    for (int idx : unmatchedDets) {
        remainDets.push_back(highDets[idx]);
    }

    costMatrix.clear();
    for (const auto& track : m_lostTracks) {
        std::vector<float> row;
        float tx1, ty1, tx2, ty2;
        track.getSmoothedBox(tx1, ty1, tx2, ty2);
        for (const auto& det : remainDets) {
            float iouVal = iou(tx1, ty1, tx2, ty2, det.x1, det.y1, det.x2, det.y2);
            row.push_back(1.0f - iouVal);
        }
        costMatrix.push_back(row);
    }

    std::vector<std::pair<int, int>> matches3;
    std::vector<int> unmatchedLost, unmatchedNew;
    linearAssignment(costMatrix, 1.0f - m_matchThresh, matches3, unmatchedLost, unmatchedNew, remainDets.size());

    for (const auto& match : matches3) {
        m_lostTracks[match.first].reActivate(remainDets[match.second], frameId);
    }

    // Move reactivated tracks back to tracked
    std::vector<STrack> stillLost;
    for (size_t i = 0; i < m_lostTracks.size(); i++) {
        if (m_lostTracks[i].state == STrack::Tracked) {
            // Reactivated
        } else if (m_lostTracks[i].timeSinceUpdate < m_maxAge) {
            stillLost.push_back(m_lostTracks[i]);
        }
        // else: removed
    }

    // Rebuild tracked tracks list properly
    std::vector<STrack> newTracked;

    // 1. Add matched tracks from first association (updated with highDets)
    for (const auto& track : matchedTracks) {
        newTracked.push_back(track);
    }

    // 2. Add matched tracks from second association (remainTracks updated with lowDets)
    for (const auto& match : matches2) {
        newTracked.push_back(remainTracks[match.first]);
    }

    // 3. Add reactivated tracks from third association
    for (const auto& match : matches3) {
        if (m_lostTracks[match.first].state == STrack::Tracked) {
            newTracked.push_back(m_lostTracks[match.first]);
        }
    }

    // 4. Add new tracks from unmatched detections
    for (int idx : unmatchedNew) {
        STrack newTrack = remainDets[idx];
        newTrack.activate(frameId, ++m_trackIdCount);
        newTracked.push_back(newTrack);
    }

    m_trackedTracks = newTracked;
    m_lostTracks = stillLost;

    return m_trackedTracks;
}

} // namespace mmoment
