"""
Pose Detection Service

GPU-accelerated pose detection using YOLOv8 pose model.
Provides body keypoints for CV apps in real-time.
"""

import cv2
import logging
import numpy as np
from typing import Optional, List, Dict
import threading

logger = logging.getLogger("PoseService")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available")


class PoseService:
    """
    Service for real-time pose detection.
    Runs YOLO pose model on GPU, provides keypoints to apps.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PoseService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.pose_model = None
        self.enabled = False
        self.visualization_enabled = True  # Draw skeletons by default

        # Storage for latest pose detections (for visualization)
        self._results_lock = threading.Lock()
        self._latest_poses = []

        # YOLO pose keypoint indices (COCO format)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        self.KEYPOINT_NAMES = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder',
            7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle'
        }

        # Skeleton connections for visualization
        self.skeleton_connections = [
            (5, 6),   # shoulders
            (5, 7),   # left shoulder - elbow
            (7, 9),   # left elbow - wrist
            (6, 8),   # right shoulder - elbow
            (8, 10),  # right elbow - wrist
            (5, 11),  # left shoulder - hip
            (6, 12),  # right shoulder - hip
            (11, 12), # hips
            (11, 13), # left hip - knee
            (13, 15), # left knee - ankle
            (12, 14), # right hip - knee
            (14, 16), # right knee - ankle
        ]

        logger.info("PoseService initialized")

    def start(self) -> bool:
        """Initialize and start pose detection"""
        if not YOLO_AVAILABLE:
            logger.error("Cannot start pose service - YOLO not available")
            return False

        try:
            # Load YOLOv8n-pose (nano model for speed)
            self.pose_model = YOLO('yolov8n-pose.pt')
            self.pose_model.to('cuda')
            self.enabled = True
            logger.info("Pose detection started on GPU")
            return True
        except Exception as e:
            logger.error(f"Failed to start pose service: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection on frame with tracking.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of detected poses with keypoints:
            [{
                'keypoints': np.ndarray,  # (17, 3) - x, y, confidence
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'track_id': int  # Persistent tracking ID
            }]
        """
        if not self.enabled or self.pose_model is None:
            return []

        try:
            # Enable tracking with BoT-SORT for persistent IDs (same as face detection used)
            results = self.pose_model.track(
                frame,
                verbose=False,
                persist=True,
                tracker='/app/botsort_reid.yaml',
                conf=0.25,
                iou=0.5
            )

            if not results or len(results) == 0:
                return []

            result = results[0]

            if result.keypoints is None or len(result.keypoints) == 0:
                return []

            poses = []

            # Process each detected person
            for i in range(len(result.keypoints)):
                keypoints_data = result.keypoints[i]

                if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
                    continue

                # Get keypoints as numpy array (17, 2)
                kpts_xy = keypoints_data.xy[0].cpu().numpy()

                # Get confidence scores if available
                if keypoints_data.conf is not None:
                    kpts_conf = keypoints_data.conf[0].cpu().numpy()
                else:
                    kpts_conf = np.ones(17)

                # Combine into (17, 3) array
                keypoints = np.zeros((17, 3))
                keypoints[:, :2] = kpts_xy
                keypoints[:, 2] = kpts_conf

                # Get bbox and track_id if available
                bbox = None
                track_id = None
                if result.boxes is not None and i < len(result.boxes):
                    box = result.boxes[i]
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    # Extract track ID from BoT-SORT tracking
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0]) if hasattr(box.id, '__iter__') else int(box.id)

                poses.append({
                    'keypoints': keypoints,
                    'bbox': bbox,
                    'confidence': kpts_conf.mean(),
                    'track_id': track_id  # Persistent ID from BoT-SORT
                })

            return poses

        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            return []

    def store_poses(self, poses: List[Dict]):
        """Store detected poses for visualization"""
        with self._results_lock:
            self._latest_poses = poses

    def get_poses(self) -> List[Dict]:
        """Get stored poses"""
        with self._results_lock:
            return self._latest_poses.copy()

    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw pose skeletons on frame.
        Reads stored pose data and draws visualization.
        OPTIMIZED: Frame is already a copy from buffer_service, draw directly on it.
        """
        if frame is None or not self.visualization_enabled:
            return frame

        with self._results_lock:
            poses = self._latest_poses.copy()

        if not poses:
            return frame

        # Draw skeletons directly on frame (it's already a copy)
        for pose in poses:
            keypoints = pose['keypoints']

            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:  # Only draw confident keypoints
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Draw skeleton connections
            for start_idx, end_idx in self.skeleton_connections:
                start_kpt = keypoints[start_idx]
                end_kpt = keypoints[end_idx]

                # Only draw if both keypoints are confident
                if start_kpt[2] > 0.5 and end_kpt[2] > 0.5:
                    start_point = (int(start_kpt[0]), int(start_kpt[1]))
                    end_point = (int(end_kpt[0]), int(end_kpt[1]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        return frame

    def stop(self):
        """Stop pose detection"""
        self.enabled = False
        self.pose_model = None
        logger.info("Pose detection stopped")


# Global instance
_pose_service = None

def get_pose_service() -> PoseService:
    """Get the pose service singleton instance"""
    global _pose_service
    if _pose_service is None:
        _pose_service = PoseService()
    return _pose_service
