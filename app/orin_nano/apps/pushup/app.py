"""
Push-up Counter App

Angle-based push-up counting that auto-detects viewing angle.
Uses elbow angles to validate proper form from front, left, or right views.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sdk import BaseApp
from sdk.base_app import CompetitionApp
from typing import Dict, Any, List, Optional
import numpy as np
import time
import logging

logger = logging.getLogger("PushupApp")


class PushupApp(CompetitionApp):
    """
    Angle-based push-up counter with automatic view detection.

    Detects push-ups from front, left, or right camera angles by:
    1. Auto-detecting which view based on keypoint visibility
    2. Tracking elbow angles to validate form
    3. Using state machine to count complete reps
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Tracking state per competitor
        self.competitor_states = {}  # wallet -> state dict

        # Thresholds
        self.elbow_down_angle = 90    # Elbow must bend below 90° for "down"
        self.elbow_up_angle = 150     # Elbow must extend above 150° for "up"
        self.min_rep_time = 0.6       # Minimum time between reps (anti-cheat)
        self.confidence_threshold = 0.4  # Keypoint confidence threshold

        # Body position validation thresholds (relative to body size, scale-invariant)
        # These ensure person is actually in push-up position, not standing
        # Ratios are relative to shoulder width (visible reference in front view)
        self.max_torso_vertical_ratio = 1.5   # torso_vertical / shoulder_width - standing is typically 2.0+
        self.max_wrist_above_ratio = 1.0      # wrist_above_shoulder / shoulder_width

        logger.info("PushupApp initialized with angle-based detection")

    def init_competitor_stats(self) -> Dict:
        """Initialize stats for a competitor"""
        return {
            'reps': 0,
            'last_rep_time': None,
            'in_down_position': False,
            'current_angle': None,
            'view': None,  # 'front', 'left', 'right', or None
            'position_valid': False,  # Whether in valid push-up position
            'rejection_reason': None,  # Why position was rejected (if any)
        }

    def start_competition(self, competitors: List[Dict], duration_limit: int = None, competition_meta: dict = None):
        """Start competition"""
        super().start_competition(competitors, duration_limit, competition_meta)

        # Initialize state for each competitor
        for wallet in self.competitors.keys():
            self.competitor_states[wallet] = {
                'in_down': False,
                'last_rep_time': None,
                'view': None
            }

    def process(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process frame - count push-ups based on elbow angles.

        Args:
            frame_data: {
                'detections': [...],
                'keypoints': [...],  # List of pose dicts with 'keypoints' (17, 3) arrays
                'timestamp': float
            }
        """
        if not self.active:
            return {'state': self.get_state(), 'visualization': {}}

        detections = frame_data.get('detections', [])
        all_keypoints = frame_data.get('keypoints', [])
        timestamp = frame_data.get('timestamp', time.time())

        # Match keypoints to competitors by track_id
        for detection in detections:
            wallet = detection.get('wallet_address')
            if not wallet or wallet not in self.competitors:
                continue

            track_id = detection.get('track_id')
            if track_id is None:
                continue

            # Find matching pose by track_id
            pose = self._find_pose_by_track_id(all_keypoints, track_id)
            if pose is None:
                continue

            keypoints = pose.get('keypoints')
            if keypoints is None:
                continue

            # Process this competitor's push-up
            self._process_competitor(wallet, keypoints, timestamp)

        # Build visualization
        viz = self._build_visualization()

        return {
            'state': self.get_state(),
            'visualization': viz
        }

    def _find_pose_by_track_id(self, all_keypoints: List[Dict], track_id: int) -> Optional[Dict]:
        """Find pose data matching a track_id"""
        for pose in all_keypoints:
            if pose.get('track_id') == track_id:
                return pose
        return None

    def _process_competitor(self, wallet: str, keypoints: np.ndarray, timestamp: float):
        """Process push-up for one competitor using angle detection"""
        competitor = self.competitors[wallet]
        state = self.competitor_states[wallet]
        stats = competitor['stats']

        # Auto-detect view and get elbow angle
        view, elbow_angle = self._detect_view_and_angle(keypoints)

        if view is None or elbow_angle is None:
            # Can't detect view or angle - skip this frame
            stats['view'] = None
            stats['current_angle'] = None
            return

        # Validate body position - must be in push-up position, not standing
        is_valid_position, rejection_reason = self._is_in_pushup_position(keypoints, view)

        if not is_valid_position:
            # Not in push-up position - don't count anything
            stats['view'] = view
            stats['current_angle'] = elbow_angle
            stats['position_valid'] = False
            stats['rejection_reason'] = rejection_reason
            return

        # Update stats
        stats['view'] = view
        stats['current_angle'] = elbow_angle
        stats['position_valid'] = True
        stats['rejection_reason'] = None

        # State machine: UP → DOWN → UP (completes rep)
        if elbow_angle < self.elbow_down_angle:
            # Elbows bent - in down position
            if not state['in_down']:
                state['in_down'] = True
                stats['in_down_position'] = True
                logger.info(f"{competitor['display_name']}: DOWN ({view}, angle={elbow_angle:.1f}°)")

        elif elbow_angle > self.elbow_up_angle:
            # Elbows extended - back up
            if state['in_down']:
                # Check rep timing (anti-cheat)
                if state['last_rep_time'] is None or (timestamp - state['last_rep_time']) >= self.min_rep_time:
                    # Count rep!
                    stats['reps'] += 1
                    state['last_rep_time'] = timestamp
                    stats['last_rep_time'] = timestamp
                    logger.info(f"{competitor['display_name']}: REP #{stats['reps']} ({view})")

                state['in_down'] = False
                stats['in_down_position'] = False

    def _detect_view_and_angle(self, keypoints: np.ndarray) -> tuple[Optional[str], Optional[float]]:
        """
        Auto-detect viewing angle and calculate elbow angle.

        Returns:
            (view, elbow_angle) where view is 'front', 'left', 'right', or None
        """
        # Keypoint indices (COCO format):
        # 5: left_shoulder, 6: right_shoulder
        # 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]

        # Check confidence
        conf_thresh = self.confidence_threshold
        left_conf = min(left_shoulder[2], left_elbow[2], left_wrist[2])
        right_conf = min(right_shoulder[2], right_elbow[2], right_wrist[2])

        # Determine view based on keypoint visibility
        left_visible = left_conf > conf_thresh
        right_visible = right_conf > conf_thresh

        if left_visible and right_visible:
            # Both sides visible - front view, use average of both elbows
            left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)

            if left_angle is not None and right_angle is not None:
                return 'front', (left_angle + right_angle) / 2
            elif left_angle is not None:
                return 'front', left_angle
            elif right_angle is not None:
                return 'front', right_angle
            else:
                return None, None

        elif left_visible:
            # Only left side visible - left side view
            angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            return ('left', angle) if angle is not None else (None, None)

        elif right_visible:
            # Only right side visible - right side view
            angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            return ('right', angle) if angle is not None else (None, None)

        else:
            # No clear view
            return None, None

    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> Optional[float]:
        """
        Calculate angle at point2 formed by point1-point2-point3.

        Args:
            point1, point2, point3: Keypoints as [x, y, confidence]

        Returns:
            Angle in degrees, or None if calculation fails
        """
        try:
            # Extract coordinates
            p1 = point1[:2]
            p2 = point2[:2]
            p3 = point3[:2]

            # Create vectors
            v1 = p1 - p2  # shoulder -> elbow
            v2 = p3 - p2  # wrist -> elbow

            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            # Clamp to valid range to handle numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angle = np.degrees(np.arccos(cos_angle))
            return float(angle)

        except Exception as e:
            logger.debug(f"Angle calculation failed: {e}")
            return None

    def _is_in_pushup_position(self, keypoints: np.ndarray, view: str) -> tuple[bool, str]:
        """
        Validate that person is actually in a push-up position, not standing.

        Uses SCALE-INVARIANT ratios based on shoulder width, so it works
        regardless of how far the person is from the camera.

        Key insight for front-facing view:
        - When STANDING: torso_vertical / shoulder_width is HIGH (1.5+)
        - When in PUSH-UP: torso_vertical / shoulder_width is LOW (<0.8)

        Args:
            keypoints: COCO format keypoints array
            view: Detected view ('front', 'left', 'right')

        Returns:
            (is_valid, reason) - True if in valid push-up position
        """
        conf_thresh = self.confidence_threshold

        # COCO keypoint indices
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        if view == 'front':
            # Check shoulder visibility (required for reference measurement)
            shoulders_visible = (left_shoulder[2] > conf_thresh and
                               right_shoulder[2] > conf_thresh)
            hips_visible = (left_hip[2] > conf_thresh and
                          right_hip[2] > conf_thresh)
            wrists_visible = (left_wrist[2] > conf_thresh and
                            right_wrist[2] > conf_thresh)

            if not shoulders_visible:
                return False, "shoulders_not_visible"

            # Calculate shoulder width as our scale reference
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            if shoulder_width < 10:  # Too small to be reliable
                return False, "shoulders_too_close"

            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            wrist_ratio = None  # Track for logging

            # Check 1: Wrists should not be far above shoulders (rules out overhead arm movements)
            if wrists_visible:
                avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
                wrist_above_shoulder = avg_shoulder_y - avg_wrist_y  # positive = wrists above shoulders

                wrist_ratio = wrist_above_shoulder / shoulder_width
                if wrist_ratio > self.max_wrist_above_ratio:
                    logger.info(f"REJECTED: wrists above shoulders (ratio={wrist_ratio:.2f} > {self.max_wrist_above_ratio})")
                    return False, "wrists_above_head"

            # Check 2: Body should be roughly horizontal
            # When standing: torso_vertical / shoulder_width is HIGH (person is upright)
            # When in push-up: torso_vertical / shoulder_width is LOW (body is horizontal, going into camera)
            if hips_visible:
                avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                torso_vertical = abs(avg_hip_y - avg_shoulder_y)

                torso_ratio = torso_vertical / shoulder_width

                # Only log occasionally to reduce spam (every ~30 frames worth)
                # logger.debug(f"POSITION CHECK: torso_ratio={torso_ratio:.2f}, wrist_ratio={wrist_ratio if wrist_ratio else 'N/A'}")

                if torso_ratio > self.max_torso_vertical_ratio:
                    logger.info(f"REJECTED: standing position (torso_ratio={torso_ratio:.2f} > {self.max_torso_vertical_ratio})")
                    return False, "standing_position"

            return True, "valid"

        elif view in ('left', 'right'):
            # For side views, use shoulder-to-hip distance as reference
            shoulder = left_shoulder if view == 'left' else right_shoulder
            wrist = left_wrist if view == 'left' else right_wrist
            hip = left_hip if view == 'left' else right_hip

            if shoulder[2] < conf_thresh:
                return False, "shoulder_not_visible"

            # For side view, we need a different reference - use torso length when horizontal
            # This is trickier, so we'll be more permissive
            if hip[2] > conf_thresh:
                # In side view during push-up, the torso should be roughly horizontal
                # meaning shoulder and hip have similar Y values
                torso_vertical = abs(hip[1] - shoulder[1])
                torso_horizontal = abs(hip[0] - shoulder[0])

                # If torso is more vertical than horizontal, probably standing
                if torso_horizontal > 10:  # Avoid division issues
                    verticality = torso_vertical / torso_horizontal
                    if verticality > 1.5:  # More vertical than horizontal = standing
                        logger.info(f"REJECTED: standing in side view (verticality={verticality:.2f})")
                        return False, "standing_position"

            # Check wrist position relative to shoulder
            if wrist[2] > conf_thresh and hip[2] > conf_thresh:
                body_scale = abs(hip[1] - shoulder[1]) + abs(hip[0] - shoulder[0])
                if body_scale > 20:
                    wrist_above = shoulder[1] - wrist[1]
                    wrist_ratio = wrist_above / body_scale
                    if wrist_ratio > self.max_wrist_above_ratio:
                        logger.info(f"REJECTED: wrist above shoulder ({view} view, ratio={wrist_ratio:.2f})")
                        return False, "wrist_above_head"

            return True, "valid"

        return True, "unknown_view"

    def _build_visualization(self) -> Dict:
        """
        Build visualization commands.

        Just shows skeleton - no text overlays.
        Frontend will display stats via API polling.
        """
        viz = {
            'skeleton': True,  # Show pose skeleton
            'text': []  # No text overlays - frontend handles UI
        }

        return viz


def get_app():
    """Factory function for app manager"""
    return PushupApp()
