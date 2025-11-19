#!/usr/bin/env python3
"""
Push-up Competition App for mmoment Camera Network

Real-time push-up counting using MediaPipe pose estimation.
Perfect for "money where your mouth is" betting competitions.

Features:
- Face-based identity tracking (face visible during push-ups)
- Rep counting with form validation
- Real-time feedback
- Crypto betting integration
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_competition_app import BaseCompetitionApp, Competitor

try:
    import mediapipe as mp
except ImportError:
    mp = None
    logging.warning("MediaPipe not available - pose estimation disabled")

logger = logging.getLogger("PushupApp")


@dataclass
class PushupState:
    """State for tracking push-up reps"""
    count: int = 0
    in_down_position: bool = False
    in_up_position: bool = True  # Start in up position
    last_rep_time: Optional[float] = None
    elbow_angle_history: deque = None  # Last N frames of elbow angles
    form_score: float = 100.0  # Quality of form (100 = perfect)
    bad_form_warnings: int = 0

    def __post_init__(self):
        if self.elbow_angle_history is None:
            self.elbow_angle_history = deque(maxlen=10)


class PushupCompetitionApp(BaseCompetitionApp):
    """
    Push-up competition with real-time rep counting.

    Uses MediaPipe pose estimation to track body position and count reps.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Push-up detection parameters
        self.down_angle_threshold = 90  # Elbow angle for "down" position (degrees)
        self.up_angle_threshold = 160   # Elbow angle for "up" position (degrees)
        self.min_rep_time = 0.5        # Minimum time between reps (anti-cheat)

        # Form validation parameters
        self.body_alignment_threshold = 20  # Max degrees for body alignment
        self.hip_sag_threshold = 0.15      # Max hip sag ratio

        # Initialize MediaPipe Pose
        if mp:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=lite, 1=full, 2=heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose initialized")
        else:
            self.mp_pose = None
            self.pose_estimator = None
            logger.warning("MediaPipe not available")

        # Initialize push-up states for each competitor
        self.pushup_states: Dict[str, PushupState] = {}

    def start_competition(
        self,
        competitors: List[Dict],
        duration_limit: Optional[float] = None,
        bet_amount: Optional[float] = None
    ) -> Dict:
        """Start push-up competition"""
        result = super().start_competition(competitors, duration_limit, bet_amount)

        # Initialize push-up state for each competitor
        self.pushup_states.clear()
        for wallet in self.competitors.keys():
            self.pushup_states[wallet] = PushupState()
            self.competitors[wallet].stats = {
                'reps': 0,
                'form_score': 100.0,
                'bad_form_warnings': 0,
                'last_rep_time': None
            }

        return result

    def process_competitor_frame(
        self,
        competitor: Competitor,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> Dict[str, Any]:
        """Process frame for a single competitor doing push-ups"""

        if not self.pose_estimator:
            return {'error': 'Pose estimation not available'}

        pushup_state = self.pushup_states.get(competitor.wallet_address)
        if not pushup_state:
            return {'error': 'Push-up state not initialized'}

        # Get competitor's body region
        if not competitor.body_bbox:
            return {'tracked': False}

        x1, y1, x2, y2 = competitor.body_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Expand bbox for better pose detection
        h, w = frame.shape[:2]
        x1 = max(0, x1 - 20)
        y1 = max(0, y1 - 20)
        x2 = min(w, x2 + 20)
        y2 = min(h, y2 + 20)

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return {'tracked': False}

        # Run pose estimation
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.pose_estimator.process(person_rgb)

        if not results.pose_landmarks:
            return {'tracked': True, 'pose_detected': False}

        # Extract key landmarks (in crop coordinates)
        landmarks = results.pose_landmarks.landmark

        # Get shoulder, elbow, wrist positions (both sides)
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Get hip and knee for form validation
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]

        # Calculate elbow angles (average both arms)
        left_angle = self._calculate_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y)
        )

        right_angle = self._calculate_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )

        avg_elbow_angle = (left_angle + right_angle) / 2
        pushup_state.elbow_angle_history.append(avg_elbow_angle)

        # Validate form
        form_valid, form_score = self._validate_form(
            left_shoulder, left_hip, left_knee, left_ankle
        )

        pushup_state.form_score = form_score
        if not form_valid:
            pushup_state.bad_form_warnings += 1

        # Rep counting state machine
        now = time.time()
        rep_detected = False

        # Check if in down position
        if avg_elbow_angle < self.down_angle_threshold:
            if pushup_state.in_up_position:
                pushup_state.in_down_position = True
                pushup_state.in_up_position = False
                logger.debug(f"{competitor.display_name} went DOWN (angle: {avg_elbow_angle:.1f}°)")

        # Check if back in up position (completes a rep)
        elif avg_elbow_angle > self.up_angle_threshold:
            if pushup_state.in_down_position:
                # Validate rep timing (anti-cheat)
                if pushup_state.last_rep_time is None or \
                   (now - pushup_state.last_rep_time) >= self.min_rep_time:

                    pushup_state.count += 1
                    pushup_state.last_rep_time = now
                    pushup_state.in_down_position = False
                    pushup_state.in_up_position = True
                    rep_detected = True

                    logger.info(
                        f"REP! {competitor.display_name} - {pushup_state.count} "
                        f"(form: {form_score:.0f}%)"
                    )

        # Update competitor stats
        competitor.stats.update({
            'reps': pushup_state.count,
            'form_score': pushup_state.form_score,
            'bad_form_warnings': pushup_state.bad_form_warnings,
            'last_rep_time': pushup_state.last_rep_time,
            'current_angle': avg_elbow_angle,
            'in_down_position': pushup_state.in_down_position,
            'in_up_position': pushup_state.in_up_position
        })

        return {
            'tracked': True,
            'pose_detected': True,
            'rep_detected': rep_detected,
            'elbow_angle': avg_elbow_angle,
            'form_score': form_score,
            'reps': pushup_state.count
        }

    def _calculate_angle(
        self,
        point1: tuple,
        point2: tuple,
        point3: tuple
    ) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3.

        Args:
            point1, point2, point3: (x, y) coordinates

        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _validate_form(
        self,
        shoulder,
        hip,
        knee,
        ankle
    ) -> tuple[bool, float]:
        """
        Validate push-up form.

        Checks:
        - Body alignment (shoulder-hip-ankle should be roughly straight)
        - Hip position (not sagging or too high)

        Returns:
            (form_valid, form_score)
        """
        # Calculate body alignment angle (shoulder-hip-ankle)
        alignment_angle = self._calculate_angle(
            (shoulder.x, shoulder.y),
            (hip.x, hip.y),
            (ankle.x, ankle.y)
        )

        # Ideal alignment is ~180 degrees (straight line)
        alignment_deviation = abs(180 - alignment_angle)

        # Check hip sag (hip should be in line with shoulder and ankle)
        # Calculate perpendicular distance of hip from shoulder-ankle line
        shoulder_pos = np.array([shoulder.x, shoulder.y])
        ankle_pos = np.array([ankle.x, ankle.y])
        hip_pos = np.array([hip.x, hip.y])

        # Line from shoulder to ankle
        line_vec = ankle_pos - shoulder_pos
        line_length = np.linalg.norm(line_vec)

        if line_length > 0:
            # Project hip onto line
            hip_vec = hip_pos - shoulder_pos
            projection = np.dot(hip_vec, line_vec) / (line_length ** 2) * line_vec
            perpendicular = hip_vec - projection
            hip_deviation = np.linalg.norm(perpendicular) / line_length
        else:
            hip_deviation = 0

        # Calculate form score (0-100)
        alignment_score = max(0, 100 - (alignment_deviation * 2))
        hip_score = max(0, 100 - (hip_deviation * 500))  # Scale up hip deviation

        form_score = (alignment_score + hip_score) / 2

        # Form is valid if score > 60
        form_valid = form_score > 60

        return form_valid, form_score

    def check_competition_end(self) -> bool:
        """Check if competition should end"""
        # Check time limit
        if self.state.duration_limit:
            elapsed = time.time() - self.state.start_time
            if elapsed >= self.state.duration_limit:
                logger.info("Competition ended: Time limit reached")
                return True

        # Check if any competitor reached target reps
        target_reps = self.config.get('target_reps')
        if target_reps:
            for comp in self.competitors.values():
                if comp.stats.get('reps', 0) >= target_reps:
                    logger.info(
                        f"Competition ended: {comp.display_name} reached "
                        f"{target_reps} reps"
                    )
                    return True

        return False

    def determine_winner(self) -> Optional[str]:
        """Determine winner (most reps)"""
        if not self.competitors:
            return None

        # Sort by reps, then by form score (tie-breaker)
        sorted_competitors = sorted(
            self.competitors.items(),
            key=lambda x: (
                x[1].stats.get('reps', 0),
                x[1].stats.get('form_score', 0)
            ),
            reverse=True
        )

        winner_wallet, winner_comp = sorted_competitors[0]

        # Check for tie
        if len(sorted_competitors) > 1:
            second_wallet, second_comp = sorted_competitors[1]
            if (winner_comp.stats.get('reps', 0) ==
                second_comp.stats.get('reps', 0)):
                logger.info("Competition ended in a tie!")
                return None  # Tie - return bets

        return winner_wallet

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw push-up competition overlay"""
        # Draw base overlay (timer, pot, fps)
        frame = self.draw_base_overlay(frame)

        if not self.state.active:
            return frame

        # Draw competitor stats
        y_offset = 100

        # Sort competitors by reps
        sorted_competitors = sorted(
            self.competitors.items(),
            key=lambda x: x[1].stats.get('reps', 0),
            reverse=True
        )

        for wallet, comp in sorted_competitors:
            stats = comp.stats

            # Competitor name and reps
            text = f"{comp.display_name}: {stats.get('reps', 0)} reps"

            # Color based on position
            if comp == sorted_competitors[0][1]:
                color = (0, 255, 0)  # Green for leader
            else:
                color = (255, 255, 255)  # White

            cv2.putText(
                frame, text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

            # Form score indicator
            form_score = stats.get('form_score', 100)
            form_text = f"Form: {form_score:.0f}%"

            form_color = (0, 255, 0) if form_score > 80 else (0, 165, 255) if form_score > 60 else (0, 0, 255)

            cv2.putText(
                frame, form_text,
                (300, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, form_color, 2
            )

            # Position indicator
            current_angle = stats.get('current_angle', 180)
            if stats.get('in_down_position'):
                position_text = "DOWN"
                pos_color = (0, 255, 255)
            elif stats.get('in_up_position'):
                position_text = "UP"
                pos_color = (255, 255, 0)
            else:
                position_text = f"{current_angle:.0f}°"
                pos_color = (255, 255, 255)

            cv2.putText(
                frame, position_text,
                (500, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, pos_color, 2
            )

            y_offset += 35

        # Draw instructions at bottom
        instructions = "Get in push-up position. Rep = down (elbows < 90°) then up (elbows > 160°)"
        cv2.putText(
            frame, instructions,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1
        )

        return frame


def create_app(config: Dict = None) -> PushupCompetitionApp:
    """Factory function to create push-up app"""
    return PushupCompetitionApp(config)


if __name__ == "__main__":
    # Test the app
    logging.basicConfig(level=logging.INFO)

    app = create_app()

    print("Push-up Competition App initialized")
    print(f"Down threshold: {app.down_angle_threshold}°")
    print(f"Up threshold: {app.up_angle_threshold}°")
    print(f"Min rep time: {app.min_rep_time}s")
