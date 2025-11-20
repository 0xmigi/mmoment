"""
Push-up Counter App (SDK v2)

Simple push-up counting using vertical position tracking (like Clearspace).
Works from any camera angle - front, side, or angled.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sdk import BaseApp
from sdk.base_app import CompetitionApp
from typing import Dict, Any, List
import numpy as np
import time
import logging

logger = logging.getLogger("PushupApp")


class PushupApp(CompetitionApp):
    """
    Push-up counter using Y-position tracking.

    Counts reps when nose/shoulders drop below threshold and come back up.
    Simple, robust, works from any angle.
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Tracking state per competitor
        self.competitor_states = {}  # wallet -> PushupState

        # Thresholds (will auto-calibrate per person)
        self.drop_ratio = 0.15  # Drop 15% of standing height to count as "down"
        self.min_rep_time = 0.5  # Anti-cheat: min time between reps

        logger.info("PushupApp (v2) initialized with Y-position tracking")

    def init_competitor_stats(self) -> Dict:
        """Initialize stats for a competitor"""
        return {
            'reps': 0,
            'last_rep_time': None,
            'current_y': None,
            'baseline_y': None,  # Standing position
            'in_down_position': False
        }

    def start_competition(self, competitors: List[Dict], duration_limit: int = None):
        """Start competition"""
        super().start_competition(competitors, duration_limit)

        # Initialize state for each competitor
        for wallet in self.competitors.keys():
            self.competitor_states[wallet] = {
                'baseline_y': None,
                'in_down': False,
                'last_rep_time': None
            }

    def process(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process frame - count push-ups based on Y-position.

        Args:
            frame_data: {
                'detections': [...],
                'keypoints': [...],  # List of (17, 3) arrays
                'timestamp': float
            }
        """
        if not self.active:
            return {'state': self.get_state(), 'visualization': {}}

        detections = frame_data.get('detections', [])
        all_keypoints = frame_data.get('keypoints', [])
        timestamp = frame_data.get('timestamp', time.time())

        # Match keypoints to competitors
        for i, detection in enumerate(detections):
            wallet = detection.get('wallet_address')
            if not wallet or wallet not in self.competitors:
                continue

            if i >= len(all_keypoints):
                continue

            keypoints = all_keypoints[i]

            # Process this competitor's push-up
            self._process_competitor(wallet, keypoints, timestamp)

        # Build visualization
        viz = self._build_visualization()

        return {
            'state': self.get_state(),
            'visualization': viz
        }

    def _process_competitor(self, wallet: str, keypoints: np.ndarray, timestamp: float):
        """Process push-up for one competitor"""
        competitor = self.competitors[wallet]
        state = self.competitor_states[wallet]
        stats = competitor['stats']

        # Get nose Y-position (keypoint 0)
        nose = keypoints[0]
        if nose[2] < 0.5:  # Low confidence
            return

        current_y = nose[1]

        # Initialize baseline (standing position) on first valid frame
        if state['baseline_y'] is None:
            state['baseline_y'] = current_y
            logger.info(f"{competitor['display_name']}: Baseline Y = {current_y:.1f}")
            stats['baseline_y'] = current_y
            stats['current_y'] = current_y
            return

        # Calculate drop from baseline
        drop = current_y - state['baseline_y']
        drop_threshold = state['baseline_y'] * self.drop_ratio

        stats['current_y'] = current_y

        # State machine: UP → DOWN → UP (completes rep)
        if drop > drop_threshold:
            # Gone down
            if not state['in_down']:
                state['in_down'] = True
                stats['in_down_position'] = True
                logger.info(f"{competitor['display_name']}: DOWN (y={current_y:.1f}, drop={drop:.1f})")

        elif drop < drop_threshold * 0.5:
            # Back up
            if state['in_down']:
                # Check rep timing
                if state['last_rep_time'] is None or (timestamp - state['last_rep_time']) >= self.min_rep_time:
                    # Count rep!
                    stats['reps'] += 1
                    state['last_rep_time'] = timestamp
                    stats['last_rep_time'] = timestamp
                    logger.info(f"{competitor['display_name']}: REP #{stats['reps']}")

                state['in_down'] = False
                stats['in_down_position'] = False

    def _build_visualization(self) -> Dict:
        """Build visualization commands"""
        viz = {
            'skeleton': True,  # Always show skeleton
            'text': [],
            'lines': []
        }

        # Show rep count for each competitor
        y_offset = 50
        for wallet, competitor in self.competitors.items():
            stats = competitor['stats']
            state = self.competitor_states[wallet]

            # Rep count
            text = f"{competitor['display_name']}: {stats['reps']} reps"
            color = (0, 255, 0) if state['in_down'] else (255, 255, 255)

            viz['text'].append({
                'content': text,
                'pos': (10, y_offset),
                'color': color
            })

            # Show baseline line if calibrated
            if stats.get('baseline_y'):
                viz['lines'].append({
                    'start': (0, int(stats['baseline_y'])),
                    'end': (1280, int(stats['baseline_y'])),
                    'color': (255, 255, 0),
                    'thickness': 1
                })

            y_offset += 40

        return viz


def get_app():
    """Factory function for app manager"""
    return PushupApp()
