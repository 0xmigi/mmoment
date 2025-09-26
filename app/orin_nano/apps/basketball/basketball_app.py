#!/usr/bin/env python3
"""
Basketball Shot Counter App for mmoment Camera Network

Uses the identity tracking system to attribute shots to specific
wallet-authenticated users. Demonstrates the power of on-chain
identity + computer vision.
"""

import cv2
import numpy as np
import time
import json
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("BasketballApp")

@dataclass
class Player:
    """Tracked player with stats"""
    wallet_address: str
    display_name: str
    track_id: Optional[int] = None
    stats: Dict = field(default_factory=lambda: {
        'shots_made': 0,
        'shots_missed': 0,
        'attempts': 0,
        'fg_percentage': 0.0,
        'streak': 0,
        'best_streak': 0,
        'last_shot_time': None,
        'points': 0
    })

class BasketballApp:
    """
    Basketball tracking and scoring app that runs on mmoment cameras.
    Tracks shots, scores, and player statistics with on-chain identity.
    """

    def __init__(self, config_path: str = None):
        """Initialize basketball app with configuration"""
        self.config = self.load_config(config_path)

        # Game state
        self.game_active = False
        self.game_start_time = None
        self.game_duration = self.config.get('game_duration', 300)  # 5 minutes default

        # Players tracked in current game
        self.players = {}  # wallet_address -> Player
        self.track_to_wallet = {}  # track_id -> wallet_address

        # Ball tracking
        self.ball_history = deque(maxlen=30)  # 1 second at 30fps
        self.last_ball_pos = None
        self.shot_in_progress = False
        self.shot_start_pos = None
        self.shot_player = None  # Who took the shot

        # Court zones
        self.hoop_zone = self.config.get('hoop_zone', {
            'x': 640,  # Default center of 1280 width
            'y': 200,  # Upper portion of frame
            'radius': 60,
            'made_radius': 30  # Smaller zone for made shots
        })

        self.three_point_line = self.config.get('three_point_line', {
            'enabled': False,
            'distance': 300  # Pixels from hoop
        })

        # Shot detection parameters
        self.min_shot_height = 100  # Minimum upward movement
        self.shot_timeout = 3.0  # Max time for shot to complete

        logger.info("Basketball App initialized")

    def load_config(self, config_path: str) -> dict:
        """Load app configuration"""
        default_config = {
            'game_duration': 300,
            'score_to_win': 21,
            'enable_three_pointers': False,
            'require_check_in': True
        }

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        return default_config

    def start_game(self, players: List[Dict]) -> Dict:
        """
        Start a new basketball game

        Args:
            players: List of checked-in players with wallet addresses

        Returns:
            Game initialization status
        """
        self.game_active = True
        self.game_start_time = time.time()

        # Initialize players
        self.players.clear()
        self.track_to_wallet.clear()

        for player_data in players:
            player = Player(
                wallet_address=player_data['wallet_address'],
                display_name=player_data.get('display_name', player_data['wallet_address'][:8])
            )
            self.players[player.wallet_address] = player

        logger.info(f"Game started with {len(self.players)} players")

        return {
            'success': True,
            'game_id': f"game_{int(self.game_start_time)}",
            'players': len(self.players),
            'duration': self.game_duration
        }

    def process_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Process a frame with tracking data

        Args:
            frame: Current video frame
            detections: List of detections with identity tracking

        Returns:
            Frame processing results with game state
        """
        if not self.game_active:
            return {'game_active': False}

        # Check game time
        elapsed = time.time() - self.game_start_time
        if elapsed > self.game_duration:
            return self.end_game()

        # Update player tracking
        self._update_player_tracking(detections)

        # Track ball
        ball_detection = self._find_ball(detections)
        if ball_detection:
            self._track_ball(ball_detection)

            # Check for shots
            shot_result = self._detect_shot(frame)
            if shot_result:
                self._process_shot_result(shot_result)

        # Generate game state
        game_state = {
            'game_active': True,
            'time_remaining': self.game_duration - elapsed,
            'players': self._get_player_stats(),
            'ball_tracked': ball_detection is not None,
            'shot_in_progress': self.shot_in_progress
        }

        return game_state

    def _update_player_tracking(self, detections: List[Dict]):
        """Update which track IDs belong to which players"""
        for detection in detections:
            if detection.get('class') != 'person':
                continue

            track_id = detection.get('track_id')
            wallet = detection.get('wallet_address')

            if track_id and wallet and wallet in self.players:
                self.track_to_wallet[track_id] = wallet
                self.players[wallet].track_id = track_id

    def _find_ball(self, detections: List[Dict]) -> Optional[Dict]:
        """Find basketball in detections"""
        for detection in detections:
            # Look for sports ball class (YOLOv8)
            if detection.get('class') in ['sports ball', 'basketball', 'ball']:
                return detection
        return None

    def _track_ball(self, ball_detection: Dict):
        """Track ball position over time"""
        x = (ball_detection['x1'] + ball_detection['x2']) // 2
        y = (ball_detection['y1'] + ball_detection['y2']) // 2

        ball_pos = {
            'x': x,
            'y': y,
            'time': time.time(),
            'bbox': (ball_detection['x1'], ball_detection['y1'],
                    ball_detection['x2'], ball_detection['y2'])
        }

        self.ball_history.append(ball_pos)
        self.last_ball_pos = ball_pos

    def _detect_shot(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect basketball shots by analyzing ball trajectory

        Returns:
            Shot result if detected (made/missed/none)
        """
        if len(self.ball_history) < 5:
            return None

        positions = list(self.ball_history)

        # Calculate vertical movement
        y_positions = [p['y'] for p in positions]
        y_velocities = np.diff(y_positions)

        # Not in shot - check for upward movement (shot start)
        if not self.shot_in_progress:
            # Look for significant upward movement
            if np.any(y_velocities[:3] < -10):  # Ball going up
                self.shot_in_progress = True
                self.shot_start_pos = positions[-1]

                # Determine who's shooting based on closest player
                self.shot_player = self._find_shooter(positions[-1])

                logger.info(f"Shot detected by {self.shot_player or 'unknown'}")
                return None

        # Shot in progress - check for completion
        else:
            # Check for timeout
            shot_duration = time.time() - self.shot_start_pos['time']
            if shot_duration > self.shot_timeout:
                self.shot_in_progress = False
                return {'result': 'missed', 'player': self.shot_player}

            # Check if ball is falling
            if y_velocities[-1] > 5:  # Ball falling down
                last_pos = positions[-1]

                # Check distance to hoop
                dist_to_hoop = np.sqrt(
                    (last_pos['x'] - self.hoop_zone['x'])**2 +
                    (last_pos['y'] - self.hoop_zone['y'])**2
                )

                self.shot_in_progress = False

                # Determine if shot was made
                if dist_to_hoop < self.hoop_zone['made_radius']:
                    # Check if 3-pointer
                    is_three = False
                    if self.three_point_line['enabled'] and self.shot_start_pos:
                        shot_distance = np.sqrt(
                            (self.shot_start_pos['x'] - self.hoop_zone['x'])**2 +
                            (self.shot_start_pos['y'] - self.hoop_zone['y'])**2
                        )
                        is_three = shot_distance > self.three_point_line['distance']

                    return {
                        'result': 'made',
                        'player': self.shot_player,
                        'points': 3 if is_three else 2,
                        'three_pointer': is_three
                    }
                elif dist_to_hoop < self.hoop_zone['radius']:
                    return {'result': 'missed', 'player': self.shot_player}

        return None

    def _find_shooter(self, ball_pos: Dict) -> Optional[str]:
        """Find which player is closest to the ball (likely shooter)"""
        # This would use the current frame detections to find closest player
        # For now, return the first active player
        if self.track_to_wallet:
            return list(self.track_to_wallet.values())[0]
        return None

    def _process_shot_result(self, shot_result: Dict):
        """Update player stats based on shot result"""
        player_wallet = shot_result.get('player')
        if not player_wallet or player_wallet not in self.players:
            return

        player = self.players[player_wallet]
        player.stats['attempts'] += 1
        player.stats['last_shot_time'] = time.time()

        if shot_result['result'] == 'made':
            player.stats['shots_made'] += 1
            player.stats['streak'] += 1
            player.stats['points'] += shot_result.get('points', 2)

            if player.stats['streak'] > player.stats['best_streak']:
                player.stats['best_streak'] = player.stats['streak']

            logger.info(f"MADE! {player.display_name} scores {shot_result.get('points', 2)} points!")
        else:
            player.stats['shots_missed'] += 1
            player.stats['streak'] = 0
            logger.info(f"MISSED! {player.display_name}")

        # Update FG%
        if player.stats['attempts'] > 0:
            player.stats['fg_percentage'] = (
                player.stats['shots_made'] / player.stats['attempts'] * 100
            )

    def _get_player_stats(self) -> List[Dict]:
        """Get current stats for all players"""
        stats_list = []
        for wallet, player in self.players.items():
            stats_list.append({
                'wallet_address': wallet,
                'display_name': player.display_name,
                'track_id': player.track_id,
                'stats': player.stats
            })

        # Sort by points
        stats_list.sort(key=lambda x: x['stats']['points'], reverse=True)
        return stats_list

    def draw_overlay(self, frame: np.ndarray, game_state: Dict) -> np.ndarray:
        """
        Draw game overlay on frame

        Args:
            frame: Video frame
            game_state: Current game state

        Returns:
            Frame with overlay
        """
        if not game_state.get('game_active'):
            return frame

        # Draw hoop zone
        cv2.circle(frame,
                  (self.hoop_zone['x'], self.hoop_zone['y']),
                  self.hoop_zone['radius'],
                  (0, 255, 0), 2)

        # Draw made zone (inner circle)
        cv2.circle(frame,
                  (self.hoop_zone['x'], self.hoop_zone['y']),
                  self.hoop_zone['made_radius'],
                  (0, 255, 255), 1)

        # Draw scoreboard
        y_offset = 30

        # Game timer
        time_remaining = game_state.get('time_remaining', 0)
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2)
        y_offset += 40

        # Player stats
        for player_stats in game_state.get('players', [])[:3]:  # Top 3 players
            name = player_stats['display_name']
            stats = player_stats['stats']

            text = f"{name}: {stats['points']}pts ({stats['shots_made']}/{stats['attempts']}) {stats['fg_percentage']:.0f}%"

            cv2.putText(frame, text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            y_offset += 30

        # Shot in progress indicator
        if game_state.get('shot_in_progress'):
            cv2.putText(frame, "SHOOTING!",
                       (frame.shape[1]//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 255, 255), 3)

        # Ball trail
        if len(self.ball_history) > 1:
            points = [(p['x'], p['y']) for p in self.ball_history]
            for i in range(1, len(points)):
                # Fade older points
                alpha = i / len(points)
                color = (255 * alpha, 165 * alpha, 0)
                cv2.line(frame, points[i-1], points[i], color, 2)

        return frame

    def end_game(self) -> Dict:
        """End the current game and return final stats"""
        self.game_active = False

        final_stats = {
            'game_active': False,
            'game_ended': True,
            'duration': time.time() - self.game_start_time,
            'final_stats': self._get_player_stats()
        }

        # Log final results
        logger.info("Game ended! Final scores:")
        for player_stats in final_stats['final_stats']:
            logger.info(f"  {player_stats['display_name']}: {player_stats['stats']['points']} points")

        return final_stats

    def get_attestation_data(self) -> Dict:
        """
        Generate on-chain attestation data for the game
        This can be stored on Solana as proof of the game
        """
        if not self.game_start_time:
            return {}

        return {
            'game_id': f"game_{int(self.game_start_time)}",
            'timestamp': datetime.now().isoformat(),
            'duration': time.time() - self.game_start_time if self.game_start_time else 0,
            'players': [
                {
                    'wallet': wallet,
                    'stats': player.stats
                }
                for wallet, player in self.players.items()
            ],
            'verified': True,  # Camera-verified game
            'camera_pda': self.config.get('camera_pda', 'unknown')
        }