#!/usr/bin/env python3
"""
Basketball Service Integration

Connects the basketball app to the camera service's identity tracking system.
"""

import sys
import os
import logging
import threading
from typing import Optional, Dict, List

# Add apps directory to path
sys.path.append('/app/apps/basketball')

from basketball_app import BasketballApp

logger = logging.getLogger("BasketballService")

class BasketballService:
    """Service to integrate basketball app with camera system"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BasketballService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.app = None
        self.enabled = False
        self.current_game = None
        self.gpu_face_service = None

        logger.info("Basketball Service initialized")

    def set_gpu_face_service(self, service):
        """Set reference to GPU face service for identity tracking"""
        self.gpu_face_service = service

    def start_app(self, config: Dict = None) -> Dict:
        """Start the basketball app"""
        try:
            self.app = BasketballApp(config_path=None)
            self.app.config.update(config or {})
            self.enabled = True

            logger.info("Basketball app started")
            return {'success': True, 'app': 'basketball', 'enabled': True}

        except Exception as e:
            logger.error(f"Failed to start basketball app: {e}")
            return {'success': False, 'error': str(e)}

    def stop_app(self) -> Dict:
        """Stop the basketball app"""
        self.enabled = False
        if self.current_game:
            self.end_game()

        logger.info("Basketball app stopped")
        return {'success': True, 'app': 'basketball', 'enabled': False}

    def start_game(self, players: List[Dict] = None) -> Dict:
        """
        Start a new basketball game

        Args:
            players: Optional list of players. If not provided, uses checked-in users

        Returns:
            Game start status
        """
        if not self.app:
            return {'success': False, 'error': 'App not started'}

        # Get checked-in users from identity tracker if no players specified
        if not players and self.gpu_face_service:
            active = self.gpu_face_service.identity_tracker.get_active_identities()
            players = []

            for wallet in active['sessions']:
                session = self.gpu_face_service.identity_tracker._active_sessions.get(wallet, {})
                players.append({
                    'wallet_address': wallet,
                    'display_name': session.get('metadata', {}).get('display_name', wallet[:8])
                })

        if not players:
            return {'success': False, 'error': 'No players checked in'}

        result = self.app.start_game(players)
        if result['success']:
            self.current_game = result['game_id']

        return result

    def process_frame(self, frame, detections: List[Dict]) -> Dict:
        """
        Process a frame with the basketball app

        Args:
            frame: Video frame
            detections: Enhanced detections from identity tracker

        Returns:
            Game state and processing results
        """
        if not self.enabled or not self.app or not self.current_game:
            return {'enabled': False}

        # Process with basketball app
        game_state = self.app.process_frame(frame, detections)

        # Check if game ended
        if game_state.get('game_ended'):
            self.current_game = None

            # Store attestation on-chain (future feature)
            attestation = self.app.get_attestation_data()
            logger.info(f"Game attestation ready: {attestation['game_id']}")

        return game_state

    def draw_overlay(self, frame) -> None:
        """Draw basketball app overlay on frame"""
        if not self.enabled or not self.app:
            return frame

        # Get current game state
        game_state = {'game_active': self.current_game is not None}

        if self.current_game:
            # Get latest player stats
            game_state['players'] = self.app._get_player_stats()
            game_state['time_remaining'] = self.app.game_duration - (
                time.time() - self.app.game_start_time
            ) if self.app.game_start_time else 0

        return self.app.draw_overlay(frame, game_state)

    def end_game(self) -> Dict:
        """End the current game"""
        if not self.app or not self.current_game:
            return {'success': False, 'error': 'No game in progress'}

        result = self.app.end_game()
        self.current_game = None

        return result

    def get_status(self) -> Dict:
        """Get current app status"""
        return {
            'enabled': self.enabled,
            'app': 'basketball',
            'game_active': self.current_game is not None,
            'game_id': self.current_game
        }

# Singleton instance
_basketball_service = None

def get_basketball_service() -> BasketballService:
    """Get the basketball service singleton"""
    global _basketball_service
    if _basketball_service is None:
        _basketball_service = BasketballService()
    return _basketball_service