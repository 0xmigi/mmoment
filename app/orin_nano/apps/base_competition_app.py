#!/usr/bin/env python3
"""
Base Competition App for mmoment Camera Network

Extensible base class for CV-based competition apps with:
- Identity tracking integration
- Wallet-based player management
- Escrow/betting support
- Real-time pose/action detection
- On-chain attestation
"""

import cv2
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger("BaseCompetitionApp")


@dataclass
class Competitor:
    """Tracked competitor with stats and wallet"""
    wallet_address: str
    display_name: str
    track_id: Optional[int] = None
    face_bbox: Optional[tuple] = None  # (x1, y1, x2, y2)
    body_bbox: Optional[tuple] = None  # Full body detection
    last_seen: float = field(default_factory=time.time)
    stats: Dict[str, Any] = field(default_factory=dict)
    bet_amount: float = 0.0  # SOL wagered


@dataclass
class CompetitionState:
    """Current competition state"""
    active: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_limit: Optional[float] = None
    winner: Optional[str] = None
    pot_amount: float = 0.0  # Total SOL in pot
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCompetitionApp(ABC):
    """
    Base class for competition apps using CV tracking and crypto betting.

    Provides:
    - Competitor registration and tracking
    - Competition lifecycle management
    - Betting/escrow integration hooks
    - Identity tracking integration
    - On-chain attestation generation
    """

    def __init__(self, config: Dict = None):
        """Initialize competition app"""
        self.config = config or {}

        # Competition state
        self.state = CompetitionState()

        # Competitors
        self.competitors: Dict[str, Competitor] = {}  # wallet -> Competitor
        self.track_to_wallet: Dict[int, str] = {}  # track_id -> wallet

        # Frame counter for performance
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def process_competitor_frame(
        self,
        competitor: Competitor,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> Dict[str, Any]:
        """
        Process frame data for a single competitor.

        This is where CV-specific logic goes (pose estimation, action detection, etc.)

        Args:
            competitor: The competitor being processed
            frame: Current video frame
            detections: Detection data for this competitor

        Returns:
            Dict with competition-specific updates (reps, score, etc.)
        """
        pass

    @abstractmethod
    def check_competition_end(self) -> bool:
        """
        Check if competition should end.

        Returns:
            True if competition should end
        """
        pass

    @abstractmethod
    def determine_winner(self) -> Optional[str]:
        """
        Determine winner based on stats.

        Returns:
            Wallet address of winner, or None for tie
        """
        pass

    def start_competition(
        self,
        competitors: List[Dict],
        duration_limit: Optional[float] = None,
        bet_amount: Optional[float] = None
    ) -> Dict:
        """
        Start a new competition.

        Args:
            competitors: List of dicts with 'wallet_address', optional 'display_name'
            duration_limit: Max duration in seconds (None = unlimited)
            bet_amount: Amount each competitor wagered (SOL)

        Returns:
            Competition initialization status
        """
        self.state = CompetitionState(
            active=True,
            start_time=time.time(),
            duration_limit=duration_limit
        )

        # Register competitors
        self.competitors.clear()
        self.track_to_wallet.clear()

        for comp_data in competitors:
            competitor = Competitor(
                wallet_address=comp_data['wallet_address'],
                display_name=comp_data.get('display_name', comp_data['wallet_address'][:8]),
                bet_amount=bet_amount or 0.0
            )
            self.competitors[competitor.wallet_address] = competitor
            self.state.pot_amount += competitor.bet_amount

        logger.info(
            f"Competition started: {len(self.competitors)} competitors, "
            f"pot: {self.state.pot_amount} SOL"
        )

        return {
            'success': True,
            'competition_id': f"comp_{int(self.state.start_time)}",
            'competitors': len(self.competitors),
            'pot_amount': self.state.pot_amount,
            'duration_limit': duration_limit
        }

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> Dict[str, Any]:
        """
        Main frame processing loop.

        Args:
            frame: Current video frame
            detections: List of detections with identity tracking

        Returns:
            Current competition state with all stats
        """
        if not self.state.active:
            return {'active': False}

        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            self.fps = 30 / (now - self.last_fps_time)
            self.last_fps_time = now

        # Update competitor tracking from detections
        self._update_competitor_tracking(detections)

        # Process each competitor
        updates = {}
        for wallet, competitor in self.competitors.items():
            if competitor.track_id is not None:
                # Get detections for this competitor
                comp_detections = [
                    d for d in detections
                    if d.get('track_id') == competitor.track_id
                ]

                if comp_detections:
                    result = self.process_competitor_frame(
                        competitor, frame, comp_detections
                    )
                    updates[wallet] = result

        # Check if competition should end
        if self.check_competition_end():
            return self.end_competition()

        # Build state response
        return self._get_competition_state()

    def _update_competitor_tracking(self, detections: List[Dict]):
        """Update which track IDs belong to which competitors"""
        for detection in detections:
            if detection.get('class') != 'person':
                continue

            track_id = detection.get('track_id')
            wallet = detection.get('wallet_address')

            if track_id and wallet and wallet in self.competitors:
                self.track_to_wallet[track_id] = wallet
                competitor = self.competitors[wallet]
                competitor.track_id = track_id
                competitor.last_seen = time.time()

                # Update bboxes
                competitor.face_bbox = detection.get('face_bbox')
                competitor.body_bbox = (
                    detection.get('x1'),
                    detection.get('y1'),
                    detection.get('x2'),
                    detection.get('y2')
                )

    def _get_competition_state(self) -> Dict[str, Any]:
        """Get current competition state for API response"""
        elapsed = time.time() - self.state.start_time if self.state.start_time else 0
        time_remaining = None

        if self.state.duration_limit:
            time_remaining = max(0, self.state.duration_limit - elapsed)

        return {
            'active': self.state.active,
            'elapsed': elapsed,
            'time_remaining': time_remaining,
            'pot_amount': self.state.pot_amount,
            'competitors': [
                {
                    'wallet_address': wallet,
                    'display_name': comp.display_name,
                    'track_id': comp.track_id,
                    'tracked': comp.track_id is not None,
                    'last_seen': comp.last_seen,
                    'bet_amount': comp.bet_amount,
                    'stats': comp.stats
                }
                for wallet, comp in self.competitors.items()
            ],
            'fps': self.fps,
            'metadata': self.state.metadata
        }

    def end_competition(self) -> Dict[str, Any]:
        """End competition and determine winner"""
        self.state.active = False
        self.state.end_time = time.time()

        # Determine winner
        self.state.winner = self.determine_winner()

        final_state = self._get_competition_state()
        final_state.update({
            'ended': True,
            'winner': self.state.winner,
            'duration': self.state.end_time - self.state.start_time
        })

        logger.info(f"Competition ended. Winner: {self.state.winner}")

        return final_state

    def get_attestation_data(self) -> Dict[str, Any]:
        """
        Generate on-chain attestation data.

        This can be stored on Solana as cryptographic proof of the competition.
        """
        if not self.state.start_time:
            return {}

        return {
            'competition_id': f"comp_{int(self.state.start_time)}",
            'app_type': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'start_time': self.state.start_time,
            'end_time': self.state.end_time,
            'duration': (
                self.state.end_time - self.state.start_time
                if self.state.end_time else None
            ),
            'pot_amount': self.state.pot_amount,
            'winner': self.state.winner,
            'competitors': [
                {
                    'wallet': wallet,
                    'display_name': comp.display_name,
                    'bet_amount': comp.bet_amount,
                    'stats': comp.stats
                }
                for wallet, comp in self.competitors.items()
            ],
            'verified': True,  # Camera-verified competition
            'camera_pda': self.config.get('camera_pda', 'unknown'),
            'metadata': self.state.metadata
        }

    def draw_base_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw common overlay elements.

        Override and call super() to add competition-specific overlays.
        """
        if not self.state.active:
            return frame

        # Competition timer
        elapsed = time.time() - self.state.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        cv2.putText(
            frame,
            f"Time: {minutes:02d}:{seconds:02d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        # Pot amount
        cv2.putText(
            frame,
            f"Pot: {self.state.pot_amount:.2f} SOL",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 215, 0), 2
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        return frame
