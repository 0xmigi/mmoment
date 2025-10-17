#!/usr/bin/env python3
"""
Identity Tracking Service

Maintains persistent identity tracking across frames by combining:
- YOLOv8 object tracking (ByteTrack/BoT-SORT)
- Face verification from enrolled users
- Wallet address → body tracking association

This is the core building block for all CV apps on the mmoment network.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger("IdentityTracker")

class IdentityTracker:
    """
    Tracks checked-in users persistently across frames.
    Links YOLOv8 tracker IDs to verified wallet addresses.
    """

    def __init__(self, face_service=None, re_verify_interval: float = 5.0):
        """
        Initialize identity tracker

        Args:
            face_service: Reference to GPUFaceService for face verification
            re_verify_interval: How often to re-verify faces (seconds)
        """
        self.face_service = face_service
        self.re_verify_interval = re_verify_interval

        # Tracking state
        self._lock = threading.Lock()
        self._tracked_identities = {}  # track_id -> wallet_address
        self._identity_metadata = {}   # wallet_address -> metadata
        self._track_history = defaultdict(list)  # track_id -> position history
        self._last_verified = {}  # track_id -> timestamp
        self._confidence_scores = {}  # track_id -> confidence

        # Re-identification state
        self._lost_tracks = {}  # wallet_address -> last known state
        self._pending_reacquisition = set()  # wallet addresses to re-acquire

        # Session management
        self._active_sessions = {}  # wallet_address -> session_data

        logger.info("Identity Tracker initialized")

    def check_in_user(self, wallet_address: str, face_embedding: np.ndarray,
                      initial_bbox: Tuple[int, int, int, int], metadata: Dict = None) -> Dict:
        """
        Check in a user and start tracking them

        Args:
            wallet_address: User's wallet address
            face_embedding: Face embedding from enrollment
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            metadata: Additional user metadata (display_name, etc)

        Returns:
            Dict with check-in status and assigned track_id
        """
        with self._lock:
            # Store session data
            self._active_sessions[wallet_address] = {
                'face_embedding': face_embedding,
                'check_in_time': time.time(),
                'metadata': metadata or {},
                'stats': defaultdict(int)  # For app-specific stats
            }

            # Will assign track_id when detection matches
            self._pending_reacquisition.add(wallet_address)

            logger.info(f"User checked in: {wallet_address[:8]}...")

            return {
                'success': True,
                'wallet_address': wallet_address,
                'status': 'pending_acquisition',
                'metadata': metadata
            }

    def process_frame_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Process YOLOv8 detections and maintain identity tracking

        Args:
            detections: List of YOLOv8 detections with track IDs
            frame: Current frame for face verification

        Returns:
            List of detections with wallet addresses added
        """
        current_time = time.time()
        enhanced_detections = []

        with self._lock:
            current_track_ids = set()

            for detection in detections:
                if detection.get('class') != 'person':
                    enhanced_detections.append(detection)
                    continue

                track_id = detection.get('track_id')
                if track_id is None:
                    # YOLOv8 tracking not enabled - enable it!
                    enhanced_detections.append(detection)
                    continue

                current_track_ids.add(track_id)
                bbox = (detection['x1'], detection['y1'], detection['x2'], detection['y2'])

                # Update track history
                self._track_history[track_id].append({
                    'bbox': bbox,
                    'timestamp': current_time,
                    'confidence': detection.get('confidence', 0.0)
                })

                # Keep history limited
                if len(self._track_history[track_id]) > 30:  # ~1 second at 30fps
                    self._track_history[track_id].pop(0)

                # Check if this track is already identified
                if track_id in self._tracked_identities:
                    wallet_address = self._tracked_identities[track_id]
                    detection['wallet_address'] = wallet_address
                    detection['identity_confidence'] = self._confidence_scores.get(track_id, 0.0)

                    # Check if re-verification needed
                    last_verify = self._last_verified.get(track_id, 0)
                    if current_time - last_verify > self.re_verify_interval:
                        self._schedule_reverification(track_id, bbox, frame)

                # Check if we need to acquire this track
                elif self._pending_reacquisition:
                    # Try to identify this person
                    identified_wallet = self._attempt_identification(bbox, frame)
                    if identified_wallet:
                        self._tracked_identities[track_id] = identified_wallet
                        self._confidence_scores[track_id] = 1.0
                        self._last_verified[track_id] = current_time
                        self._pending_reacquisition.discard(identified_wallet)

                        detection['wallet_address'] = identified_wallet
                        detection['identity_confidence'] = 1.0
                        logger.info(f"Acquired track {track_id} as {identified_wallet[:8]}...")

                enhanced_detections.append(detection)

            # Handle lost tracks
            lost_tracks = set(self._tracked_identities.keys()) - current_track_ids
            for lost_track_id in lost_tracks:
                wallet_address = self._tracked_identities.pop(lost_track_id)
                last_state = {
                    'last_bbox': self._track_history[lost_track_id][-1]['bbox'] if self._track_history[lost_track_id] else None,
                    'last_seen': current_time,
                    'confidence': self._confidence_scores.get(lost_track_id, 0.0)
                }
                self._lost_tracks[wallet_address] = last_state
                self._pending_reacquisition.add(wallet_address)
                logger.warning(f"Lost track of {wallet_address[:8]}... scheduling reacquisition")

        return enhanced_detections

    def _attempt_identification(self, bbox: Tuple, frame: np.ndarray) -> Optional[str]:
        """
        Attempt to identify a person by their face

        Args:
            bbox: Bounding box of person (x1, y1, x2, y2)
            frame: Current frame

        Returns:
            Wallet address if identified, None otherwise
        """
        if not self.face_service or not self._active_sessions:
            return None

        try:
            # Extract face region from bbox (upper portion)
            x1, y1, x2, y2 = bbox
            height = y2 - y1

            # Face is typically in upper 40% of person bbox
            face_y2 = y1 + int(height * 0.4)
            face_region = frame[y1:face_y2, x1:x2]

            if face_region.size == 0:
                return None

            # Get face embedding
            embedding = self.face_service.extract_face_embedding(face_region)
            if embedding is None:
                return None

            # Compare with all active sessions
            best_match = None
            best_similarity = 0.0
            threshold = 0.6

            for wallet_address, session_data in self._active_sessions.items():
                stored_embedding = session_data['face_embedding']
                similarity = float(np.dot(embedding, stored_embedding))

                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = wallet_address

            if best_match:
                logger.info(f"Identified person as {best_match[:8]}... (similarity: {best_similarity:.3f})")
                return best_match

        except Exception as e:
            logger.error(f"Error in identification attempt: {e}")

        return None

    def _schedule_reverification(self, track_id: int, bbox: Tuple, frame: np.ndarray):
        """
        Schedule face re-verification for a tracked person
        """
        # This would run async in production
        wallet_address = self._tracked_identities.get(track_id)
        if not wallet_address:
            return

        identified = self._attempt_identification(bbox, frame)
        if identified == wallet_address:
            # Boost confidence
            self._confidence_scores[track_id] = min(1.0, self._confidence_scores.get(track_id, 0.5) + 0.1)
            self._last_verified[track_id] = time.time()
        elif identified and identified != wallet_address:
            # Wrong person! Track was mixed up
            logger.error(f"Track {track_id} changed identity! Was {wallet_address[:8]}, now {identified[:8]}")
            # Handle track confusion...
            self._confidence_scores[track_id] *= 0.5

    def check_out_user(self, wallet_address: str) -> Dict:
        """
        Check out a user and stop tracking them

        Args:
            wallet_address: User's wallet address

        Returns:
            Dict with checkout status and session stats
        """
        with self._lock:
            session_data = self._active_sessions.pop(wallet_address, None)
            if not session_data:
                return {'success': False, 'error': 'User not checked in'}

            # Remove from tracking
            self._pending_reacquisition.discard(wallet_address)
            self._lost_tracks.pop(wallet_address, None)

            # Find and remove track ID
            track_to_remove = None
            for track_id, tracked_wallet in self._tracked_identities.items():
                if tracked_wallet == wallet_address:
                    track_to_remove = track_id
                    break

            if track_to_remove:
                self._tracked_identities.pop(track_to_remove)
                self._confidence_scores.pop(track_to_remove, None)
                self._last_verified.pop(track_to_remove, None)

            duration = time.time() - session_data['check_in_time']

            logger.info(f"User checked out: {wallet_address[:8]}... (duration: {duration:.1f}s)")

            return {
                'success': True,
                'wallet_address': wallet_address,
                'duration': duration,
                'stats': dict(session_data['stats'])
            }

    def get_active_identities(self) -> Dict:
        """
        Get all currently tracked identities

        Returns:
            Dict of track_id -> wallet_address mappings
        """
        with self._lock:
            return {
                'tracked': dict(self._tracked_identities),
                'pending': list(self._pending_reacquisition),
                'lost': list(self._lost_tracks.keys()),
                'sessions': list(self._active_sessions.keys())
            }

    def update_user_stats(self, wallet_address: str, stat_key: str, value: int = 1):
        """
        Update stats for a checked-in user (for app-specific tracking)

        Args:
            wallet_address: User's wallet address
            stat_key: Stat to update (e.g., 'shots_made', 'climbs_completed')
            value: Amount to add to stat
        """
        with self._lock:
            if wallet_address in self._active_sessions:
                self._active_sessions[wallet_address]['stats'][stat_key] += value