#!/usr/bin/env python3
"""
Identity Tracking Service

Maintains persistent identity tracking across frames by combining:
- YOLOv8 object tracking (ByteTrack/BoT-SORT)
- Face verification from enrolled users
- Wallet address → body tracking association

This is the core building block for all CV apps on the mmoment network.

NOTE: This module now uses IdentityStore as the single source of truth for
all identity data. It focuses purely on the tracking logic.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
import threading
import cv2

from .identity_store import get_identity_store

logger = logging.getLogger("IdentityTracker")

# Re-ID model for appearance-based tracking
_reid_model = None
_reid_transform = None

def get_reid_model():
    """Get or initialize the Re-ID model for appearance matching"""
    global _reid_model, _reid_transform

    if _reid_model is None:
        try:
            import torch
            import timm
            from torchvision import transforms

            logger.info("Initializing Re-ID model (mobilenetv4) on CUDA...")

            # Use a lightweight but effective model
            _reid_model = timm.create_model(
                'mobilenetv4_conv_small.e1200_r224_in1k',
                pretrained=True,
                num_classes=0  # Remove classification head, get embeddings
            )
            _reid_model = _reid_model.to('cuda').eval()

            # Standard ImageNet preprocessing
            _reid_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("Re-ID model initialized successfully (1280-dim embeddings)")

        except Exception as e:
            logger.error(f"Failed to initialize Re-ID model: {e}")
            return None, None

    return _reid_model, _reid_transform


class IdentityTracker:
    """
    Tracks checked-in users persistently across frames.
    Links YOLOv8 tracker IDs to verified wallet addresses.

    Uses IdentityStore as the single source of truth for identity data.
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
        self._lock = threading.Lock()

        # Get identity store reference
        self.store = get_identity_store()

        # Track-specific state (not identity-related)
        self._track_history = {}  # track_id -> position history
        self._last_verified = {}  # track_id -> timestamp of last verification attempt

        # Initialize Re-ID model
        get_reid_model()

        logger.info("Identity Tracker initialized with unified IdentityStore")

    def check_in_user(self, wallet_address: str, face_embedding: np.ndarray,
                      initial_bbox: Tuple[int, int, int, int], metadata: Dict = None) -> Dict:
        """
        Check in a user and start tracking them.

        Args:
            wallet_address: User's wallet address
            face_embedding: Face embedding from enrollment
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            metadata: Additional user metadata (display_name, etc)

        Returns:
            Dict with check-in status
        """
        # Build profile from metadata
        profile = {}
        if metadata:
            profile['display_name'] = metadata.get('display_name') or metadata.get('name')
            profile['username'] = metadata.get('username')

        # Use IdentityStore for check-in
        identity = self.store.check_in(wallet_address, face_embedding, profile)

        # Update initial bbox
        if initial_bbox:
            self.store.update_last_bbox(wallet_address, initial_bbox)

        logger.info(f"User checked in via tracker: {wallet_address[:8]}...")

        return {
            'success': True,
            'wallet_address': wallet_address,
            'status': 'pending_acquisition',
            'metadata': metadata
        }

    def check_out_user(self, wallet_address: str) -> Dict:
        """
        Check out a user and clear ALL their identity data.

        Args:
            wallet_address: User's wallet address

        Returns:
            Dict with checkout status and session stats
        """
        # Clear track-specific state
        with self._lock:
            identity = self.store.get_identity(wallet_address)
            if identity and identity.active_track_id is not None:
                track_id = identity.active_track_id
                self._track_history.pop(track_id, None)
                self._last_verified.pop(track_id, None)

        # Use IdentityStore for checkout
        stats = self.store.check_out(wallet_address)

        if stats:
            logger.info(f"User checked out via tracker: {wallet_address[:8]}...")
            return {
                'success': True,
                'wallet_address': wallet_address,
                'duration': stats['duration'],
                'stats': stats['stats']
            }
        else:
            return {
                'success': False,
                'error': 'User not checked in'
            }

    def process_frame_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Process YOLOv8 detections and maintain identity tracking.

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

            import sys
            pending_count = len(self.store.get_pending_reacquisitions())
            active_count = len(self.store.get_active_identities())
            print(f"[IDENTITY-TRACKER] Processing {len(detections)} detections, pending={pending_count}, active={active_count}", flush=True, file=sys.stderr)

            for detection in detections:
                if detection.get('class') != 'person':
                    enhanced_detections.append(detection)
                    continue

                track_id = detection.get('track_id')

                if track_id is None:
                    enhanced_detections.append(detection)
                    continue

                current_track_ids.add(track_id)
                bbox = (detection['x1'], detection['y1'], detection['x2'], detection['y2'])

                # Update track history
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append({
                    'bbox': bbox,
                    'timestamp': current_time,
                    'confidence': detection.get('confidence', 0.0)
                })

                # Keep history limited
                if len(self._track_history[track_id]) > 30:
                    self._track_history[track_id].pop(0)

                # Check if this track is already identified
                identity = self.store.get_identity_by_track(track_id)

                if identity:
                    wallet_address = identity.wallet_address
                    print(f"[IDENTITY-TRACKER] Track {track_id} FOUND: {wallet_address[:8]}", flush=True, file=sys.stderr)

                    # Check face timeout
                    if self.store.check_face_timeout(wallet_address):
                        logger.warning(f"Track {track_id} ({wallet_address[:8]}...) expired - no face seen")

                        # Store bbox for re-acquisition
                        self.store.update_last_bbox(wallet_address, bbox)

                        # Release track
                        self.store.release_track(track_id)
                        continue

                    # Add identity to detection
                    detection['wallet_address'] = wallet_address
                    detection['identity_confidence'] = identity.identity_confidence
                    print(f"[IDENTITY-TRACKER] ✅ Assigned {wallet_address[:8]} to detection", flush=True, file=sys.stderr)

                    # Update appearance periodically
                    last_verify = self._last_verified.get(track_id, 0)
                    if current_time - last_verify > 2.0:
                        self._update_appearance_embedding(wallet_address, frame, bbox)

                    # Check if re-verification needed
                    if current_time - last_verify > self.re_verify_interval:
                        self._schedule_reverification(track_id, bbox, frame, current_time)

                # Check if we need to acquire this track
                elif self.store.get_pending_reacquisitions():
                    import sys
                    matched_wallet = None
                    match_method = None

                    # FIRST: Try proximity-based re-acquisition
                    proximity_match = self.store.find_by_proximity(bbox, current_time)
                    if proximity_match:
                        matched_wallet = proximity_match
                        match_method = 'proximity'
                        print(f"[IDENTITY-TRACKER] ✅ PROXIMITY: track_id={track_id} -> {proximity_match[:8]}", flush=True, file=sys.stderr)

                    # SECOND: Try appearance-based Re-ID
                    if not matched_wallet:
                        appearance_embedding = self._extract_appearance_embedding(frame, bbox)
                        if appearance_embedding is not None:
                            appearance_match = self.store.find_by_appearance(appearance_embedding, current_time)
                            if appearance_match:
                                matched_wallet = appearance_match
                                match_method = 'appearance'
                                print(f"[IDENTITY-TRACKER] ✅ APPEARANCE: track_id={track_id} -> {appearance_match[:8]}", flush=True, file=sys.stderr)

                    # THIRD: Try face-based identification
                    if not matched_wallet:
                        print(f"[IDENTITY-TRACKER] Attempting face identification for track_id={track_id}", flush=True, file=sys.stderr)
                        face_match = self._attempt_identification(bbox, frame)
                        if face_match:
                            matched_wallet = face_match
                            match_method = 'face'
                            print(f"[IDENTITY-TRACKER] ✅ FACE: track_id={track_id} -> {face_match[:8]}", flush=True, file=sys.stderr)

                    # Apply the match
                    if matched_wallet:
                        confidence = 1.0 if match_method == 'face' else 0.85
                        self.store.assign_track(matched_wallet, track_id, confidence, match_method)
                        self._last_verified[track_id] = current_time

                        detection['wallet_address'] = matched_wallet
                        detection['identity_confidence'] = confidence
                        detection['reacquired_by'] = match_method
                        logger.info(f"{match_method.upper()} acquired track {track_id} as {matched_wallet[:8]}...")

                        # Update appearance embedding
                        self._update_appearance_embedding(matched_wallet, frame, bbox)

                enhanced_detections.append(detection)

            # Handle lost tracks
            for track_id in list(self._track_history.keys()):
                if track_id not in current_track_ids:
                    # Get wallet before releasing
                    wallet = self.store.get_wallet_by_track(track_id)

                    if wallet:
                        # Get last bbox from history
                        if self._track_history.get(track_id):
                            last_bbox = self._track_history[track_id][-1]['bbox']
                            self.store.update_last_bbox(wallet, last_bbox)

                        # Release track
                        self.store.release_track(track_id)

                        import sys
                        print(f"[IDENTITY-TRACKER] 📍 Lost track {track_id} ({wallet[:8]}...)", flush=True, file=sys.stderr)
                        logger.warning(f"Lost track of {wallet[:8]}... - scheduling re-acquisition")

                    # Clean up local state
                    self._track_history.pop(track_id, None)
                    self._last_verified.pop(track_id, None)

        return enhanced_detections

    def _attempt_identification(self, bbox: Tuple, frame: np.ndarray) -> Optional[str]:
        """
        Attempt to identify a person by their face.

        Args:
            bbox: Bounding box of person (x1, y1, x2, y2)
            frame: Current frame

        Returns:
            Wallet address if identified, None otherwise
        """
        if not self.face_service:
            return None

        # Check if any identities are pending
        if not self.store.get_pending_reacquisitions():
            return None

        try:
            # Extract person region
            x1, y1, x2, y2 = bbox
            person_region = frame[y1:y2, x1:x2]

            if person_region.size == 0:
                return None

            # Get face embedding
            embedding = self.face_service.extract_face_embedding(person_region)

            import sys
            print(f"[IDENTITY-TRACKER] extract_face_embedding returned: {embedding is not None}", flush=True, file=sys.stderr)

            if embedding is None:
                # Try direct recognition as fallback
                print(f"[IDENTITY-TRACKER] No embedding, trying direct face recognition", flush=True, file=sys.stderr)

                recognized_name, similarity = self.face_service.recognize_face(person_region)
                print(f"[IDENTITY-TRACKER] Direct recognition: name={recognized_name}, sim={similarity}", flush=True, file=sys.stderr)

                if recognized_name and similarity > 0.6:
                    # Check if this wallet is pending
                    if self.store.is_pending_reacquisition(recognized_name) or self.store.is_checked_in(recognized_name):
                        print(f"[IDENTITY-TRACKER] Matched to {recognized_name[:8]}...", flush=True, file=sys.stderr)
                        return recognized_name
                return None

            # Find match using IdentityStore
            matched_wallet = self.store.find_by_face(embedding, threshold=0.6)

            if matched_wallet:
                print(f"[IDENTITY-TRACKER] ✅ IDENTIFIED: {matched_wallet[:8]}...", flush=True, file=sys.stderr)
                logger.info(f"Identified person as {matched_wallet[:8]}...")
                return matched_wallet
            else:
                print(f"[IDENTITY-TRACKER] ❌ No face match found", flush=True, file=sys.stderr)

        except Exception as e:
            logger.error(f"Error in identification attempt: {e}")

        return None

    def _extract_appearance_embedding(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extract appearance embedding from a person crop using Re-ID model.

        Args:
            frame: Full frame
            bbox: Person bounding box (x1, y1, x2, y2)

        Returns:
            Normalized appearance embedding or None if extraction fails
        """
        import sys
        import torch

        reid_model, reid_transform = get_reid_model()
        if reid_model is None:
            return None

        try:
            x1, y1, x2, y2 = bbox
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0 or person_crop.shape[0] < 50 or person_crop.shape[1] < 50:
                return None

            # Convert BGR to RGB
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Transform and run through model
            input_tensor = reid_transform(person_rgb).unsqueeze(0).cuda()

            with torch.no_grad():
                embedding = reid_model(input_tensor)

            # Normalize embedding
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            return embedding

        except Exception as e:
            print(f"[RE-ID] Error extracting appearance: {e}", flush=True, file=sys.stderr)
            return None

    def _update_appearance_embedding(self, wallet_address: str, frame: np.ndarray, bbox: Tuple):
        """
        Update the stored appearance embedding for a tracked user.
        """
        embedding = self._extract_appearance_embedding(frame, bbox)
        if embedding is not None:
            self.store.update_appearance(wallet_address, embedding)

    def _schedule_reverification(self, track_id: int, bbox: Tuple, frame: np.ndarray, current_time: float):
        """
        Schedule face re-verification for a tracked person.

        Args:
            track_id: YOLO track ID
            bbox: Person bounding box
            frame: Current frame
            current_time: Current timestamp
        """
        wallet_address = self.store.get_wallet_by_track(track_id)
        if not wallet_address:
            return

        identified = self._attempt_identification(bbox, frame)

        if identified == wallet_address:
            # Face visible and matches - update timers
            self.store.update_face_seen(wallet_address)
            self._last_verified[track_id] = current_time

            identity = self.store.get_identity(wallet_address)
            if identity:
                new_confidence = min(1.0, identity.identity_confidence + 0.1)
                self.store.assign_track(wallet_address, track_id, new_confidence, 'face')

            logger.info(f"Re-verified track {track_id} as {wallet_address[:8]}...")

        elif identified and identified != wallet_address:
            # Face visible but WRONG PERSON - drop track
            logger.error(f"Track {track_id} changed identity! Was {wallet_address[:8]}..., now {identified[:8]}...")
            self.store.release_track(track_id)

        elif identified is None:
            # No face detected - trust track_id, just update verify time
            self._last_verified[track_id] = current_time
            logger.debug(f"Track {track_id} ({wallet_address[:8]}...) face not visible - trusting track")

    def get_active_identities(self) -> Dict:
        """
        Get all currently tracked identities.

        Returns:
            Dict with tracking state information
        """
        status = self.store.get_status()
        return {
            'tracked': {i['active_track_id']: i['wallet_address']
                       for i in status['identities'] if i['active_track_id']},
            'pending': self.store.get_pending_reacquisitions(),
            'sessions': self.store.get_checked_in_wallets()
        }

    def update_user_stats(self, wallet_address: str, stat_key: str, value: int = 1):
        """
        Update stats for a checked-in user (for app-specific tracking).

        Args:
            wallet_address: User's wallet address
            stat_key: Stat to update
            value: Amount to add to stat
        """
        self.store.update_stats(wallet_address, stat_key, value)
