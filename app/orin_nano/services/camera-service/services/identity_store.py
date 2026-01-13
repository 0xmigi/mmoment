#!/usr/bin/env python3
"""
Unified Identity Store

Single source of truth for all user identity data on the camera.
This consolidates face embeddings, appearance embeddings, profiles, and tracking state
into a clean, atomic model that cv-apps can easily consume.

Key Design Principles:
- One UserIdentity object per checked-in user
- Atomic check-in/check-out (no partial states)
- Thread-safe access to all identity data
- Clean API for cv-apps to query identities
"""

import numpy as np
import time
import logging
import threading
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger("IdentityStore")


@dataclass
class UserIdentity:
    """
    Complete identity representation for a checked-in user.
    All identity-related data for a user in one place.
    """
    wallet_address: str

    # Profile data (from Farcaster/X auth)
    display_name: Optional[str] = None
    username: Optional[str] = None
    profile_image: Optional[str] = None

    # Biometric embeddings
    face_embedding: Optional[np.ndarray] = None  # 512-dim from InsightFace
    appearance_embedding: Optional[np.ndarray] = None  # 1280-dim from Re-ID model

    # Tracking state
    active_track_id: Optional[int] = None
    identity_confidence: float = 0.0
    last_face_seen: float = 0.0
    last_appearance_update: float = 0.0

    # Session metadata
    checked_in_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Lost track recovery data (for re-acquisition)
    last_known_bbox: Optional[Tuple[int, int, int, int]] = None
    last_known_center: Optional[Tuple[float, float]] = None

    # App-specific stats (for cv-apps to use)
    stats: Dict[str, Any] = field(default_factory=dict)

    def get_display_name(self) -> str:
        """Get the best available display name for this user"""
        if self.display_name:
            return self.display_name
        if self.username:
            return self.username
        return self.wallet_address[:8]

    def update_profile(self, profile: Dict) -> None:
        """Update profile data, preserving existing values if new ones are empty"""
        logger.info(f"🔍 [UPDATE-PROFILE] Input: {profile}")
        logger.info(f"🔍 [UPDATE-PROFILE] Before: display_name={self.display_name}, username={self.username}")
        if profile.get('display_name'):
            self.display_name = profile['display_name']
        if profile.get('username'):
            self.username = profile['username']
        if profile.get('profile_image'):
            self.profile_image = profile['profile_image']
        logger.info(f"🔍 [UPDATE-PROFILE] After: display_name={self.display_name}, username={self.username}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization/API responses"""
        return {
            'wallet_address': self.wallet_address,
            'display_name': self.get_display_name(),
            'username': self.username,
            'profile_image': self.profile_image,
            'has_face_embedding': self.face_embedding is not None,
            'has_appearance_embedding': self.appearance_embedding is not None,
            'active_track_id': self.active_track_id,
            'identity_confidence': self.identity_confidence,
            'checked_in_at': self.checked_in_at,
            'last_seen': self.last_seen,
            'session_duration': time.time() - self.checked_in_at,
            'stats': self.stats
        }


class IdentityStore:
    """
    Unified store for all user identities on this camera.

    Single source of truth that replaces:
    - gpu_face_service._face_embeddings
    - gpu_face_service._user_profiles
    - identity_tracker._active_sessions
    - identity_tracker._appearance_embeddings
    - identity_tracker._tracked_identities
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(IdentityStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._store_lock = threading.RLock()  # RLock for nested calls

        # Main identity storage
        self._identities: Dict[str, UserIdentity] = {}  # wallet_address -> UserIdentity

        # Reverse lookup for tracking
        self._track_to_wallet: Dict[int, str] = {}  # track_id -> wallet_address

        # Pending re-acquisition
        self._pending_reacquisition: set = set()  # wallet_addresses needing re-acquisition

        # Settings
        self._proximity_threshold = 250  # pixels for re-acquisition
        self._temporal_threshold = 5.0   # seconds for re-acquisition
        self._appearance_threshold = 0.7  # cosine similarity for appearance match
        self._face_timeout = 30.0  # seconds without face before expiry

        # Persistence
        self._data_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/faces")
        Path(self._data_dir).mkdir(parents=True, exist_ok=True)

        logger.info("IdentityStore initialized")

    # =========================================================================
    # Core Check-in/Check-out API
    # =========================================================================

    def check_in(self, wallet_address: str, face_embedding: np.ndarray = None,
                 profile: Dict = None) -> UserIdentity:
        """
        Check in a user atomically with all their identity data.

        Args:
            wallet_address: User's wallet address
            face_embedding: Face embedding from InsightFace (optional if already stored)
            profile: User profile dict with display_name, username, etc.

        Returns:
            UserIdentity object for the checked-in user
        """
        with self._store_lock:
            # Create or update identity
            if wallet_address in self._identities:
                identity = self._identities[wallet_address]
                # Update with new data
                if face_embedding is not None:
                    identity.face_embedding = face_embedding
                if profile:
                    identity.update_profile(profile)
                identity.last_seen = time.time()
                logger.info(f"Updated existing identity for {wallet_address[:8]}...")
            else:
                # Create new identity
                identity = UserIdentity(
                    wallet_address=wallet_address,
                    face_embedding=face_embedding,
                    checked_in_at=time.time(),
                    last_seen=time.time()
                )
                if profile:
                    identity.update_profile(profile)
                self._identities[wallet_address] = identity
                logger.info(f"Created new identity for {wallet_address[:8]}...")

            # Mark for tracking acquisition
            self._pending_reacquisition.add(wallet_address)

            # Save to disk
            if face_embedding is not None:
                self._save_identity(identity)

            return identity

    def check_out(self, wallet_address: str) -> Optional[Dict]:
        """
        Check out a user and clear ALL their identity data atomically.

        This ensures complete privacy - no traces left after checkout.

        Args:
            wallet_address: User's wallet address

        Returns:
            Session stats dict if user was checked in, None otherwise
        """
        with self._store_lock:
            if wallet_address not in self._identities:
                logger.warning(f"Checkout failed: {wallet_address[:8]}... not checked in")
                return None

            identity = self._identities[wallet_address]

            # Capture session stats before deletion
            session_stats = {
                'wallet_address': wallet_address,
                'display_name': identity.get_display_name(),
                'duration': time.time() - identity.checked_in_at,
                'stats': identity.stats.copy()
            }

            # Clear tracking association
            if identity.active_track_id is not None:
                self._track_to_wallet.pop(identity.active_track_id, None)

            # Clear from pending re-acquisition
            self._pending_reacquisition.discard(wallet_address)

            # Delete the identity
            del self._identities[wallet_address]

            # Delete from disk
            self._delete_identity_files(wallet_address)

            logger.info(f"✅ Checked out {wallet_address[:8]}... - all identity data cleared")

            return session_stats

    def update_profile(self, wallet_address: str, profile: Dict) -> bool:
        """
        Update user profile data for a checked-in user.

        Args:
            wallet_address: User's wallet address
            profile: Profile dict with display_name, username, etc.

        Returns:
            True if profile was updated, False if user not found
        """
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            self._identities[wallet_address].update_profile(profile)
            logger.info(f"Updated profile for {wallet_address[:8]}...")
            return True

    # =========================================================================
    # Tracking API (for IdentityTracker to use)
    # =========================================================================

    def assign_track(self, wallet_address: str, track_id: int, confidence: float = 1.0,
                    method: str = 'face') -> bool:
        """
        Assign a YOLO track_id to a wallet address.

        Args:
            wallet_address: User's wallet address
            track_id: YOLO track ID
            confidence: Initial confidence score
            method: How identity was confirmed ('face', 'proximity', 'appearance')

        Returns:
            True if assignment succeeded
        """
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            identity = self._identities[wallet_address]

            # Clear old track association if exists
            if identity.active_track_id is not None:
                self._track_to_wallet.pop(identity.active_track_id, None)

            # Set new track
            identity.active_track_id = track_id
            identity.identity_confidence = confidence
            identity.last_seen = time.time()

            # Set face timer based on method
            if method == 'face':
                identity.last_face_seen = time.time()
            else:
                # Grace period for non-face methods
                identity.last_face_seen = time.time() - 15.0

            self._track_to_wallet[track_id] = wallet_address
            self._pending_reacquisition.discard(wallet_address)

            return True

    def release_track(self, track_id: int, store_for_reacquisition: bool = True) -> Optional[str]:
        """
        Release a track_id (when YOLO loses the track).

        Args:
            track_id: YOLO track ID to release
            store_for_reacquisition: Whether to store state for re-acquisition

        Returns:
            Wallet address that was tracking if found
        """
        with self._store_lock:
            if track_id not in self._track_to_wallet:
                return None

            wallet_address = self._track_to_wallet.pop(track_id)

            if wallet_address in self._identities:
                identity = self._identities[wallet_address]
                identity.active_track_id = None

                if store_for_reacquisition:
                    self._pending_reacquisition.add(wallet_address)

            return wallet_address

    def get_identity_by_track(self, track_id: int) -> Optional[UserIdentity]:
        """Get identity associated with a track_id"""
        with self._store_lock:
            wallet = self._track_to_wallet.get(track_id)
            if wallet:
                return self._identities.get(wallet)
            return None

    def get_wallet_by_track(self, track_id: int) -> Optional[str]:
        """Get wallet address for a track_id"""
        with self._store_lock:
            return self._track_to_wallet.get(track_id)

    def is_pending_reacquisition(self, wallet_address: str) -> bool:
        """Check if a wallet needs track re-acquisition"""
        with self._store_lock:
            return wallet_address in self._pending_reacquisition

    def get_pending_reacquisitions(self) -> List[str]:
        """Get all wallets pending re-acquisition"""
        with self._store_lock:
            return list(self._pending_reacquisition)

    def update_appearance(self, wallet_address: str, embedding: np.ndarray) -> bool:
        """Update appearance embedding for a user"""
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            self._identities[wallet_address].appearance_embedding = embedding
            self._identities[wallet_address].last_appearance_update = time.time()
            return True

    def update_last_bbox(self, wallet_address: str, bbox: Tuple[int, int, int, int]) -> bool:
        """Update last known bounding box for re-acquisition"""
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            identity = self._identities[wallet_address]
            identity.last_known_bbox = bbox
            identity.last_known_center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            )
            identity.last_seen = time.time()
            return True

    def update_face_seen(self, wallet_address: str) -> bool:
        """Mark that a face was seen for this user (resets timeout)"""
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            self._identities[wallet_address].last_face_seen = time.time()
            self._identities[wallet_address].last_seen = time.time()
            return True

    def check_face_timeout(self, wallet_address: str) -> bool:
        """Check if user has gone too long without face being seen"""
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            identity = self._identities[wallet_address]
            time_without_face = time.time() - identity.last_face_seen
            return time_without_face > self._face_timeout

    # =========================================================================
    # Query API (for cv-apps and services)
    # =========================================================================

    def get_identity(self, wallet_address: str) -> Optional[UserIdentity]:
        """Get identity by wallet address"""
        with self._store_lock:
            return self._identities.get(wallet_address)

    def get_all_identities(self) -> List[UserIdentity]:
        """Get all checked-in identities"""
        with self._store_lock:
            return list(self._identities.values())

    def get_active_identities(self) -> List[UserIdentity]:
        """Get identities with active tracking"""
        with self._store_lock:
            return [i for i in self._identities.values() if i.active_track_id is not None]

    def get_display_name(self, wallet_address: str) -> str:
        """Get display name for a wallet address"""
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            if identity:
                return identity.get_display_name()
            return wallet_address[:8]

    def get_face_embedding(self, wallet_address: str) -> Optional[np.ndarray]:
        """Get face embedding for a wallet address"""
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            return identity.face_embedding if identity else None

    def get_appearance_embedding(self, wallet_address: str) -> Optional[np.ndarray]:
        """Get appearance embedding for a wallet address"""
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            return identity.appearance_embedding if identity else None

    def get_checked_in_wallets(self) -> List[str]:
        """Get all checked-in wallet addresses"""
        with self._store_lock:
            return list(self._identities.keys())

    def is_checked_in(self, wallet_address: str) -> bool:
        """Check if a wallet is currently checked in"""
        with self._store_lock:
            return wallet_address in self._identities

    def get_identity_count(self) -> int:
        """Get number of checked-in identities"""
        with self._store_lock:
            return len(self._identities)

    def update_stats(self, wallet_address: str, stat_key: str, value: Any = 1) -> bool:
        """Update app-specific stats for a user"""
        with self._store_lock:
            if wallet_address not in self._identities:
                return False

            identity = self._identities[wallet_address]
            if isinstance(value, (int, float)) and stat_key in identity.stats:
                identity.stats[stat_key] += value
            else:
                identity.stats[stat_key] = value
            return True

    def get_stats(self, wallet_address: str) -> Optional[Dict]:
        """Get stats for a user"""
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            return identity.stats.copy() if identity else None

    # =========================================================================
    # Re-acquisition helpers
    # =========================================================================

    def find_by_proximity(self, bbox: Tuple[int, int, int, int],
                         current_time: float) -> Optional[str]:
        """
        Find a pending re-acquisition identity by proximity to bbox.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            current_time: Current timestamp

        Returns:
            Wallet address if match found
        """
        with self._store_lock:
            if not self._pending_reacquisition:
                return None

            new_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            best_match = None
            best_distance = float('inf')

            for wallet in list(self._pending_reacquisition):
                identity = self._identities.get(wallet)
                if not identity or not identity.last_known_center:
                    continue

                # Check temporal threshold
                time_since_lost = current_time - identity.last_seen
                if time_since_lost > self._temporal_threshold:
                    continue

                # Calculate distance
                distance = (
                    (new_center[0] - identity.last_known_center[0]) ** 2 +
                    (new_center[1] - identity.last_known_center[1]) ** 2
                ) ** 0.5

                if distance < self._proximity_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = wallet

            return best_match

    def find_by_appearance(self, embedding: np.ndarray,
                          current_time: float) -> Optional[str]:
        """
        Find a pending re-acquisition identity by appearance similarity.

        Args:
            embedding: Appearance embedding from Re-ID model
            current_time: Current timestamp

        Returns:
            Wallet address if match found
        """
        with self._store_lock:
            if not self._pending_reacquisition:
                return None

            best_match = None
            best_similarity = 0.0

            for wallet in list(self._pending_reacquisition):
                identity = self._identities.get(wallet)
                if not identity or identity.appearance_embedding is None:
                    continue

                # Check temporal threshold
                time_since_lost = current_time - identity.last_seen
                if time_since_lost > self._temporal_threshold:
                    continue

                # Cosine similarity
                similarity = float(np.dot(embedding, identity.appearance_embedding))

                if similarity > self._appearance_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = wallet

            return best_match

    def find_by_face(self, embedding: np.ndarray, threshold: float = 0.70) -> Optional[str]:
        """
        Find an identity by face embedding similarity.

        Args:
            embedding: Face embedding from InsightFace
            threshold: Similarity threshold

        Returns:
            Wallet address if match found
        """
        with self._store_lock:
            best_match = None
            best_similarity = 0.0

            for wallet, identity in self._identities.items():
                if identity.face_embedding is None:
                    continue

                similarity = float(np.dot(embedding, identity.face_embedding))

                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = wallet

            return best_match

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_identity(self, identity: UserIdentity) -> None:
        """Save identity to disk"""
        try:
            # Save face embedding
            if identity.face_embedding is not None:
                embedding_path = os.path.join(
                    self._data_dir, f"{identity.wallet_address}.npy"
                )
                np.save(embedding_path, identity.face_embedding)

            # Save metadata
            metadata = {
                'display_name': identity.display_name,
                'username': identity.username,
                'profile_image': identity.profile_image
            }
            metadata_path = os.path.join(
                self._data_dir, f"{identity.wallet_address}_metadata.json"
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            logger.debug(f"Saved identity data for {identity.wallet_address[:8]}...")

        except Exception as e:
            logger.error(f"Failed to save identity: {e}")

    def _delete_identity_files(self, wallet_address: str) -> None:
        """Delete identity files from disk"""
        try:
            embedding_path = os.path.join(self._data_dir, f"{wallet_address}.npy")
            metadata_path = os.path.join(self._data_dir, f"{wallet_address}_metadata.json")

            if os.path.exists(embedding_path):
                os.unlink(embedding_path)
                logger.debug(f"Deleted embedding file for {wallet_address[:8]}...")

            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
                logger.debug(f"Deleted metadata file for {wallet_address[:8]}...")

        except Exception as e:
            logger.error(f"Failed to delete identity files: {e}")

    def load_from_disk(self) -> int:
        """
        Load face embeddings from disk (for service restart).
        Note: This only loads embeddings, not active sessions.

        Returns:
            Number of embeddings loaded
        """
        count = 0

        for embedding_file in Path(self._data_dir).glob("*.npy"):
            try:
                wallet_address = embedding_file.stem
                embedding = np.load(embedding_file)

                # Load metadata
                metadata_file = embedding_file.with_name(f"{wallet_address}_metadata.json")
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                # Create identity (but NOT checked in - this is just for recognition)
                # Actual check-in happens via blockchain events
                with self._store_lock:
                    if wallet_address not in self._identities:
                        identity = UserIdentity(
                            wallet_address=wallet_address,
                            face_embedding=embedding,
                            display_name=metadata.get('display_name'),
                            username=metadata.get('username'),
                            profile_image=metadata.get('profile_image')
                        )
                        self._identities[wallet_address] = identity
                        count += 1

            except Exception as e:
                logger.warning(f"Failed to load {embedding_file}: {e}")

        logger.info(f"Loaded {count} face embeddings from disk")
        return count

    # =========================================================================
    # Status/Debug
    # =========================================================================

    def get_status(self) -> Dict:
        """Get store status for debugging"""
        with self._store_lock:
            return {
                'total_identities': len(self._identities),
                'active_tracks': len(self._track_to_wallet),
                'pending_reacquisition': len(self._pending_reacquisition),
                'identities': [i.to_dict() for i in self._identities.values()]
            }


# Global singleton accessor
def get_identity_store() -> IdentityStore:
    """Get the IdentityStore singleton instance"""
    return IdentityStore()
