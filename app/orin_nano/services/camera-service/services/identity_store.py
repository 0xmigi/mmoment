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

    # =========================================================================
    # Part 3: Unified ReID - Appearance History & Confirmation Levels
    # =========================================================================

    # Appearance history for robust body-based re-acquisition
    # Rolling buffer of recent body embeddings (512-dim from OSNet)
    appearance_history: List[np.ndarray] = field(default_factory=list)
    appearance_history_max: int = 10  # Keep last N body snapshots

    # Averaged appearance embedding for efficient cluster matching
    last_appearance_cluster: Optional[np.ndarray] = None

    # Identity confirmation level - determines trust in body-only tracking
    # 'none': Never face-confirmed (must face camera to be recognized)
    # 'face_once': Face matched 1-2 times (basic recognition)
    # 'face_confirmed': Face matched 3+ times (high trust, body-only OK)
    # 'body_maintained': Currently maintained by body tracking (face not visible)
    confirmation_level: str = 'none'
    face_confirmations: int = 0  # How many distinct face matches

    # Time of last face confirmation (for confidence decay)
    last_face_confirmation: float = 0.0

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
            # Part 3: Unified ReID fields
            'confirmation_level': self.confirmation_level,
            'face_confirmations': self.face_confirmations,
            'appearance_history_size': len(self.appearance_history),
            'has_appearance_cluster': self.last_appearance_cluster is not None,
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
    # Part 3: Unified ReID - Appearance History & Confirmation
    # =========================================================================

    def update_appearance_history(self, wallet_address: str, embedding: np.ndarray) -> bool:
        """
        Add a body embedding to the rolling appearance history.
        Also updates the appearance cluster (average of recent embeddings).

        Args:
            wallet_address: User's wallet address
            embedding: Body appearance embedding (512-dim from OSNet)

        Returns:
            True if successful
        """
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            if identity is None:
                return False

            # Validate embedding
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.size != 512:
                logger.warning(f"Invalid appearance embedding size: {embedding.size}")
                return False

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Add to history
            identity.appearance_history.append(embedding.copy())

            # Trim to max size
            while len(identity.appearance_history) > identity.appearance_history_max:
                identity.appearance_history.pop(0)

            # Update cluster center (average of recent appearances)
            if len(identity.appearance_history) >= 3:
                cluster = np.mean(identity.appearance_history, axis=0)
                # Re-normalize the cluster
                cluster_norm = np.linalg.norm(cluster)
                if cluster_norm > 0:
                    identity.last_appearance_cluster = cluster / cluster_norm
                else:
                    identity.last_appearance_cluster = cluster

            identity.last_appearance_update = time.time()
            return True

    def record_face_confirmation(self, wallet_address: str) -> str:
        """
        Record a face confirmation and update confirmation level.

        Confirmation levels:
        - 'none': Never face-confirmed
        - 'face_once': Face matched 1-2 times
        - 'face_confirmed': Face matched 3+ times (high trust)

        Args:
            wallet_address: User's wallet address

        Returns:
            New confirmation level
        """
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            if identity is None:
                return 'none'

            identity.face_confirmations += 1
            identity.last_face_confirmation = time.time()

            # Update confirmation level
            if identity.face_confirmations >= 3:
                identity.confirmation_level = 'face_confirmed'
            elif identity.face_confirmations >= 1:
                identity.confirmation_level = 'face_once'

            return identity.confirmation_level

    def get_confirmation_level(self, wallet_address: str) -> str:
        """Get confirmation level for a wallet"""
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            return identity.confirmation_level if identity else 'none'

    def find_by_appearance_cluster(
        self,
        embedding: np.ndarray,
        threshold: float = 0.65,
        require_confirmation: bool = True,
        exclude_active: bool = True
    ) -> Optional[Tuple[str, float]]:
        """
        Find an identity by matching against appearance clusters.

        This is used for extended re-acquisition - matching against the
        averaged appearance of face-confirmed users who have lost their track.

        Args:
            embedding: Current body appearance embedding
            threshold: Similarity threshold for matching
            require_confirmation: Only match 'face_once' or 'face_confirmed' users
            exclude_active: Skip users who already have an active track

        Returns:
            Tuple of (wallet_address, similarity) if match found, None otherwise
        """
        with self._store_lock:
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_match = None
            best_similarity = threshold

            for wallet, identity in self._identities.items():
                # Skip if no appearance cluster
                if identity.last_appearance_cluster is None:
                    continue

                # Skip if requires confirmation and not confirmed
                if require_confirmation:
                    if identity.confirmation_level not in ('face_once', 'face_confirmed'):
                        continue

                # Skip if has active track (unless we want to include active)
                if exclude_active and identity.active_track_id is not None:
                    continue

                # Cosine similarity against cluster
                similarity = float(np.dot(embedding, identity.last_appearance_cluster))

                # Apply confirmation-based threshold adjustment
                # face_confirmed users: use base threshold
                # face_once users: require higher similarity
                effective_threshold = threshold
                if identity.confirmation_level == 'face_once':
                    effective_threshold = threshold + 0.10  # More strict for less-confirmed

                if similarity > effective_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = wallet

            if best_match:
                return (best_match, best_similarity)
            return None

    def get_all_checked_in_with_cluster(self) -> List[Tuple[str, 'UserIdentity']]:
        """
        Get all checked-in identities that have an appearance cluster.
        Used for unified re-acquisition.

        Returns:
            List of (wallet_address, identity) tuples
        """
        with self._store_lock:
            return [
                (wallet, identity)
                for wallet, identity in self._identities.items()
                if identity.last_appearance_cluster is not None
            ]

    def set_confirmation_level(self, wallet_address: str, level: str) -> bool:
        """
        Manually set confirmation level (e.g., for body_maintained state).

        Args:
            wallet_address: User's wallet address
            level: New confirmation level

        Returns:
            True if successful
        """
        with self._store_lock:
            identity = self._identities.get(wallet_address)
            if identity is None:
                return False

            if level in ('none', 'face_once', 'face_confirmed', 'body_maintained'):
                identity.confirmation_level = level
                return True
            return False

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
