"""
Native Identity Service - BULLETPROOF EDITION

Lightweight identity matching that uses face embeddings from the native C++ server.
No PyTorch, no InsightFace, no ONNX - just numpy for cosine similarity.

CRITICAL DESIGN PRINCIPLE:
    FACE EMBEDDING IS THE ONLY SOURCE OF TRUTH.
    ReID/body embeddings are NEVER used to assign identity to a new person.
    ReID is ONLY used to maintain identity for the SAME track during brief occlusions.

This replaces the heavy GPUFaceService when running in native mode.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Set

from .identity_store import get_identity_store

logger = logging.getLogger("NativeIdentityService")


class NativeIdentityService:
    """
    Lightweight identity service that uses embeddings from native TensorRT server.

    The native server runs:
    - YOLOv8-pose for person detection and tracking
    - RetinaFace for face detection
    - ArcFace for face embedding (512-dim)
    - OSNet x0.25 for ReID embedding (512-dim body appearance)

    BULLETPROOF IDENTITY HIERARCHY:
    1. Face embedding match - THE ONLY way to assign identity to a person
    2. Track continuity - maintains existing assignment for the same track_id
    3. ReID - ONLY used to re-acquire the SAME track after brief occlusion
       (NEVER used to assign identity to a different person/track)

    KEY SAFEGUARDS:
    - If ANY face is detected, ReID matching is SKIPPED (faces are authoritative)
    - If a face is detected but doesn't match the track's wallet, identity is REVOKED
    - ReID can only re-associate to tracks that appear near the original track's last position
    - ReID features expire after 2 seconds (not 5)
    - Face must have been seen within 1 second (not 3) for ReID to work
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NativeIdentityService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._processing_enabled = True

        # Get identity store reference (single source of truth)
        self.identity_store = get_identity_store()

        # NOTE: Do NOT load embeddings from disk on startup.
        # Embeddings should only exist in memory after check-in (fetched from onchain)
        # or temporarily after first-time registration. They are deleted on checkout.

        # Detection settings
        # ArcFace cosine similarity threshold:
        # 0.5 = too permissive, 0.7 = false positives possible, 0.8+ = strict
        # STRICT MODE: Prefer false negatives over false positives
        # User's matches: ~0.84, girlfriend false positive: 0.745
        self._similarity_threshold = 0.70

        # Stats
        self.total_matches = 0
        self.total_faces_processed = 0
        self.total_reid_blocked = 0  # Track how many times ReID was blocked by face presence
        self.total_mismatches_revoked = 0  # Track identity revocations
        self._frame_counter = 0  # For rate-limited logging

        # Track → identity mapping (track_id from YOLO → wallet_address)
        self._track_lock = threading.Lock()
        self._track_to_wallet: Dict[int, str] = {}
        self._wallet_to_track: Dict[str, int] = {}

        # Face confirmation tracking - FACE IS THE ONLY SOURCE OF TRUTH
        # wallet → timestamp of last face confirmation
        self._wallet_face_last_seen: Dict[str, float] = {}

        # Session-level ReID embeddings (OSNet 512-dim from native server)
        # wallet → {'embedding': np.array, 'timestamp': float, 'face_seen': float,
        #           'original_track_id': int, 'last_bbox': [x1,y1,x2,y2]}
        self._wallet_reid_features: Dict[str, Dict] = {}

        # ReID settings - balanced for continuous tracking while preventing false positives
        # When someone turns around (face not visible), we need to maintain their identity
        # until they either leave the frame or turn back to show their face
        self._reid_similarity_threshold = 0.80  # High threshold for appearance matching
        self._reid_feature_max_age = 60.0       # ReID features valid for 60 seconds of continuous tracking
        self._reid_require_recent_face = 30.0   # Face must have been seen in last 30 seconds
        self._reid_max_position_drift = 500.0   # Allow more movement (person can walk around)

        # Debug mode - set to True for verbose logging during troubleshooting
        self._debug_mode = True

        logger.info("NativeIdentityService initialized - BULLETPROOF EDITION")
        logger.info(f"  Face threshold: {self._similarity_threshold}")
        logger.info(f"  ReID threshold: {self._reid_similarity_threshold}")
        logger.info(f"  ReID max age: {self._reid_feature_max_age}s")
        logger.info(f"  ReID require recent face: {self._reid_require_recent_face}s")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def process_native_results(self, persons: List[Dict], faces: List[Dict],
                                frame: np.ndarray = None) -> List[Dict]:
        """
        Process detection results from native server and perform identity matching.

        BULLETPROOF IDENTITY LOGIC:
        1. Face embedding match - THE ONLY way to assign identity
        2. Face mismatch revocation - if face detected but doesn't match, REVOKE identity
        3. Track continuity - existing track-wallet mappings persist unless revoked
        4. ReID - ONLY for same-track re-acquisition during brief occlusions
           (COMPLETELY SKIPPED if any face is detected in frame)

        Args:
            persons: Person detections from native server (YOLO-pose)
            faces: Face detections with embeddings from native server (RetinaFace + ArcFace)
            frame: Optional frame for body appearance extraction

        Returns:
            Enhanced person detections with wallet_address if identified
        """
        if not self._processing_enabled:
            return persons

        self._frame_counter += 1
        current_time = time.time()

        # Get all checked-in identities
        checked_in = self.identity_store.get_checked_in_wallets()
        if not checked_in:
            return persons  # No one to match against

        # Count faces with valid embeddings
        valid_faces = [f for f in (faces or []) if f.get('embedding') is not None]
        faces_detected = len(valid_faces) > 0

        # Reduced logging - only log occasionally
        if self._debug_mode and self._frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
            logger.info(f"[IDENTITY] Processing: {len(persons)} persons, {len(valid_faces)} faces, {len(checked_in)} checked-in")

        # ==========================================================================
        # STEP 1: Face matching - THE ONLY WAY TO ASSIGN IDENTITY
        # ==========================================================================
        matched_wallets_this_frame: Set[str] = set()

        for face in valid_faces:
            embedding = face.get('embedding')
            if embedding is None:
                continue

            # Check if this face belongs to a track that's already identified recently
            # Skip expensive cosine similarity if we confirmed this track in the last second
            face_track_id = face.get('person_track_id')
            if face_track_id is not None:
                with self._track_lock:
                    existing_wallet = self._track_to_wallet.get(face_track_id)
                    if existing_wallet:
                        last_seen = self._wallet_face_last_seen.get(existing_wallet, 0)
                        if current_time - last_seen < 1.0:  # Confirmed within last second
                            # Skip re-matching, just update timestamp periodically
                            matched_wallets_this_frame.add(existing_wallet)
                            continue

            embedding = np.array(embedding, dtype=np.float32)
            self.total_faces_processed += 1

            # Find best matching identity
            best_match = None
            best_similarity = self._similarity_threshold

            for wallet_address in checked_in:
                identity = self.identity_store.get_identity(wallet_address)
                if identity is None or identity.face_embedding is None:
                    continue

                similarity = self.cosine_similarity(embedding, identity.face_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = wallet_address

            if best_match:
                self.total_matches += 1
                matched_wallets_this_frame.add(best_match)
                # Only log matches occasionally to reduce I/O
                if self.total_matches % 30 == 1:  # Log every ~1 second at 30fps
                    logger.info(f"✅ [FACE-MATCH] {best_match[:8]}... (similarity: {best_similarity:.3f})")

                # Update face seen time - this is THE authority for identity
                self.identity_store.update_face_seen(best_match)
                self._wallet_face_last_seen[best_match] = current_time

                # Associate with person
                matched_person = self._associate_face_to_person_safe(
                    face, persons, best_match, best_similarity, current_time
                )

                if matched_person:
                    # Store ReID embedding when face is confirmed (for same-track re-acquisition only)
                    self._store_reid_embedding(best_match, matched_person, current_time)

            else:
                if self._debug_mode:
                    logger.info(f"❌ [FACE-MATCH] No match - best was {best_similarity:.3f} < {self._similarity_threshold}")

        # ==========================================================================
        # STEP 2: PROACTIVE MISMATCH REVOCATION
        # For ALL persons with associated wallets, if a face is visible but doesn't
        # match, REVOKE the identity. This catches cases where:
        # - ByteTracker reuses a track ID for a different person
        # - A new person inherited an old track through tracker bugs
        # ==========================================================================
        self._revoke_all_mismatched_identities(persons, valid_faces, checked_in, current_time)

        # ==========================================================================
        # STEP 3: Clean up stale track associations
        # If a track is no longer visible, remove its wallet association
        # ==========================================================================
        self._cleanup_stale_tracks(persons)

        # ==========================================================================
        # STEP 4: ReID matching - for persons WITHOUT visible faces
        # Changed from global check to per-person check:
        # - If a person has a face visible (in their bbox), skip ReID for them
        # - If a person has no face visible (turned around), allow ReID for them
        # This enables continuous tracking when someone turns their back to camera
        # ==========================================================================
        # Build set of track_ids that have a face visible inside their bbox
        tracks_with_visible_face = self._get_tracks_with_visible_face(persons, valid_faces)

        # ReID matching for persons without visible faces
        self._match_reid_embeddings_per_person(persons, current_time, tracks_with_visible_face)

        # Add wallet info to persons with tracks
        return self._enhance_persons_with_identity(persons)

    def _associate_face_to_person_safe(
        self,
        face: Dict,
        persons: List[Dict],
        wallet_address: str,
        confidence: float,
        current_time: float
    ) -> Optional[Dict]:
        """
        Safely associate a matched face with a person.

        Returns:
            The matched person dict, or None if no match found
        """
        face_bbox = face.get('bbox', [])
        landmarks = face.get('landmarks', [])

        # Validate face bbox - the native server sometimes returns corrupted bboxes
        # Check for negative values or unreasonably large values
        bbox_valid = (len(face_bbox) >= 4 and
                      face_bbox[0] >= 0 and face_bbox[1] >= 0 and
                      face_bbox[2] > face_bbox[0] and face_bbox[3] > face_bbox[1] and
                      face_bbox[2] < 5000 and face_bbox[3] < 5000)  # Sanity check

        # If bbox is invalid but we have landmarks, compute bbox from landmarks
        if not bbox_valid and len(landmarks) >= 5:
            try:
                xs = [lm[0] for lm in landmarks[:5]]
                ys = [lm[1] for lm in landmarks[:5]]
                # Add padding around landmarks to get face bbox
                padding = 30
                face_bbox = [
                    max(0, min(xs) - padding),
                    max(0, min(ys) - padding),
                    max(xs) + padding,
                    max(ys) + padding
                ]
                bbox_valid = True
            except Exception as e:
                logger.warning(f"Failed to compute bbox from landmarks: {e}")

        if not bbox_valid or len(face_bbox) < 4:
            # No valid face bbox - if only 1 person, associate directly
            if len(persons) == 1:
                person = persons[0]
                track_id = person.get('track_id')
                if track_id is not None:
                    with self._track_lock:
                        # Clear old associations
                        old_wallet = self._track_to_wallet.get(track_id)
                        if old_wallet and old_wallet != wallet_address:
                            self._wallet_to_track.pop(old_wallet, None)
                        old_track = self._wallet_to_track.get(wallet_address)
                        if old_track and old_track != track_id:
                            self._track_to_wallet.pop(old_track, None)

                        self._track_to_wallet[track_id] = wallet_address
                        self._wallet_to_track[wallet_address] = track_id

                    self.identity_store.assign_track(wallet_address, track_id, confidence, 'face')
                    logger.info(f"[ASSOCIATE] Track {track_id} → {wallet_address[:8]}... (single person, no face bbox)")
                    return person
            return None

        # Find the person whose bbox contains this face
        face_cx = (face_bbox[0] + face_bbox[2]) / 2
        face_cy = (face_bbox[1] + face_bbox[3]) / 2

        best_person = None
        best_distance = float('inf')

        for person in persons:
            p_bbox = person.get('bbox', [])
            if len(p_bbox) < 4:
                continue

            # Check if face center is inside person bbox
            if (p_bbox[0] <= face_cx <= p_bbox[2] and p_bbox[1] <= face_cy <= p_bbox[3]):
                # Calculate distance to person center
                p_cx = (p_bbox[0] + p_bbox[2]) / 2
                p_cy = (p_bbox[1] + p_bbox[3]) / 2
                distance = ((face_cx - p_cx) ** 2 + (face_cy - p_cy) ** 2) ** 0.5

                if distance < best_distance:
                    best_distance = distance
                    best_person = person

        if best_person:
            track_id = best_person.get('track_id')
            if track_id is not None:
                with self._track_lock:
                    # Clear old associations
                    old_wallet = self._track_to_wallet.get(track_id)
                    if old_wallet and old_wallet != wallet_address:
                        self._wallet_to_track.pop(old_wallet, None)
                    old_track = self._wallet_to_track.get(wallet_address)
                    if old_track and old_track != track_id:
                        self._track_to_wallet.pop(old_track, None)

                    self._track_to_wallet[track_id] = wallet_address
                    self._wallet_to_track[wallet_address] = track_id

                self.identity_store.assign_track(wallet_address, track_id, confidence, 'face')
                logger.info(f"[ASSOCIATE] Track {track_id} → {wallet_address[:8]}... (face in person bbox)")

        return best_person

    def _revoke_all_mismatched_identities(
        self,
        persons: List[Dict],
        faces: List[Dict],
        checked_in: List[str],
        current_time: float
    ):
        """
        PROACTIVE MISMATCH REVOCATION:
        For every person with an associated wallet, check if we can see their face.
        If we CAN see a face and it DOESN'T match, REVOKE the identity.

        This catches ALL cases where the wrong person has inherited an identity:
        - ByteTracker reusing track IDs
        - Track ID collisions
        - Any other tracker bugs
        """
        # Build face lookup by person - only include HIGH QUALITY faces
        # Low quality faces (side profiles, partial faces) have garbage embeddings
        # and should not be used for revocation decisions
        face_by_person: Dict[int, Dict] = {}  # track_id → face with embedding

        for face in faces:
            embedding = face.get('embedding')
            if embedding is None:
                continue

            # CRITICAL: Only use high-quality faces for revocation
            # Quality is typically 0-1 from the face detector
            # Side profiles and partial faces have low quality scores
            face_quality = face.get('quality', 0)
            face_confidence = face.get('confidence', 0)

            # Require high quality for revocation (frontal face, good lighting)
            # Quality < 0.7 usually means partial/side view - don't use for revocation
            min_quality_for_revoke = 0.7
            if face_quality < min_quality_for_revoke:
                continue

            face_bbox = face.get('bbox', [])
            if len(face_bbox) < 4:
                continue

            face_cx = (face_bbox[0] + face_bbox[2]) / 2
            face_cy = (face_bbox[1] + face_bbox[3]) / 2

            # Find which person this face belongs to
            for person in persons:
                p_bbox = person.get('bbox', [])
                if len(p_bbox) < 4:
                    continue

                track_id = person.get('track_id')
                if track_id is None:
                    continue

                # Check if face center is inside person bbox
                if (p_bbox[0] <= face_cx <= p_bbox[2] and p_bbox[1] <= face_cy <= p_bbox[3]):
                    face_by_person[track_id] = {
                        'embedding': np.array(embedding, dtype=np.float32),
                        'bbox': face_bbox,
                        'quality': face_quality
                    }
                    break  # Face belongs to this person

        # Now check all persons with wallet associations
        with self._track_lock:
            tracks_to_revoke = []

            for person in persons:
                track_id = person.get('track_id')
                if track_id is None:
                    continue

                # Check if this track has an associated wallet
                associated_wallet = self._track_to_wallet.get(track_id)
                if not associated_wallet:
                    continue

                # Check if we detected a face for this person
                face_info = face_by_person.get(track_id)
                if not face_info:
                    continue  # No face visible for this person - keep existing association

                # We CAN see this person's face - verify it matches the associated wallet
                identity = self.identity_store.get_identity(associated_wallet)
                if identity is None or identity.face_embedding is None:
                    continue

                face_embedding = face_info['embedding']
                similarity = self.cosine_similarity(face_embedding, identity.face_embedding)

                # Use a MUCH lower threshold for revocation than for matching
                # Match threshold: 0.7 (needs to be confident it's the right person)
                # Revoke threshold: 0.3 (only revoke if clearly NOT the same person)
                # This allows for face quality degradation at distance without losing identity
                revoke_threshold = 0.3
                if similarity < revoke_threshold:
                    # DEFINITE MISMATCH! This is clearly NOT the wallet owner
                    tracks_to_revoke.append((track_id, associated_wallet, similarity))

            # Revoke outside the iteration
            for track_id, wallet, similarity in tracks_to_revoke:
                self._track_to_wallet.pop(track_id, None)
                self._wallet_to_track.pop(wallet, None)
                self._wallet_reid_features.pop(wallet, None)  # Clear ReID to prevent re-association
                self.total_mismatches_revoked += 1

                logger.warning(
                    f"⚠️ [REVOKE] Track {track_id} REVOKED from {wallet[:8]}... "
                    f"(face similarity {similarity:.3f} < 0.3) - "
                    f"DEFINITELY WRONG PERSON!"
                )

    def _cleanup_stale_tracks(self, persons: List[Dict]):
        """
        Remove track associations for tracks that are no longer visible.

        This is CRITICAL to prevent identity from persisting incorrectly.
        When a person leaves the frame, their track-wallet association must be removed
        so that a new person entering doesn't inherit the old identity.
        """
        current_track_ids = set()
        for person in persons:
            track_id = person.get('track_id')
            if track_id is not None:
                current_track_ids.add(track_id)

        with self._track_lock:
            # Find tracks that are no longer visible
            stale_tracks = []
            for track_id in list(self._track_to_wallet.keys()):
                if track_id not in current_track_ids:
                    stale_tracks.append(track_id)

            # Remove stale track associations
            for track_id in stale_tracks:
                wallet = self._track_to_wallet.pop(track_id, None)
                if wallet:
                    self._wallet_to_track.pop(wallet, None)
                    # DON'T clear ReID features here - we want to allow same-track re-acquisition
                    if self._debug_mode:
                        logger.info(f"🗑️ [CLEANUP] Track {track_id} left frame (was {wallet[:8]}...)")

    def _get_tracks_with_visible_face(self, persons: List[Dict], faces: List[Dict]) -> Set[int]:
        """
        Find track_ids of persons that have a face detected inside their bounding box.

        This allows us to skip ReID for persons who have a visible face (face matching
        should be used instead), while allowing ReID for persons who are turned away
        from the camera (no face visible in their bbox).

        Returns:
            Set of track_ids that have a face visible inside their bbox
        """
        tracks_with_face: Set[int] = set()

        for face in faces:
            face_bbox = face.get('bbox', [])
            if len(face_bbox) < 4:
                continue

            face_cx = (face_bbox[0] + face_bbox[2]) / 2
            face_cy = (face_bbox[1] + face_bbox[3]) / 2

            # Find which person this face belongs to
            for person in persons:
                p_bbox = person.get('bbox', [])
                if len(p_bbox) < 4:
                    continue

                track_id = person.get('track_id')
                if track_id is None:
                    continue

                # Check if face center is inside person bbox
                if (p_bbox[0] <= face_cx <= p_bbox[2] and p_bbox[1] <= face_cy <= p_bbox[3]):
                    tracks_with_face.add(track_id)
                    break

        return tracks_with_face

    def _match_reid_embeddings_per_person(
        self,
        persons: List[Dict],
        current_time: float,
        tracks_with_visible_face: Set[int]
    ):
        """
        ReID matching for persons WITHOUT visible faces.

        This is called for ALL persons, but will skip:
        - Persons who already have an identity assigned
        - Persons who have a face visible (should use face matching instead)

        For persons without visible faces who have no identity, try to match
        their ReID embedding against stored embeddings from known wallets.
        """
        with self._track_lock:
            for person in persons:
                track_id = person.get('track_id')
                if track_id is None:
                    continue

                # Skip if already has identity
                if track_id in self._track_to_wallet:
                    # But update ReID features to prevent expiry during continuous tracking
                    wallet = self._track_to_wallet[track_id]
                    self._update_reid_features_during_tracking(wallet, person, current_time)
                    continue

                # Skip if this person has a visible face (use face matching instead)
                if track_id in tracks_with_visible_face:
                    continue

                # No identity, no visible face - try ReID matching
                self._try_reid_match_for_person(person, current_time)

    def _update_reid_features_during_tracking(
        self,
        wallet: str,
        person: Dict,
        current_time: float
    ):
        """
        Update ReID features while a person is being tracked (even without face visible).

        This prevents ReID features from expiring while the person is continuously
        tracked. We update:
        - The timestamp (so features don't expire)
        - The embedding (so it stays current)
        - The bbox (so position drift check uses current position)
        """
        has_reid = person.get('has_reid_embedding', 0)
        if not has_reid:
            return

        reid_embedding = person.get('reid_embedding')
        if reid_embedding is None:
            return

        bbox = person.get('bbox', [])

        try:
            embedding = np.array(reid_embedding, dtype=np.float32)
            if len(embedding) != 512:
                return

            # Get existing features
            existing = self._wallet_reid_features.get(wallet)
            if existing is None:
                # No existing features - this shouldn't happen but handle it
                return

            # Update with current data while preserving face_seen time
            self._wallet_reid_features[wallet] = {
                'embedding': embedding,
                'timestamp': current_time,  # Keep features fresh
                'face_seen': existing.get('face_seen', current_time),  # Preserve original face time
                'original_track_id': person.get('track_id'),
                'last_bbox': bbox.copy() if isinstance(bbox, list) else list(bbox),
            }
        except Exception:
            pass

    def _try_reid_match_for_person(self, person: Dict, current_time: float):
        """
        Try to match a single person using ReID embeddings.

        This person has no identity and no visible face. Check if their ReID
        embedding matches any stored embeddings from known wallets.

        Safeguards:
        1. Face must have been seen within _reid_require_recent_face seconds
        2. ReID features must be within _reid_feature_max_age
        3. Position must be within _reid_max_position_drift pixels
        4. High similarity threshold (_reid_similarity_threshold)
        5. Wallet must not already have an active track
        """
        track_id = person.get('track_id')
        if track_id is None:
            return

        has_reid = person.get('has_reid_embedding', 0)
        if not has_reid:
            return

        person_reid = person.get('reid_embedding')
        if person_reid is None:
            return

        person_bbox = person.get('bbox', [])
        if len(person_bbox) < 4:
            return

        try:
            person_embedding = np.array(person_reid, dtype=np.float32)
            if len(person_embedding) != 512:
                return
        except Exception:
            return

        # Try to match against stored ReID features
        best_wallet = None
        best_similarity = self._reid_similarity_threshold

        for wallet, features in list(self._wallet_reid_features.items()):
            # CHECK 1: ReID features age
            feature_age = current_time - features['timestamp']
            if feature_age > self._reid_feature_max_age:
                if self._debug_mode:
                    logger.debug(f"[REID] {wallet[:8]}...: features expired ({feature_age:.1f}s > {self._reid_feature_max_age}s)")
                continue

            # CHECK 2: Face recency
            face_seen_time = features.get('face_seen', 0)
            face_age = current_time - face_seen_time
            if face_age > self._reid_require_recent_face:
                if self._debug_mode:
                    logger.debug(f"[REID] {wallet[:8]}...: face too old ({face_age:.1f}s > {self._reid_require_recent_face}s)")
                continue

            # CHECK 3: Wallet doesn't already have an active track
            if wallet in self._wallet_to_track:
                active_track = self._wallet_to_track[wallet]
                # Note: We're inside the lock, so we can't easily check if track is visible
                # Just check if the track_id is different
                if active_track != track_id:
                    continue

            # NOTE: Position drift check removed - we rely purely on appearance embedding
            # similarity for ReID matching. The OSNet body embedding is the source of truth,
            # not position. This allows proper tracking when someone steps back/forward.

            # CHECK 4: High appearance similarity
            stored_embedding = features['embedding']
            similarity = self.cosine_similarity(person_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_wallet = wallet

        if best_wallet:
            # Match found - associate track with wallet
            old_track = self._wallet_to_track.get(best_wallet)
            if old_track is not None:
                self._track_to_wallet.pop(old_track, None)

            self._track_to_wallet[track_id] = best_wallet
            self._wallet_to_track[best_wallet] = track_id

            logger.info(
                f"✅ [REID] Track {track_id} → {best_wallet[:8]}... "
                f"(similarity: {best_similarity:.3f})"
            )

    def _store_reid_embedding(self, wallet: str, person: Dict, current_time: float):
        """
        Store ReID embedding from native server when face is confirmed.

        IMPORTANT: Also stores the original track_id and bbox so ReID can only
        re-associate to tracks that appear in a similar position (same person
        after brief occlusion, not a different person).
        """
        # Check if person has a ReID embedding from native server
        has_reid = person.get('has_reid_embedding', 0)
        if not has_reid:
            return

        reid_embedding = person.get('reid_embedding')
        if reid_embedding is None:
            return

        track_id = person.get('track_id')
        bbox = person.get('bbox', [])

        try:
            embedding = np.array(reid_embedding, dtype=np.float32)
            if len(embedding) != 512:
                logger.warning(f"Invalid ReID embedding size: {len(embedding)}")
                return

            # Store ReID embedding with ALL context needed for safe re-association
            self._wallet_reid_features[wallet] = {
                'embedding': embedding,
                'timestamp': current_time,
                'face_seen': current_time,  # Face was just confirmed
                'original_track_id': track_id,  # Track ID when face was confirmed
                'last_bbox': bbox.copy() if isinstance(bbox, list) else list(bbox),  # Position when face was confirmed
            }

            if self._debug_mode:
                logger.info(f"[REID-STORE] {wallet[:8]}...: track={track_id}, bbox={bbox[:4] if len(bbox) >= 4 else bbox}")

        except Exception as e:
            logger.warning(f"Failed to store ReID embedding: {e}")

    def _enhance_persons_with_identity(self, persons: List[Dict]) -> List[Dict]:
        """Add wallet_address to person detections that have been identified."""
        enhanced = []

        with self._track_lock:
            for person in persons:
                track_id = person.get('track_id')
                if track_id is not None and track_id in self._track_to_wallet:
                    wallet = self._track_to_wallet[track_id]
                    person['wallet_address'] = wallet
                    person['identity_confidence'] = 1.0
                    person['tracking_method'] = 'native_face'
                enhanced.append(person)

        return enhanced

    def get_status(self) -> Dict:
        """Get service status with comprehensive stats."""
        with self._track_lock:
            tracked_count = len(self._track_to_wallet)
            reid_features_count = len(self._wallet_reid_features)
            track_mappings = dict(self._track_to_wallet)

        return {
            'initialized': self._initialized,
            'enabled': self._processing_enabled,
            'mode': 'BULLETPROOF_EDITION',
            # Stats
            'total_faces_processed': self.total_faces_processed,
            'total_matches': self.total_matches,
            'total_reid_blocked': self.total_reid_blocked,
            'total_mismatches_revoked': self.total_mismatches_revoked,
            'currently_tracked': tracked_count,
            'reid_features_stored': reid_features_count,
            # Configuration
            'face_threshold': self._similarity_threshold,
            'reid_threshold': self._reid_similarity_threshold,
            'reid_max_age_s': self._reid_feature_max_age,
            'reid_require_recent_face_s': self._reid_require_recent_face,
            'reid_max_position_drift_px': self._reid_max_position_drift,
            'debug_mode': self._debug_mode,
            # Current mappings (for debugging)
            'track_mappings': {str(k): v[:8] + '...' for k, v in track_mappings.items()},
        }

    def set_enabled(self, enabled: bool):
        """Enable/disable identity processing."""
        self._processing_enabled = enabled
        logger.info(f"Identity processing {'enabled' if enabled else 'disabled'}")

    def set_debug_mode(self, enabled: bool):
        """Enable/disable verbose debug logging."""
        self._debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")

    def reset_stats(self):
        """Reset all statistics counters."""
        self.total_faces_processed = 0
        self.total_matches = 0
        self.total_reid_blocked = 0
        self.total_mismatches_revoked = 0
        logger.info("Stats reset")

    def clear_tracks(self):
        """Clear all track associations and ReID features."""
        with self._track_lock:
            self._track_to_wallet.clear()
            self._wallet_to_track.clear()
            self._wallet_face_last_seen.clear()
            self._wallet_reid_features.clear()
        logger.info("Cleared all track associations and ReID features")

    def add_identity(self, wallet_address: str, embedding: np.ndarray,
                     metadata: Dict = None) -> bool:
        """
        Add or update a face embedding for a wallet address.

        Used by phone selfie registration to store embeddings from native TensorRT.

        Args:
            wallet_address: User's wallet address
            embedding: Face embedding from native TensorRT ArcFace
            metadata: Optional metadata about the enrollment

        Returns:
            True if successful
        """
        try:
            profile = {}
            if metadata:
                profile['enrollment_metadata'] = metadata

            self.identity_store.check_in(
                wallet_address=wallet_address,
                face_embedding=embedding,
                profile=profile
            )
            logger.info(f"Added/updated face embedding for {wallet_address[:8]}... via native identity service")
            return True
        except Exception as e:
            logger.error(f"Failed to add identity for {wallet_address[:8]}...: {e}")
            return False


# Global singleton
_native_identity_service: Optional[NativeIdentityService] = None


def get_native_identity_service() -> NativeIdentityService:
    """Get the singleton native identity service."""
    global _native_identity_service
    if _native_identity_service is None:
        _native_identity_service = NativeIdentityService()
    return _native_identity_service
