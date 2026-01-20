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

# Face alignment for embedding compatibility with phone selfies
try:
    from .face_alignment_service import align_face, is_available as alignment_available
    ALIGNMENT_AVAILABLE = alignment_available()
except ImportError:
    ALIGNMENT_AVAILABLE = False
    align_face = None

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
        # User's matches: ~0.84, girlfriend false positive: 0.745
        # Current issue: stored embeddings giving ~0.5 similarity (should be 0.8+)
        # Need to fix root cause - likely quantization or preprocessing mismatch
        self._similarity_threshold = 0.65

        # Stats
        self.total_matches = 0
        self.total_faces_processed = 0
        self._frame_counter = 0  # For rate-limited logging

        # Track → identity mapping (track_id from YOLO → wallet_address)
        # MUST be RLock (reentrant) because _maintain_reid_for_recognized_tracks
        # holds this lock and calls functions that also try to acquire it
        self._track_lock = threading.RLock()
        self._track_to_wallet: Dict[int, str] = {}
        self._wallet_to_track: Dict[str, int] = {}

        # Face confirmation tracking - FACE IS THE ONLY SOURCE OF TRUTH
        # wallet → timestamp of last face confirmation
        self._wallet_face_last_seen: Dict[str, float] = {}

        # Session-level ReID embeddings (OSNet 512-dim from native server)
        # wallet → {'embedding': np.array, 'timestamp': float, ...}
        self._wallet_reid_features: Dict[str, Dict] = {}

        # ReID settings for track recovery after brief occlusions
        self._reid_similarity_threshold = 0.80  # High threshold for appearance matching
        self._reid_recovery_max_time = 5.0      # Max seconds since track lost for recovery
        self._reid_recovery_min_iou = 0.3       # Spatial overlap required (same region)

        # Part 3: Unified ReID settings
        self._unified_reid_threshold = 0.65     # Threshold for appearance cluster matching
        self._unified_recovery_max_time = 30.0  # Extended window for confirmed users
        self._body_maintenance_threshold = 0.60 # Body-only maintenance for face_confirmed users
        self._face_confirmed_count = 3          # Face matches needed for 'face_confirmed' status
        self._last_appearance_update: Dict[str, float] = {}  # Rate-limit appearance history updates

        # Part 3.5: Body-assisted recognition when face is visible but weak
        # For face_confirmed users: if face is detected but similarity is weak (not strong enough
        # to match, but not so low it's clearly a different person), fall back to body matching
        self._body_assisted_min_face_sim = 0.30  # Face must not be clearly different person
        self._body_assisted_body_threshold = 0.70  # Body must match strongly to accept

        # Pending recovery - stores recently lost tracks for ReID re-acquisition
        # wallet → {last_track_id, last_bbox, reid_embedding, lost_time}
        self._pending_recovery: Dict[str, Dict] = {}

        # Track continuity protection - trust body tracking when face is unreliable at distance
        # track_id -> timestamp when track was FIRST assigned via face (not recovered)
        self._track_original_assignment_time: Dict[int, float] = {}
        # track_id -> True if track was recovered (less trustworthy than original assignment)
        self._track_was_recovered: Dict[int, bool] = {}

        # Similarity tracking for display
        # track_id -> latest similarity score (for showing on annotated stream)
        self._track_face_similarity: Dict[int, float] = {}  # Face embedding similarity
        self._track_reid_similarity: Dict[int, float] = {}  # ReID embedding similarity
        self._track_identity_source: Dict[int, str] = {}    # 'face' or 'reid' - how identity was last confirmed

        # Debug mode - set to True for verbose logging during troubleshooting
        self._debug_mode = False  # Production: False, enable via API for debugging

        logger.info("NativeIdentityService initialized - BULLETPROOF EDITION + Part 3/3.5 Unified ReID")
        logger.info(f"  Face threshold: {self._similarity_threshold}")
        logger.info(f"  ReID threshold: {self._reid_similarity_threshold}")
        logger.info(f"  ReID recovery window: {self._reid_recovery_max_time}s")
        logger.info(f"  ReID recovery min IoU: {self._reid_recovery_min_iou}")
        logger.info(f"  Part 3 - Unified ReID threshold: {self._unified_reid_threshold}")
        logger.info(f"  Part 3 - Unified recovery window: {self._unified_recovery_max_time}s")
        logger.info(f"  Part 3 - Body maintenance threshold: {self._body_maintenance_threshold}")
        logger.info(f"  Part 3 - Face confirmations for 'confirmed': {self._face_confirmed_count}")
        logger.info(f"  Part 3.5 - Body-assisted min face sim: {self._body_assisted_min_face_sim}")
        logger.info(f"  Part 3.5 - Body-assisted body threshold: {self._body_assisted_body_threshold}")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union between two bounding boxes [x1,y1,x2,y2]."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def process_native_results(self, persons: List[Dict], faces: List[Dict],
                                frame: np.ndarray = None) -> List[Dict]:
        """
        Process detection results from native server and perform identity matching.

        IDENTITY LOGIC:
        1. Face embedding match - THE ONLY way to assign identity
        2. Track continuity - existing track-wallet mappings persist
        3. ReID - ONLY for same-track re-acquisition during brief occlusions
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
        start_time = current_time  # For timing diagnostics

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

            # Use the C++ TensorRT embedding directly
            # Python alignment experiment didn't improve similarity - environmental factors
            # (lighting, distance, angle) have more impact than alignment algorithm
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

            # Store face similarity for debug display (even if no match)
            # Compute best similarity for display purposes
            display_similarity = best_similarity if best_match else 0.0
            if not best_match and checked_in:
                # No match - compute best similarity for display
                for wallet_address in checked_in:
                    identity = self.identity_store.get_identity(wallet_address)
                    if identity is not None and identity.face_embedding is not None:
                        sim = self.cosine_similarity(embedding, identity.face_embedding)
                        display_similarity = max(display_similarity, sim)

            # Store similarity for the track this face belongs to
            if face_track_id is not None:
                self._track_face_similarity[face_track_id] = display_similarity

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

                    # Part 3: Record face confirmation and update appearance history
                    # RATE-LIMITED: Only update once per second to prevent lock contention
                    last_history_update = self._last_appearance_update.get(best_match, 0)
                    if current_time - last_history_update >= 1.0:
                        confirmation_level = self.identity_store.record_face_confirmation(best_match)
                        reid_embedding = matched_person.get('reid_embedding')
                        if reid_embedding is not None:
                            self.identity_store.update_appearance_history(best_match, reid_embedding)
                            self._last_appearance_update[best_match] = current_time
                            if self._debug_mode:
                                identity = self.identity_store.get_identity(best_match)
                                if identity:
                                    logger.info(f"📊 [CONFIRM] {best_match[:8]}... level={confirmation_level}, "
                                               f"face_count={identity.face_confirmations}, "
                                               f"history_size={len(identity.appearance_history)}")

            else:
                if self._debug_mode:
                    # Log actual computed similarities for debugging weak recognition
                    live_norm = np.linalg.norm(embedding)
                    for wallet_address in checked_in:
                        identity = self.identity_store.get_identity(wallet_address)
                        if identity is not None and identity.face_embedding is not None:
                            stored_norm = np.linalg.norm(identity.face_embedding)
                            actual_sim = self.cosine_similarity(embedding, identity.face_embedding)
                            # Debug: show first 5 embedding values to diagnose low similarity
                            live_preview = embedding.flatten()[:5]
                            stored_preview = identity.face_embedding.flatten()[:5]
                            logger.info(f"❌ [FACE-MATCH] {wallet_address[:8]}... actual_sim={actual_sim:.3f} (need>{self._similarity_threshold}) | "
                                       f"norms: live={live_norm:.3f} stored={stored_norm:.3f}")
                            logger.info(f"   🔍 [EMB-DEBUG] live[0:5]={[f'{v:.4f}' for v in live_preview]} stored[0:5]={[f'{v:.4f}' for v in stored_preview]}")

        # ==========================================================================
        # STEP 2: Clean up stale track associations
        # If a track is no longer visible, store recovery data and remove mapping
        # ==========================================================================
        self._cleanup_stale_tracks(persons, current_time)

        # ==========================================================================
        # STEP 3: ReID maintenance, recovery, and body-assisted recognition
        # - For recognized tracks: update ReID features, show confidence when face not visible
        # - For unrecognized tracks: attempt recovery if recently lost (same person, brief occlusion)
        # - For unrecognized tracks with weak face: try body-assisted recognition
        # ==========================================================================
        tracks_with_visible_face = self._get_tracks_with_visible_face(persons, valid_faces)
        self._maintain_reid_for_recognized_tracks(persons, current_time, tracks_with_visible_face, valid_faces)

        # Check for slow processing - warn if taking more than 50ms
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 50:
            logger.warning(f"⚠️ [IDENTITY] Slow processing: {elapsed_ms:.1f}ms (faces={len(valid_faces)}, checked_in={len(checked_in)})")

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
                        self._track_identity_source[track_id] = 'face'

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
                    self._track_identity_source[track_id] = 'face'

                    # Track when this assignment was made (for track continuity protection)
                    # Only set if this is a NEW assignment (not updating existing)
                    if track_id not in self._track_original_assignment_time:
                        self._track_original_assignment_time[track_id] = current_time
                        self._track_was_recovered[track_id] = False

                self.identity_store.assign_track(wallet_address, track_id, confidence, 'face')
                logger.info(f"[ASSOCIATE] Track {track_id} → {wallet_address[:8]}... (face in person bbox)")

        return best_person


    def _cleanup_stale_tracks(self, persons: List[Dict], current_time: float):
        """
        Remove track associations for tracks that are no longer visible.
        Store recovery data for ReID re-acquisition during brief occlusions.
        """
        current_track_ids = set()
        for person in persons:
            track_id = person.get('track_id')
            if track_id is not None:
                current_track_ids.add(track_id)

        with self._track_lock:
            # DIAGNOSTIC: Log every frame when we have wallet mappings
            if self._track_to_wallet:
                wallet_tracks = list(self._track_to_wallet.keys())
                missing = [t for t in wallet_tracks if t not in current_track_ids]
                if missing:
                    logger.warning(f"⚠️ [DIAG] Wallet tracks {wallet_tracks} but persons only has {list(current_track_ids)} - MISSING: {missing}")

            # Find tracks that are no longer visible
            stale_tracks = []
            for track_id in list(self._track_to_wallet.keys()):
                if track_id not in current_track_ids:
                    stale_tracks.append(track_id)

            # Remove stale track associations but store recovery data
            for track_id in stale_tracks:
                wallet = self._track_to_wallet.pop(track_id, None)
                if wallet:
                    self._wallet_to_track.pop(wallet, None)

                    # Store recovery data for ReID re-acquisition
                    reid_features = self._wallet_reid_features.get(wallet)
                    if reid_features:
                        self._pending_recovery[wallet] = {
                            'last_track_id': track_id,
                            'last_bbox': reid_features.get('last_bbox', []),
                            'reid_embedding': reid_features.get('embedding'),
                            'lost_time': current_time,
                        }
                        # ALWAYS log recovery storage (diagnostic)
                        logger.warning(f"🔄 [TRACK-LOST] Track {track_id} disappeared from persons! Stored for recovery ({wallet[:8]}...)")
                    else:
                        # ALWAYS log cleanup (diagnostic)
                        logger.warning(f"🗑️ [TRACK-LOST] Track {track_id} disappeared, NO ReID data for recovery ({wallet[:8]}...)")

                # Clear per-track state
                self._track_face_similarity.pop(track_id, None)
                self._track_reid_similarity.pop(track_id, None)
                self._track_identity_source.pop(track_id, None)
                self._track_original_assignment_time.pop(track_id, None)
                self._track_was_recovered.pop(track_id, None)

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

    def _maintain_reid_for_recognized_tracks(
        self,
        persons: List[Dict],
        current_time: float,
        tracks_with_visible_face: Set[int],
        valid_faces: List[Dict] = None
    ):
        """
        Maintain ReID display confidence for already-recognized tracks.

        Also handles recovery and body-assisted recognition for unidentified tracks:
        1. Updates ReID features for continuous tracking (prevents expiry)
        2. Switches to ReID confidence display when face is not visible
        3. Attempts recovery/reacquisition when face is not visible
        4. Attempts body-assisted recognition when face IS visible but weak
        """
        # Build mapping of track_id -> face for body-assisted recognition
        track_to_face = {}
        if valid_faces:
            for face in valid_faces:
                face_bbox = face.get('bbox', [])
                if len(face_bbox) < 4:
                    continue
                # Find which person this face belongs to
                for person in persons:
                    person_bbox = person.get('bbox', [])
                    if len(person_bbox) < 4:
                        continue
                    track_id = person.get('track_id')
                    if track_id is None:
                        continue
                    # Check if face is inside person bbox
                    face_cx = (face_bbox[0] + face_bbox[2]) / 2
                    face_cy = (face_bbox[1] + face_bbox[3]) / 2
                    if (person_bbox[0] <= face_cx <= person_bbox[2] and
                        person_bbox[1] <= face_cy <= person_bbox[3]):
                        track_to_face[track_id] = face
                        break

        with self._track_lock:
            for person in persons:
                track_id = person.get('track_id')
                if track_id is None:
                    continue

                if track_id in self._track_to_wallet:
                    # Already has identity - update ReID features and display
                    wallet = self._track_to_wallet[track_id]
                    self._update_reid_features_during_tracking(wallet, person, current_time)

                    if track_id not in tracks_with_visible_face:
                        self._switch_to_reid_mode_for_display(track_id, wallet, person)
                else:
                    # No identity - try recovery methods
                    if track_id not in tracks_with_visible_face:
                        # Face NOT visible - try standard recovery and unified reacquisition
                        self._attempt_track_recovery(person, current_time)

                        if track_id not in self._track_to_wallet:
                            self._attempt_unified_reacquisition(person, current_time)
                    else:
                        # Face IS visible but didn't match during face matching
                        # For face_confirmed users, try body-assisted recognition
                        # This handles the case where face quality is too poor at distance
                        face = track_to_face.get(track_id)
                        if face is not None:
                            self._attempt_body_assisted_recognition(person, face, current_time)

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

            # Get existing features or create new ones
            existing = self._wallet_reid_features.get(wallet)

            # Preserve original face_seen time if exists, otherwise use current time
            face_seen_time = existing.get('face_seen', current_time) if existing else current_time

            # Update/create with current data
            self._wallet_reid_features[wallet] = {
                'embedding': embedding,
                'timestamp': current_time,  # Keep features fresh
                'face_seen': face_seen_time,  # Preserve original face time
                'original_track_id': person.get('track_id'),
                'last_bbox': bbox.copy() if isinstance(bbox, list) else list(bbox),
            }

            if existing is None and self._debug_mode:
                logger.info(f"[REID-CREATE] {wallet[:8]}...: created ReID features (was missing)")

            # Part 3: Rate-limited appearance history update (1x/second max)
            # Prevents lock contention that was causing WebRTC freezes
            last_history_update = self._last_appearance_update.get(wallet, 0)
            if current_time - last_history_update >= 1.0:
                self.identity_store.update_appearance_history(wallet, embedding)
                self.identity_store.update_last_bbox(wallet, tuple(bbox[:4]) if len(bbox) >= 4 else None)
                self._last_appearance_update[wallet] = current_time

        except Exception:
            pass

    def _switch_to_reid_mode_for_display(self, track_id: int, wallet: str, person: Dict):
        """
        Switch display to ReID mode when face is not visible.

        Computes similarity between current ReID embedding and stored ReID features
        to show accurate confidence when person is turned away from camera.
        """
        # Check if we have stored ReID features for this wallet
        stored_features = self._wallet_reid_features.get(wallet)
        if stored_features is None:
            return

        # Get current ReID embedding from person
        has_reid = person.get('has_reid_embedding', 0)
        if not has_reid:
            return

        current_reid = person.get('reid_embedding')
        if current_reid is None:
            return

        try:
            current_embedding = np.array(current_reid, dtype=np.float32)
            stored_embedding = stored_features['embedding']

            if len(current_embedding) != 512 or len(stored_embedding) != 512:
                return

            # Compute ReID similarity
            similarity = self.cosine_similarity(current_embedding, stored_embedding)

            # Switch to ReID mode for display
            self._track_identity_source[track_id] = 'reid'
            self._track_reid_similarity[track_id] = similarity

        except Exception:
            pass

    def _attempt_track_recovery(self, person: Dict, current_time: float):
        """
        Attempt to recover identity for a new track using ReID.

        This ONLY works for recently-lost tracks (within recovery window) that
        appear in the same spatial region. It restores identity to the SAME
        person after brief occlusion - it does NOT assign to new people.

        Recovery requires:
        1. Track was lost within _reid_recovery_max_time seconds
        2. New track appears in same region (IoU > _reid_recovery_min_iou)
        3. ReID embedding similarity > _reid_similarity_threshold
        """
        track_id = person.get('track_id')
        if track_id is None:
            return

        person_bbox = person.get('bbox', [])
        if len(person_bbox) < 4:
            return

        person_reid = person.get('reid_embedding')
        if person_reid is None:
            return

        try:
            person_embedding = np.array(person_reid, dtype=np.float32)
            if len(person_embedding) != 512:
                return
        except Exception:
            return

        # Try to match against pending recovery data (people who WERE recognized)
        best_wallet = None
        best_similarity = self._reid_similarity_threshold
        best_iou = 0.0

        expired_wallets = []

        for wallet, recovery_data in self._pending_recovery.items():
            # CHECK 1: Time window
            time_since_lost = current_time - recovery_data['lost_time']
            if time_since_lost > self._reid_recovery_max_time:
                expired_wallets.append(wallet)
                continue

            # CHECK 2: Wallet doesn't already have an active track
            if wallet in self._wallet_to_track:
                continue

            # CHECK 3: Spatial proximity (same region)
            last_bbox = recovery_data.get('last_bbox', [])
            iou = self.compute_iou(person_bbox, last_bbox)
            if iou < self._reid_recovery_min_iou:
                continue

            # CHECK 4: ReID similarity
            stored_embedding = recovery_data.get('reid_embedding')
            if stored_embedding is None:
                continue

            similarity = self.cosine_similarity(person_embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_wallet = wallet
                best_iou = iou

        # Clean up expired recovery data
        for wallet in expired_wallets:
            del self._pending_recovery[wallet]

        if best_wallet:
            # Recovery successful - restore identity
            # NOTE: We don't acquire _track_lock here because this function is called from
            # _maintain_reid_for_recognized_tracks() which already holds the lock.
            self._track_to_wallet[track_id] = best_wallet
            self._wallet_to_track[best_wallet] = track_id
            self._track_identity_source[track_id] = 'reid_recovery'
            self._track_reid_similarity[track_id] = best_similarity

            # Mark this track as RECOVERED (less trustworthy than original face assignment)
            # This means we should be more careful about trusting body-only tracking
            self._track_original_assignment_time[track_id] = current_time
            self._track_was_recovered[track_id] = True

            del self._pending_recovery[best_wallet]

            logger.info(
                f"✅ [RECOVERY] Track {track_id} ← {best_wallet[:8]}... "
                f"(similarity: {best_similarity:.3f}, IoU: {best_iou:.2f})"
            )

    def _attempt_unified_reacquisition(self, person: Dict, current_time: float) -> bool:
        """
        Part 3: Unified appearance-based re-acquisition.

        This runs for unidentified persons and attempts to match them against
        ALL checked-in users who don't currently have an active track.

        Unlike _attempt_track_recovery (which requires IoU overlap), this
        uses appearance cluster matching with NO spatial requirement.
        This allows re-acquisition even if the person entered from a different
        location than where they were last seen.

        Requirements:
        - User must be face_once or face_confirmed
        - Appearance cluster must exist (at least 3 body embeddings captured)
        - Extended time window: 30s for face_confirmed, 10s for face_once

        Returns:
            True if re-acquisition succeeded
        """
        track_id = person.get('track_id')
        if track_id is None:
            return False

        # Already has identity?
        # NOTE: We don't acquire _track_lock here because this function is called from
        # _maintain_reid_for_recognized_tracks() which already holds the lock.
        # Using _track_lock here would cause a deadlock (it's a Lock, not RLock).
        if track_id in self._track_to_wallet:
            return False

        # Get body embedding
        person_reid = person.get('reid_embedding')
        if person_reid is None:
            return False

        try:
            person_embedding = np.array(person_reid, dtype=np.float32)
            if len(person_embedding) != 512:
                return False
        except Exception:
            return False

        # Use the IdentityStore's cluster matching
        result = self.identity_store.find_by_appearance_cluster(
            embedding=person_embedding,
            threshold=self._unified_reid_threshold,  # 0.65
            require_confirmation=True,
            exclude_active=True
        )

        if result is None:
            return False

        wallet, similarity = result
        identity = self.identity_store.get_identity(wallet)

        if identity is None:
            return False

        # Check time-based eligibility
        # face_confirmed: 30 second window
        # face_once: 10 second window
        time_since_last_seen = current_time - identity.last_seen

        if identity.confirmation_level == 'face_confirmed':
            max_time = self._unified_recovery_max_time  # 30s
        elif identity.confirmation_level == 'face_once':
            max_time = 10.0
        else:
            # Should not happen due to require_confirmation, but be safe
            return False

        if time_since_last_seen > max_time:
            if self._debug_mode:
                logger.info(f"[UNIFIED-SKIP] {wallet[:8]}... expired: {time_since_last_seen:.1f}s > {max_time}s")
            return False

        # Re-acquisition successful
        # NOTE: We don't acquire _track_lock here because this function is called from
        # _maintain_reid_for_recognized_tracks() which already holds the lock.
        
        # Clear any old track association for this wallet
        old_track = self._wallet_to_track.get(wallet)
        if old_track is not None:
            self._track_to_wallet.pop(old_track, None)

        self._track_to_wallet[track_id] = wallet
        self._wallet_to_track[wallet] = track_id
        self._track_identity_source[track_id] = 'unified_reid'
        self._track_reid_similarity[track_id] = similarity

        # Mark as recovered (less trustworthy than original face assignment)
        self._track_original_assignment_time[track_id] = current_time
        self._track_was_recovered[track_id] = True

        # Update identity store
        self.identity_store.assign_track(wallet, track_id, similarity, 'appearance')

        logger.info(
            f"✅ [UNIFIED-REACQ] Track {track_id} ← {wallet[:8]}... "
            f"(similarity: {similarity:.3f}, level: {identity.confirmation_level}, "
            f"last_seen: {time_since_last_seen:.1f}s ago)"
        )

        return True

    def _attempt_body_assisted_recognition(
        self,
        person: Dict,
        face: Dict,
        current_time: float
    ) -> bool:
        """
        Part 3.5: Body-assisted recognition when face is visible but weak.

        For face_confirmed users: if face is detected but similarity is in the "weak" range
        (not strong enough to match, but not so low it's clearly a different person),
        fall back to body matching.

        Requirements:
        - User must be face_confirmed with appearance cluster
        - Face similarity must be >= _body_assisted_min_face_sim (0.30) - not clearly different
        - Face similarity must be < _similarity_threshold (0.65) - didn't match normally
        - Body similarity must be >= _body_assisted_body_threshold (0.70) - strong body match

        Returns:
            True if body-assisted recognition succeeded
        """
        track_id = person.get('track_id')
        if track_id is None:
            return False

        if self._debug_mode:
            logger.info(f"[BODY-ASSIST] Called for track {track_id}")

        # Already has identity?
        if track_id in self._track_to_wallet:
            if self._debug_mode:
                logger.info(f"[BODY-ASSIST] Track {track_id} already has identity, skip")
            return False

        # Get face embedding
        face_embedding = face.get('embedding')
        if face_embedding is None:
            if self._debug_mode:
                logger.info(f"[BODY-ASSIST] Track {track_id} no face embedding, skip")
            return False

        # Get body embedding
        body_embedding = person.get('reid_embedding')
        if body_embedding is None:
            if self._debug_mode:
                logger.info(f"[BODY-ASSIST] Track {track_id} no body embedding, skip")
            return False

        try:
            face_emb = np.array(face_embedding, dtype=np.float32)
            body_emb = np.array(body_embedding, dtype=np.float32)

            if len(face_emb) != 512 or len(body_emb) != 512:
                return False

            # Normalize
            face_norm = np.linalg.norm(face_emb)
            body_norm = np.linalg.norm(body_emb)
            if face_norm > 0:
                face_emb = face_emb / face_norm
            if body_norm > 0:
                body_emb = body_emb / body_norm

        except Exception:
            return False

        # Check each face_confirmed user
        best_wallet = None
        best_body_sim = 0.0  # Track best among valid matches (threshold checked per-candidate)
        best_face_sim = 0.0

        checked_in = self.identity_store.get_checked_in_wallets()
        if self._debug_mode:
            logger.info(f"[BODY-ASSIST] Checking {len(checked_in)} checked-in wallets")

        for wallet in checked_in:
            identity = self.identity_store.get_identity(wallet)
            if identity is None:
                continue

            # Must be at least face_once with appearance cluster
            # face_confirmed: standard thresholds
            # face_once: require higher body similarity (adds 0.10 to threshold)
            if identity.confirmation_level not in ('face_confirmed', 'face_once'):
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST] {wallet[:8]}... not face_confirmed/face_once (level={identity.confirmation_level})")
                continue

            # Determine extra threshold for face_once users
            body_threshold_boost = 0.10 if identity.confirmation_level == 'face_once' else 0.0

            if identity.last_appearance_cluster is None:
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST] {wallet[:8]}... no appearance cluster")
                continue
            if identity.face_embedding is None:
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST] {wallet[:8]}... no face embedding")
                continue

            # Skip if already has an ACTUALLY ACTIVE track (track still exists in our tracking)
            if identity.active_track_id is not None:
                # Check if that track is still valid (exists in our tracking)
                if identity.active_track_id in self._track_to_wallet:
                    if self._debug_mode:
                        logger.info(f"[BODY-ASSIST] {wallet[:8]}... has active track {identity.active_track_id}")
                    continue
                # Old track is stale - this user is eligible for body-assisted recognition
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST] {wallet[:8]}... old track {identity.active_track_id} is stale, proceeding")

            # Check face similarity - determine required body threshold
            stored_face = identity.face_embedding
            face_sim = float(np.dot(face_emb, stored_face / np.linalg.norm(stored_face)))

            if face_sim >= self._similarity_threshold:
                # Face should have matched normally - something is wrong
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST] {wallet[:8]}... face_sim={face_sim:.3f} >= threshold (should have matched normally)")
                continue

            # Determine body threshold based on face similarity
            # - face_sim >= 0.30: normal body threshold (0.70) - weak face, trust body
            # - face_sim 0.15-0.30: very high body threshold (0.85) - face might be angle change
            # - face_sim 0.05-0.15: extreme angle (90° profile) - require exceptional body (0.90)
            # - face_sim < 0.05: truly different person, skip entirely
            if face_sim < 0.05:
                # Face is truly a different person - even exceptional body can't override
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST-SKIP] {wallet[:8]}... face_sim={face_sim:.3f} < 0.05 (truly different person)")
                continue

            if face_sim < 0.15:
                # Very low face similarity - extreme angle (e.g., 90° side profile)
                # Require exceptional body similarity to match (0.90)
                required_body_threshold = 0.90 + body_threshold_boost
            elif face_sim < self._body_assisted_min_face_sim:  # 0.30
                # Low face similarity - might be angle change
                # Require very high body similarity to match (0.85)
                required_body_threshold = 0.85 + body_threshold_boost
            else:
                # Normal weak face range (0.30-0.65) - use standard body threshold
                required_body_threshold = self._body_assisted_body_threshold + body_threshold_boost  # 0.70 + boost

            # Cap threshold at 0.88 - even exceptional body similarity rarely exceeds 0.90
            required_body_threshold = min(required_body_threshold, 0.88)

            # Check body similarity against appearance cluster
            body_sim = float(np.dot(body_emb, identity.last_appearance_cluster))

            if self._debug_mode:
                level_str = f"[{identity.confirmation_level}]" if identity.confirmation_level == 'face_once' else ""
                logger.info(f"[BODY-ASSIST] {wallet[:8]}...{level_str} face_sim={face_sim:.3f}, body_sim={body_sim:.3f} (need>{required_body_threshold:.2f})")

            # Check if this candidate meets its required body threshold
            if body_sim < required_body_threshold:
                if self._debug_mode:
                    logger.info(f"[BODY-ASSIST-SKIP] {wallet[:8]}... body_sim={body_sim:.3f} < {required_body_threshold:.2f}")
                continue

            # This candidate meets threshold - track if best so far
            if body_sim > best_body_sim:
                best_body_sim = body_sim
                best_wallet = wallet
                best_face_sim = face_sim

        if best_wallet is None:
            if self._debug_mode:
                logger.info(f"[BODY-ASSIST] No match found for track {track_id}")
            return False

        # Body-assisted recognition successful!
        identity = self.identity_store.get_identity(best_wallet)

        self._track_to_wallet[track_id] = best_wallet
        self._wallet_to_track[best_wallet] = track_id
        self._track_identity_source[track_id] = 'body_assisted'
        self._track_reid_similarity[track_id] = best_body_sim
        self._track_face_similarity[track_id] = best_face_sim

        # Mark as recovered (since it wasn't a direct face match)
        self._track_original_assignment_time[track_id] = current_time
        self._track_was_recovered[track_id] = True

        # Update identity store
        self.identity_store.assign_track(best_wallet, track_id, best_body_sim, 'body_assisted')

        logger.info(
            f"✅ [BODY-ASSIST] Track {track_id} ← {best_wallet[:8]}... "
            f"(face_sim={best_face_sim:.3f} [weak], body_sim={best_body_sim:.3f} [strong])"
        )

        return True

    def _store_reid_embedding(self, wallet: str, person: Dict, current_time: float):
        """
        Store ReID embedding from native server when face is confirmed.
        Used for display confidence and track recovery after brief occlusions.
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
        """Add wallet_address and display_name to person detections that have been identified.

        IMPORTANT: Caches display_name here to avoid repeated lock acquisitions
        in _draw_native_annotations which runs on every WebRTC frame.
        """
        enhanced = []

        with self._track_lock:
            for person in persons:
                track_id = person.get('track_id')
                if track_id is not None:
                    # Add similarity scores for display
                    if track_id in self._track_face_similarity:
                        person['face_similarity'] = self._track_face_similarity[track_id]
                    if track_id in self._track_reid_similarity:
                        person['reid_similarity'] = self._track_reid_similarity[track_id]

                    # Add identity source (face or reid)
                    if track_id in self._track_identity_source:
                        person['identity_source'] = self._track_identity_source[track_id]

                    if track_id in self._track_to_wallet:
                        wallet = self._track_to_wallet[track_id]
                        person['wallet_address'] = wallet

                        # Cache display_name to avoid lock contention in drawing code
                        # This is critical - drawing runs at 30fps and would otherwise
                        # acquire identity_store lock on every frame for every person
                        identity = self.identity_store.get_identity(wallet)
                        if identity:
                            person['display_name'] = identity.get_display_name()
                        else:
                            person['display_name'] = wallet[:8]

                        # Compute UNIFIED confidence score combining face and body
                        # This gives a holistic view of recognition confidence
                        face_sim = self._track_face_similarity.get(track_id)
                        reid_sim = self._track_reid_similarity.get(track_id)

                        if face_sim is not None and reid_sim is not None:
                            # Both available - weighted average (face weighted higher)
                            unified_confidence = 0.6 * face_sim + 0.4 * reid_sim
                            confidence_source = None  # Combined, no indicator needed
                        elif face_sim is not None:
                            # Face only
                            unified_confidence = face_sim
                            confidence_source = "F"
                        elif reid_sim is not None:
                            # Body only
                            unified_confidence = reid_sim
                            confidence_source = "B"
                        else:
                            # Fallback
                            unified_confidence = 0.7
                            confidence_source = None

                        person['unified_confidence'] = unified_confidence
                        person['confidence_source'] = confidence_source
                        person['identity_confidence'] = unified_confidence  # Keep for compatibility

                        source = self._track_identity_source.get(track_id, 'face')
                        person['tracking_method'] = f'native_{source}'
                enhanced.append(person)

        return enhanced

    def get_status(self) -> Dict:
        """Get service status with comprehensive stats."""
        with self._track_lock:
            tracked_count = len(self._track_to_wallet)
            reid_features_count = len(self._wallet_reid_features)
            pending_recovery_count = len(self._pending_recovery)
            track_mappings = dict(self._track_to_wallet)

        # Part 3: Get confirmation level stats
        identities = self.identity_store.get_all_identities()
        confirmation_stats = {'none': 0, 'face_once': 0, 'face_confirmed': 0, 'body_maintained': 0}
        appearance_clusters = 0
        for identity in identities:
            level = identity.confirmation_level
            confirmation_stats[level] = confirmation_stats.get(level, 0) + 1
            if identity.last_appearance_cluster is not None:
                appearance_clusters += 1

        return {
            'initialized': self._initialized,
            'enabled': self._processing_enabled,
            'mode': 'BULLETPROOF_EDITION_v2_UNIFIED_REID',
            # Stats
            'total_faces_processed': self.total_faces_processed,
            'total_matches': self.total_matches,
            'currently_tracked': tracked_count,
            'reid_features_stored': reid_features_count,
            'pending_recovery': pending_recovery_count,
            # Configuration
            'face_threshold': self._similarity_threshold,
            'reid_threshold': self._reid_similarity_threshold,
            'reid_recovery_window_s': self._reid_recovery_max_time,
            'reid_recovery_min_iou': self._reid_recovery_min_iou,
            'debug_mode': self._debug_mode,
            # Part 3: Unified ReID config
            'unified_reid_threshold': self._unified_reid_threshold,
            'unified_recovery_window_s': self._unified_recovery_max_time,
            'body_maintenance_threshold': self._body_maintenance_threshold,
            # Part 3: Stats
            'confirmation_levels': confirmation_stats,
            'appearance_clusters_stored': appearance_clusters,
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
        logger.info("Stats reset")

    def clear_tracks(self):
        """Clear all track associations, ReID features, and pending recovery."""
        with self._track_lock:
            self._track_to_wallet.clear()
            self._wallet_to_track.clear()
            self._wallet_face_last_seen.clear()
            self._wallet_reid_features.clear()
            self._pending_recovery.clear()
            self._track_face_similarity.clear()
            self._track_reid_similarity.clear()
            self._track_identity_source.clear()
            self._track_original_assignment_time.clear()
            self._track_was_recovered.clear()
        logger.info("Cleared all track associations and ReID features")

    def clear_wallet_tracks(self, wallet_address: str):
        """
        Clear all track associations and ReID features for a specific wallet.
        Called on checkout to ensure the user is no longer recognized.

        Args:
            wallet_address: User's wallet address to clear
        """
        with self._track_lock:
            # Find and remove track_id -> wallet mapping
            track_id = self._wallet_to_track.pop(wallet_address, None)
            if track_id is not None:
                self._track_to_wallet.pop(track_id, None)
                self._track_face_similarity.pop(track_id, None)
                self._track_reid_similarity.pop(track_id, None)
                self._track_identity_source.pop(track_id, None)
                self._track_original_assignment_time.pop(track_id, None)
                self._track_was_recovered.pop(track_id, None)

            # Clear wallet-specific data
            self._wallet_face_last_seen.pop(wallet_address, None)
            self._wallet_reid_features.pop(wallet_address, None)
            self._pending_recovery.pop(wallet_address, None)
            self._last_appearance_update.pop(wallet_address, None)

        logger.info(f"Cleared track associations for {wallet_address[:8]}... (track_id={track_id})")

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
