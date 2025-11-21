#!/usr/bin/env python3
"""
GPU-Accelerated Face Recognition Service

Unified service that combines:
- YOLOv8 for person detection (GPU accelerated)
- InsightFace for high-quality face embeddings (GPU accelerated) 
- Integration with the main camera service architecture
- Blockchain-based face enrollment and recognition
"""

import cv2
import time
import numpy as np
import torch
import threading
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from .identity_tracker import IdentityTracker
from .identity_store import get_identity_store

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPUFaceService")

class GPUFaceService:
    """
    GPU-accelerated face recognition service with blockchain integration.
    Singleton service that integrates with the main camera service.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPUFaceService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        self._processing_enabled = False
        self._detection_enabled = True
        self._visualization_enabled = False  # OFF by default
        self._boxes_enabled = False
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        # GPU Models
        self.face_detector = None  # YOLOv8
        self.face_embedder = None  # InsightFace
        self._models_loaded = False
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.recognition_count = 0
        self.last_recognition = None
        
        # Detection settings - Optimized for real-time performance
        self._detection_interval = 0.1   # Face detection every 0.1s (10 FPS for smooth overlays)
        self._recognition_interval = 0.1  # Face recognition every 0.1s (merged with detection)
        self._similarity_threshold = 0.6  # Higher threshold for GPU model
        self._last_detection_time = 0
        self._last_recognition_time = 0
        
        # Get reference to unified IdentityStore (single source of truth)
        self.identity_store = get_identity_store()

        # Lock for thread safety in recognition results
        self._faces_lock = threading.Lock()
        
        # Detection results
        self._results_lock = threading.Lock()
        self._detected_faces = []
        self._recognized_faces = {}
        
        # Database paths
        self._faces_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/faces")
        Path(self._faces_dir).mkdir(parents=True, exist_ok=True)

        # Initialize identity tracker for persistent tracking
        self.identity_tracker = IdentityTracker(face_service=self)

        # Initialize GPU and models
        self.initialize_models()
        self.load_face_database()

    def initialize_gpu(self) -> bool:
        """Initialize and verify GPU setup - STRICT GPU ONLY"""
        logger.info("=== GPU FACE SERVICE INITIALIZATION (GPU ONLY) ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            logger.error("CUDA NOT AVAILABLE! GPU Face Service DISABLED - NO CPU FALLBACK!")
            raise RuntimeError("GPU Face Service requires CUDA - CPU fallback disabled")
            
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True

    def initialize_models(self) -> bool:
        """Initialize YOLOv8 and InsightFace models - STRICT GPU ONLY"""
        # STRICT: GPU must be available or service fails completely
        if not self.initialize_gpu():
            return False
        
        try:
            # REMOVED: Generic YOLOv8 detection - we reuse pose detections instead
            # This saves ~100MB GPU memory and eliminates redundant YOLO inference
            logger.info("Skipping generic YOLO loading - using pose service detections")
            
            # Initialize InsightFace for face embeddings - STRICT GPU ONLY
            logger.info("Loading InsightFace model on GPU with STRICT validation...")
            try:
                import insightface
                import onnxruntime as ort

                # Log available providers
                available_providers = ort.get_available_providers()
                logger.info(f"ONNX Runtime available providers: {available_providers}")

                # CRITICAL FIX: Force ONNX Runtime to use GPU for Jetson
                # InsightFace has a bug where it ignores the providers parameter for individual models
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'

                # Check for CUDA provider
                if 'CUDAExecutionProvider' not in available_providers:
                    raise RuntimeError("CUDAExecutionProvider not available! InsightFace requires GPU.")

                logger.info("Initializing InsightFace with CUDA provider...")

                # Step 1: Create FaceAnalysis with providers parameter
                self.face_embedder = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])

                # Step 2: Prepare models (loads ONNX models with ctx_id=0 for GPU)
                self.face_embedder.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

                # Step 3: CRITICAL FIX - Force each model to use CUDA after prepare()
                # InsightFace's prepare() ignores the providers parameter due to a bug
                # We must manually set providers on each model's ONNX session
                cuda_count = 0
                total_models = 0

                for model_name, model in self.face_embedder.models.items():
                    total_models += 1
                    try:
                        # Check current providers
                        current_providers = model.session.get_providers()

                        # If not using CUDA, recreate session with CUDA
                        if 'CUDAExecutionProvider' not in current_providers:
                            logger.warning(f"{model_name} using {current_providers[0]}, forcing CUDA...")

                            # Get model file path
                            model_file = model.model_file if hasattr(model, 'model_file') else None

                            if model_file and os.path.exists(model_file):
                                # Recreate ONNX session with CUDA provider
                                model.session = ort.InferenceSession(
                                    model_file,
                                    providers=['CUDAExecutionProvider']
                                )
                                logger.info(f"✅ {model_name}: Forced to CUDA")
                                cuda_count += 1
                            else:
                                logger.error(f"❌ {model_name}: Could not find model file")
                        else:
                            logger.info(f"✅ {model_name}: Already using CUDA")
                            cuda_count += 1

                    except Exception as e:
                        logger.error(f"❌ {model_name}: Failed to set CUDA - {e}")

                if cuda_count == 0:
                    raise RuntimeError(f"InsightFace CUDA initialization failed! 0/{total_models} models on GPU")

                logger.info(f"✅ InsightFace GPU validation: {cuda_count}/{total_models} models using CUDA")
                
                logger.info("InsightFace model loaded successfully on GPU with STRICT validation")
                
            except ImportError:
                logger.error("InsightFace not available. Install with: pip install insightface")
                raise RuntimeError("InsightFace required for GPU Face Service")
            
            self._models_loaded = True
            logger.info("All GPU face recognition models loaded successfully - NO CPU FALLBACK")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU models: {e}")
            logger.error("GPU Face Service DISABLED - NO CPU FALLBACK AVAILABLE")
            raise RuntimeError(f"GPU Face Service failed to initialize: {e}")

    def extract_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract high-quality face embedding using InsightFace"""
        logger.info(f"[EXTRACT-EMBEDDING] Called with shape={face_img.shape}")

        if not self._models_loaded:
            logger.warning("[EXTRACT-EMBEDDING] Models not loaded!")
            return None

        try:
            # Ensure face image is in the right format and size
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                logger.warning(f"[EXTRACT-EMBEDDING] Face region too small: {face_img.shape}")
                return None

            # Convert BGR to RGB if needed (InsightFace expects RGB)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_img

            logger.info(f"[EXTRACT-EMBEDDING] Calling InsightFace.get() on {face_rgb.shape} image...")

            # Extract embedding with GPU
            faces = self.face_embedder.get(face_rgb)

            logger.info(f"[EXTRACT-EMBEDDING] InsightFace found {len(faces)} faces")

            if len(faces) > 0:
                # Get the largest face (most confident detection)
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = face.normed_embedding
                logger.info(f"[EXTRACT-EMBEDDING] ✅ Successfully extracted embedding, shape={embedding.shape}")
                return embedding
            else:
                logger.warning("[EXTRACT-EMBEDDING] InsightFace detected 0 faces in the cropped person region")
                return None

        except Exception as e:
            logger.error(f"[EXTRACT-EMBEDDING] Exception: {e}", exc_info=True)
            return None

    def extract_compact_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract compact 128-dimension face embedding for blockchain storage
        
        This method generates a smaller embedding suitable for on-chain storage
        while maintaining reasonable recognition accuracy for cross-camera use.
        
        Args:
            face_img: Face image as numpy array
            
        Returns:
            128-dimension embedding array or None if extraction fails
        """
        # First get the full 512-dimension embedding
        full_embedding = self.extract_face_embedding(face_img)
        
        if full_embedding is None:
            return None
            
        try:
            # Reduce to 128 dimensions using simple truncation
            # This keeps the most significant features while fitting blockchain limits
            compact_embedding = full_embedding[:128]

            # Renormalize the compact embedding to maintain similarity properties
            norm = np.linalg.norm(compact_embedding)
            if norm > 0:
                compact_embedding = compact_embedding / norm

            return compact_embedding

        except Exception as e:
            return None

    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        OPTIMIZED v2: Use YOLO-pose detections instead of separate YOLO inference.

        This eliminates redundant person detection - YOLO-pose already gives us:
        - Person bounding boxes
        - Track IDs (for persistence)
        - Confidence scores

        Performance improvement: 31ms saved per frame (eliminates one YOLO call)
        """
        if not self._models_loaded:
            return []

        faces = []

        try:
            # CRITICAL OPTIMIZATION: Get person detections from pose service
            # This reuses YOLO-pose results instead of running YOLOv8n again
            if not hasattr(self, '_pose_service') or self._pose_service is None:
                # Fallback: pose service not available (shouldn't happen in production)
                import sys
                print(f"⚠️ WARNING: Pose service not available, skipping face detection", flush=True, file=sys.stderr)
                return []

            # Get latest pose detections (already computed by pose service)
            pose_detections = self._pose_service.get_poses()

            if not pose_detections or len(pose_detections) == 0:
                return []  # No persons detected

            # PERFORMANCE: Pre-compute frame dimensions to avoid repeated lookups
            frame_h, frame_w = frame.shape[:2]

            # Process each detected person from YOLO-pose
            for pose_data in pose_detections:
                bbox = pose_data.get('bbox')
                conf = pose_data.get('confidence', 0.5)

                if bbox is None:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)

                # Track ID handling (YOLO-pose may provide this in future)
                # For now, we'll use bbox matching for tracking
                track_id = pose_data.get('track_id', None)

                # OPTIMIZED: Extract face region with padding (clipped to frame bounds)
                padding = 20
                y1_padded = max(0, y1 - padding)
                y2_padded = min(frame_h, y2 + padding)
                x1_padded = max(0, x1 - padding)
                x2_padded = min(frame_w, x2 + padding)

                # Extract face region (numpy slice - zero copy operation)
                face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]

                if face_region.size == 0:
                    continue  # Skip invalid regions

                # Attempt face recognition (InsightFace on GPU)
                recognized_name, similarity = self.recognize_face(face_region)

                face_data = {
                    'box': (x1, y1, x2, y2),
                    'confidence': conf,
                    'track_id': track_id,
                    'recognized_name': recognized_name,
                    'similarity': similarity,
                    'face_region': face_region
                }

                faces.append(face_data)

        except Exception as e:
            import sys
            print(f"🚨 EXCEPTION in detect_and_recognize_faces: {e}", flush=True, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        return faces

    def detect_and_track_identities(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons and maintain persistent identity tracking.
        This is the main method for CV apps that need continuous identity tracking.
        """
        if not self._models_loaded:
            return []

        try:
            # Get detections with track IDs
            detections = self.detect_and_recognize_faces(frame)
            logger.info(f"[IDENTITY-TRACK] Got {len(detections)} detections from detect_and_recognize_faces")

            # Convert to format for identity tracker
            tracker_detections = []
            for det in detections:
                tracker_det = {
                    'class': 'person',
                    'x1': det['box'][0],
                    'y1': det['box'][1],
                    'x2': det['box'][2],
                    'y2': det['box'][3],
                    'confidence': det['confidence'],
                    'track_id': det.get('track_id')
                }
                tracker_detections.append(tracker_det)

            logger.info(f"[IDENTITY-TRACK] Processing {len(tracker_detections)} detections through identity tracker")

            # Process through identity tracker for persistent tracking
            enhanced_detections = self.identity_tracker.process_frame_detections(
                tracker_detections, frame
            )

            logger.info(f"[IDENTITY-TRACK] Got {len(enhanced_detections)} enhanced detections from tracker")

        except Exception as e:
            logger.error(f"[IDENTITY-TRACK] Error in detection phase: {e}", exc_info=True)
            return []

        try:
            # Merge back face recognition data
            import sys
            final_detections = []
            for enhanced, original in zip(enhanced_detections, detections):
                # Ensure 'box' tuple exists for visualization (visualization expects this format)
                if 'x1' in enhanced:
                    enhanced['box'] = (enhanced['x1'], enhanced['y1'], enhanced['x2'], enhanced['y2'])
                elif 'box' not in enhanced and 'box' in original:
                    enhanced['box'] = original['box']

                # Merge face recognition data
                enhanced['recognized_name'] = original.get('recognized_name')
                enhanced['similarity'] = original.get('similarity', 0.0)

                # If identity tracker found wallet, use that over face recognition
                if 'wallet_address' in enhanced:
                    enhanced['identity'] = enhanced['wallet_address']
                    enhanced['tracking_method'] = 'persistent_tracking'
                    print(f"[FINAL-DETECTION] Using PERSISTENT TRACKING: wallet={enhanced['wallet_address'][:8]}, track_id={enhanced.get('track_id')}", flush=True, file=sys.stderr)
                elif original.get('recognized_name'):
                    enhanced['identity'] = original['recognized_name']
                    enhanced['tracking_method'] = 'face_recognition'
                    print(f"[FINAL-DETECTION] Using FACE RECOGNITION: name={original['recognized_name']}, track_id={enhanced.get('track_id')}", flush=True, file=sys.stderr)
                else:
                    enhanced['identity'] = None
                    enhanced['tracking_method'] = None
                    print(f"[FINAL-DETECTION] NO IDENTITY: track_id={enhanced.get('track_id')}", flush=True, file=sys.stderr)

                final_detections.append(enhanced)

            logger.info(f"[IDENTITY-TRACK] Merged {len(final_detections)} final detections")

            # Forward to CV apps for competition processing
            try:
                from services.cv_apps_client import get_cv_apps_client
                cv_client = get_cv_apps_client()
                if cv_client.enabled and cv_client.active_app:
                    cv_client.process_frame(frame, final_detections)
            except Exception as cv_error:
                logger.debug(f"CV apps processing skipped: {cv_error}")

            return final_detections

        except Exception as e:
            logger.error(f"[IDENTITY-TRACK] Error in merge phase: {e}", exc_info=True)
            return []

    def recognize_face(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face by comparing embeddings"""
        # Check if we have any identities
        if self.identity_store.get_identity_count() == 0:
            return None, 0.0

        # Extract embedding for input face
        query_embedding = self.extract_face_embedding(face_img)
        if query_embedding is None:
            return None, 0.0

        # Find match using IdentityStore
        best_match = self.identity_store.find_by_face(query_embedding, self._similarity_threshold)

        if best_match:
            # Calculate similarity for return value
            identity = self.identity_store.get_identity(best_match)
            if identity and identity.face_embedding is not None:
                similarity = float(np.dot(query_embedding, identity.face_embedding))
                return best_match, similarity

        return None, 0.0

    def enroll_face(self, frame: np.ndarray, wallet_address: str, metadata: Dict = None) -> Dict:
        """Enroll a face for recognition using current frame"""
        if not self._models_loaded:
            return {
                'success': False,
                'error': 'Models not loaded'
            }

        try:
            # Detect faces in the frame
            faces = self.detect_and_recognize_faces(frame)

            if not faces:
                return {
                    'success': False,
                    'error': 'No faces detected in frame'
                }

            # Use the first detected face
            face_data = faces[0]
            face_region = face_data['face_region']

            # Extract embedding
            embedding = self.extract_face_embedding(face_region)
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Failed to extract face embedding'
                }

            # Build profile from metadata
            profile = {}
            if metadata:
                profile['display_name'] = metadata.get('name') or metadata.get('display_name')

            # Enroll via IdentityStore
            identity = self.identity_store.check_in(wallet_address, embedding, profile)

            logger.info(f"Successfully enrolled face for wallet: {wallet_address}")

            return {
                'success': True,
                'wallet_address': wallet_address,
                'embedding_shape': embedding.shape,
                'display_name': identity.get_display_name()
            }

        except Exception as e:
            logger.error(f"Error enrolling face: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def save_face_embedding(self, wallet_address: str, embedding: np.ndarray, metadata: Dict = None):
        """Save face embedding to disk (handled by IdentityStore now)"""
        # This is now handled by IdentityStore._save_identity()
        # Keeping method for API compatibility
        pass

    def load_face_database(self):
        """Load face database from stored embeddings"""
        # Delegate to IdentityStore
        count = self.identity_store.load_from_disk()
        logger.info(f"Loaded {count} face embeddings via IdentityStore")

    def inject_app_services(self, pose_service=None, app_manager=None):
        """Inject pose service and app manager for CV apps"""
        if pose_service:
            self._pose_service = pose_service
            logger.info("Injected pose service into GPU face service")
        if app_manager:
            self._app_manager = app_manager
            logger.info("Injected app manager into GPU face service")

    def start(self, buffer_service) -> bool:
        """Start the GPU face service with buffer integration"""
        if self._processing_enabled:
            logger.info("GPU face service already running")
            return True

        if not self._models_loaded:
            logger.error("Cannot start - models not loaded")
            return False

        self._buffer_service = buffer_service
        self._processing_enabled = True
        self._stop_event.clear()

        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

        logger.info("GPU face service started")
        return True

    def stop(self) -> None:
        """Stop the GPU face service"""
        if not self._processing_enabled:
            return
        
        self._processing_enabled = False
        self._stop_event.set()
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        logger.info("GPU face service stopped")

    def _processing_loop(self) -> None:
        """
        OPTIMIZED: Maximum performance processing loop.

        Key optimizations:
        1. Smart sleep - only sleep remaining time until next processing
        2. Batch GPU work - face + pose detection together when possible
        3. No redundant frame fetches - single fetch per cycle
        4. Precise timing - no drift from fixed sleep intervals
        """
        logger.info("Starting GPU face processing loop (OPTIMIZED)")

        next_detection_time = time.time()

        while self._processing_enabled and not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Smart sleep: Only sleep if we're ahead of schedule
                time_until_next = next_detection_time - current_time
                if time_until_next > 0:
                    # Use wait with timeout on stop event (allows clean shutdown)
                    if self._stop_event.wait(timeout=time_until_next):
                        break  # Stop event triggered
                    current_time = time.time()

                # Skip if still too early (in case of spurious wakeup)
                if current_time < next_detection_time:
                    continue

                # Get current frame from buffer service (single fetch)
                if not hasattr(self._buffer_service, 'get_latest_frame'):
                    time.sleep(0.01)
                    continue

                frame = self._buffer_service.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Run pose detection first - face detection will reuse these results
                if hasattr(self, '_pose_service') and self._pose_service and self._pose_service.enabled:
                    poses = self._pose_service.detect(frame)
                    self._pose_service.store_poses(poses)

                # Face detection using pose bboxes (no separate YOLO call)
                if self._detection_enabled:
                    self._detect_faces(frame)
                    self._update_recognition_from_detections()

                    # Process with active app
                    if hasattr(self, '_app_manager') and self._app_manager and self._app_manager.active_app:
                        with self._results_lock:
                            detections = self._detected_faces.copy()

                        frame_data = {
                            'detections': detections,
                            'keypoints': poses,
                            'timestamp': current_time
                        }
                        self._app_manager.process_frame(frame_data)

                # Update performance metrics
                self.frame_count += 1
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                # Schedule next detection (precise timing, no drift)
                next_detection_time += self._detection_interval

                # If we fell behind, catch up (don't accumulate delays)
                if next_detection_time < current_time:
                    next_detection_time = current_time + self._detection_interval

            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                # On error, reset timing to avoid cascading issues
                next_detection_time = time.time() + self._detection_interval

        logger.warning(f"🛑 Processing loop exited - enabled: {self._processing_enabled}, stop_event: {self._stop_event.is_set()}")

    def _detect_faces(self, frame: np.ndarray) -> None:
        """Detect faces in frame and update results"""
        try:
            # Use identity tracker for temporal tracking
            faces = self.detect_and_track_identities(frame)

            with self._results_lock:
                self._detected_faces = faces

        except Exception as e:
            logger.error(f"Error in _detect_faces: {e}", exc_info=True)

    def _update_recognition_from_detections(self) -> None:
        """
        Update recognition results from detected faces (merged pass optimization).
        This eliminates the separate _recognize_faces call.
        """
        try:
            with self._results_lock:
                faces = self._detected_faces.copy()

            recognized = {}
            for face in faces:
                # Check for persistent tracking first (wallet_address), then face recognition (recognized_name)
                wallet_address = face.get('wallet_address') or face.get('identity')

                if wallet_address:
                    # Got identity from persistent tracking
                    box = face.get('box')
                    similarity = face.get('similarity', 0)
                    confidence = face.get('identity_confidence', similarity)

                    recognized[wallet_address] = {
                        'box': box,
                        'similarity': confidence,
                        'name': self.get_user_display_name(wallet_address),
                        'last_seen': time.time(),
                        'tracking_method': face.get('tracking_method', 'persistent_tracking')
                    }

                    self.recognition_count += 1
                    self.last_recognition = time.time()
                elif face.get('recognized_name') and face.get('similarity', 0) > self._similarity_threshold:
                    # Fallback to face recognition
                    wallet_address = face['recognized_name']
                    box = face['box']
                    similarity = face['similarity']

                    recognized[wallet_address] = {
                        'box': box,
                        'similarity': similarity,
                        'name': self.get_user_display_name(wallet_address),
                        'last_seen': time.time(),
                        'tracking_method': 'face_recognition'
                    }

                    self.recognition_count += 1
                    self.last_recognition = time.time()

            with self._results_lock:
                self._recognized_faces = recognized

            # ✅ Notify blockchain sync service of recognized users
            try:
                from services.blockchain_session_sync import get_blockchain_session_sync
                blockchain_sync = get_blockchain_session_sync()
                for wallet_address in recognized.keys():
                    blockchain_sync.update_user_seen(wallet_address)
            except Exception as notify_error:
                pass

        except Exception as e:
            logger.error(f"Error in _update_recognition_from_detections: {e}")

    def _recognize_faces(self, frame: np.ndarray) -> None:
        """Recognize faces in frame and update results"""
        try:
            with self._results_lock:
                faces = self._detected_faces.copy()
            
            recognized = {}
            for face in faces:
                # Check for persistent tracking first (wallet_address), then face recognition (recognized_name)
                wallet_address = face.get('wallet_address') or face.get('identity')

                if wallet_address:
                    # Got identity from persistent tracking
                    box = face.get('box')
                    similarity = face.get('similarity', 0)
                    confidence = face.get('identity_confidence', similarity)

                    recognized[wallet_address] = {
                        'box': box,
                        'similarity': confidence,
                        'name': self.get_user_display_name(wallet_address),
                        'last_seen': time.time(),
                        'tracking_method': face.get('tracking_method', 'persistent_tracking')
                    }

                    self.recognition_count += 1
                    self.last_recognition = time.time()
                elif face.get('recognized_name') and face.get('similarity', 0) > self._similarity_threshold:
                    # Fallback to face recognition
                    wallet_address = face['recognized_name']
                    box = face['box']
                    similarity = face['similarity']

                    recognized[wallet_address] = {
                        'box': box,
                        'similarity': similarity,
                        'name': self.get_user_display_name(wallet_address),
                        'last_seen': time.time(),
                        'tracking_method': 'face_recognition'
                    }

                    self.recognition_count += 1
                    self.last_recognition = time.time()
            
            with self._results_lock:
                self._recognized_faces = recognized

            # ✅ NEW: Notify blockchain sync service of recognized users
            # This updates last_seen timestamps for face-based auto-checkout
            try:
                from services.blockchain_session_sync import get_blockchain_session_sync
                blockchain_sync = get_blockchain_session_sync()
                for wallet_address in recognized.keys():
                    blockchain_sync.update_user_seen(wallet_address)
            except Exception as notify_error:
                # Don't let notification errors break recognition
                pass

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")

    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply face detection and recognition overlays to the frame.
        OPTIMIZED: Frame is already a copy from buffer_service, draw directly on it.
        """
        # Show boxes if EITHER visualization OR boxes are enabled (not both required)
        if frame is None or not (self._visualization_enabled or self._boxes_enabled):
            return frame

        # Draw directly on frame (it's already a copy from buffer)
        output = frame
        
        with self._results_lock:
            # Draw recognized faces first (green boxes with names)
            for wallet_address, face_data in self._recognized_faces.items():
                if 'box' in face_data:
                    # face_data['box'] is (x1, y1, x2, y2), convert to (top, right, bottom, left)
                    x1, y1, x2, y2 = face_data['box']
                    face_location = (y1, x2, y2, x1)  # (top, right, bottom, left)
                    
                    # Use the stored display name from recognition results
                    identity_info = {
                        'name': face_data['name'],  # This already contains the display name
                        'confidence': face_data.get('similarity', 0) * 100,
                        'wallet_address': wallet_address
                    }
                    
                    # Draw face box with proper display name
                    self._draw_face_box(
                        output, 
                        face_location, 
                        identity_info=identity_info
                    )
            
            # Draw unrecognized detected faces (red boxes)
            for face_data in self._detected_faces:
                if isinstance(face_data, dict) and 'box' in face_data:
                    # Check if this face was already drawn as recognized
                    is_recognized = False

                    # Check persistent tracking (wallet_address/identity) first
                    wallet_address = face_data.get('wallet_address') or face_data.get('identity')
                    if wallet_address and wallet_address in self._recognized_faces:
                        is_recognized = True

                    # Also check face recognition
                    if not is_recognized and face_data.get('recognized_name'):
                        wallet_address = face_data['recognized_name']
                        if wallet_address in self._recognized_faces:
                            is_recognized = True
                    
                    # Only draw if not already recognized
                    if not is_recognized:
                        x1, y1, x2, y2 = face_data['box']
                        face_location = (y1, x2, y2, x1)  # (top, right, bottom, left)
                        
                        # Draw unrecognized face box
                        self._draw_face_box(
                            output, 
                            face_location, 
                            identity_info=None
                        )
        
        return output
    
    def _is_same_face_location(self, location1, location2, threshold=50):
        """
        Check if two face locations are roughly the same (within threshold pixels).
        """
        if not location1 or not location2:
            return False
        
        # Calculate center points
        center1 = ((location1[1] + location1[3]) // 2, (location1[0] + location1[2]) // 2)
        center2 = ((location2[1] + location2[3]) // 2, (location2[0] + location2[2]) // 2)
        
        # Calculate distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance < threshold

    def get_status(self) -> Dict:
        """Get service status"""
        enrolled_count = self.identity_store.get_identity_count()

        with self._results_lock:
            detected_count = len(self._detected_faces)
            recognized_count = len(self._recognized_faces)

        return {
            'gpu_available': torch.cuda.is_available(),
            'models_loaded': self._models_loaded,
            'processing_enabled': self._processing_enabled,
            'detection_enabled': self._detection_enabled,
            'visualization_enabled': self._visualization_enabled,
            'boxes_enabled': self._boxes_enabled,
            'fps': round(self.fps, 2),
            'enrolled_faces': enrolled_count,
            'detected_faces': detected_count,
            'recognized_faces': recognized_count,
            'recognition_count': self.recognition_count,
            'last_recognition': self.last_recognition,
            'similarity_threshold': self._similarity_threshold
        }

    def set_similarity_threshold(self, threshold: float) -> bool:
        """Set similarity threshold for recognition"""
        if 0.0 <= threshold <= 1.0:
            self._similarity_threshold = threshold
            logger.info(f"Similarity threshold set to: {threshold}")
            return True
        return False

    def enable_detection(self, enabled: bool) -> None:
        """Enable/disable face detection"""
        self._detection_enabled = enabled
        logger.info(f"Face detection {'enabled' if enabled else 'disabled'}")

    def enable_visualization(self, enabled: bool) -> None:
        """Enable/disable face visualization overlay"""
        self._visualization_enabled = enabled
        logger.info(f"Face visualization {'enabled' if enabled else 'disabled'}")

    def enable_boxes(self, enabled: bool) -> None:
        """Enable/disable face bounding boxes"""
        self._boxes_enabled = enabled
        logger.info(f"Face boxes {'enabled' if enabled else 'disabled'}")

    def get_enrolled_faces(self) -> List[Dict]:
        """Get list of enrolled faces"""
        faces = []
        for identity in self.identity_store.get_all_identities():
            faces.append({
                'wallet_address': identity.wallet_address,
                'name': identity.get_display_name(),
                'has_face': identity.face_embedding is not None,
                'checked_in_at': identity.checked_in_at
            })
        return faces

    def store_user_profile(self, wallet_address: str, user_profile: Dict) -> None:
        """Store user profile for display name resolution"""
        self.identity_store.update_profile(wallet_address, user_profile)
        display_name = self.identity_store.get_display_name(wallet_address)
        logger.info(f"Stored user profile for {wallet_address[:8]}...: {display_name}")

    def get_user_profile(self, wallet_address: str) -> Optional[Dict]:
        """Get user profile by wallet address"""
        identity = self.identity_store.get_identity(wallet_address)
        if identity:
            return identity.to_dict()
        return None

    def remove_user_profile(self, wallet_address: str) -> bool:
        """Remove user profile (for session cleanup)"""
        # Full identity removal is done via identity_store.check_out()
        # This is kept for API compatibility
        return self.identity_store.is_checked_in(wallet_address)

    def get_user_display_name(self, wallet_address: str) -> str:
        """Get the best display name for a wallet address"""
        return self.identity_store.get_display_name(wallet_address)

    def clear_enrolled_faces(self) -> bool:
        """Clear all enrolled faces"""
        try:
            # Check out all identities
            for wallet in self.identity_store.get_checked_in_wallets():
                self.identity_store.check_out(wallet)

            logger.info("Cleared all enrolled faces via IdentityStore")
            return True

        except Exception as e:
            logger.error(f"Error clearing enrolled faces: {e}")
            return False

    def remove_face_embedding(self, wallet_address: str) -> bool:
        """
        Remove ALL traces of a user's face data.
        Used when a user checks out on-chain.
        """
        try:
            # Remove from active recognition state
            with self._results_lock:
                self._recognized_faces.pop(wallet_address, None)

            # Full checkout is handled by IdentityStore
            stats = self.identity_store.check_out(wallet_address)

            if stats:
                logger.info(f"✅ Removed face data for {wallet_address[:8]}...")
                return True
            else:
                logger.info(f"ℹ️  No face data found for {wallet_address[:8]}...")
                return True  # Goal achieved

        except Exception as e:
            logger.error(f"❌ Error removing face embedding for {wallet_address}: {e}")
            return False

    def get_faces(self) -> Dict:
        """
        Get current face detection and recognition results.
        This method provides compatibility with the recognize_face route.
        """
        with self._results_lock:
            detected_faces = self._detected_faces.copy()
            recognized_faces = self._recognized_faces.copy()
        
        # Convert to expected format
        detected_count = len(detected_faces)
        recognized_count = len(recognized_faces)
        
        # Format recognized faces for compatibility
        recognized_faces_formatted = {}
        for wallet_address, face_data in recognized_faces.items():
            # Format: [top, right, bottom, left, confidence]
            box = face_data['box']  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = box
            # Convert to (top, right, bottom, left) format
            top, right, bottom, left = y1, x2, y2, x1
            confidence = face_data.get('similarity', 0.0)
            recognized_faces_formatted[wallet_address] = [top, right, bottom, left, confidence]
        
        return {
            'detected_count': detected_count,
            'recognized_count': recognized_count,
            'detected_faces': detected_faces,
            'recognized_faces': recognized_faces_formatted
        }

    def process_frame_for_recognition(self, frame: np.ndarray) -> None:
        """
        Process a frame for face detection and recognition.
        This is the proper public API for on-demand face processing.
        """
        try:
            # First detect faces if needed
            self._detect_faces(frame)
            # Then recognize them
            self._recognize_faces(frame)
        except Exception as e:
            logger.error(f"Error processing frame for recognition: {e}")

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame using YOLO model
        
        Args:
            frame: Input frame
            
        Returns:
            List of face detection dictionaries with 'box' key
        """
        try:
            if not self._models_loaded:
                logger.warning("Models not loaded for face detection")
                return []
            
            # Use pose service detections instead of separate YOLO
            if not hasattr(self, '_pose_service') or self._pose_service is None:
                logger.warning("Pose service not available for detect_faces()")
                return []

            pose_detections = self._pose_service.get_poses()

            faces = []
            for pose_data in pose_detections:
                bbox = pose_data.get('bbox')
                conf = pose_data.get('confidence', 0.5)

                if bbox and conf > 0.5:
                    x1, y1, x2, y2 = bbox
                    faces.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def get_current_embedding(self, frame: np.ndarray = None) -> Optional[List[float]]:
        """
        Extract face embedding from current frame for NFT creation
        
        Args:
            frame: Optional frame to process (if None, gets current frame from buffer)
            
        Returns:
            Face embedding as list of floats, or None if no face detected
        """
        try:
            # If no frame provided, we'll need to get it from buffer service
            if frame is None:
                logger.warning("get_current_embedding called without frame - need buffer service integration")
                return None
            
            # Detect faces in the frame
            faces = self.detect_faces(frame)
            
            if not faces or len(faces) == 0:
                logger.warning("No faces detected in current frame for embedding extraction")
                return None
            
            # Use the largest face (first face from YOLO detection)
            largest_face = faces[0]
            
            # Extract face region
            x1, y1, x2, y2 = largest_face['box']
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning("Face region is empty")
                return None
            
            # Extract embedding using InsightFace
            embedding = self.extract_face_embedding(face_region)
            
            if embedding is None:
                logger.warning("Failed to extract face embedding")
                return None
            
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            logger.info(f"Successfully extracted face embedding, shape: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error extracting current embedding: {e}")
            return None

    def get_current_embedding_with_buffer(self, buffer_service) -> Optional[List[float]]:
        """
        Extract face embedding from current buffer frame
        
        Args:
            buffer_service: Buffer service to get current frame from
            
        Returns:
            Face embedding as list of floats, or None if no face detected
        """
        try:
            # Get current frame from buffer
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                logger.warning("No frame available from buffer service")
                return None
            
            return self.get_current_embedding(frame)
            
        except Exception as e:
            logger.error(f"Error extracting embedding with buffer: {e}")
            return None

    def get_current_compact_embedding(self, frame: np.ndarray = None) -> Optional[List[float]]:
        """
        Extract compact 128-dimension face embedding for blockchain storage
        
        Args:
            frame: Optional frame to process (if None, gets current frame from buffer)
            
        Returns:
            Compact face embedding as list of floats, or None if no face detected
        """
        try:
            # If no frame provided, we'll need to get it from buffer service
            if frame is None:
                logger.warning("get_current_compact_embedding called without frame - need buffer service integration")
                return None
            
            # Detect faces in the frame
            faces = self.detect_faces(frame)
            
            if not faces or len(faces) == 0:
                logger.warning("No faces detected in current frame for compact embedding extraction")
                return None
            
            # Use the largest face (first face from YOLO detection)
            largest_face = faces[0]
            
            # Extract face region
            x1, y1, x2, y2 = largest_face['box']
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                logger.warning("Face region is empty")
                return None
            
            # Extract compact embedding using InsightFace
            compact_embedding = self.extract_compact_face_embedding(face_region)
            
            if compact_embedding is None:
                logger.warning("Failed to extract compact face embedding")
                return None
            
            # Convert numpy array to list for JSON serialization
            embedding_list = compact_embedding.tolist()
            
            logger.info(f"Successfully extracted compact face embedding, shape: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error extracting current compact embedding: {e}")
            return None

    def get_current_compact_embedding_with_buffer(self, buffer_service) -> Optional[List[float]]:
        """
        Extract compact face embedding from current buffer frame for blockchain
        
        Args:
            buffer_service: Buffer service to get current frame from
            
        Returns:
            Compact face embedding as list of floats, or None if no face detected
        """
        try:
            # Get current frame from buffer
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                logger.warning("No frame available from buffer service")
                return None
            
            return self.get_current_compact_embedding(frame)
            
        except Exception as e:
            logger.error(f"Error extracting compact embedding with buffer: {e}")
            return None

    def _draw_face_box(self, frame, face_location, identity_info=None):
        """
        Draw a face detection box on the frame with proper color coding and labels.
        
        Args:
            frame: The frame to draw on
            face_location: (top, right, bottom, left) coordinates
            identity_info: Optional identity information for recognized faces
        """
        top, right, bottom, left = face_location
        
        # Determine box color and label based on recognition status
        if identity_info and identity_info.get('name'):
            # Recognized face - GREEN box
            box_color = (0, 255, 0)  # Green
            name = identity_info['name']
            confidence = identity_info.get('confidence', 0)
            label = f"{name} ({confidence:.1f}%)"
        else:
            # Unrecognized face - RED box
            box_color = (0, 0, 255)  # Red
            label = "Not Recognized"
        
        # Draw the face box
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - 30), (left + label_size[0] + 10, top), box_color, -1)
        
        # Draw label text (white text for better visibility)
        cv2.putText(frame, label, (left + 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Global service instance
_gpu_face_service = None

def get_gpu_face_service() -> GPUFaceService:
    """Get the GPU face service singleton instance"""
    global _gpu_face_service
    if _gpu_face_service is None:
        _gpu_face_service = GPUFaceService()
    return _gpu_face_service