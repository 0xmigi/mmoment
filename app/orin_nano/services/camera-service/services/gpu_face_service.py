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
        
        # Detection settings - Restored to original performance
        self._detection_interval = 0.5   # Face detection every 0.5s (restored from 5s)
        self._recognition_interval = 1.0  # Face recognition every 1s (restored from 10s)
        self._similarity_threshold = 0.6  # Higher threshold for GPU model
        self._last_detection_time = 0
        self._last_recognition_time = 0
        
        # Face database (blockchain-based)
        self._faces_lock = threading.Lock()
        self._face_embeddings = {}  # wallet_address -> embedding
        self._face_names = {}       # wallet_address -> display_name
        self._face_metadata = {}    # wallet_address -> metadata
        self._user_profiles = {}    # wallet_address -> user_profile (display_name, username)
        
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
            # Initialize YOLOv8 for person detection - GPU ONLY
            logger.info("Loading YOLOv8 detection model on GPU...")
            from ultralytics import YOLO
            
            model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt')
            self.face_detector = YOLO(model_path)
            
            # FORCE GPU ONLY - no CPU fallback
            self.face_detector.to('cuda')
            
            # VALIDATE YOLOv8 is actually on GPU
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for YOLOv8!")
            
            # Check model device
            model_device = next(self.face_detector.model.parameters()).device
            if model_device.type != 'cuda':
                raise RuntimeError(f"YOLOv8 failed to load on GPU! Device: {model_device}")
            
            logger.info(f"YOLOv8 VERIFIED on GPU device: {model_device}")
            
            # Initialize InsightFace for face embeddings - STRICT GPU ONLY
            logger.info("Loading InsightFace model on GPU with STRICT validation...")
            try:
                import insightface
                
                # FORCE ONNX Runtime to fail if CUDA unavailable
                os.environ['ORT_CUDA_UNAVAILABLE'] = '0'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
                # STRICT: Only CUDA provider, WILL FAIL if GPU unavailable
                providers = ['CUDAExecutionProvider']
                self.face_embedder = insightface.app.FaceAnalysis(providers=providers)
                self.face_embedder.prepare(ctx_id=0, det_size=(640, 640))
                
                # VALIDATE that CUDA provider is actually being used
                try:
                    actual_providers = self.face_embedder.models['det'].get_providers()
                    if 'CUDAExecutionProvider' not in actual_providers:
                        raise RuntimeError(f"InsightFace failed to use CUDA! Using: {actual_providers}")
                    logger.info(f"InsightFace VERIFIED using GPU providers: {actual_providers}")
                except Exception as validation_error:
                    logger.warning(f"Could not validate providers: {validation_error}")
                
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
        import sys
        print(f"🔍 extract_face_embedding called, shape={face_img.shape}", flush=True, file=sys.stderr)
        if not self._models_loaded:
            print(f"🔍 Models not loaded", flush=True, file=sys.stderr)
            return None

        try:
            # Ensure face image is in the right format and size
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                print(f"🔍 Face too small: {face_img.shape}", flush=True, file=sys.stderr)
                return None

            # Convert BGR to RGB if needed (InsightFace expects RGB)
            print(f"🔍 Converting BGR to RGB...", flush=True, file=sys.stderr)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_img

            # Extract embedding with GPU
            print(f"🔍 BEFORE InsightFace get()...", flush=True, file=sys.stderr)
            faces = self.face_embedder.get(face_rgb)
            print(f"🔍 AFTER InsightFace get(): {len(faces)} faces", flush=True, file=sys.stderr)

            if len(faces) > 0:
                # Get the largest face (most confident detection)
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

                # Return normalized embedding
                embedding = face.normed_embedding
                return embedding
            else:
                return None

        except Exception as e:
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
        """Detect persons and extract faces for recognition"""
        import sys
        print(f"🔍 detect_and_recognize_faces called, models_loaded={self._models_loaded}", flush=True, file=sys.stderr)
        if not self._models_loaded:
            print(f"🔍 Models not loaded, returning empty", flush=True, file=sys.stderr)
            return []

        faces = []

        try:
            # GPU inference with YOLOv8
            import torch
            device = next(self.face_detector.model.parameters()).device

            # Check if CUDA is being used
            if device.type != 'cuda':
                raise RuntimeError("GPU detection failed - model not on CUDA!")

            # CRITICAL FIX: Force YOLOv8 to use GPU for inference
            # YOLOv8 bug: ignores .to('cuda') without explicit device parameter!
            # Enable tracking with persist=True for identity tracking
            with torch.cuda.nvtx.range("YOLOv8_inference"):  # GPU profiling marker
                results = self.face_detector.track(frame, device=0, verbose=False, persist=True)  # device=0 forces GPU, persist enables tracking!
            
            import sys
            for result in results:
                if result.boxes is not None:
                    print(f"🔍 Found {len(result.boxes)} boxes", flush=True, file=sys.stderr)
                    for i, box in enumerate(result.boxes):
                        cls = int(box.cls)
                        conf = float(box.conf)
                        print(f"🔍 Box {i}: class={cls}, conf={conf}", flush=True, file=sys.stderr)
                        # Filter for person class (class 0 in COCO)
                        if cls == 0 and conf > 0.5:
                            print(f"🔍 Person detected! Processing...", flush=True, file=sys.stderr)
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            confidence = float(box.conf)

                            # Get track ID if available (from tracking)
                            track_id = None
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0]) if hasattr(box.id, '__iter__') else int(box.id)

                            # Extract face region with some padding
                            padding = 20
                            y1_padded = max(0, y1 - padding)
                            y2_padded = min(frame.shape[0], y2 + padding)
                            x1_padded = max(0, x1 - padding)
                            x2_padded = min(frame.shape[1], x2 + padding)

                            # Extract face region
                            face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            print(f"🔍 Face region size: {face_region.size}", flush=True, file=sys.stderr)

                            if face_region.size > 0:
                                # Attempt face recognition
                                print(f"🔍 BEFORE calling recognize_face()...", flush=True, file=sys.stderr)
                                recognized_name, similarity = self.recognize_face(face_region)
                                print(f"🔍 AFTER recognize_face(): name={recognized_name}, sim={similarity}", flush=True, file=sys.stderr)

                                face_data = {
                                    'box': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'track_id': track_id,  # Add track ID for identity tracking
                                    'recognized_name': recognized_name,
                                    'similarity': similarity,
                                    'face_region': face_region
                                }

                                faces.append(face_data)
                                print(f"🔍 Face added to list! Total: {len(faces)}", flush=True, file=sys.stderr)

        except Exception as e:
            # REMOVED: logger causes deadlock
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

        # Get detections with track IDs
        detections = self.detect_and_recognize_faces(frame)

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

        # Process through identity tracker for persistent tracking
        enhanced_detections = self.identity_tracker.process_frame_detections(
            tracker_detections, frame
        )

        # Merge back face recognition data
        final_detections = []
        for enhanced, original in zip(enhanced_detections, detections):
            enhanced['face_data'] = {
                'recognized_name': original['recognized_name'],
                'similarity': original['similarity']
            }
            # If identity tracker found wallet, use that over face recognition
            if 'wallet_address' in enhanced:
                enhanced['identity'] = enhanced['wallet_address']
                enhanced['tracking_method'] = 'persistent_tracking'
            elif original['recognized_name']:
                enhanced['identity'] = original['recognized_name']
                enhanced['tracking_method'] = 'face_recognition'
            else:
                enhanced['identity'] = None
                enhanced['tracking_method'] = None

            final_detections.append(enhanced)

        return final_detections

    def recognize_face(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face by comparing embeddings"""
        # Quick check if we have any embeddings (minimal lock time)
        with self._faces_lock:
            if not self._face_embeddings:
                return None, 0.0
            # Make a copy of embeddings dict to avoid holding lock during embedding extraction
            embeddings_copy = self._face_embeddings.copy()

        # Extract embedding for input face (outside lock)
        query_embedding = self.extract_face_embedding(face_img)
        if query_embedding is None:
            return None, 0.0

        # Compare with all database faces using cosine similarity (no lock needed, using copy)
        best_match = None
        best_similarity = 0.0

        for wallet_address, db_embedding in embeddings_copy.items():
            # Cosine similarity (higher is better for normalized embeddings)
            similarity = float(np.dot(query_embedding, db_embedding))

            if similarity > best_similarity and similarity > self._similarity_threshold:
                best_similarity = similarity
                best_match = wallet_address

        return best_match, best_similarity

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
            
            # Store embedding and metadata
            with self._faces_lock:
                            self._face_embeddings[wallet_address] = embedding
            
            # Use display name from user profile if available, otherwise use metadata or fallback
            if wallet_address in self._user_profiles:
                profile = self._user_profiles[wallet_address]
                display_name = profile.get('display_name') or profile.get('username') or wallet_address[:8]
            else:
                display_name = metadata.get('name', wallet_address[:8]) if metadata else wallet_address[:8]
            
            self._face_names[wallet_address] = display_name
            self._face_metadata[wallet_address] = metadata or {}
            
            # Save to disk
            self.save_face_embedding(wallet_address, embedding, metadata)
            
            logger.info(f"Successfully enrolled face for wallet: {wallet_address}")
            
            return {
                'success': True,
                'wallet_address': wallet_address,
                'embedding_shape': embedding.shape,
                'metadata': self._face_metadata[wallet_address]
            }
            
        except Exception as e:
            logger.error(f"Error enrolling face: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def save_face_embedding(self, wallet_address: str, embedding: np.ndarray, metadata: Dict = None):
        """Save face embedding to disk"""
        try:
            # Save embedding
            embedding_path = os.path.join(self._faces_dir, f"{wallet_address}.npy")
            np.save(embedding_path, embedding)
            
            # Save metadata
            metadata_path = os.path.join(self._faces_dir, f"{wallet_address}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata or {}, f)
                
            logger.info(f"Saved face embedding: {embedding_path}")
            
        except Exception as e:
            logger.error(f"Failed to save face embedding: {e}")

    def load_face_database(self):
        """Load face database from stored embeddings"""
        with self._faces_lock:
            self._face_embeddings = {}
            self._face_names = {}
            self._face_metadata = {}
        
        # Load from .npy files (saved embeddings)
        for embedding_file in Path(self._faces_dir).glob("*.npy"):
            try:
                # Load embedding
                embedding = np.load(embedding_file)
                wallet_address = embedding_file.stem
                
                # Load metadata if available
                metadata_file = embedding_file.with_name(f"{wallet_address}_metadata.json")
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                with self._faces_lock:
                                    self._face_embeddings[wallet_address] = embedding
                
                # Use display name from user profile if available, otherwise use metadata or fallback
                if wallet_address in self._user_profiles:
                    profile = self._user_profiles[wallet_address]
                    display_name = profile.get('display_name') or profile.get('username') or wallet_address[:8]
                else:
                    display_name = metadata.get('name', wallet_address[:8])
                
                self._face_names[wallet_address] = display_name
                self._face_metadata[wallet_address] = metadata
                
                logger.info(f"Loaded face embedding: {wallet_address}")
                
            except Exception as e:
                logger.warning(f"Failed to load {embedding_file}: {e}")
        
        with self._faces_lock:
            count = len(self._face_embeddings)
        
        logger.info(f"Loaded {count} face embeddings from database")

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
        """Main processing loop for face detection and recognition"""
        logger.info("Starting GPU face processing loop")

        while self._processing_enabled and not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Get current frame from buffer service
                if hasattr(self._buffer_service, 'get_latest_frame'):
                    frame = self._buffer_service.get_latest_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                else:
                    time.sleep(0.1)
                    continue

                # Perform face detection at intervals
                if current_time - self._last_detection_time >= self._detection_interval:
                    if self._detection_enabled:
                        self._detect_faces(frame)
                    self._last_detection_time = current_time

                # Perform face recognition at intervals
                if current_time - self._last_recognition_time >= self._recognition_interval:
                    if self._detection_enabled:
                        self._recognize_faces(frame)
                    self._last_recognition_time = current_time

                # Update performance metrics
                self.frame_count += 1
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

                # Increased sleep to prevent excessive CPU/GPU usage
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.5)

        logger.warning(f"🛑 Processing loop exited - enabled: {self._processing_enabled}, stop_event: {self._stop_event.is_set()}")

    def _detect_faces(self, frame: np.ndarray) -> None:
        """Detect faces in frame and update results"""
        try:
            faces = self.detect_and_recognize_faces(frame)

            with self._results_lock:
                self._detected_faces = faces

        except Exception as e:
            pass  # Silent - logging here causes deadlock

    def _recognize_faces(self, frame: np.ndarray) -> None:
        """Recognize faces in frame and update results"""
        try:
            with self._results_lock:
                faces = self._detected_faces.copy()
            
            recognized = {}
            for face in faces:
                if face.get('recognized_name') and face.get('similarity', 0) > self._similarity_threshold:
                    wallet_address = face['recognized_name']
                    box = face['box']
                    similarity = face['similarity']
                    
                    recognized[wallet_address] = {
                        'box': box,
                        'similarity': similarity,
                        'name': self.get_user_display_name(wallet_address),
                        'last_seen': time.time()
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
        Apply face detection and recognition overlays to a copy of the frame.
        This does NOT modify the original frame from the buffer.
        """
        # Show boxes if EITHER visualization OR boxes are enabled (not both required)
        if frame is None or not (self._visualization_enabled or self._boxes_enabled):
            return frame
            
        # Make a copy to avoid modifying the original
        output = frame.copy()
        
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
                    if face_data.get('recognized_name'):
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
        with self._faces_lock:
            enrolled_count = len(self._face_embeddings)
        
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
        """Enable/disable visualization"""
        self._visualization_enabled = enabled
        logger.info(f"Face visualization {'enabled' if enabled else 'disabled'}")

    def enable_boxes(self, enabled: bool) -> None:
        """Enable/disable bounding boxes"""
        self._boxes_enabled = enabled
        logger.info(f"Face boxes {'enabled' if enabled else 'disabled'}")

    def get_enrolled_faces(self) -> List[Dict]:
        """Get list of enrolled faces"""
        with self._faces_lock:
            faces = []
            for wallet_address, metadata in self._face_metadata.items():
                faces.append({
                    'wallet_address': wallet_address,
                    'name': self.get_user_display_name(wallet_address),
                    'metadata': metadata
                })
            return faces

    def store_user_profile(self, wallet_address: str, user_profile: Dict) -> None:
        """Store user profile for display name resolution"""
        with self._faces_lock:
            self._user_profiles[wallet_address] = user_profile
            
            # Update face name with display name if available
            display_name = user_profile.get('display_name')
            username = user_profile.get('username')
            
            if display_name:
                self._face_names[wallet_address] = display_name
            elif username:
                self._face_names[wallet_address] = username
            else:
                self._face_names[wallet_address] = wallet_address[:8]
                
            logger.info(f"Stored user profile for {wallet_address}: {display_name or username or wallet_address[:8]}")

    def get_user_profile(self, wallet_address: str) -> Optional[Dict]:
        """Get user profile by wallet address"""
        with self._faces_lock:
            return self._user_profiles.get(wallet_address)

    def remove_user_profile(self, wallet_address: str) -> bool:
        """Remove user profile (for session cleanup)"""
        with self._faces_lock:
            profile_removed = wallet_address in self._user_profiles
            if profile_removed:
                self._user_profiles.pop(wallet_address, None)
                self._face_names.pop(wallet_address, None)
                logger.info(f"[GPU-FACE] Removed user profile for {wallet_address}")
            return profile_removed

    def get_user_display_name(self, wallet_address: str) -> str:
        """Get the best display name for a wallet address"""
        import sys
        with self._faces_lock:
            # First check if we have a stored user profile
            if wallet_address in self._user_profiles:
                profile = self._user_profiles[wallet_address]
                print(f"🔍 DISPLAY_NAME: Found profile for {wallet_address[:8]}: {profile}", flush=True, file=sys.stderr)
                if profile.get('display_name'):
                    print(f"🔍 DISPLAY_NAME: Returning display_name: {profile['display_name']}", flush=True, file=sys.stderr)
                    return profile['display_name']
                elif profile.get('username'):
                    print(f"🔍 DISPLAY_NAME: Returning username: {profile['username']}", flush=True, file=sys.stderr)
                    return profile['username']
            else:
                print(f"🔍 DISPLAY_NAME: No profile for {wallet_address[:8]}, _user_profiles has {len(self._user_profiles)} entries", flush=True, file=sys.stderr)

            # Fallback to stored face names
            if wallet_address in self._face_names:
                print(f"🔍 DISPLAY_NAME: Returning from _face_names: {self._face_names[wallet_address]}", flush=True, file=sys.stderr)
                return self._face_names[wallet_address]

            # Final fallback to shortened wallet address
            print(f"🔍 DISPLAY_NAME: Fallback to wallet[:8]: {wallet_address[:8]}", flush=True, file=sys.stderr)
            return wallet_address[:8]

    def clear_enrolled_faces(self) -> bool:
        """Clear all enrolled faces"""
        try:
            with self._faces_lock:
                self._face_embeddings.clear()
                self._face_names.clear()
                self._face_metadata.clear()
                self._user_profiles.clear()

            # Clear database files
            for file_path in Path(self._faces_dir).glob("*.npy"):
                file_path.unlink()
            for file_path in Path(self._faces_dir).glob("*_metadata.json"):
                file_path.unlink()

            logger.info("Cleared all enrolled faces")
            return True

        except Exception as e:
            logger.error(f"Error clearing enrolled faces: {e}")
            return False

    def remove_face_embedding(self, wallet_address: str) -> bool:
        """
        Remove ALL traces of a user's face data from memory and disk.
        Used when a user checks out on-chain.

        Ensures complete "leave no trace" cleanup:
        - Facial embeddings (memory + disk)
        - User profiles and display names
        - Active recognition state
        - Metadata files
        """
        try:
            # Remove from memory (all face-related data structures)
            with self._faces_lock:
                removed_embedding = self._face_embeddings.pop(wallet_address, None)
                removed_name = self._face_names.pop(wallet_address, None)
                removed_metadata = self._face_metadata.pop(wallet_address, None)
                removed_profile = self._user_profiles.pop(wallet_address, None)

            # Remove from active recognition state
            with self._results_lock:
                removed_recognition = self._recognized_faces.pop(wallet_address, None)

            # Remove from disk
            embedding_path = os.path.join(self._faces_dir, f"{wallet_address}.npy")
            metadata_path = os.path.join(self._faces_dir, f"{wallet_address}_metadata.json")

            files_removed = 0
            if os.path.exists(embedding_path):
                os.unlink(embedding_path)
                files_removed += 1
                logger.info(f"🗑️  Removed embedding file: {embedding_path}")

            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
                files_removed += 1
                logger.info(f"🗑️  Removed metadata file: {metadata_path}")

            # Log comprehensive cleanup summary
            removed_items = {
                'embedding': removed_embedding is not None,
                'name': removed_name is not None,
                'metadata': removed_metadata is not None,
                'profile': removed_profile is not None,
                'recognition_state': removed_recognition is not None,
                'disk_files': files_removed
            }

            if removed_embedding is not None:
                logger.info(f"✅ Successfully removed ALL face data for {wallet_address[:8]}... "
                           f"(embedding: ✓, profile: {'✓' if removed_profile else '✗'}, "
                           f"recognition: {'✓' if removed_recognition else '✗'}, "
                           f"disk files: {files_removed})")
                return True
            else:
                logger.info(f"ℹ️  No face embedding found for {wallet_address[:8]}... (already clean)")
                return True  # Consider this success since the goal is achieved

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
            
            # Use the existing _detect_faces method logic
            results = self.face_detector(frame)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = box.conf.item()
                        if conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            faces.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': conf
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