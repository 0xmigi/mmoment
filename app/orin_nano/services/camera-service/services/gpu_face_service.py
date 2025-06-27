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
        
        # Detection settings
        self._detection_interval = 0.5   # Face detection every 0.5s
        self._recognition_interval = 1.0  # Face recognition every 1s
        self._similarity_threshold = 0.6  # Higher threshold for GPU model
        self._last_detection_time = 0
        self._last_recognition_time = 0
        
        # Face database (blockchain-based)
        self._faces_lock = threading.Lock()
        self._face_embeddings = {}  # wallet_address -> embedding
        self._face_names = {}       # wallet_address -> display_name
        self._face_metadata = {}    # wallet_address -> metadata
        
        # Detection results
        self._results_lock = threading.Lock()
        self._detected_faces = []
        self._recognized_faces = {}
        
        # Database paths
        self._faces_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/faces")
        Path(self._faces_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU and models
        self.initialize_models()
        self.load_face_database()

    def initialize_gpu(self) -> bool:
        """Initialize and verify GPU setup"""
        logger.info("=== GPU FACE SERVICE INITIALIZATION ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            logger.error("CUDA not available! Falling back to CPU mode.")
            return False

    def initialize_models(self) -> bool:
        """Initialize YOLOv8 and InsightFace models"""
        if not self.initialize_gpu():
            # GPU not available, but continue with CPU
            pass
        
        try:
            # Initialize YOLOv8 for person detection
            logger.info("Loading YOLOv8 detection model...")
            from ultralytics import YOLO
            
            model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt')
            self.face_detector = YOLO(model_path)
            
            if torch.cuda.is_available():
                self.face_detector.to('cuda')
                logger.info("YOLOv8 loaded on GPU")
            else:
                logger.info("YOLOv8 loaded on CPU")
            
            # Initialize InsightFace for face embeddings
            logger.info("Loading InsightFace model...")
            try:
                import insightface
                
                # Create InsightFace app with GPU support if available
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                self.face_embedder = insightface.app.FaceAnalysis(providers=providers)
                self.face_embedder.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
                
                logger.info(f"InsightFace model loaded successfully ({'GPU' if torch.cuda.is_available() else 'CPU'})")
                
            except ImportError:
                logger.error("InsightFace not available. Install with: pip install insightface")
                return False
            
            self._models_loaded = True
            logger.info("All GPU face recognition models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False

    def extract_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract high-quality face embedding using InsightFace"""
        if not self._models_loaded:
            logger.warning("Models not loaded")
            return None
            
        try:
            # Ensure face image is in the right format and size
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                logger.warning(f"Face image too small: {face_img.shape}")
                return None
            
            # Convert BGR to RGB if needed (InsightFace expects RGB)
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_img
            
            # Use InsightFace to get face embedding
            faces = self.face_embedder.get(face_rgb)
            
            if len(faces) > 0:
                # Get the largest face (most confident detection)
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                
                # Return normalized embedding
                embedding = face.normed_embedding
                logger.debug(f"Successfully extracted embedding, shape: {embedding.shape}")
                return embedding
            else:
                logger.warning("InsightFace found no faces in the region")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
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
            
            logger.debug(f"Successfully extracted compact embedding, shape: {compact_embedding.shape}")
            return compact_embedding
            
        except Exception as e:
            logger.warning(f"Failed to create compact embedding: {e}")
            return None

    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons and extract faces for recognition"""
        if not self._models_loaded:
            return []
            
        faces = []
        
        try:
            # Detect persons using YOLOv8
            results = self.face_detector(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Filter for person class (class 0 in COCO)
                        if int(box.cls) == 0 and float(box.conf) > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            confidence = float(box.conf)
                            
                            # Extract face region with some padding
                            padding = 20
                            y1_padded = max(0, y1 - padding)
                            y2_padded = min(frame.shape[0], y2 + padding)
                            x1_padded = max(0, x1 - padding)
                            x2_padded = min(frame.shape[1], x2 + padding)
                            
                            # Extract face region
                            face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            
                            if face_region.size > 0:
                                # Attempt face recognition
                                recognized_name, similarity = self.recognize_face(face_region)
                                
                                face_data = {
                                    'box': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'recognized_name': recognized_name,
                                    'similarity': similarity,
                                    'face_region': face_region
                                }
                                
                                faces.append(face_data)
        
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
        
        return faces

    def recognize_face(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face by comparing embeddings"""
        with self._faces_lock:
            if not self._face_embeddings:
                return None, 0.0
        
        # Extract embedding for input face
        query_embedding = self.extract_face_embedding(face_img)
        if query_embedding is None:
            return None, 0.0
        
        # Compare with all database faces using cosine similarity
        best_match = None
        best_similarity = 0.0
        
        with self._faces_lock:
            for wallet_address, db_embedding in self._face_embeddings.items():
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
                self._face_names[wallet_address] = metadata.get('name', wallet_address[:8]) if metadata else wallet_address[:8]
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
                    self._face_names[wallet_address] = metadata.get('name', wallet_address[:8])
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
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.5)

    def _detect_faces(self, frame: np.ndarray) -> None:
        """Detect faces in frame and update results"""
        try:
            faces = self.detect_and_recognize_faces(frame)
            
            with self._results_lock:
                self._detected_faces = faces
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")

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
                        'name': self._face_names.get(wallet_address, wallet_address[:8]),
                        'last_seen': time.time()
                    }
                    
                    self.recognition_count += 1
                    self.last_recognition = time.time()
            
            with self._results_lock:
                self._recognized_faces = recognized
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")

    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply face detection and recognition overlays to a copy of the frame.
        This does NOT modify the original frame from the buffer.
        """
        if frame is None or not self._visualization_enabled or not self._boxes_enabled:
            return frame
            
        # Make a copy to avoid modifying the original
        output = frame.copy()
        
        with self._results_lock:
            if self._detected_faces:
                for face_data in self._detected_faces:
                    # Extract face location from the face data dictionary
                    if isinstance(face_data, dict) and 'box' in face_data:
                        # face_data['box'] is (x1, y1, x2, y2), convert to (top, right, bottom, left)
                        x1, y1, x2, y2 = face_data['box']
                        face_location = (y1, x2, y2, x1)  # (top, right, bottom, left)
                        
                        # Check if this face has been recognized
                        identity_info = None
                        if face_data.get('recognized_name') and face_data.get('similarity', 0) > self._similarity_threshold:
                            identity_info = {
                                'name': self._face_names.get(face_data['recognized_name'], face_data['recognized_name'][:8]),
                                'confidence': face_data.get('similarity', 0) * 100,
                                'wallet_address': face_data['recognized_name']
                            }
                        
                        # Draw face box with wallet address tagging
                        self._draw_face_box(
                            output, 
                            face_location, 
                            identity_info=identity_info
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
                    'name': self._face_names.get(wallet_address, wallet_address[:8]),
                    'metadata': metadata
                })
            return faces

    def clear_enrolled_faces(self) -> bool:
        """Clear all enrolled faces"""
        try:
            with self._faces_lock:
                self._face_embeddings.clear()
                self._face_names.clear()
                self._face_metadata.clear()
            
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