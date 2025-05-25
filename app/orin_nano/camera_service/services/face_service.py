"""
Face Recognition Service

Provides face detection and recognition functionality.
Works with the buffer service to process frames without modifying the source.
Uses OpenCV for face detection and a simple distance-based approach for face recognition.
"""

import os
import cv2
import logging
import numpy as np
import threading
import time
import json
import base64
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceService")

# Import face detector
from services.face_detector import get_face_detector

# Initialize face detector
face_detector = None

try:
    face_detector = get_face_detector()
    logger.info("Face detector loaded successfully")
except Exception as e:
    logger.error(f"Error initializing face detector: {str(e)}")
    logger.error("Face detection will not be available")

# Try to import Solana integration
try:
    from services.solana_integration import get_solana_integration_service
    SOLANA_INTEGRATION_AVAILABLE = True
    logger.info("Solana integration service imported successfully")
except ImportError:
    SOLANA_INTEGRATION_AVAILABLE = False
    logger.warning("Solana integration service not available, will use local storage only")

# Simple face recognizer using feature-based distance
class SimpleFaceRecognizer:
    """Simple face recognizer using feature-based distance"""
    
    def __init__(self):
        self.face_features = {}  # wallet_address -> face features (avg pixel values)
        self.is_trained = False
        
    def get_face_features(self, face_img):
        """Extract simple features from face image"""
        # Resize to standard size
        resized = cv2.resize(face_img, (50, 50))
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        # Extract features: pixel values flattened
        features = gray.flatten().astype(np.float32)
        
        # Normalize features
        if np.max(features) > 0:
            features = features / np.max(features)
            
        return features
        
    def train(self, face_chips, wallet_addresses):
        """Train the recognizer with face images"""
        if not face_chips or len(face_chips) != len(wallet_addresses):
            return False
            
        # Extract features for each face
        for i, face_chip in enumerate(face_chips):
            wallet = wallet_addresses[i]
            features = self.get_face_features(face_chip)
            self.face_features[wallet] = features
            
        self.is_trained = len(self.face_features) > 0
        return self.is_trained
        
    def predict(self, face_img):
        """Recognize a face and return wallet address and confidence"""
        if not self.is_trained or not self.face_features:
            return None, 0
            
        # Get features from face
        test_features = self.get_face_features(face_img)
        
        # Compare with all enrolled faces
        best_match = None
        best_similarity = -1
        
        for wallet, features in self.face_features.items():
            # Calculate distance (Euclidean)
            distance = np.sqrt(np.sum((test_features - features) ** 2))
            
            # Convert distance to similarity (0-1, higher is better)
            similarity = max(0, 1.0 - (distance / 10.0))  # Scale distance to similarity
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = wallet
                
        return best_match, best_similarity
        
    def save(self, file_path):
        """Save the face features to disk"""
        if not self.is_trained:
            return False
            
        # Convert numpy arrays to lists for serialization
        serializable_features = {}
        for wallet, features in self.face_features.items():
            serializable_features[wallet] = features.tolist()
            
        with open(file_path, 'w') as f:
            json.dump(serializable_features, f)
            
        return True
        
    def load(self, file_path):
        """Load face features from disk"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    serializable_features = json.load(f)
                    
                # Convert lists back to numpy arrays
                for wallet, features in serializable_features.items():
                    self.face_features[wallet] = np.array(features, dtype=np.float32)
                    
                self.is_trained = len(self.face_features) > 0
                return True
                
        except Exception as e:
            logger.error(f"Error loading face features: {e}")
            
        return False

# Global instance of the simple recognizer
simple_recognizer = SimpleFaceRecognizer()

class FaceService:
    """
    Service for face detection and recognition.
    This is a processor that consumes frames from the buffer service.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        self._processing_enabled = False
        self._visualization_enabled = True  # Enable visualization by default
        self._detection_enabled = True  # Face detection is enabled by default
        self._boxes_enabled = False  # Face boxes are disabled by default (turn on after login)
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        # Face detection settings
        self._detection_interval = 0.03  # Faster detection (33fps) for smoother tracking
        self._recognition_interval = 0.5  # Faster recognition (2fps) for better responsiveness
        self._last_detection_time = 0
        self._last_recognition_time = 0
        self._similarity_threshold = 0.5  # Threshold for face recognition
        
        # Face detection results
        self._results_lock = threading.Lock()
        self._detected_faces = []  # List of face dictionaries with 'box', 'confidence', etc.
        self._recognized_faces = {}  # Dict of name -> (top, right, bottom, left, confidence)
        
        # Enrolled faces storage
        self._faces_lock = threading.Lock()
        self._enrolled_faces = {}  # Dict of name -> face image (for OpenCV)
        self._faces_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/faces")
        Path(self._faces_dir).mkdir(parents=True, exist_ok=True)
        
        # Recognition model path
        self._model_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/models/simple_model")
        Path(self._model_dir).mkdir(parents=True, exist_ok=True)
        self._model_path = os.path.join(self._model_dir, "simple_face_features.json")
        
        # Store buffer service reference
        self._buffer_service = None
        
        # Initialize face detector
        self._face_detector = face_detector
        
        # Load enrolled faces if available
        self._load_enrolled_faces()
        
        logger.info("FaceService initialized with simple face recognition")
        
    def start(self, buffer_service) -> bool:
        """
        Start the face processing thread.
        """
        with self._lock:
            if self._processing_enabled:
                logger.info("FaceService already running")
                return True
                
            if self._face_detector is None:
                logger.warning("Face detector not available, face detection will not work")
                return False
            
            # Store reference to buffer service
            self._buffer_service = buffer_service
            
            # Make sure enrolled faces are loaded
            logger.info("[SERVICE] Loading enrolled faces at service start")
            self._load_enrolled_faces()
            with self._faces_lock:
                logger.info(f"[SERVICE] Loaded {len(self._enrolled_faces)} enrolled faces: {list(self._enrolled_faces.keys())}")
            
            # Reset stop event
            self._stop_event.clear()
            
            # Start processing thread
            self._processing_enabled = True
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="FaceProcessingThread"
            )
            self._processing_thread.start()
            
            logger.info("FaceService started")
            return True
    
    def stop(self) -> None:
        """
        Stop the face processing thread.
        """
        logger.info("Stopping FaceService")
        
        # Signal the processing thread to stop
        self._stop_event.set()
        
        # Wait for the processing thread to stop
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        with self._lock:
            self._processing_enabled = False
            
        logger.info("FaceService stopped")
    
    def _processing_loop(self) -> None:
        """
        Main processing loop that runs face detection and recognition.
        """
        logger.info("Face processing loop started")
        
        try:
            while not self._stop_event.is_set():
                # Get the latest frame from the buffer
                frame, timestamp = self._buffer_service.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                
                # Run face detection at regular intervals
                if current_time - self._last_detection_time >= self._detection_interval:
                    self._detect_faces(frame)
                    self._last_detection_time = current_time
                
                # Run face recognition at regular intervals (less frequently)
                if current_time - self._last_recognition_time >= self._recognition_interval:
                    if len(self._detected_faces) > 0:
                        self._recognize_faces(frame)
                    self._last_recognition_time = current_time
                
                # Sleep a minimal amount to avoid consuming too much CPU
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in face processing loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Face processing loop stopped")
            with self._lock:
                self._processing_enabled = False
    
    def _detect_faces(self, frame: np.ndarray) -> None:
        """
        Detect faces in the given frame using the face detector.
        Updates self._detected_faces with the results.
        """
        # Skip if detection is disabled
        if not self._detection_enabled or self._face_detector is None:
            return
            
        try:
            # Use face detector
            faces = self._face_detector.detect_faces(frame)
            
            # Update detected faces
            with self._results_lock:
                self._detected_faces = faces
                
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            with self._results_lock:
                self._detected_faces = []
    
    def _recognize_faces(self, frame: np.ndarray) -> None:
        """
        Recognize faces in the given frame using simple face recognition.
        Updates self._recognized_faces with the results.
        """
        if len(self._detected_faces) == 0 or not simple_recognizer.is_trained:
            return
            
        try:
            # Get face chips from detector
            face_chips = self._face_detector.get_face_chips(frame, self._detected_faces)
            
            if not face_chips:
                return
                
            # Clear previous recognitions
            with self._results_lock:
                self._recognized_faces = {}
                
                # Process each face
                for i, face_chip in enumerate(face_chips):
                    if i >= len(self._detected_faces):
                        continue
                        
                    # Use simple recognizer to identify face
                    wallet_address, similarity = simple_recognizer.predict(face_chip)
                    
                    if wallet_address and similarity > self._similarity_threshold:
                        # Get face box
                        x, y, w, h = self._detected_faces[i]['box']
                        
                        # Store recognition result
                        self._recognized_faces[wallet_address] = (y, x+w, y+h, x, similarity)
                        logger.debug(f"Recognized face: {wallet_address} with confidence {similarity:.2f}")
                    
        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with face detection and recognition visualization.
        """
        if frame is None or not self._visualization_enabled:
            return frame
            
        try:
            # Create a copy of the frame to avoid modifying the original
            output = frame.copy()
            
            # Get current detection and recognition results
            with self._results_lock:
                detected_faces = self._detected_faces.copy()
                recognized_faces = self._recognized_faces.copy()
            
            # Only draw boxes if enabled
            if self._boxes_enabled:
                # Draw boxes around detected faces (but not recognized ones)
                # These are the RED boxes (unrecognized faces)
                if detected_faces:
                    # Create a set of recognized face boxes to avoid drawing red boxes for recognized faces
                    recognized_boxes = set()
                    for (top, right, bottom, left, _) in recognized_faces.values():
                        recognized_boxes.add((int(left), int(top), int(right-left), int(bottom-top)))
                    
                    for face in detected_faces:
                        # Get face box
                        x, y, w, h = face['box']
                        
                        # Skip if this face is already recognized (to avoid double-boxing)
                        if (x, y, w, h) in recognized_boxes:
                            continue
                        
                        # Draw rectangle around face (RED for unrecognized)
                        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
                        # Add confidence if available
                        confidence = face.get('confidence', 0)
                        if confidence > 0:
                            confidence_text = f"{confidence:.2f}"
                            cv2.putText(output, confidence_text, (x, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw GREEN boxes for recognized faces
                for name, (top, right, bottom, left, confidence) in recognized_faces.items():
                    # Draw rectangle in green for recognized faces
                    cv2.rectangle(output, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    
                    # Add name label
                    display_name = name
                    if len(display_name) > 12:
                        display_name = name[:10] + ".."
                        
                    label = f"{display_name} ({confidence:.2f})"
                    cv2.putText(output, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            return output
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def enroll_face(self, frame: np.ndarray, wallet_address: str) -> Dict:
        """
        Enroll a face from the given frame using simple face recognition.
        """
        logger.info(f"[ENROLL-SVC] Starting face enrollment for wallet: {wallet_address}")
        
        if self._face_detector is None:
            logger.error("[ENROLL-SVC] Face detector not available, cannot enroll face")
            return {
                "success": False,
                "error": "Face detection is not available. Please ensure the service is properly installed."
            }
            
        if frame is None:
            logger.error("[ENROLL-SVC] Cannot enroll face, frame is None")
            return {
                "success": False,
                "error": "No camera frame available. Please ensure your camera is connected and working properly."
            }
            
        try:
            # First detect faces in the frame
            logger.info(f"[ENROLL-SVC] Detecting faces for wallet: {wallet_address}")
            faces = self._face_detector.detect_faces(frame)
            
            if not faces:
                logger.warning(f"[ENROLL-SVC] No face detected for enrollment for wallet: {wallet_address}")
                return {
                    "success": False,
                    "error": "No face detected. Please ensure your face is clearly visible, well-lit, and centered in the frame."
                }
                
            if len(faces) > 1:
                logger.warning(f"[ENROLL-SVC] Multiple faces ({len(faces)}) detected during enrollment for wallet: {wallet_address}")
                return {
                    "success": False,
                    "error": "Multiple faces detected. Please ensure only your face is visible in the camera frame."
                }
                
            logger.info(f"[ENROLL-SVC] Detected {len(faces)} faces for wallet: {wallet_address}")
                
            # Get face chips for detected faces
            face_chips = self._face_detector.get_face_chips(frame, faces)
            
            if not face_chips:
                logger.warning(f"[ENROLL-SVC] Failed to extract face chip for wallet: {wallet_address}")
                return {
                    "success": False,
                    "error": "Failed to extract face details. Please try again with better lighting and position yourself directly in front of the camera."
                }
                
            face_chip = face_chips[0]
            logger.info(f"[ENROLL-SVC] Successfully extracted face for wallet: {wallet_address}")
            
            # Save face image for face recognition
            with self._faces_lock:
                self._enrolled_faces[wallet_address] = face_chip
                
            # Save face image to disk
            face_image_path = os.path.join(self._faces_dir, f"{wallet_address}.jpg")
            cv2.imwrite(face_image_path, face_chip)
            logger.info(f"[ENROLL-SVC] Saved face image to disk: {face_image_path}")
            
            # Train the recognizer with all enrolled faces
            self._train_recognizer()
            
            # Save the trained model
            if simple_recognizer.is_trained:
                simple_recognizer.save(self._model_path)
                logger.info(f"[ENROLL-SVC] Saved trained model to {self._model_path}")
            
            logger.info(f"[ENROLL-SVC] Face enrolled successfully for wallet: {wallet_address}")
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "encrypted": False,
                "nft_verified": False
            }
            
        except Exception as e:
            logger.error(f"[ENROLL-SVC] Error enrolling face for wallet {wallet_address}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Face enrollment failed: {str(e)}. Please try again or contact support if the issue persists."
            }
    
    def _train_recognizer(self):
        """Train the recognizer with all enrolled faces"""
        with self._faces_lock:
            if not self._enrolled_faces:
                logger.warning("[TRAIN] No enrolled faces to train with")
                return False
                
            # Collect face chips and wallet addresses
            face_chips = []
            wallet_addresses = []
            
            for wallet_address, face_chip in self._enrolled_faces.items():
                face_chips.append(face_chip)
                wallet_addresses.append(wallet_address)
                
            # Train the simple recognizer
            if simple_recognizer.train(face_chips, wallet_addresses):
                logger.info(f"[TRAIN] Successfully trained recognizer with {len(face_chips)} faces")
                return True
            else:
                logger.warning("[TRAIN] Failed to train recognizer")
                return False
    
    def _load_enrolled_faces(self) -> None:
        """
        Load enrolled faces from disk.
        """
        try:
            # Get all .jpg files in the faces directory
            face_files = list(Path(self._faces_dir).glob("*.jpg"))
            logger.info(f"[LOAD-FACES] Found {len(face_files)} face image files in {self._faces_dir}")
            
            loaded_count = 0
            with self._faces_lock:
                # Don't clear the enrolled faces dictionary if already populated
                if len(self._enrolled_faces) > 0:
                    logger.info(f"[LOAD-FACES] Already have {len(self._enrolled_faces)} enrolled faces, keeping them")
                else:
                    self._enrolled_faces = {}
                
                for face_file in face_files:
                    try:
                        # Extract wallet address from filename
                        wallet_address = face_file.stem
                        logger.info(f"[LOAD-FACES] Loading face for {wallet_address}")
                        
                        # Check if wallet already loaded
                        if wallet_address in self._enrolled_faces:
                            logger.info(f"[LOAD-FACES] Face for {wallet_address} already loaded, skipping")
                            loaded_count += 1
                            continue
                        
                        # Load the face image
                        face_image = cv2.imread(str(face_file))
                        if face_image is not None:
                            self._enrolled_faces[wallet_address] = face_image
                            loaded_count += 1
                            logger.info(f"[LOAD-FACES] Successfully loaded face image for {wallet_address}")
                        else:
                            logger.warning(f"[LOAD-FACES] Failed to load face image for {wallet_address}")
                        
                    except Exception as e:
                        logger.error(f"[LOAD-FACES] Error processing face {face_file}: {e}")
                        
            logger.info(f"[LOAD-FACES] Loaded {loaded_count} enrolled faces out of {len(face_files)} available files")
            logger.info(f"[LOAD-FACES] Enrolled faces now: {list(self._enrolled_faces.keys())}")
            
            # Load or train the recognizer
            if loaded_count > 0:
                # Try to load existing model first
                if os.path.exists(self._model_path) and simple_recognizer.load(self._model_path):
                    logger.info(f"[LOAD-FACES] Loaded existing recognition model from {self._model_path}")
                else:
                    # Train a new model with loaded faces
                    logger.info("[LOAD-FACES] Training new recognition model with loaded faces")
                    self._train_recognizer()
                    if simple_recognizer.is_trained:
                        simple_recognizer.save(self._model_path)
                        logger.info(f"[LOAD-FACES] Saved newly trained model to {self._model_path}")
            
        except Exception as e:
            logger.error(f"[LOAD-FACES] Error loading enrolled faces: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_faces(self) -> Dict:
        """
        Get current face detection and recognition results.
        """
        with self._results_lock:
            # Convert detected faces to a simpler format for API compatibility
            simplified_detections = []
            for face in self._detected_faces:
                x, y, w, h = face['box']
                # Convert to (top, right, bottom, left) format for compatibility
                simplified_detections.append((y, x+w, y+h, x))
            
            # Convert recognized faces to ensure JSON serialization works
            serializable_recognized_faces = {}
            for name, face_data in self._recognized_faces.items():
                # Ensure all values are plain Python types, not numpy types
                serializable_recognized_faces[name] = tuple(float(v) if isinstance(v, np.number) else v for v in face_data)
                
            return {
                "detected_count": len(self._detected_faces),
                "detected_faces": simplified_detections,
                "recognized_count": len(self._recognized_faces),
                "recognized_faces": serializable_recognized_faces
            }
    
    def get_enrolled_faces(self) -> List[str]:
        """
        Get a list of enrolled face names.
        """
        with self._faces_lock:
            return list(self._enrolled_faces.keys())
            
    def clear_enrolled_faces(self) -> bool:
        """
        Clear all enrolled faces.
        """
        try:
            with self._faces_lock:
                self._enrolled_faces = {}
                
            # Delete all face files
            face_files = list(Path(self._faces_dir).glob("*.*"))
            for face_file in face_files:
                try:
                    face_file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting face file {face_file}: {e}")
                    
            # Also delete model files
            if os.path.exists(self._model_path):
                os.remove(self._model_path)
                
            # Reset the recognizer
            global simple_recognizer
            simple_recognizer = SimpleFaceRecognizer()
                
            logger.info("All enrolled faces cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing enrolled faces: {e}")
            return False
            
    def enable_visualization(self, enabled: bool) -> None:
        """
        Enable or disable face visualization.
        """
        self._visualization_enabled = enabled
        logger.info(f"Face visualization {'enabled' if enabled else 'disabled'}")
        
    def enable_detection(self, enabled: bool) -> None:
        """
        Enable or disable face detection.
        Note: Face detection is now always enabled and this method is kept for compatibility.
        """
        logger.info(f"Face detection enable request: {enabled} (always on)")
        # Face detection is always enabled, ignore the request
        self._detection_enabled = True
        
    def enable_boxes(self, enabled: bool) -> None:
        """
        Enable or disable face box visualization.
        """
        self._boxes_enabled = enabled
        logger.info(f"Face boxes {'enabled' if enabled else 'disabled'}")
        
        # Make sure visualization is enabled if boxes are enabled
        if enabled and not self._visualization_enabled:
            self._visualization_enabled = True
            logger.info("Face visualization automatically enabled to show boxes")
        
    def get_settings(self) -> Dict:
        """
        Get current face detection settings.
        """
        return {
            "detection_enabled": self._detection_enabled,
            "visualization_enabled": self._visualization_enabled,
            "boxes_enabled": self._boxes_enabled,
            "enrolled_faces_count": len(self._enrolled_faces),
            "facenet_available": False,  # Always False now
            "detection_method": self._face_detector.get_detection_method() if self._face_detector else "none",
            "similarity_threshold": self._similarity_threshold
        }

# Global function to get the face service instance
def get_face_service() -> FaceService:
    """Get the singleton FaceService instance."""
    return FaceService() 