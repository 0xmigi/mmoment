"""
Face Detector Module

Provides optimized face detection using MTCNN or OpenCV for edge devices.
"""

import os
import cv2
import time
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceDetector")

# Try to import MTCNN
MTCNN_AVAILABLE = False
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    logger.info("MTCNN face detector loaded successfully")
except ImportError:
    logger.warning("MTCNN not available. Install with: pip install mtcnn")

class FaceDetector:
    """
    Face detector implementation using either MTCNN (preferred) or OpenCV Cascade.
    Optimized for performance on edge devices.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._detector = None
        self._face_cascade = None
        self._detection_method = "unknown"
        
        # Try to initialize MTCNN first
        if MTCNN_AVAILABLE:
            try:
                self._detector = MTCNN(min_face_size=60, scale_factor=0.709)
                self._detection_method = "mtcnn"
                logger.info("Using MTCNN for face detection")
            except Exception as e:
                logger.error(f"Error initializing MTCNN: {e}")
                self._detector = None
        
        # Fall back to OpenCV if MTCNN failed
        if self._detector is None:
            try:
                self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self._detection_method = "opencv"
                logger.info("Using OpenCV Cascade for face detection")
            except Exception as e:
                logger.error(f"Error initializing OpenCV Cascade: {e}")
    
    def detect_faces(self, frame: np.ndarray, confidence_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image (BGR format for OpenCV)
            confidence_threshold: Minimum confidence threshold (for MTCNN)
            
        Returns:
            List of dictionaries with face information:
            - 'box': [x, y, width, height]
            - 'confidence': detection confidence
            - 'keypoints': facial keypoints (for MTCNN)
        """
        start_time = time.time()
        
        if self._detection_method == "mtcnn":
            # MTCNN detector expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self._detector.detect_faces(rgb_frame)
            
            # Filter by confidence
            faces = [face for face in faces if face.get('confidence', 0) > confidence_threshold]
            
        elif self._detection_method == "opencv":
            # Use OpenCV's Cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            opencv_faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to same format as MTCNN
            faces = []
            for (x, y, w, h) in opencv_faces:
                face_dict = {
                    'box': [x, y, w, h],
                    'confidence': 1.0,  # OpenCV doesn't provide confidence
                    'keypoints': {}  # OpenCV doesn't provide keypoints
                }
                faces.append(face_dict)
        else:
            logger.error("No face detection method available")
            return []
        
        elapsed = time.time() - start_time
        logger.debug(f"Face detection completed in {elapsed:.3f} seconds, found {len(faces)} faces")
        
        return faces
    
    def get_face_chips(self, frame: np.ndarray, faces: List[Dict[str, Any]], 
                      size: Tuple[int, int] = (160, 160), margin: float = 0.3) -> List[np.ndarray]:
        """
        Extract aligned face chips from the frame based on detected faces.
        
        Args:
            frame: Input image (BGR format)
            faces: List of detected faces from detect_faces()
            size: Output size for face chips
            margin: Margin to add around faces (as a fraction of face size)
            
        Returns:
            List of face chip images
        """
        face_chips = []
        
        for face in faces:
            try:
                # Get face box
                x, y, w, h = face['box']
                
                # Add margin
                margin_x = int(w * margin)
                margin_y = int(h * margin)
                
                # Calculate coordinates with margin
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + w + margin_x)
                y2 = min(frame.shape[0], y + h + margin_y)
                
                # Extract face chip
                face_chip = frame[y1:y2, x1:x2]
                
                # Resize to required size
                face_chip = cv2.resize(face_chip, size)
                
                face_chips.append(face_chip)
                
            except Exception as e:
                logger.error(f"Error extracting face chip: {e}")
        
        return face_chips
    
    def get_detection_method(self) -> str:
        """Get the current detection method."""
        return self._detection_method

# Global function to get the detector instance
def get_face_detector() -> FaceDetector:
    """Get the singleton FaceDetector instance."""
    return FaceDetector() 