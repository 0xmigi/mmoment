#!/usr/bin/env python3
"""
FaceNet Test Script

This script tests the FaceNet implementation by:
1. Loading the FaceNet model
2. Capturing a frame from the camera
3. Detecting faces in the frame
4. Generating embeddings for the detected faces
5. Comparing embeddings and displaying results

This is a comprehensive test to ensure all parts of the facial recognition system work.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceNetTest")

# Import our modules
try:
    from services.face_detector import get_face_detector
    from services.facenet_model import get_facenet_model
    logger.info("Successfully imported face modules")
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    sys.exit(1)

def test_face_detection():
    """Test face detection functionality"""
    logger.info("Testing face detection...")
    
    face_detector = get_face_detector()
    logger.info(f"Face detector initialized with method: {face_detector.get_detection_method()}")
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Could not open camera!")
        return False
    
    # Capture a frame
    logger.info("Capturing frame from camera...")
    ret, frame = camera.read()
    if not ret or frame is None:
        logger.error("Failed to capture frame!")
        camera.release()
        return False
    
    # Detect faces
    logger.info("Detecting faces in frame...")
    faces = face_detector.detect_faces(frame)
    
    logger.info(f"Detected {len(faces)} faces in frame")
    if len(faces) == 0:
        logger.warning("No faces detected in frame!")
    
    # Extract face chips
    if len(faces) > 0:
        logger.info("Extracting face chips...")
        face_chips = face_detector.get_face_chips(frame, faces)
        logger.info(f"Extracted {len(face_chips)} face chips")
        
        # Save a sample face for verification
        if len(face_chips) > 0:
            sample_path = "test_face_detected.jpg"
            cv2.imwrite(sample_path, face_chips[0])
            logger.info(f"Saved sample face to {sample_path}")
    
    # Clean up
    camera.release()
    
    return len(faces) > 0

def test_facenet_model():
    """Test FaceNet model loading and embedding generation"""
    logger.info("Testing FaceNet model...")
    
    # Get FaceNet model
    facenet_model = get_facenet_model()
    if not hasattr(facenet_model, 'generate_embedding'):
        logger.error("FaceNet model does not have generate_embedding method!")
        return False
    
    # Create a test image if no faces detected previously
    test_image_path = "test_face_detected.jpg"
    if not os.path.exists(test_image_path):
        logger.info("No test face available, using a blank test image")
        # Create a blank test image
        test_image = np.ones((160, 160, 3), dtype=np.uint8) * 128
        cv2.imwrite(test_image_path, test_image)
    
    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        logger.error(f"Could not load test image from {test_image_path}")
        return False
    
    # Generate embedding
    logger.info("Generating embedding for test face...")
    start_time = time.time()
    embedding = facenet_model.generate_embedding(test_image)
    elapsed = time.time() - start_time
    
    if embedding is None:
        logger.error("Failed to generate embedding!")
        return False
    
    logger.info(f"Successfully generated embedding in {elapsed:.3f} seconds")
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(embedding)}")
    
    # Generate a second embedding to test comparison
    logger.info("Generating a second embedding for comparison...")
    embedding2 = facenet_model.generate_embedding(test_image)
    
    if embedding2 is None:
        logger.error("Failed to generate second embedding!")
        return False
    
    # Compare embeddings
    logger.info("Testing embedding comparison...")
    similarity = facenet_model.compute_similarity(embedding, embedding2)
    logger.info(f"Self-similarity: {similarity:.6f} (should be close to 1.0)")
    
    # Test with a slightly modified image
    logger.info("Testing with a modified image...")
    modified_image = test_image.copy()
    modified_image = cv2.GaussianBlur(modified_image, (5, 5), 0)
    embedding_modified = facenet_model.generate_embedding(modified_image)
    
    if embedding_modified is None:
        logger.error("Failed to generate embedding for modified image!")
        return False
    
    similarity_modified = facenet_model.compute_similarity(embedding, embedding_modified)
    logger.info(f"Modified image similarity: {similarity_modified:.6f} (should be less than 1.0)")
    
    return True

def test_face_service():
    """Test face service functionality (simplified version)"""
    logger.info("Testing full face service functionality...")
    
    # Create a test directory structure
    test_faces_dir = Path("test_faces")
    test_faces_dir.mkdir(exist_ok=True)
    
    try:
        # Import the face service
        from services.face_service import get_face_service
        
        # Get face service
        face_service = get_face_service()
        logger.info("Face service initialized")
        
        # Test face detection
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Could not open camera!")
            return False
        
        # Capture a frame
        ret, frame = camera.read()
        camera.release()
        
        if not ret or frame is None:
            logger.error("Failed to capture frame!")
            return False
        
        # Test face detection
        face_service._detect_faces(frame)
        with face_service._results_lock:
            detected_count = len(face_service._detected_faces)
        
        logger.info(f"Detected {detected_count} faces with face service")
        
        # Test face recognition if faces were detected
        if detected_count > 0:
            face_service._recognize_faces(frame)
            
            # Get recognition results
            faces_info = face_service.get_faces()
            logger.info(f"Recognition result: {faces_info['recognized_count']} faces recognized")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing face service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("Starting FaceNet test suite")
    
    # Test face detection
    detection_success = test_face_detection()
    logger.info(f"Face detection test {'PASSED' if detection_success else 'FAILED'}")
    
    # Test FaceNet model
    model_success = test_facenet_model()
    logger.info(f"FaceNet model test {'PASSED' if model_success else 'FAILED'}")
    
    # Test face service
    service_success = test_face_service()
    logger.info(f"Face service test {'PASSED' if service_success else 'FAILED'}")
    
    # Overall result
    if detection_success and model_success and service_success:
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.error("Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 