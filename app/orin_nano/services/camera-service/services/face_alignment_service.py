"""
Face Alignment Service - Uses InsightFace's battle-tested norm_crop for face alignment.

This ensures embeddings from phone selfies match embeddings from camera stream
by using the exact same alignment algorithm that ArcFace was trained with.
"""

import numpy as np
import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Import InsightFace's alignment function
# This uses skimage.transform.SimilarityTransform (Umeyama algorithm)
try:
    from insightface.utils.face_align import norm_crop
    INSIGHTFACE_AVAILABLE = True
    logger.info("InsightFace face_align module loaded successfully")
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning(f"InsightFace face_align not available: {e}")
    norm_crop = None


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> Optional[np.ndarray]:
    """
    Align face using InsightFace's norm_crop.

    This produces a 112x112 aligned face image that is compatible with ArcFace embeddings.
    Uses the standard 5-point landmark alignment (Umeyama similarity transform).

    Args:
        image_bgr: BGR image (H, W, 3) - full frame or crop containing the face
        landmarks: 5 facial landmarks as [[x,y], ...] in order:
                   [left_eye, right_eye, nose, left_mouth, right_mouth]
        image_size: Output size (default 112 for ArcFace)

    Returns:
        112x112 BGR aligned face image, or None if alignment fails
    """
    if not INSIGHTFACE_AVAILABLE:
        logger.error("InsightFace not available - cannot align face")
        return None

    if image_bgr is None or len(image_bgr.shape) != 3:
        logger.error("Invalid input image")
        return None

    # Ensure landmarks are in correct format: (5, 2) float32 array
    try:
        landmarks = np.array(landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2):
            # Try to reshape if flattened
            if landmarks.size == 10:
                landmarks = landmarks.reshape(5, 2)
            else:
                logger.error(f"Invalid landmarks shape: {landmarks.shape}, expected (5, 2)")
                return None
    except Exception as e:
        logger.error(f"Failed to convert landmarks: {e}")
        return None

    # Validate landmarks are within image bounds
    h, w = image_bgr.shape[:2]
    if np.any(landmarks < 0) or np.any(landmarks[:, 0] >= w) or np.any(landmarks[:, 1] >= h):
        logger.warning(f"Landmarks partially outside image bounds ({w}x{h}): {landmarks}")
        # Clamp to valid range - norm_crop handles edge cases
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, w - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, h - 1)

    try:
        # norm_crop(img, landmark, image_size=112, mode='arcface')
        # Uses skimage.transform.SimilarityTransform to compute optimal
        # rotation, scale, and translation to align face to standard template
        aligned = norm_crop(image_bgr, landmarks, image_size=image_size)

        if aligned is None or aligned.shape != (image_size, image_size, 3):
            logger.error(f"norm_crop returned invalid result: {aligned.shape if aligned is not None else None}")
            return None

        return aligned

    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        return None


def align_face_from_detection(image_bgr: np.ndarray, face_dict: dict, image_size: int = 112) -> Optional[np.ndarray]:
    """
    Convenience function to align face from a detection result dict.

    Args:
        image_bgr: BGR image (H, W, 3)
        face_dict: Detection result with 'landmarks' key containing 5 points
        image_size: Output size (default 112)

    Returns:
        112x112 BGR aligned face image, or None if alignment fails
    """
    landmarks = face_dict.get('landmarks')
    if landmarks is None:
        logger.error("No landmarks in face detection result")
        return None

    return align_face(image_bgr, landmarks, image_size)


def is_available() -> bool:
    """Check if the alignment service is available."""
    return INSIGHTFACE_AVAILABLE
