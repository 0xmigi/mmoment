"""
Dummy FaceNet Model

This is a placeholder that doesn't use any resources.
FaceNet has been disabled and replaced with a lightweight OpenCV solution.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceNetModel")

class FaceNetModel:
    """
    Dummy FaceNet model that does nothing.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceNetModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        logger.info("FaceNet is disabled. Using lightweight OpenCV recognition instead.")
        
    def _load_model(self):
        """Dummy method"""
        return False
    
    def generate_embedding(self, face_img):
        """Dummy method"""
        return None
    
    def compute_similarity(self, embedding1, embedding2):
        """Dummy method"""
        return 0.0

# Global function to get the model instance
def get_facenet_model():
    """Returns a dummy model instance that does nothing"""
    logger.warning("FaceNet is disabled. Using lightweight OpenCV recognition instead.")
    return FaceNetModel() 