import os
import logging

class Settings:
    # Base configuration
    DEBUG = False
    SYNTHETIC_FACE_TEST = False
    
    # API settings
    CAMERA_API_HOST = "0.0.0.0"
    CAMERA_API_PORT = 5000
    
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    @classmethod
    def setup(cls):
        """Set up the settings and validate configuration"""
        logger = logging.getLogger(__name__)
        logger.info("Initializing settings")
        
    @classmethod
    def load_env_file(cls, path):
        """Load environment variables from a file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value 