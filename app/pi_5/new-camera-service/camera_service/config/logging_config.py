import logging

def configure_logging():
    # Configure the root logger
    logging.basicConfig(level=logging.INFO)
    
    # Set Picamera2 logger to WARNING to reduce noise
    logging.getLogger('picamera2.picamera2').setLevel(logging.WARNING)