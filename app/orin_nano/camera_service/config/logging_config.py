import logging
import os
from .settings import Settings
import sys

def configure_logging():
    """Configure logging for the application"""
    # Ensure logs directory exists
    os.makedirs(Settings.LOGS_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Settings.LOGS_DIR, 'app.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set different log levels for some modules
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Set our app modules to INFO
    logging.getLogger('camera_service').setLevel(logging.INFO)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {Settings.APP_NAME} v{Settings.VERSION}")
    logger.info(f"Logs directory: {Settings.LOGS_DIR}")
    logger.info(f"Videos directory: {Settings.VIDEOS_DIR}")

def reconfigure_logging_for_absl_compat():
    """Re-apply basic logging config, potentially after absl/mediapipe import."""
    logger = logging.getLogger() # Get root logger
    # Clear existing handlers from root logger to avoid duplicate messages if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Re-apply basicConfig (or a more targeted handler setup)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Settings.LOGS_DIR, 'app.log')),
            logging.StreamHandler()
        ],
        force=True # Add force=True to override existing root logger config if any
    )
    logging.getLogger('camera_service').setLevel(logging.INFO)
    # logging.getLogger('absl').setLevel(logging.WARNING) # Optionally silence absl more
    print("LOGGING_CONFIG.PY: Reconfigured logging for absl compat.", file=sys.stderr, flush=True) 