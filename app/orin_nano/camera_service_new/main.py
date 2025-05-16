#!/usr/bin/env python3
"""
Jetson Camera API - Main Entry Point

This is the main entry point for the lightweight, optimized camera service.
It initializes all services and starts the API server.
"""

import os
import time
import logging
import argparse
import threading
from pathlib import Path
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser('~/mmoment/app/orin_nano/camera_service_new/logs/camera_service.log'))
    ]
)

logger = logging.getLogger('CameraService')

# Create required directories
def create_directories():
    """Create required directories for the camera service"""
    base_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service_new')
    
    dirs = [
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'photos'),
        os.path.join(base_dir, 'videos'),
        os.path.join(base_dir, 'faces')
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")

# Initialize all services
def init_services():
    """Initialize all services and return them in a dict"""
    from services.buffer_service import get_buffer_service
    from services.face_service import get_face_service
    from services.face_detector import get_face_detector
    from services.gesture_service import get_gesture_service
    from services.capture_service import get_capture_service
    from services.session_service import get_session_service
    
    # Try to import Solana integration
    solana_integration_service = None
    try:
        from services.solana_integration import get_solana_integration_service
        solana_integration_service = get_solana_integration_service()
        logger.info("Solana integration service initialized")
    except ImportError:
        logger.warning("Solana integration service not available")
    
    logger.info("Initializing services...")
    
    # Get service instances (they are singletons)
    buffer_service = get_buffer_service()
    face_service = get_face_service()
    face_detector = get_face_detector()
    gesture_service = get_gesture_service()
    capture_service = get_capture_service()
    session_service = get_session_service()
    
    # Start the buffer service first (it's the source of truth)
    logger.info("Starting buffer service...")
    if not buffer_service.start():
        logger.error("Failed to start buffer service")
        return None
    
    # Allow buffer service to initialize
    time.sleep(1)
    
    # Start the face service
    logger.info("Starting face service...")
    face_service.start(buffer_service)
    
    # Start the gesture service if MediaPipe is available
    logger.info("Starting gesture service...")
    gesture_service.start(buffer_service)
    
    # Inject services into the buffer service for processing
    logger.info("Injecting services into buffer service...")
    buffer_service.inject_services(
        face_service=face_service,
        gesture_service=gesture_service
    )
    
    # Build service dictionary
    services = {
        'buffer': buffer_service,
        'face': face_service,
        'face_detector': face_detector,
        'gesture': gesture_service,
        'capture': capture_service,
        'session': session_service
    }
    
    # Add Solana integration if available
    if solana_integration_service:
        services['solana'] = solana_integration_service
        solana_status = solana_integration_service.get_health_status()
        logger.info(f"Solana middleware status: {solana_status['status']}, Camera PDA: {solana_status['camera_pda']}")
    
    return services

# Create Flask application
def create_app(services):
    """Create and configure the Flask application"""
    app = Flask(__name__, static_folder='static')
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['JSON_SORT_KEYS'] = False
    
    # Add services to app context
    app.config['SERVICES'] = services
    
    # Register routes
    from routes import register_routes
    register_routes(app)
    
    return app

# Main entry point
def main():
    """Main entry point for the camera service"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Jetson Camera API Server')
    parser.add_argument('--port', type=int, default=5003, help='Port to run the server on (default: 5003)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Create required directories
    create_directories()
    
    # Initialize services
    services = init_services()
    
    if not services:
        logger.error("Failed to initialize services. Exiting.")
        return 1
    
    # Create Flask app
    app = create_app(services)
    
    # Run the Flask app
    host = os.environ.get('FLASK_HOST', args.host)
    port = int(os.environ.get('FLASK_PORT', args.port))
    debug = args.debug
    
    logger.info(f"Starting Jetson Camera API on {host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return 1
    finally:
        # Shut down services
        logger.info("Stopping services...")
        if 'gesture' in services:
            services['gesture'].stop()
        if 'face' in services:
            services['face'].stop()
        if 'session' in services:
            services['session'].stop()
        if 'buffer' in services:
            services['buffer'].stop()
    
    return 0

if __name__ == "__main__":
    exit(main()) 