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

# Use environment-configured RTP ports from docker-compose (respects external configuration)
# Only set defaults if not already configured - use range that includes our target ports
if 'AIORTC_RTP_MIN_PORT' not in os.environ:
    os.environ['AIORTC_RTP_MIN_PORT'] = '10000'
if 'AIORTC_RTP_MAX_PORT' not in os.environ:
    os.environ['AIORTC_RTP_MAX_PORT'] = '10100'

logger = logging.getLogger('CameraService')
logger.info(f"Using RTP port range: {os.environ.get('AIORTC_RTP_MIN_PORT')}-{os.environ.get('AIORTC_RTP_MAX_PORT')}")

from flask import Flask, jsonify
from flask_cors import CORS

# Native Mode - C++ TensorRT server runs inside this container (started by entrypoint.sh)
print("=" * 60)
print("  NATIVE MODE - C++ TensorRT inference (in-container)")
print("  Camera capture + YOLOv8 + InsightFace handled by native server")
print("  Python handles WebRTC streaming + identity matching")
print("=" * 60)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/camera_service.log')
    ]
)

# Create required directories
def create_directories():
    """Create required directories for the camera service"""
    base_dir = '/app'  # Use the Docker volume mount path
    
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
    # Native mode - use NativeBufferService for frames from in-container C++ server
    from services.native_buffer_service import get_native_buffer_service as get_buffer_service
    logger.info("Using NativeBufferService - frames from native C++ server")

    from services.capture_service import get_capture_service
    from services.session_service import get_session_service
    from services.webrtc_service import get_webrtc_service
    from services.whip_publisher import WHIPPublisher, CleanVideoTrack, AnnotatedVideoTrack

    logger.info("Initializing services...")

    # Get service instances (they are singletons)
    buffer_service = get_buffer_service()
    capture_service = get_capture_service()
    session_service = get_session_service()
    webrtc_service = get_webrtc_service()

    # Start the buffer service first (connects to native server)
    logger.info("Starting buffer service...")
    if not buffer_service.start():
        logger.error("Failed to start buffer service")
        return None

    # Allow buffer service to initialize
    time.sleep(1)

    # Initialize WebRTC service with buffer service
    logger.info("🚀 Initializing WebRTC service...")
    logger.info(f"🔧 Setting buffer service for WebRTC: {buffer_service}")
    webrtc_service.set_buffer_service(buffer_service)
    
    logger.info("🚀 Starting WebRTC service...")
    webrtc_start_result = webrtc_service.start()
    if webrtc_start_result:
        logger.info("✅ WebRTC service started successfully - ready for sub-second latency streaming")
    else:
        logger.error("❌ CRITICAL: WebRTC service failed to start!")
        logger.error("❌ This will prevent real-time video streaming functionality")
        # Don't exit - let other services continue, but log the failure
    
    # Give WebRTC service time to initialize async components
    time.sleep(2)

    # Initialize DUAL WHIP Publishers for remote streaming via MediaMTX
    # One for clean stream (no CV annotations), one for annotated stream (with CV overlays)
    whip_publisher_clean = None
    whip_publisher_annotated = None
    whip_enabled = os.environ.get('WHIP_ENABLED', 'true').lower() == 'true'
    # Load camera PDA from device config (never use hardcoded fallbacks)
    camera_pda = os.environ.get('CAMERA_PDA')
    if not camera_pda:
        try:
            import json
            config_path = '/app/config/device_config.json'
            with open(config_path, 'r') as f:
                device_config = json.load(f)
                camera_pda = device_config.get('camera_pda')
                logger.info(f"📋 Loaded camera_pda from device config: {camera_pda}")
        except Exception as e:
            logger.error(f"❌ Failed to load camera_pda from config: {e}")
            camera_pda = None

    if not camera_pda:
        logger.error("❌ No CAMERA_PDA configured - WHIP streaming disabled")
        whip_enabled = False

    if whip_enabled:
        logger.info("📡 Initializing DUAL WHIP publishers for remote streaming...")

        # Clean stream publisher (default, no CV annotations)
        try:
            whip_publisher_clean = WHIPPublisher(
                stream_name=camera_pda,  # Default stream name
                video_track_class=CleanVideoTrack,
                fps=15  # Reduced to 15 to decrease CPU load (no NVENC on Orin Nano)
            )
            whip_publisher_clean.set_buffer_service(buffer_service)
            if whip_publisher_clean.start():
                logger.info(f"✅ WHIP [clean] started - stream at: {whip_publisher_clean.whep_url}")
            else:
                logger.warning("⚠️ WHIP [clean] failed to start")
                whip_publisher_clean = None
        except Exception as e:
            logger.error(f"❌ WHIP [clean] initialization failed: {e}")
            whip_publisher_clean = None

        # Annotated stream publisher (with CV overlays)
        try:
            whip_publisher_annotated = WHIPPublisher(
                stream_name=f"{camera_pda}-annotated",  # Annotated stream suffix
                video_track_class=AnnotatedVideoTrack,
                fps=15  # Reduced to 15 to decrease CPU load (no NVENC on Orin Nano)
            )
            whip_publisher_annotated.set_buffer_service(buffer_service)
            if whip_publisher_annotated.start():
                logger.info(f"✅ WHIP [annotated] started - stream at: {whip_publisher_annotated.whep_url}")
            else:
                logger.warning("⚠️ WHIP [annotated] failed to start")
                whip_publisher_annotated = None
        except Exception as e:
            logger.error(f"❌ WHIP [annotated] initialization failed: {e}")
            whip_publisher_annotated = None
    else:
        logger.info("📡 WHIP publishers disabled via WHIP_ENABLED=false")

    # Initialize Blockchain Session Sync (for recognition token loading)
    from services.blockchain_session_sync import get_blockchain_session_sync, reset_blockchain_session_sync
    reset_blockchain_session_sync()  # Ensure fresh instance with current environment variables
    blockchain_sync = get_blockchain_session_sync()
    blockchain_sync.set_services(session_service, None)  # No face_service in native mode
    blockchain_sync.start()
    logger.info("🔗 Blockchain session sync initialized - camera will auto-enable for on-chain check-ins")

    # Initialize Device Registration Service (for QR-based setup flow)
    from services.device_registration import get_device_registration_service
    device_registration = get_device_registration_service()
    logger.info("📱 Device registration service initialized - ready for QR-based device setup")

    # Initialize App Manager for CV apps (push-up tracker, etc.)
    from services.app_manager import get_app_manager
    app_manager = get_app_manager()
    logger.info("🎮 App Manager initialized - CV apps ready")

    # Inject app_manager into buffer service so apps receive frame data
    buffer_service.inject_services(app_manager=app_manager)
    logger.info("🔗 App Manager injected into buffer service")

    # Build service dictionary (native mode - no legacy CV services)
    services = {
        'buffer': buffer_service,
        'capture': capture_service,
        'session': session_service,
        'webrtc': webrtc_service,
        'whip': whip_publisher_clean,  # Default WHIP is clean stream
        'whip_clean': whip_publisher_clean,
        'whip_annotated': whip_publisher_annotated,
        'app_manager': app_manager
    }

    return services

# Create Flask application
def create_app(services):
    """Create and configure the Flask application"""
    app = Flask(__name__, static_folder='static')
    
    # Configure CORS for frontend integration
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:5173", 
                "http://localhost:3000", 
                "https://mmoment.xyz", 
                "http://localhost:5002",
                "https://jetson.mmoment.xyz",
                "http://jetson.mmoment.xyz",
                "https://middleware.mmoment.xyz",
                "http://middleware.mmoment.xyz",
                "*"  # Allow all origins for development
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Origin", "Accept", "X-Requested-With"],
            "supports_credentials": True
        }
    })
    
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
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on (default: 5002)')
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
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return 1
    finally:
        # Shut down services
        logger.info("Stopping services...")
        # Stop both WHIP publishers
        if 'whip_clean' in services and services['whip_clean']:
            services['whip_clean'].stop()
        if 'whip_annotated' in services and services['whip_annotated']:
            services['whip_annotated'].stop()
        if 'webrtc' in services:
            services['webrtc'].stop()
        if 'session' in services:
            services['session'].stop()
        if 'buffer' in services:
            services['buffer'].stop()
    
    return 0

if __name__ == "__main__":
    exit(main()) 