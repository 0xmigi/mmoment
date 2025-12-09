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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser('~/mmoment/app/orin_nano/camera_service/logs/camera_service.log'))
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
    from services.buffer_service import get_buffer_service
    from services.gesture_service import get_gesture_service
    from services.capture_service import get_capture_service
    from services.session_service import get_session_service
    from services.livepeer_stream_service import LivepeerStreamService
    from services.webrtc_service import get_webrtc_service
    from services.whip_publisher import get_whip_publisher
    
    # GPU Face service for YOLOv8 + InsightFace face detection and recognition
    gpu_face_service = None
    try:
        from services.gpu_face_service import get_gpu_face_service
        gpu_face_service = get_gpu_face_service()
        logger.info("GPU Face service (YOLOv8 + InsightFace) initialized on GPU - NO CPU FALLBACK")
    except (ImportError, RuntimeError) as e:
        logger.error(f"GPU Face service FAILED - GPU not available or CPU fallback detected: {e}")
        logger.error("SERVICE WILL NOT START WITHOUT PROPER GPU ACCELERATION")
        gpu_face_service = None
    
    # Try to import DeepFace service (legacy)
    deepface_service = None
    try:
        from services.deepface_service import get_deepface_service
        deepface_service = get_deepface_service()
        logger.info("DeepFace GPU service initialized")
    except ImportError:
        logger.warning("DeepFace GPU service not available")
    
    # Solana integration is handled by the dedicated solana-middleware container
    # Camera service only communicates with it via HTTP API calls
    
    logger.info("Initializing services...")
    
    # Get service instances (they are singletons)
    buffer_service = get_buffer_service()
    gesture_service = get_gesture_service()
    capture_service = get_capture_service()
    session_service = get_session_service()
    
    # Reset and create Livepeer service to ensure it picks up current environment variables
    LivepeerStreamService.reset_instance()
    livepeer_service = LivepeerStreamService()
    
    # Get WebRTC service instance
    webrtc_service = get_webrtc_service()
    
    # Start the buffer service first (it's the source of truth)
    logger.info("Starting buffer service...")
    if not buffer_service.start():
        logger.error("Failed to start buffer service")
        return None
    
    # Allow buffer service to initialize
    time.sleep(1)
    
    # Start the GPU face service (only GPU-accelerated face recognition)
    if gpu_face_service:
        logger.info("Starting GPU face service...")
        gpu_face_service.start(buffer_service)
        logger.info("GPU face recognition enabled")
    else:
        logger.warning("GPU face service not available - face recognition disabled")
    
    # DISABLED: Gesture service causing excessive CPU usage (200%+)
    logger.info("DISABLED gesture service - was causing 200% CPU usage")
    # gesture_service.start(buffer_service)

    # Initialize pose service for CV apps
    pose_service = None
    try:
        from services.pose_service import get_pose_service
        pose_service = get_pose_service()
        pose_service.start()
        logger.info("Pose service initialized on GPU")
    except Exception as e:
        logger.error(f"Failed to initialize pose service: {e}")

    # Initialize app manager for CV apps
    app_manager = None
    try:
        from services.app_manager import get_app_manager
        app_manager = get_app_manager()
        logger.info("App manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize app manager: {e}")

    # Initialize Livepeer service with buffer service
    logger.info("Initializing Livepeer service...")
    livepeer_service.set_buffer_service(buffer_service)
    
    # Inject all services into Livepeer service for visual effects
    temp_services = {
        'gesture': gesture_service,
        'face': gpu_face_service
    }
    livepeer_service.set_services(temp_services)
    logger.info("Injected visual services into Livepeer service for overlay support")
    
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

    # Initialize WHIP Publisher for remote streaming via MediaMTX
    whip_publisher = None
    whip_enabled = os.environ.get('WHIP_ENABLED', 'true').lower() == 'true'
    if whip_enabled:
        logger.info("📡 Initializing WHIP publisher for remote streaming...")
        try:
            whip_publisher = get_whip_publisher()
            whip_publisher.set_buffer_service(buffer_service)
            whip_start_result = whip_publisher.start()
            if whip_start_result:
                logger.info(f"✅ WHIP publisher started - stream available at: {whip_publisher.whep_url}")
            else:
                logger.warning("⚠️ WHIP publisher failed to start - remote streaming unavailable")
        except Exception as e:
            logger.error(f"❌ WHIP publisher initialization failed: {e}")
            whip_publisher = None
    else:
        logger.info("📡 WHIP publisher disabled via WHIP_ENABLED=false")

    # Initialize Blockchain Session Sync
    from services.blockchain_session_sync import get_blockchain_session_sync, reset_blockchain_session_sync
    reset_blockchain_session_sync()  # Ensure fresh instance with current environment variables
    blockchain_sync = get_blockchain_session_sync()
    blockchain_sync.set_services(session_service, gpu_face_service)
    blockchain_sync.start()
    logger.info("🔗 Blockchain session sync initialized - camera will auto-enable for on-chain check-ins")
    
    # Initialize Device Registration Service (for QR-based setup flow)
    from services.device_registration import get_device_registration_service
    device_registration = get_device_registration_service()
    logger.info("📱 Device registration service initialized - ready for QR-based device setup")
    
    # Inject pose service and app manager into GPU face service (for background processing)
    if gpu_face_service and (pose_service or app_manager):
        logger.info("Injecting CV app services into GPU face service...")
        gpu_face_service.inject_app_services(
            pose_service=pose_service,
            app_manager=app_manager
        )

    # Inject services into the buffer service for processing
    logger.info("Injecting services into buffer service...")
    buffer_service.inject_services(
        gesture_service=gesture_service,
        gpu_face_service=gpu_face_service,
        pose_service=pose_service,
        app_manager=app_manager
    )
    
    # Build service dictionary
    services = {
        'buffer': buffer_service,
        'gesture': gesture_service,
        'capture': capture_service,
        'session': session_service,
        'livepeer': livepeer_service,
        'webrtc': webrtc_service,
        'whip': whip_publisher
    }
    
    # Add GPU Face service if available
    if gpu_face_service:
        services['gpu_face'] = gpu_face_service
        services['face'] = gpu_face_service  # Also register as 'face' for route compatibility
        services['face_detector'] = gpu_face_service  # Also register as 'face_detector' for route compatibility
        gpu_face_status = gpu_face_service.get_status()
        logger.info(f"GPU Face service status: GPU={gpu_face_status['gpu_available']}, Models={gpu_face_status['models_loaded']}, Enrolled={gpu_face_status['enrolled_faces']}")
    
    # Add DeepFace service if available (legacy)
    if deepface_service:
        services['deepface'] = deepface_service
        deepface_status = deepface_service.get_status()
        logger.info(f"DeepFace service status: Available={deepface_status['available']}, GPU={deepface_status['gpu_enabled']}, Model={deepface_status['current_model']}")

    # Add pose service and app manager
    if pose_service:
        services['pose'] = pose_service
    if app_manager:
        services['app_manager'] = app_manager

    # Solana integration is handled by the dedicated solana-middleware container
    # Camera service communicates with it via HTTP API calls when needed

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
        if 'whip' in services and services['whip']:
            services['whip'].stop()
        if 'webrtc' in services:
            services['webrtc'].stop()
        if 'livepeer' in services:
            services['livepeer'].cleanup_stream()
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