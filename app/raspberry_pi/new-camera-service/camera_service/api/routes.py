# camera_service/api/routes.py
import os
from flask import Flask, jsonify, send_file, request, redirect, Blueprint, make_response
from flask_cors import CORS, cross_origin
import io
import logging
from functools import wraps
from ..services.camera_service import CameraService
from ..services.stream_service import LivepeerStreamService
from ..services.solana_auth import SolanaAuthService
from ..config.settings import Settings
from .system_checks import SystemChecks, check_system_resources
from ..services.buffer_service import BufferService
import time
from typing import Optional, Dict, List
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize services
camera_service = CameraService()
stream_service = LivepeerStreamService()
solana_auth = SolanaAuthService()
buffer_service = BufferService()

# Create Blueprint
api = Blueprint('api', __name__)

def validate_solana_public_key(public_key):
    """Validate that a string looks like a Solana public key"""
    if not public_key or not isinstance(public_key, str):
        return False
    # Basic check: Solana public keys are base58 encoded and 32-44 characters long
    if not 32 <= len(public_key) <= 44:
        return False
    # Check for valid base58 characters
    valid_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return all(c in valid_chars for c in public_key)

def require_auth(f):
    """Decorator to require Solana authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == "OPTIONS":
            return "", 204
            
        # TEMPORARY: Always skip authentication for testing
        logger.warning("Skipping Solana authentication (development mode)")
        return f(*args, **kwargs)
            
        # Get user's public key from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.error("No Authorization header")
            return jsonify({"error": "Authentication required", "auth_type": "solana"}), 401
            
        try:
            # Basic validation of the public key format
            if not validate_solana_public_key(auth_header):
                logger.error(f"Invalid Solana public key format: {auth_header}")
                return jsonify({"error": "Invalid Solana public key format"}), 400
                
            # Check if user is authorized using the solana_auth instance
            if not solana_auth.check_user_authorized(auth_header):
                logger.error(f"User {auth_header} not authorized")
                return jsonify({"error": "Not authorized to access this camera"}), 403
                
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({"error": "Authentication failed"}), 401
            
    return decorated_function

@api.after_request
def after_request(response):
    """Add CORS headers to every response"""
    # Add CORS headers to every response
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin,Cache-Control,Pragma,Range')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Authorization,Content-Length,Accept-Ranges,Content-Range')
    response.headers.add('Access-Control-Max-Age', '86400')
    
    # Ensure no caching for dynamic content
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    return response

@api.route("/api/system/status", methods=["GET", "OPTIONS"])
def get_system_status():
    """Get current system status"""
    if request.method == "OPTIONS":
        return "", 204
    try:
        status = SystemChecks.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/auth/status", methods=["GET", "OPTIONS"])
def get_auth_status():
    """Get Solana authentication status"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        camera_pubkey = solana_auth.get_camera_public_key()
        is_registered = solana_auth.check_camera_registered()
        
        # Always return the camera key (even if mocked)
        if not camera_pubkey:
            camera_pubkey = "5omKvXxzsMkPJh7HZbozJXHR4h7TGRQXcNgRbTngd1Ww"
            
        # If we're in development mode, ensure we return something useful
        if Settings.SKIP_SOLANA_AUTH:
            return jsonify({
                "camera_pubkey": camera_pubkey,
                "registered": True,
                "auth_enabled": False,
                "program_id": Settings.SOLANA_PROGRAM_ID,
                "mode": "development"
            })
            
        return jsonify({
            "camera_pubkey": camera_pubkey,
            "registered": is_registered,
            "auth_enabled": not Settings.SKIP_SOLANA_AUTH,
            "program_id": str(solana_auth.program_id) if solana_auth.program_id else Settings.SOLANA_PROGRAM_ID,
            "mode": "production"
        })
    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/auth/check", methods=["GET", "OPTIONS"])
def check_auth():
    """Check if a user is authorized"""
    if request.method == "OPTIONS":
        return "", 204
        
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"authorized": False, "error": "No Authorization header"}), 401
        
    try:
        # Basic validation of the public key format
        if not validate_solana_public_key(auth_header):
            return jsonify({
                "authorized": False, 
                "error": "Invalid Solana public key format",
                "pubkey": auth_header
            }), 400
            
        # Get camera public key
        camera_pubkey = solana_auth.get_camera_public_key()
        
        # Check if development mode is enabled
        if Settings.SKIP_SOLANA_AUTH:
            return jsonify({
                "authorized": True,
                "reason": "Development mode enabled",
                "user_pubkey": auth_header,
                "camera_pubkey": camera_pubkey
            })
            
        # Check if user is authorized
        authorized = solana_auth.check_user_authorized(auth_header)
        return jsonify({
            "authorized": authorized,
            "user_pubkey": auth_header,
            "camera_pubkey": camera_pubkey
        })
    except Exception as e:
        logger.error(f"Error checking authorization: {e}")
        return jsonify({"authorized": False, "error": str(e)}), 500

@api.route("/api/capture", methods=["POST", "OPTIONS"])
@require_auth
@check_system_resources(priority_level=4)
def capture():
    """Handle image capture with proper CORS"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        jpeg_data = camera_service.take_picture()
        response = send_file(
            io.BytesIO(jpeg_data),
            mimetype='image/jpeg',
            as_attachment=False
        )
        return response
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/stream/info", methods=["GET", "OPTIONS"])
def get_stream_info():
    """Get stream information"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        stream_info = stream_service.get_stream_info()
        return jsonify(stream_info)
    except Exception as e:
        logger.error(f"Error getting stream info: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/stream/start", methods=["POST", "OPTIONS"])
@require_auth
@check_system_resources(priority_level=1)
def start_stream():
    """Start streaming to Livepeer"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        # Initialize the stream service
        stream_service = LivepeerStreamService()
        
        # Start the stream
        result = stream_service.start_stream()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/stream/stop", methods=["POST", "OPTIONS"])
@require_auth
def stop_stream():
    """Stop the stream"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        # Initialize the stream service
        stream_service = LivepeerStreamService()
        
        # Stop the stream
        result = stream_service.stop_stream()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/video/start", methods=["POST", "OPTIONS"])
@require_auth
@check_system_resources(priority_level=2)
def start_recording():
    """Start video recording"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        video_info = camera_service.start_recording()
        return jsonify(video_info)
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/video/stop", methods=["POST", "OPTIONS"])
@require_auth
def stop_recording():
    """Stop video recording"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        video_info = camera_service.stop_recording()
        return jsonify(video_info)
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/video/download/<filename>", methods=["GET", "OPTIONS"])
@require_auth
def download_video(filename):
    """Download a recorded video"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        video_path = os.path.join(Settings.VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
            
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return jsonify({"error": "Video file too large"}), 413
            
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/health", methods=['GET', 'OPTIONS'])
@check_system_resources(priority_level=1)
def health_check():
    """Handle health check with proper CORS"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return response, 204
    
    try:
        # Get buffer service status
        buffer_active = buffer_service.is_running
        camera_id = solana_auth.get_camera_public_key()
        
        return jsonify({
            "status": "ok",
            "camera_id": camera_id,
            "buffer_active": buffer_active
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@api.route("/api/camera/status", methods=["GET", "OPTIONS"])
def get_camera_status():
    """Get camera status including stream and buffer information"""
    try:
        # Get stream info
        stream_info = stream_service.get_stream_info()
        
        # Get buffer status
        buffer_status = {
            "active": buffer_service.is_running,
            "frame_count": buffer_service.get_frame_count() if buffer_service.is_running else 0
        }
        
        # Get audio status
        audio_status = camera_service.get_audio_status() if hasattr(camera_service, 'get_audio_status') else {"running": False}
        
        # Get camera ID
        camera_id = solana_auth.get_camera_public_key()
        
        return jsonify({
            "online": buffer_service.is_running,
            "camera_id": camera_id,
            "stream": stream_info,
            "buffer": buffer_status,
            "audio": audio_status
        })
    except Exception as e:
        logger.error(f"Error getting camera status: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/audio/status", methods=["GET", "OPTIONS"])
@require_auth
def get_audio_status():
    """Get audio buffer status"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        audio_status = camera_service.get_audio_status()
        return jsonify(audio_status)
    except Exception as e:
        logger.error(f"Error getting audio status: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/record", methods=["POST", "OPTIONS"])
@require_auth
@check_system_resources(priority_level=1)
def record_video():
    """Record a video clip"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        # Get duration from request or use default
        data = request.get_json(silent=True) or {}
        duration = data.get("duration", 30)  # Default to 30 seconds
        
        # Ensure duration is within strict limits
        duration = min(max(1, duration), 30)  # Cap at 30 seconds maximum
        
        # Initialize camera service
        camera_service = CameraService()
        
        # Start recording
        result = camera_service.start_recording(duration)
        
        # If successful, add some additional info for the frontend
        if result.get("status") == "recorded" and result.get("path"):
            # Get base filename without path
            filename = result.get("filename")
            filepath = result.get("path")
            
            # Verify the file exists and has content
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                # Calculate a relative URL that will work with the frontend
                # Include full path information for middleware
                result["file_url"] = f"/videos/{filename}"
                result["full_path"] = filepath
                result["file_size"] = os.path.getsize(filepath)
                result["content_type"] = "video/quicktime"
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error recording video: {e}")
        return jsonify({"error": str(e)}), 500

@api.route("/api/record/stop", methods=["POST", "OPTIONS"])
@require_auth
def stop_record():
    """Stop video recording"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        # Stop recording
        result = camera_service.stop_recording()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/record/latest', methods=['GET', 'OPTIONS'])
@require_auth
def get_latest_recording():
    """Get the latest recorded video file"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        # Get the videos directory from settings
        videos_dir = Settings.VIDEOS_DIR
        
        if not os.path.exists(videos_dir):
            return jsonify({"error": "No videos directory found"}), 404
            
        # List all video files in the directory
        video_files = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) 
                      if f.startswith('video_') and f.endswith('.mov')]
        
        if not video_files:
            return jsonify({"error": "No videos found"}), 404
            
        # Sort by modification time (newest first)
        video_files.sort(key=os.path.getmtime, reverse=True)
        
        # Get the newest video file
        latest_video = video_files[0]
        filename = os.path.basename(latest_video)
        
        # Check if the file exists and has content
        if not os.path.exists(latest_video):
            return jsonify({"error": "Video file not found"}), 404
            
        file_size = os.path.getsize(latest_video)
        if file_size == 0:
            return jsonify({"error": "Video file is empty"}), 500
        
        # Return the video file
        return send_file(
            latest_video,
            mimetype='video/quicktime',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error getting latest recording: {e}")
        return jsonify({"error": str(e)}), 500

def create_app():
    app = Flask(__name__)
    
    # Enhanced CORS configuration
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", 
                             "Cache-Control", "Pragma", "Range", "Access-Control-Allow-Origin", 
                             "Access-Control-Allow-Headers", "Access-Control-Allow-Methods",
                             "Cf-Connecting-Ip", "Cf-Ipcountry", "Cf-Ray", "Cf-Visitor"],
            "expose_headers": ["Content-Type", "Authorization", "Content-Length", "Accept-Ranges", 
                              "Content-Range", "Access-Control-Allow-Origin"],
            "supports_credentials": False,
            "send_wildcard": True,
            "max_age": 86400
        }
    })
    
    # Register the blueprint
    app.register_blueprint(api)
    
    return app