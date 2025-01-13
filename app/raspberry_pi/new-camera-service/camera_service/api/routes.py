# camera_service/api/routes.py
import os
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import io
import logging
from ..services.camera_service import CameraService
from ..services.stream_service import LivepeerStreamService
from ..config.settings import Settings
from .system_checks import SystemChecks, check_system_resources  # Add this import

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    # Configure CORS to allow all methods and headers
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": True,
            "max_age": 600
        }
    })

    camera_service = CameraService()
    stream_service = LivepeerStreamService()

    @app.after_request
    def after_request(response):
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        return response

    # Add new system status endpoint
    @app.route("/api/system/status", methods=["GET", "OPTIONS"])
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

    @app.route("/api/capture", methods=["POST", "OPTIONS"])
    @check_system_resources(priority_level=4)  # Add decorator
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
            logger.error(f"Capture error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/stream/start", methods=["POST", "OPTIONS"])
    @check_system_resources(priority_level=5)  # Add decorator
    def start_stream():
        """Handle stream start with proper CORS"""
        if request.method == "OPTIONS":
            return "", 204

        logger.debug("Start stream endpoint hit")
        try:
            success = stream_service.start_streaming()
            if success:
                logger.debug("Stream started successfully")
                return jsonify(stream_service.get_stream_info())
            logger.error("Failed to start stream")
            return jsonify({"error": "Failed to start streaming"}), 500
        except Exception as e:
            logger.error(f"Error in start_stream: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route("/api/stream/stop", methods=["POST", "OPTIONS"])
    @check_system_resources(priority_level=5)  # Add decorator
    def stop_stream():
        """Handle stream stop with proper CORS"""
        if request.method == "OPTIONS":
            return "", 204

        logger.debug("Stop stream endpoint hit")
        try:
            stream_service.stop_streaming()
            return jsonify({"status": "stopped"})
        except Exception as e:
            logger.error(f"Error in stop_stream: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stream/info", methods=["GET", "OPTIONS"])
    @check_system_resources(priority_level=1)  # Add decorator - low impact operation
    def get_stream_info():
        """Handle stream info with proper CORS"""
        if request.method == "OPTIONS":
            return "", 204

        try:
            info = stream_service.get_stream_info()
            if info:
                return jsonify(info)
            return jsonify({"error": "No stream info available"}), 404
        except Exception as e:
            logger.error(f"Error in stream_info: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET", "OPTIONS"])
    @check_system_resources(priority_level=1)  # Add decorator - health check should always work
    def health():
        """Handle health check with proper CORS"""
        if request.method == "OPTIONS":
            return "", 204
        return jsonify({"status": "ok"})
    
    @app.route("/api/video/start", methods=["POST", "OPTIONS"])
    @check_system_resources(priority_level=3)  # Add decorator
    def start_recording():
        """Handle video recording start with CORS"""
        if request.method == "OPTIONS":
            return "", 204

        try:
            logger.debug("Starting video recording")
            result = camera_service.start_recording()
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route("/api/video/stop", methods=["POST", "OPTIONS"])
    @check_system_resources(priority_level=3)  # Add decorator
    def stop_video():
        """Handle video recording stop with CORS"""
        if request.method == "OPTIONS":
            return "", 204

        try:
            logger.debug("Stopping video recording")
            result = camera_service.stop_recording()
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route("/api/video/download/<filename>", methods=["GET", "OPTIONS"])
    @check_system_resources(priority_level=2)  # Add decorator - important to complete downloads
    def download_video(filename):
        """Handle video download with CORS"""
        if request.method == "OPTIONS":
            return "", 204

        try:
            video_path = os.path.join(Settings.VIDEOS_DIR, filename)
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return jsonify({'error': 'Video not found'}), 404

            file_size = os.path.getsize(video_path)
            logger.info(f"Found video file: {video_path}, size: {file_size} bytes")

            if file_size == 0:
                logger.error(f"Video file is empty: {video_path}")
                return jsonify({'error': 'Video file is empty'}), 500

            response = send_file(
                video_path,
                mimetype='video/quicktime',
                as_attachment=True,
                download_name=filename
            )
            response.headers["Accept-Ranges"] = "bytes"
            return response
        except Exception as e:
            logger.error(f"Error downloading video: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    return app