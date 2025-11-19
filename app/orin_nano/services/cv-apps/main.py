#!/usr/bin/env python3
"""
CV Apps Service - Isolated container for computer vision applications

Runs CV-based competition/tracking apps separately from core camera service.
Provides fault isolation and independent scaling.

Port: 5004
"""

import os
import sys
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CVAppsService")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global state
loaded_apps: Dict[str, any] = {}
active_app: Optional[str] = None


def load_app(app_name: str) -> bool:
    """
    Dynamically load a CV app.

    Args:
        app_name: Name of app to load (e.g., 'pushup', 'basketball')

    Returns:
        True if loaded successfully
    """
    global loaded_apps

    try:
        # Add apps directory to path
        apps_dir = '/app/apps'
        if apps_dir not in sys.path:
            sys.path.insert(0, apps_dir)

        # Import the app module
        if app_name == 'pushup':
            from pushup import create_app
            loaded_apps[app_name] = create_app()
            logger.info(f"Loaded push-up competition app")
            return True

        elif app_name == 'basketball':
            from basketball.basketball_app import BasketballApp
            loaded_apps[app_name] = BasketballApp()
            logger.info(f"Loaded basketball app")
            return True

        else:
            logger.error(f"Unknown app: {app_name}")
            return False

    except Exception as e:
        logger.error(f"Failed to load app {app_name}: {e}", exc_info=True)
        return False


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'cv-apps',
        'loaded_apps': list(loaded_apps.keys()),
        'active_app': active_app
    }), 200


@app.route('/api/apps', methods=['GET'])
def list_apps():
    """List available apps"""
    available = ['pushup', 'basketball']
    return jsonify({
        'available': available,
        'loaded': list(loaded_apps.keys()),
        'active': active_app
    }), 200


@app.route('/api/apps/<app_name>/load', methods=['POST'])
def load_app_endpoint(app_name: str):
    """Load a specific app"""
    if load_app(app_name):
        return jsonify({'success': True, 'app': app_name}), 200
    else:
        return jsonify({'error': f'Failed to load {app_name}'}), 500


@app.route('/api/apps/<app_name>/activate', methods=['POST'])
def activate_app(app_name: str):
    """Set active app"""
    global active_app

    if app_name not in loaded_apps:
        if not load_app(app_name):
            return jsonify({'error': f'Failed to load {app_name}'}), 500

    active_app = app_name
    logger.info(f"Activated app: {app_name}")

    return jsonify({'success': True, 'active_app': active_app}), 200


@app.route('/api/process', methods=['POST'])
def process_frame():
    """
    Process a frame with the active app.

    Body:
    {
        "frame": "<base64_encoded_image>",  // or multipart/form-data
        "detections": [
            {
                "track_id": 1,
                "wallet_address": "...",
                "class": "person",
                "x1": 100, "y1": 200, "x2": 300, "y2": 500,
                "face_bbox": [110, 210, 150, 260]
            }
        ]
    }

    Returns:
    {
        "success": true,
        "app": "pushup",
        "state": {...},  // Competition state
        "overlay_data": {...}  // Optional: data for overlay rendering
    }
    """
    if not active_app:
        return jsonify({'error': 'No active app'}), 400

    if active_app not in loaded_apps:
        return jsonify({'error': f'App {active_app} not loaded'}), 400

    try:
        # Get frame (handle both base64 and multipart)
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Frame sent as file
            if 'frame' not in request.files:
                return jsonify({'error': 'No frame in request'}), 400

            file = request.files['frame']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Detections sent as JSON in form data
            detections = request.form.get('detections', '[]')
            import json
            detections = json.loads(detections)

        else:
            # JSON body with base64 frame
            data = request.get_json()

            if 'frame' in data:
                import base64
                frame_data = base64.b64decode(data['frame'])
                file_bytes = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                return jsonify({'error': 'No frame provided'}), 400

            detections = data.get('detections', [])

        # Process with active app
        app_instance = loaded_apps[active_app]
        result = app_instance.process_frame(frame, detections)

        return jsonify({
            'success': True,
            'app': active_app,
            'state': result
        }), 200

    except Exception as e:
        logger.error(f"Frame processing failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/apps/<app_name>/overlay', methods=['POST'])
def render_overlay(app_name: str):
    """
    Render overlay on a frame.

    Used by camera service to composite app overlay onto stream.
    """
    if app_name not in loaded_apps:
        return jsonify({'error': f'App {app_name} not loaded'}), 400

    try:
        # Get frame
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame in request'}), 400

        file = request.files['frame']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Render overlay
        app_instance = loaded_apps[app_name]
        if hasattr(app_instance, 'draw_overlay'):
            frame = app_instance.draw_overlay(frame)

        # Encode and return
        _, buffer = cv2.imencode('.jpg', frame)

        from flask import Response
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Overlay rendering failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Mount app-specific routes
def mount_app_routes():
    """Mount Flask blueprints from loaded apps"""
    global loaded_apps

    for app_name, app_instance in loaded_apps.items():
        try:
            # Check if app has routes module
            if app_name == 'pushup':
                from pushup import routes as pushup_routes
                blueprint = pushup_routes.init_routes(app_instance)
                app.register_blueprint(blueprint)
                logger.info(f"Registered routes for {app_name}")

        except Exception as e:
            logger.error(f"Failed to mount routes for {app_name}: {e}")


if __name__ == '__main__':
    logger.info("CV Apps Service starting...")

    # Pre-load push-up app by default
    if load_app('pushup'):
        active_app = 'pushup'
        mount_app_routes()
        logger.info("Push-up app pre-loaded and activated")

    # Start Flask server
    port = int(os.environ.get('PORT', 5004))
    logger.info(f"Starting server on port {port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
