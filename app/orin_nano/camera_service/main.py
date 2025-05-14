#!/usr/bin/env python3
"""
Jetson Camera API Main Module

Main entry point for the camera API service. Uses direct imports to avoid import errors.
"""

import os
import sys
import time
import traceback
import threading
from flask import Flask, request, jsonify, Response, render_template, send_file

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(current_dir)

# Import modules with direct imports
import config
import camera
import gestures
import recording
import session
import face_recognition_handler

# Initialize face recognition
print(f"Face recognition available: {face_recognition_handler.FACE_RECOGNITION_AVAILABLE}")
if face_recognition_handler.FACE_RECOGNITION_AVAILABLE:
    face_recognition_handler.load_enrolled_faces()
    print("Loaded enrolled faces successfully")

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    # Create required directories
    config.create_directories()
    
    # Initialize camera
    camera.init_camera()
    
    # Register all routes
    import routes
    routes.register_routes(app)
    
    # Start camera thread
    frame_thread = threading.Thread(target=camera.capture_frames, daemon=True)
    frame_thread.start()
    
    return app

if __name__ == "__main__":
    try:
        # Create Flask app
        app = create_app()
        
        # Set host and port
        host = os.environ.get('FLASK_HOST', '0.0.0.0')
        port = int(os.environ.get('FLASK_PORT', 5002))
        
        print(f"Starting Jetson Camera API on {host}:{port}")
        
        # Run the Flask app
        app.run(host=host, port=port, threaded=True, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1) 