#!/usr/bin/env python3
"""
Jetson Camera API Server

Main entry point for the camera API service. Runs the Flask app.
"""

import os
import sys
import time
import traceback

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try importing face_recognition_handler first so it's initialized
try:
    from face_recognition_handler import load_enrolled_faces, FACE_RECOGNITION_AVAILABLE
    print(f"Face recognition available: {FACE_RECOGNITION_AVAILABLE}")
    if FACE_RECOGNITION_AVAILABLE:
        load_enrolled_faces()
        print("Loaded enrolled faces successfully")
except ImportError as e:
    print(f"Error importing face_recognition_handler: {e}")
    print("Face recognition features will not be available")

# Import the app module
try:
    from app import run_app
    print("Imported run_app successfully")
except ImportError as e:
    try:
        # Try absolute import if relative import fails
        from jetson_system.camera_service.app import run_app
        print("Imported run_app from absolute path")
    except ImportError as e2:
        print(f"Error importing app: {e2}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Run the Flask application
        run_app()
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1) 