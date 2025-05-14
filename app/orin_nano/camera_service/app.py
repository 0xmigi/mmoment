"""
Main Application Module

Initializes the Flask app and starts the camera service.
"""

import os
import sys
import traceback
from flask import Flask, request, jsonify, Response
import threading
import time
import importlib.util

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try importing modules with absolute imports first
try:
    import jetson_system.camera_service.config as config
    import jetson_system.camera_service.camera as camera
    import jetson_system.camera_service.recording as recording
    import jetson_system.camera_service.routes as routes
    print("Using absolute imports for camera service modules")
except ImportError:
    # Fallback to relative imports
    try:
        from . import config
        from . import camera
        from . import recording
        from . import routes
        print("Using relative imports for camera service modules")
    except ImportError:
        # Direct imports as last resort
        try:
            import config
            import camera
            import recording
            import routes
            print("Using direct imports for camera service modules")
        except ImportError as e:
            print(f"Error importing modules: {e}")
            traceback.print_exc()
            sys.exit(1)

# Check for face_recognition availability
def check_face_recognition():
    """Check if face_recognition module is available and fix import if necessary"""
    try:
        # Try importing face_recognition directly
        import face_recognition
        print("Face recognition module available system-wide")
        return True
    except ImportError as e:
        print(f"Direct import failed: {e}")
        
        # Try site-packages in user directory for different Python versions
        user_paths = [
            os.path.expanduser(f'~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'),
            os.path.expanduser('~/.local/lib/python3.10/site-packages'),
            os.path.expanduser('~/.local/lib/python3.9/site-packages'),
            os.path.expanduser('~/.local/lib/python3.8/site-packages'),
            '/usr/local/lib/python3.10/dist-packages',
            '/usr/lib/python3/dist-packages'
        ]
        
        for path in user_paths:
            if os.path.exists(path):
                if path not in sys.path:
                    print(f"Adding {path} to Python path")
                    sys.path.append(path)
        
        # Try importing again
        try:
            import face_recognition
            print("Face recognition module found in user site-packages")
            return True
        except ImportError as e2:
            print(f"Failed to import face_recognition after path adjustments: {e2}")
            
            # List all packages to help debug the issue
            try:
                import pkg_resources
                installed_packages = [d.project_name for d in pkg_resources.working_set]
                print("Installed Python packages:")
                for pkg in installed_packages:
                    if 'face' in pkg.lower():
                        print(f"  - {pkg} (RELATED TO FACE RECOGNITION)")
                    else:
                        print(f"  - {pkg}")
            except Exception as pkg_err:
                print(f"Error listing packages: {pkg_err}")
                
            return False

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure app
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    app.config['PROPAGATE_EXCEPTIONS'] = True  # Ensure exceptions are properly propagated
    app.config['JSON_SORT_KEYS'] = False  # Preserve JSON order
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty printing for performance
    app.config['TRAP_HTTP_EXCEPTIONS'] = True  # Trap HTTP exceptions
    
    # Register global before_request middleware
    @app.before_request
    def handle_preflight():
        """Handle preflight OPTIONS requests properly for CORS"""
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, X-Session-ID, X-Wallet-Address')
            response.headers.add('Access-Control-Max-Age', '3600')
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            return response
    
    # Register global after_request middleware to add CORS headers
    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to all responses"""
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, X-Session-ID, X-Wallet-Address') 
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        response.headers.add('Access-Control-Max-Age', '3600')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    # Register error handlers
    @app.errorhandler(405)
    def method_not_allowed(e):
        """Better error message for method not allowed errors"""
        method = request.method
        path = request.path
        
        # Convert camelCase/kebab-case endpoints to snake_case alternatives if needed
        alternative_path = None
        if '-' in path:
            # Try snake_case alternative
            alternative_path = path.replace('-', '_')
        
        response = jsonify({
            "success": False,
            "error": f"Method {method} not allowed for endpoint {path}",
            "suggestion": f"Try using POST instead of GET or vice versa" if not alternative_path else 
                         f"Try the alternative endpoint: {alternative_path}"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 405
    
    # Register 404 handler
    @app.errorhandler(404)
    def not_found(e):
        """Better error message for not found errors"""
        path = request.path
        
        # Check if using camelCase or kebab-case instead of snake_case
        alternative_path = None
        if '-' in path:
            # Try snake_case alternative
            alternative_path = path.replace('-', '_')
        
        response = jsonify({
            "success": False,
            "error": f"Endpoint not found: {path}",
            "suggestion": "Check the documentation for available endpoints" if not alternative_path else 
                         f"Try the alternative endpoint: {alternative_path}"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 404
        
    # Register bad request handler
    @app.errorhandler(400)
    def bad_request(e):
        """Better error message for bad requests"""
        response = jsonify({
            "success": False,
            "error": f"Bad request: {str(e)}",
            "suggestion": "Check that your request data is properly formatted"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400
    
    # Register server error handler
    @app.errorhandler(500)
    def server_error(e):
        """Better error message for server errors"""
        response = jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "suggestion": "Please try again or contact the administrator"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
    
    # Create required directories
    config.create_directories()
    
    # Initialize camera
    camera.init_camera()
    
    # Check face_recognition availability
    fr_available = check_face_recognition()
    print(f"Face recognition availability: {fr_available}")
    
    # Try to import the face_recognition module from our package
    try:
        # Try importing with current structure
        from . import face_recognition
        # Load enrolled faces
        face_recognition.load_enrolled_faces()
        print("Successfully loaded face_recognition module")
    except ImportError as e:
        print(f"Error importing face_recognition module: {e}")
        traceback.print_exc()
    
    # Start camera thread
    frame_thread = threading.Thread(target=camera.capture_frames, daemon=True)
    frame_thread.start()
    
    # Register all routes
    routes.register_routes(app)
    
    # Explicitly register video recording routes
    routes.register_recording_routes(app)
    
    return app

def run_app():
    """Run the Flask application"""
    app = create_app()
    
    # Set host and port from environment variables if available
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    
    print(f"Starting Jetson Camera API on {host}:{port}")
    
    from werkzeug.serving import run_simple
    run_simple(
        hostname=host,
        port=port,
        application=app,
        use_reloader=debug,
        threaded=True,
        use_debugger=debug
    )

if __name__ == "__main__":
    run_app() 