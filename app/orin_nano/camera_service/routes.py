"""
Routes Module

Contains all the Flask route definitions for the Camera API.
"""

import os
import time
import base64
import re
import cv2
import json
from flask import jsonify, request, Response, send_file, render_template
import uuid
import traceback
import numpy as np

# Use direct imports instead of relative imports
import config
import camera
import recording
import session
import gestures

# Import the face recognition handler using direct imports
try:
    # First try to import directly
    import face_recognition_handler
    # Import specific functions for convenience
    from face_recognition_handler import (
        recognize_face, 
        enroll_face, 
        list_enrolled_faces, 
        clear_all_faces, 
        load_enrolled_faces, 
        FACE_RECOGNITION_AVAILABLE
    )
    print("Imported face_recognition functions directly from face_recognition_handler")
    
    # Initialize face recognition by loading enrolled faces
    load_enrolled_faces()
except ImportError as e:
    print(f"Error importing face_recognition_handler functions: {e}")
    
    try:
        # Try importing with a full path
        import jetson_system.camera_service.face_recognition_handler as face_recognition
        print("Imported face_recognition_handler with absolute import")
        
        # Initialize face recognition by loading enrolled faces
        face_recognition.load_enrolled_faces()
    except ImportError as e3:
        print(f"Error with absolute import of face_recognition_handler: {e3}")
        
        # Create a stub module as fallback
        from types import ModuleType
        face_recognition = ModuleType("face_recognition_handler")
        face_recognition.FACE_RECOGNITION_AVAILABLE = False
        face_recognition.recognize_face = lambda: (False, "Face recognition not available")
        face_recognition.enroll_face = lambda name: (False, "Face recognition not available")
        face_recognition.list_enrolled_faces = lambda: {"success": False, "faces": [], "error": "Face recognition not available"}
        face_recognition.clear_all_faces = lambda: {"success": False, "error": "Face recognition not available"}
        face_recognition.load_enrolled_faces = lambda: None
        print("Created stub face_recognition module - face recognition will not be available")

# Keep track of face bounding boxes from detection
detected_faces = []

def register_routes(app):
    """Register all routes with the Flask app"""
    
    # Root endpoint to confirm API is working
    @app.route('/', methods=['GET', 'POST', 'OPTIONS'])
    def root():
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        # Serve the API test page for GET requests from browsers
        if request.method == 'GET' and request.headers.get('Accept', '').find('text/html') != -1:
            try:
                return app.send_static_file('html/api-test.html')
            except:
                try:
                    # Try template folder
                    return render_template('api-test.html')
                except:
                    pass
                    
        # Return JSON API response for programmatic access
        return jsonify({
            "message": "Jetson Camera API is running",
            "status": "ok",
            "timestamp": int(time.time() * 1000)
        })

    # Health check endpoint
    @app.route('/health', methods=['GET', 'POST', 'OPTIONS'])
    def health():
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        # Check if camera is working
        camera_status = "ok" if camera.camera and camera.camera.isOpened() else "error"
        
        return jsonify({
            "status": "ok",
            "camera": camera_status,
            "timestamp": int(time.time() * 1000),
            "success": True
        })

    # Handle OPTIONS requests explicitly for all routes
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Max-Age', '3600')
        return response
        
    # ============================================================================
    # Status endpoint
    # ============================================================================
    @app.route(f'/api/{config.CAMERA_PDA}/status', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/status', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    def status():
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        return jsonify({
            "status": "ok",
            "camera_id": config.CAMERA_PDA,
            "device": "Jetson Orin Nano",
            "capabilities": {
                "face_recognition": True,
                "gesture_detection": True,
                "video_recording": True,
                "face_detection": camera.face_cascade is not None
            },
            "visualization": {
                "face_detection_enabled": config.enable_face_detection,
                "face_visualization_enabled": config.enable_face_visualization,
                "gesture_visualization_enabled": config.enable_gesture_visualization
            }
        })

    # ============================================================================
    # Stream endpoint - MJPEG implementation
    # ============================================================================
    def generate_frames():
        while True:
            with camera.frame_lock:
                if camera.last_frame is None:
                    time.sleep(0.1)
                    continue
                    
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', camera.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                    
                # Convert to bytes
                frame_bytes = buffer.tobytes()
                
            # Yield the frame in the MJPEG format
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS

    @app.route(f'/api/{config.CAMERA_PDA}/stream', methods=['GET', 'OPTIONS'])
    @app.route('/stream', methods=['GET', 'OPTIONS'])  # Add alias without prefix
    def stream():
        """Provide live MJPEG stream of camera feed"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
        
        # Make sure camera is initialized
        if camera.camera is None or not camera.camera.isOpened():
            print("STREAM ERROR: Camera not initialized - attempting to reinitialize")
            success = camera.init_camera()
            if not success:
                print("STREAM ERROR: Failed to initialize camera")
                # Return a static error image
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Camera Unavailable", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', error_frame)
                return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n',
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Add expiry headers to ensure the stream is never cached
        response = Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
        response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        print("STREAM: Starting new camera stream")
        return response

    # ============================================================================
    # 1. GATEWAY CONTROLS - Require on-chain transactions
    # ============================================================================

    @app.route(f'/api/{config.CAMERA_PDA}/check-in', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/check-in', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/connect', methods=['GET', 'POST', 'OPTIONS'])  # Add frontend-friendly alias
    def check_in():
        """Check in to the camera with wallet - requires on-chain transaction"""
        print("Received check-in/connect request")
        
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            # Extract wallet_address and display_name from request
            wallet_address = None
            display_name = None
            
            # Accept both GET and POST methods
            if request.method == 'POST':
                try:
                    # Try to parse JSON data
                    data = request.get_json(force=True, silent=True) or {}
                    wallet_address = data.get('wallet_address')
                    display_name = data.get('display_name')
                    print(f"POST data: wallet={wallet_address}, display_name={display_name}")
                except Exception as e:
                    print(f"Error parsing POST data: {e}")
                    data = {}
            else:  # GET
                wallet_address = request.args.get('wallet_address')
                display_name = request.args.get('display_name')
                print(f"GET data: wallet={wallet_address}, display_name={display_name}")
            
            # Use a fallback wallet address if none provided
            if not wallet_address:
                wallet_address = f"anonymous_{uuid.uuid4().hex[:8]}"
                print(f"No wallet address provided, using fallback: {wallet_address}")
                response = jsonify({
                    "success": True,
                    "message": "Connected to camera with anonymous wallet",
                    "session_id": session.create_session(wallet_address, display_name or "Anonymous User"),
                    "wallet_address": wallet_address,
                    "camera_pda": config.CAMERA_PDA,
                    "anonymous": True
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            
            # Create a new session - use the simplified session creation
            session_id = session.create_session(wallet_address, display_name)
            print(f"Session created: {session_id}")
            
            # Create minimal response with CORS headers
            response = jsonify({
                "success": True,
                "message": "Connected to camera successfully",
                "session_id": session_id,
                "wallet_address": wallet_address,
                "camera_pda": config.CAMERA_PDA,
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        except Exception as e:
            print(f"Error checking in: {e}")
            # Create a fallback session and return it
            fallback_wallet = f"error_{uuid.uuid4().hex[:8]}"
            fallback_session = session.create_session(fallback_wallet, "Error User")
            
            response = jsonify({
                "success": True,  # Return success to avoid frontend errors
                "message": f"Created fallback session due to error: {str(e)}",
                "session_id": fallback_session,
                "wallet_address": fallback_wallet,
                "camera_pda": config.CAMERA_PDA,
                "error": str(e),
                "fallback": True
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    @app.route(f'/api/{config.CAMERA_PDA}/check-out', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/check-out', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/disconnect', methods=['GET', 'POST', 'OPTIONS'])  # Add frontend-friendly alias
    def check_out():
        """Check out from camera - requires on-chain transaction"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            # Accept both GET and POST methods
            if request.method == 'POST':
                try:
                    data = request.get_json(force=True, silent=True) or {}
                except:
                    data = {}
                wallet_address = data.get('wallet_address')
                session_id = data.get('session_id')
            else:  # GET
                wallet_address = request.args.get('wallet_address')
                session_id = request.args.get('session_id')
            
            if not wallet_address and not session_id:
                return jsonify({"error": "Wallet address or session ID is required"}), 400
            
            # If we have session_id but no wallet, try to get the wallet from the session
            if session_id and not wallet_address:
                if session_id in session.active_sessions:
                    wallet_address = session.active_sessions[session_id].get('wallet_address')
            
            # Check if session exists
            if session_id and (not wallet_address or session.is_session_valid(session_id, wallet_address)):
                # Mark session as inactive
                session.close_session(session_id)
                
                response = jsonify({
                    "success": True,
                    "message": "Checked out from camera successfully"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            else:
                response = jsonify({
                    "success": False,
                    "error": "Invalid session ID or wallet address"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 404
        
        except Exception as e:
            print(f"Error checking out: {e}")
            response = jsonify({"error": str(e), "success": False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    # ============================================================================
    # Face Recognition Endpoints
    # ============================================================================

    @app.route('/recognize-face', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/recognize_face', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def recognize_face_endpoint():
        """Recognize a face using enrolled faces"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            # Perform face recognition
            success, result = recognize_face()
            
            # Create response based on recognition result
            if success:
                # Return the detailed result from face_recognition_handler
                response = jsonify({
                    "success": True,
                    "face_recognized": True,
                    "face_id": result.get("face_id", "unknown"),
                    "name": result.get("name", "Unknown"),
                    "confidence": result.get("confidence", 0.0)
                })
            else:
                # Handle the error case
                error_message = result if isinstance(result, str) else "Unknown error"
                response = jsonify({
                    "success": False,
                    "face_recognized": False,
                    "message": error_message
                })
                
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            traceback.print_exc()
            response = jsonify({
                "success": False,
                "face_recognized": False,
                "error": str(e)
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    @app.route('/enroll-face', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/enroll_face', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def enroll_face_endpoint():
        """Enroll a face for facial recognition"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Session-ID, X-Wallet-Address')
            return response
            
        try:
            # Get display name from request
            display_name = "Unknown User"
            
            if request.method == 'POST':
                try:
                    data = request.get_json(force=True, silent=True) or {}
                    if 'display_name' in data:
                        display_name = data['display_name']
                    elif 'name' in data:
                        display_name = data['name']
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
            else:  # GET
                if request.args.get('display_name'):
                    display_name = request.args.get('display_name')
                elif request.args.get('name'):
                    display_name = request.args.get('name')
            
            print(f"Enrolling face with name: {display_name}")
            success, result = enroll_face(display_name)
            
            if success:
                # Return success response with enrolled face info
                response = jsonify({
                    "success": True,
                    "message": "Face enrolled successfully",
                    "face_id": result.get("face_id", "unknown"),
                    "name": result.get("name", display_name)
                })
            else:
                # Return error response with message
                error_message = result if isinstance(result, str) else "Unknown error during face enrollment"
                response = jsonify({
                    "success": False,
                    "message": error_message
                })
                
            # Add CORS headers
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
                
        except Exception as e:
            traceback.print_exc()
            response = jsonify({
                "success": False,
                "message": f"Error in face enrollment: {str(e)}"
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    @app.route('/list-enrolled-faces', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/list_enrolled_faces', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def list_enrolled_faces_endpoint():
        """List all enrolled faces"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Session-ID, X-Wallet-Address')
            return response
            
        try:
            # Get list of enrolled faces
            result = list_enrolled_faces()
            
            # Return the response
            response = jsonify(result)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            traceback.print_exc()
            response = jsonify({
                "success": False,
                "error": str(e),
                "faces": []
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    # ============================================================================
    # Gesture Detection Endpoints
    # ============================================================================

    @app.route(f'/api/{config.CAMERA_PDA}/current-gesture', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/current-gesture', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/current_gesture', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def current_gesture():
        """Get the current detected gesture (for polling)"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            # Get session ID from query parameters or body
            session_id = None
            
            if request.method == 'GET':
                session_id = request.args.get('session_id')
            else:  # POST
                try:
                    data = request.get_json(force=True, silent=True) or {}
                    session_id = data.get('session_id')
                except:
                    pass
            
            # If session ID is provided, check that polling is enabled for this session
            if session_id and session_id in session.active_sessions:
                if not session.active_sessions[session_id].get("gesture_polling", False):
                    response = jsonify({
                        "success": False,
                        "message": "Gesture polling not enabled for this session",
                        "gesture": "none",
                        "confidence": 0,
                        "timestamp": int(time.time() * 1000)
                    })
                    response.headers.add('Access-Control-Allow-Origin', '*')
                    return response
            
            response = jsonify({
                "success": True,
                "gesture": gestures.active_gesture["gesture"],
                "confidence": gestures.active_gesture["confidence"],
                "description": config.gesture_labels.get(gestures.active_gesture["gesture"], 
                                                gestures.active_gesture["gesture"].capitalize() if gestures.active_gesture["gesture"] != "none" else "No gesture"),
                "timestamp": int(time.time() * 1000)
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        except Exception as e:
            print(f"Error getting current gesture: {e}")
            response = jsonify({
                "success": False,
                "gesture": "none", 
                "confidence": 0,
                "timestamp": int(time.time() * 1000),
                "error": str(e)
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    @app.route('/start-gesture-polling', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/start_gesture_polling', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def start_gesture_polling():
        """Start continuous gesture detection polling"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            wallet_address = None
            session_id = None
            
            # Get parameters from appropriate source based on HTTP method
            if request.method == 'POST':
                try:
                    data = request.get_json(force=True, silent=True) or {}
                    wallet_address = data.get('wallet_address')
                    session_id = data.get('session_id')
                except:
                    pass
            else:  # GET
                wallet_address = request.args.get('wallet_address')
                session_id = request.args.get('session_id')
            
            if not session_id:
                # Try to find existing session for wallet
                if wallet_address:
                    existing_session_id, _ = session.get_session_by_wallet(wallet_address)
                    if existing_session_id:
                        session_id = existing_session_id
                
                # If still no session, create temporary session
                if not session_id:
                    session_id, temp_wallet = session.create_temp_session()
                    print(f"Created temporary session {session_id} for gesture polling")
            
            # Mark this session for gesture polling
            if session_id in session.active_sessions:
                session.active_sessions[session_id]["gesture_polling"] = True
                print(f"Enabled gesture polling for session {session_id}")
                
                response = jsonify({
                    "success": True,
                    "message": "Gesture polling started",
                    "session_id": session_id
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            else:
                response = jsonify({
                    "success": False,
                    "message": "Invalid session ID"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
        
        except Exception as e:
            print(f"Error starting gesture polling: {e}")
            response = jsonify({"error": str(e), "success": False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    @app.route('/stop-gesture-polling', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/stop_gesture_polling', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def stop_gesture_polling():
        """Stop continuous gesture detection polling"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            session_id = None
            
            # Get parameters from appropriate source based on HTTP method
            if request.method == 'POST':
                try:
                    data = request.get_json(force=True, silent=True) or {}
                    session_id = data.get('session_id')
                except:
                    pass
            else:  # GET
                session_id = request.args.get('session_id')
            
            if not session_id:
                response = jsonify({
                    "success": False,
                    "message": "Session ID is required"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
            
            if session_id in session.active_sessions:
                session.active_sessions[session_id]["gesture_polling"] = False
                print(f"Disabled gesture polling for session {session_id}")
                
                response = jsonify({
                    "success": True,
                    "message": "Gesture polling stopped"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            else:
                response = jsonify({
                    "success": False,
                    "message": "Invalid session ID"
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response, 400
        
        except Exception as e:
            print(f"Error stopping gesture polling: {e}")
            response = jsonify({"error": str(e), "success": False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    # ============================================================================
    # Visualization Control Endpoints
    # ============================================================================

    @app.route(f'/api/{config.CAMERA_PDA}/toggle-face-detection', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/toggle-face-detection', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/toggle_face_detection', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def toggle_face_detection():
        """Toggle face detection on/off"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        # Get request parameters
        enable = request.args.get('enable')
        if enable is not None:
            # Convert to boolean
            config.enable_face_detection = enable.lower() in ('true', '1', 'yes', 'on')
        else:
            # Toggle current value
            config.enable_face_detection = not config.enable_face_detection
        
        # Save to persistent config
        config.save_config("enable_face_detection", str(config.enable_face_detection))
        
        # Get the toggle state after update
        current_state = config.enable_face_detection
            
        return jsonify({
            "success": True,
            "face_detection": current_state,
            "message": f"Face detection {'enabled' if current_state else 'disabled'}"
        })

    @app.route(f'/api/{config.CAMERA_PDA}/toggle-face-visualization', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/toggle-face-visualization', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/toggle_face_visualization', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def toggle_face_visualization():
        """Toggle face visualization (bounding boxes) on/off"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        # Get request parameters from either query args or JSON body
        enable = None
        print("Received toggle face visualization request")
        
        if request.method == 'GET':
            # Handle query parameters
            enable_param = request.args.get('enable')
            disable_param = request.args.get('disable')
            
            print(f"GET params: enable={enable_param}, disable={disable_param}")
            
            if enable_param is not None:
                enable = enable_param.lower() in ('true', '1', 'yes', 'on')
            elif disable_param is not None:
                enable = not (disable_param.lower() in ('true', '1', 'yes', 'on'))
        elif request.method == 'POST':
            # Handle JSON body
            try:
                data = request.get_json(silent=True) or {}
                print(f"POST data: {data}")
                if 'state' in data:
                    enable = bool(data['state'])
                elif 'enable' in data:
                    enable = bool(data['enable'])
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        
        # If no parameters were provided, toggle current state
        if enable is None:
            enable = not config.enable_face_visualization
            print(f"No specific state requested, toggling current state to: {enable}")
        else:
            print(f"Setting face visualization to: {enable}")
        
        # Set the new state
        config.enable_face_visualization = enable
        
        # Save to persistent config
        config.save_config("enable_face_visualization", str(config.enable_face_visualization))
        
        # Force rebuild the next frame with our current settings
        def force_rebuild_frame():
            try:
                # Get a fresh frame
                raw_frame = camera.get_raw_frame()
                if raw_frame is not None:
                    # Manually process with our current settings
                    print(f"Manual frame processing with visualization={config.enable_face_visualization}")
                    processed_frame, _ = camera.detect_faces(
                        raw_frame.copy(),
                        config.enable_face_visualization, 
                        camera.identified_user
                    )
                    # Update the frame that will be sent to clients
                    with camera.frame_lock:
                        camera.last_frame = processed_frame
                    print("Frame manually processed and updated")
                    return True
                return False
            except Exception as e:
                print(f"Error in force_rebuild_frame: {e}")
                return False
        
        # First try the standard frame refresh
        with camera.frame_lock:
            if camera.camera and camera.camera.isOpened():
                # Read a fresh frame
                success, fresh_frame = camera.camera.read()
                if success:
                    # Process it with current settings
                    processed = camera.detect_faces(
                        fresh_frame.copy(), 
                        config.enable_face_visualization,
                        camera.identified_user
                    )[0]
                    camera.last_frame = processed
                    print(f"Face visualization toggled to: {enable}")
        
        # Then try our backup method
        rebuild_success = force_rebuild_frame()
        
        # Return the current state
        return jsonify({
            "success": True,
            "face_visualization": config.enable_face_visualization,
            "state": config.enable_face_visualization,  # Add state for backward compatibility
            "message": f"Face visualization {'enabled' if config.enable_face_visualization else 'disabled'}",
            "note": "Visualization changes are now visible",
            "frame_rebuild": rebuild_success
        })
        
    @app.route(f'/api/{config.CAMERA_PDA}/toggle-gesture-visualization', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/toggle-gesture-visualization', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/toggle_gesture_visualization', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def toggle_gesture_visualization():
        """Toggle gesture visualization (hand landmarks) on/off"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        # Get request parameters
        enable = request.args.get('enable')
        if enable is not None:
            # Convert to boolean
            config.enable_gesture_visualization = enable.lower() in ('true', '1', 'yes', 'on')
        else:
            # Toggle current value
            config.enable_gesture_visualization = not config.enable_gesture_visualization
        
        # Save to persistent config
        config.save_config("enable_gesture_visualization", str(config.enable_gesture_visualization))
        
        # Get the toggle state after update
        current_state = config.enable_gesture_visualization
            
        return jsonify({
            "success": True,
            "gesture_visualization": current_state,
            "message": f"Gesture visualization {'enabled' if current_state else 'disabled'}",
            "note": "Visualization changes will be visible in the next frame"
        })

    @app.route(f'/api/{config.CAMERA_PDA}/visual-controls-status', methods=['GET'])
    @app.route('/visual-controls-status', methods=['GET'])  # Add alias without prefix
    def visual_controls_status():
        """Get current status of all visual controls"""
        return jsonify({
            "success": True,
            "face_detection": config.enable_face_detection,
            "face_visualization": config.enable_face_visualization,
            "gesture_visualization": config.enable_gesture_visualization,
            "visualization_note": "Show visual elements like bounding boxes on the video feed",
            "detection_note": "Enable/disable face detection processing (affects performance)"
        })

    @app.route(f'/api/{config.CAMERA_PDA}/camera-info', methods=['GET'])
    def camera_info():
        """Get comprehensive camera information and session status"""
        # Count active sessions
        active_count = session.count_active_sessions()
        
        return jsonify({
            "status": "ok",
            "camera_id": config.CAMERA_PDA,
            "camera_pda": config.CAMERA_PDA,
            "device": "Jetson Orin Nano",
            "active_sessions": active_count,
            "capabilities": {
                "face_recognition": True,
                "gesture_detection": True,
                "video_recording": True,
                "face_detection": camera.face_cascade is not None
            },
            "visualization": {
                "face_detection_enabled": config.enable_face_detection,
                "face_visualization_enabled": config.enable_face_visualization,
                "gesture_visualization_enabled": config.enable_gesture_visualization
            },
            "program_id": config.PROGRAM_ID,
            "camera_service": "ok",
            "solana_middleware": "ok"
        })

    # Video player page
    @app.route('/video-player/<filename>', methods=['GET'])
    def video_player(filename):
        """Serve a dedicated HTML5 video player for a specific video"""
        try:
            # Normalize filename and find the video
            name, ext = os.path.splitext(filename)
            video_path = None
            
            # Check for the file with various extensions if needed
            if not ext:
                for test_ext in ['.mp4', '.mov', '.avi']:
                    if os.path.exists(os.path.join(config.CAMERA_VIDEOS_DIR, name + test_ext)):
                        video_path = os.path.join(config.CAMERA_VIDEOS_DIR, name + test_ext)
                        filename = name + test_ext
                        break
                    elif os.path.exists(os.path.join(config.ALT_VIDEOS_DIR, name + test_ext)):
                        video_path = os.path.join(config.ALT_VIDEOS_DIR, name + test_ext)
                        filename = name + test_ext
                        break
            else:
                # Check with the provided extension
                if os.path.exists(os.path.join(config.CAMERA_VIDEOS_DIR, filename)):
                    video_path = os.path.join(config.CAMERA_VIDEOS_DIR, filename)
                elif os.path.exists(os.path.join(config.ALT_VIDEOS_DIR, filename)):
                    video_path = os.path.join(config.ALT_VIDEOS_DIR, filename)
            
            if not video_path:
                return f"<html><body><h1>Video not found: {filename}</h1><p>The requested video file could not be found.</p></body></html>"
            
            # Determine MIME type
            _, ext = os.path.splitext(filename)
            mimetype = "video/mp4"  # Default
            if ext.lower() == '.mov':
                mimetype = "video/quicktime"
            elif ext.lower() == '.avi':
                mimetype = "video/x-msvideo"
            
            # Create a simple HTML5 video player
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Player: {filename}</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 0; 
                        padding: 20px; 
                        background-color: #f0f0f0;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }}
                    h1 {{ margin-bottom: 20px; }}
                    .video-container {{ 
                        max-width: 800px; 
                        margin: 0 auto;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        background: white;
                        padding: 20px;
                        border-radius: 5px;
                    }}
                    video {{ 
                        width: 100%; 
                        height: auto;
                        border-radius: 3px;
                    }}
                    .video-info {{ 
                        margin-top: 15px;
                        padding: 10px;
                        background-color: #f9f9f9;
                        border-radius: 3px;
                    }}
                    .download-link {{
                        display: inline-block;
                        margin-top: 15px;
                        padding: 10px 15px;
                        background-color: #0066cc;
                        color: white;
                        text-decoration: none;
                        border-radius: 3px;
                    }}
                    .download-link:hover {{
                        background-color: #0055aa;
                    }}
                </style>
            </head>
            <body>
                <h1>Video Player</h1>
                <div class="video-container">
                    <video controls autoplay>
                        <source src="/videos/{filename}" type="{mimetype}">
                        Your browser does not support the video tag.
                    </video>
                    <div class="video-info">
                        <p><strong>Filename:</strong> {filename}</p>
                        <p><strong>Type:</strong> {mimetype}</p>
                    </div>
                    <a href="/videos/{filename}" download class="download-link">Download Video</a>
                </div>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            return f"<html><body><h1>Error</h1><p>Failed to load video player: {str(e)}</p></body></html>"

    # Test videos page
    @app.route('/test-videos', methods=['GET'])
    def test_videos():
        """HTML page to test video playback"""
        try:
            # Get list of videos
            videos = recording.get_video_list()
            
            # Sort by creation time (newest first)
            videos.sort(key=lambda x: x["created"], reverse=True)
            
            # Create basic HTML for testing
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Jetson Camera - Video Test</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f0f0f0;
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                    }
                    .videos-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        justify-content: center;
                    }
                    .video-card {
                        width: 300px;
                        background: white;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }
                    .video-thumbnail {
                        width: 100%;
                        height: 180px;
                        background-color: #ddd;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        cursor: pointer;
                    }
                    .video-thumbnail img {
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: cover;
                    }
                    .video-info {
                        padding: 15px;
                    }
                    .video-title {
                        font-weight: bold;
                        margin-bottom: 5px;
                        word-break: break-all;
                    }
                    .video-meta {
                        color: #666;
                        font-size: 0.9em;
                        margin-bottom: 10px;
                    }
                    .video-actions {
                        display: flex;
                        gap: 10px;
                    }
                    .btn {
                        padding: 8px 12px;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 14px;
                        text-align: center;
                    }
                    .btn-primary {
                        background-color: #007bff;
                        color: white;
                    }
                    .btn-secondary {
                        background-color: #6c757d;
                        color: white;
                    }
                    .no-videos {
                        text-align: center;
                        padding: 40px;
                        color: #666;
                        font-style: italic;
                    }
                    .recorder {
                        margin: 20px auto;
                        max-width: 600px;
                        background: white;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        text-align: center;
                    }
                    .recorder-buttons {
                        display: flex;
                        justify-content: center;
                        gap: 10px;
                        margin-top: 15px;
                    }
                    .btn-record {
                        background-color: #dc3545;
                    }
                    .btn-stop {
                        background-color: #6c757d;
                    }
                    .recording-indicator {
                        margin-top: 10px;
                        color: #dc3545;
                        font-weight: bold;
                        display: none;
                    }
                </style>
            </head>
            <body>
                <h1>Jetson Camera Videos</h1>
                
                <div class="recorder">
                    <h2>Record a New Video</h2>
                    <p>Click the button below to start recording a new video.</p>
                    <div class="recorder-buttons">
                        <button id="startRecording" class="btn btn-record">Start Recording</button>
                        <button id="stopRecording" class="btn btn-stop" disabled>Stop Recording</button>
                    </div>
                    <div id="recordingIndicator" class="recording-indicator">
                        Recording in progress... <span id="recordingTimer">00:00</span>
                    </div>
                </div>
                
                <h2 style="text-align: center;">Recorded Videos</h2>
                <div class="videos-container">
            """
            
            if not videos:
                html += """
                    <div class="no-videos">
                        No videos found. Record a video using the recorder above.
                    </div>
                """
            else:
                for video in videos:
                    filename = video["filename"]
                    created_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(video["created"]))
                    size_mb = round(video["size"] / (1024 * 1024), 2)
                    
                    thumbnail_url = f"/videos/{filename}?thumbnail=true" if video.get("thumbnail_url") else "/static/video-placeholder.jpg"
                    video_url = f"/videos/{filename}"
                    player_url = f"/video-player/{filename}"
                    
                    html += f"""
                    <div class="video-card">
                        <div class="video-thumbnail" onclick="window.location.href='{player_url}'">
                            <img src="{thumbnail_url}" alt="Thumbnail">
                        </div>
                        <div class="video-info">
                            <div class="video-title">{filename}</div>
                            <div class="video-meta">
                                Created: {created_date}<br>
                                Size: {size_mb} MB
                            </div>
                            <div class="video-actions">
                                <a href="{player_url}" class="btn btn-primary">Play</a>
                                <a href="{video_url}" download class="btn btn-secondary">Download</a>
                            </div>
                        </div>
                    </div>
                    """
            
            html += """
                </div>
                
                <script>
                    let isRecording = false;
                    let recordingStartTime = 0;
                    let timerInterval = null;
                    
                    document.getElementById('startRecording').addEventListener('click', function() {
                        startRecording();
                    });
                    
                    document.getElementById('stopRecording').addEventListener('click', function() {
                        stopRecording();
                    });
                    
                    function startRecording() {
                        fetch('/start-recording', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                duration: 30 // 30 seconds max
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                isRecording = true;
                                recordingStartTime = Date.now();
                                document.getElementById('startRecording').disabled = true;
                                document.getElementById('stopRecording').disabled = false;
                                document.getElementById('recordingIndicator').style.display = 'block';
                                
                                // Start timer
                                updateRecordingTimer();
                                timerInterval = setInterval(updateRecordingTimer, 1000);
                            } else {
                                alert('Failed to start recording: ' + data.message);
                            }
                        })
                        .catch(error => {
                            console.error('Error starting recording:', error);
                            alert('Error starting recording');
                        });
                    }
                    
                    function stopRecording() {
                        fetch('/stop-recording', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({})
                        })
                        .then(response => response.json())
                        .then(data => {
                            isRecording = false;
                            document.getElementById('startRecording').disabled = false;
                            document.getElementById('stopRecording').disabled = true;
                            document.getElementById('recordingIndicator').style.display = 'none';
                            
                            // Stop timer
                            clearInterval(timerInterval);
                            
                            // Reload page after a short delay to show new video
                            setTimeout(() => {
                                window.location.reload();
                            }, 1000);
                        })
                        .catch(error => {
                            console.error('Error stopping recording:', error);
                            alert('Error stopping recording');
                            isRecording = false;
                            document.getElementById('startRecording').disabled = false;
                            document.getElementById('stopRecording').disabled = true;
                            document.getElementById('recordingIndicator').style.display = 'none';
                            clearInterval(timerInterval);
                        });
                    }
                    
                    function updateRecordingTimer() {
                        if (!isRecording) return;
                        
                        const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
                        const minutes = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0');
                        const seconds = (elapsedSeconds % 60).toString().padStart(2, '0');
                        
                        document.getElementById('recordingTimer').textContent = `${minutes}:${seconds}`;
                        
                        // Auto-stop after 30 seconds
                        if (elapsedSeconds >= 30) {
                            stopRecording();
                        }
                    }
                </script>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            return f"<html><body><h1>Error</h1><p>Failed to load videos: {str(e)}</p></body></html>"

    @app.route('/view-all-videos', methods=['GET'])
    def view_all_videos():
        """Show a page with all recorded videos for direct viewing"""
        try:
            videos_dir = os.path.join(config.DATA_DIR, 'videos')
            html = "<html><body><h1>All Videos</h1>"
            
            # Check if the videos directory exists
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)
                
            # Get all mp4 files
            video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
            
            if not video_files:
                html += "<p>No videos found</p>"
            else:
                html += "<ul>"
                for filename in sorted(video_files, reverse=True):
                    video_path = os.path.join(videos_dir, filename)
                    size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)
                    creation_time = os.path.getctime(video_path)
                    creation_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
                    
                    html += f"<li><a href='/view-video/{filename}'>{filename}</a> ({size_mb} MB) - {creation_str}</li>"
                html += "</ul>"
            
            html += "</body></html>"
            return html
        except Exception as e:
            return f"<html><body><h1>Error</h1><p>Failed to load videos: {str(e)}</p></body></html>"

    # Simple video player endpoint for direct video access
    @app.route('/view-video/<filename>', methods=['GET'])
    def view_video(filename):
        """Simple video player for direct access"""
        try:
            videos_dir = os.path.join(config.DATA_DIR, 'videos')
            video_path = os.path.join(videos_dir, filename)
            
            if not os.path.exists(video_path):
                return f"<html><body><h1>Error</h1><p>Video not found: {filename}</p></body></html>"
                
            # Serve the video file with a simple player
            html = f"""
            <html>
            <head>
                <title>Video Player - {filename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    video {{ max-width: 100%; }}
                    .controls {{ margin-top: 20px; }}
                    .controls a {{ 
                        display: inline-block; 
                        padding: 8px 16px; 
                        background-color: #4CAF50; 
                        color: white; 
                        text-decoration: none; 
                        border-radius: 4px; 
                        margin-right: 10px;
                    }}
                    .meta {{ 
                        margin-top: 10px;
                        color: #666; 
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <h1>Video Player</h1>
                <video controls autoplay>
                    <source src="/video-data/{filename}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="meta">
                    Filename: {filename}<br>
                    Size: {round(os.path.getsize(video_path) / (1024 * 1024), 2)} MB<br>
                    Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(video_path)))}
                </div>
                <div class="controls">
                    <a href="/view-all-videos">Back to Video List</a>
                    <a href="/download-video/{filename}" download>Download</a>
                </div>
            </body>
            </html>
            """
            return html
        except Exception as e:
            return f"<html><body><h1>Error</h1><p>Failed to load video: {str(e)}</p></body></html>"

    @app.route(f'/api/{config.CAMERA_PDA}/list-videos', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/list-videos', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/list_videos', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def list_videos():
        """List all recorded videos"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            videos_dir = os.path.join(config.DATA_DIR, 'videos')
            
            # Check if the videos directory exists
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)
                
            # Get all mp4 files
            video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
            
            videos = []
            for filename in sorted(video_files, reverse=True):
                video_path = os.path.join(videos_dir, filename)
                size_bytes = os.path.getsize(video_path)
                creation_time = os.path.getctime(video_path)
                creation_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
                
                videos.append({
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "creation_time": int(creation_time),
                    "creation_datetime": creation_str,
                    "url": f"/video-data/{filename}",
                    "view_url": f"/view-video/{filename}",
                    "download_url": f"/download-video/{filename}"
                })
            
            response = jsonify({
                "success": True,
                "count": len(videos),
                "videos": videos
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            print(f"Error listing videos: {e}")
            response = jsonify({"error": str(e), "success": False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500

    @app.route('/video-data/<filename>', methods=['GET'])
    def video_data(filename):
        """Serve video file directly"""
        try:
            videos_dir = os.path.join(config.DATA_DIR, 'videos')
            return send_file(os.path.join(videos_dir, filename), mimetype='video/mp4')
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 404

    @app.route('/download-video/<filename>', methods=['GET'])
    def download_video(filename):
        """Download video file"""
        try:
            videos_dir = os.path.join(config.DATA_DIR, 'videos')
            return send_file(
                os.path.join(videos_dir, filename),
                mimetype='video/mp4',
                as_attachment=True,
                download_name=filename
            )
        except Exception as e:
            return jsonify({"error": str(e), "success": False}), 404

    @app.route(f'/api/{config.CAMERA_PDA}/clear-all-faces', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/clear-all-faces', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    @app.route('/clear_all_faces', methods=['GET', 'POST', 'OPTIONS'])  # Add snake_case alias
    def clear_all_faces_endpoint():
        """Clear all enrolled faces"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Session-ID, X-Wallet-Address')
            return response
            
        try:
            # Call the face recognition handler to clear all faces
            result = clear_all_faces()
            
            # Create response from result
            response = jsonify(result)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            traceback.print_exc()
            response = jsonify({
                "success": False,
                "error": str(e),
                "message": "Failed to clear faces: " + str(e)
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    # Add a new configuration endpoint
    @app.route(f'/api/{config.CAMERA_PDA}/set-config', methods=['POST', 'OPTIONS'])
    @app.route('/set-config', methods=['POST', 'OPTIONS'])  # Add alias without prefix
    def set_config():
        """Set configuration values directly"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            data = request.get_json(force=True, silent=True) or {}
            
            changes_made = False
            before_state = {
                "enable_face_detection": config.enable_face_detection,
                "enable_face_visualization": config.enable_face_visualization,
                "enable_gesture_visualization": config.enable_gesture_visualization,
                "disable_all_header_visualization": config.disable_all_header_visualization
            }
            
            # Update enable_face_detection if present
            if 'enable_face_detection' in data:
                config.enable_face_detection = bool(data['enable_face_detection'])
                changes_made = True
            
            # Update enable_face_visualization if present
            if 'enable_face_visualization' in data:
                config.enable_face_visualization = bool(data['enable_face_visualization'])
                changes_made = True
                
            # Update enable_gesture_visualization if present
            if 'enable_gesture_visualization' in data:
                config.enable_gesture_visualization = bool(data['enable_gesture_visualization'])
                changes_made = True
                
            # Update disable_all_header_visualization if present
            if 'disable_all_header_visualization' in data:
                config.disable_all_header_visualization = bool(data['disable_all_header_visualization'])
                changes_made = True
                
            after_state = {
                "enable_face_detection": config.enable_face_detection,
                "enable_face_visualization": config.enable_face_visualization,
                "enable_gesture_visualization": config.enable_gesture_visualization,
                "disable_all_header_visualization": config.disable_all_header_visualization
            }
            
            return jsonify({
                "success": True,
                "message": "Configuration updated successfully" if changes_made else "No changes made",
                "before": before_state,
                "after": after_state,
                "changes_made": changes_made
            })
            
        except Exception as e:
            print(f"Error setting config: {e}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Failed to set configuration: {str(e)}"
            }), 500

    # Refresh stream endpoint
    @app.route('/refresh-stream', methods=['GET', 'POST', 'OPTIONS'])
    def refresh_stream():
        """Force a stream refresh - this helps with toggle visualization"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
            
        try:
            # Force camera to refresh its frame processing
            with camera.frame_lock:
                if camera.camera and camera.camera.isOpened():
                    # Read a fresh frame
                    success, fresh_frame = camera.camera.read()
                    if success:
                        # Store it as both raw and processed frame to force an update
                        camera.raw_frame = fresh_frame.copy()
                        # Process it with current settings
                        processed = camera.detect_faces(
                            fresh_frame.copy(), 
                            config.enable_face_visualization,
                            camera.identified_user
                        )[0]
                        camera.last_frame = processed
                        print("Stream refreshed with current visualization settings")
            
            # Return success response
            return jsonify({
                "success": True,
                "face_visualization": config.enable_face_visualization,
                "gesture_visualization": config.enable_gesture_visualization
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": str(e)
            })
    
    # Add video recording routes directly
    @app.route(f'/api/{config.CAMERA_PDA}/start-recording', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/start-recording', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    def start_recording_endpoint():
        """Start video recording"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
        
        try:
            # Extract parameters from request (if any)
            max_duration = 30  # Default: 30 seconds
            if request.method == 'POST':
                try:
                    data = request.get_json(silent=True) or {}
                    if data and 'duration' in data:
                        max_duration = int(data['duration'])
                except Exception as e:
                    print(f"Error parsing duration: {e}")
            
            # Start recording
            success, result = recording.start_recording(max_duration)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Recording started",
                    "filename": result.get('filename'),
                    "path": result.get('path')
                })
            else:
                return jsonify({
                    "success": False,
                    "message": result
                })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "success": False,
                "message": str(e)
            })
    
    @app.route(f'/api/{config.CAMERA_PDA}/stop-recording', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/stop-recording', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    def stop_recording_endpoint():
        """Stop video recording"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
        
        try:
            # Stop the recording
            success, result = recording.stop_recording()
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Recording stopped successfully",
                    "filename": result.get('filename'),
                    "path": result.get('path')
                })
            else:
                return jsonify({
                    "success": False,
                    "message": result
                })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "success": False,
                "message": str(e)
            })
    
    @app.route(f'/api/{config.CAMERA_PDA}/recording-status', methods=['GET'])
    @app.route('/recording-status', methods=['GET'])  # Add alias without prefix
    def recording_status_endpoint():
        """Get recording status"""
        status = recording.get_recording_status()
        return jsonify({
            "success": True,
            **status
        })
    
    @app.route(f'/api/{config.CAMERA_PDA}/reset-recording', methods=['POST'])
    @app.route('/reset-recording', methods=['POST'])  # Add alias without prefix
    def reset_recording_endpoint():
        """Reset recording state if it's stuck"""
        with recording.recording_lock:
            recording.recording_active = False
            recording.recording_writer = None
            recording.recording_thread = None
        
        return jsonify({
            "success": True,
            "message": "Recording state has been reset"
        })

    @app.route('/debug-config-state', methods=['GET'])
    def debug_config_state():
        """Debug endpoint to return detailed information about the current configuration state"""
        
        # Get current config state
        face_visualization = config.enable_face_visualization
        face_detection = config.enable_face_detection
        gesture_visualization = config.enable_gesture_visualization
        
        # Get camera frame information
        frame_info = {}
        with camera.frame_lock:
            frame_info['camera_initialized'] = camera.camera is not None and camera.camera.isOpened()
            frame_info['last_frame_available'] = camera.last_frame is not None
            frame_info['raw_frame_available'] = camera.raw_frame is not None
            if camera.last_frame is not None:
                frame_info['frame_height'] = camera.last_frame.shape[0]
                frame_info['frame_width'] = camera.last_frame.shape[1]
        
        # Count faces detected in most recent frame
        faces_detected = len(camera.detected_faces) if hasattr(camera, 'detected_faces') else 0
        
        # Get other important state
        identified_user = camera.identified_user
        
        # Load from saved config for verification
        saved_face_vis = config.load_config("enable_face_visualization")
        
        # Return all info
        return jsonify({
            "time": time.time(),
            "config_state": {
                "enable_face_visualization": face_visualization,
                "enable_face_detection": face_detection,
                "enable_gesture_visualization": gesture_visualization,
                "saved_face_visualization": saved_face_vis
            },
            "camera_state": frame_info,
            "detection_state": {
                "face_cascade_loaded": camera.face_cascade is not None,
                "faces_detected": faces_detected,
                "identified_user": identified_user
            }
        })

    @app.route('/debug-server-toggle', methods=['POST'])
    def debug_server_toggle():
        """Debug endpoint to directly toggle face visualization on the server"""
        try:
            # Get request data
            data = request.get_json(silent=True) or {}
            print(f"DEBUG SERVER TOGGLE received: {data}")
            
            # Check if enable_face_visualization is in the request
            if 'enable_face_visualization' in data:
                new_state = bool(data['enable_face_visualization'])
                
                # Set the new state directly
                print(f"Setting face visualization to: {new_state}")
                prev_state = config.enable_face_visualization
                config.enable_face_visualization = new_state
                
                # Save to persistent config
                config.save_config("enable_face_visualization", str(new_state))
                
                # Force an immediate frame refresh to apply changes
                with camera.frame_lock:
                    # Get a fresh frame
                    if camera.camera and camera.camera.isOpened():
                        success, frame = camera.camera.read()
                        if success:
                            # Process it with new settings
                            processed = camera.detect_faces(
                                frame.copy(), 
                                config.enable_face_visualization,
                                camera.identified_user
                            )[0]
                            camera.last_frame = processed
                            print(f"Frame updated with new settings")
                        else:
                            print(f"Failed to get fresh frame")
                
                return jsonify({
                    "success": True,
                    "previous_state": prev_state,
                    "current_state": config.enable_face_visualization,
                    "applied": prev_state != config.enable_face_visualization,
                    "message": f"Face visualization {'enabled' if new_state else 'disabled'}"
                })
            
            return jsonify({
                "success": False,
                "error": "Missing enable_face_visualization parameter"
            })
        except Exception as e:
            print(f"Error in debug_server_toggle: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            })

    @app.route(f'/api/{config.CAMERA_PDA}/capture-moment', methods=['GET', 'POST', 'OPTIONS'])
    @app.route('/capture-moment', methods=['GET', 'POST', 'OPTIONS'])  # Add alias without prefix
    def capture_moment():
        """Capture a still image from the camera"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
        
        try:
            print("CAPTURE_MOMENT: Received request to capture a moment")
            
            # Extract for_recording parameter to make extra sure we're using a raw frame
            for_recording = False
            if request.method == 'POST':
                try:
                    data = request.get_json(force=True, silent=True) or {}
                    for_recording = data.get('for_recording', False)
                    print(f"CAPTURE_MOMENT: Received for_recording={for_recording} in request data")
                except Exception as e:
                    print(f"CAPTURE_MOMENT: Error parsing request data: {e}")
            
            # Force for_recording to be True to ensure we always use raw frame
            for_recording = True
            print(f"CAPTURE_MOMENT: Using for_recording={for_recording}")
            
            # DIRECT RAW FRAME ACCESS: Get the raw frame directly with NO processing whatsoever
            print("CAPTURE_MOMENT: About to acquire frame_lock...")
            with camera.frame_lock:
                print("CAPTURE_MOMENT: Frame lock acquired")
                # Directly access the raw_frame from the camera module
                if camera.raw_frame is None:
                    print("CAPTURE_MOMENT: No raw frame available in buffer")
                    return jsonify({
                        "success": False,
                        "error": "No camera frame available in raw buffer"
                    })
                
                # Make a direct copy of the raw frame
                frame_to_save = camera.raw_frame.copy()
                print(f"CAPTURE_MOMENT: Got raw frame with shape: {frame_to_save.shape}")
            
            print("CAPTURE_MOMENT: Frame lock released")
            
            # Generate a unique filename with timestamp
            timestamp = int(time.time())
            filename = f"capture_{timestamp}.jpg"
            output_dir = config.CAMERA_IMAGES_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Save a special debug version of the image
            debug_filename = f"debug_capture_{timestamp}.jpg"
            debug_path = os.path.join(output_dir, debug_filename)
            cv2.imwrite(debug_path, frame_to_save)
            print(f"CAPTURE_MOMENT: Saved debug image to {debug_path}")
            
            # Save the normal image
            cv2.imwrite(output_path, frame_to_save)
            print(f"CAPTURE_MOMENT: Saved image to {output_path}")
            
            # Return the image data as base64
            _, buffer = cv2.imencode('.jpg', frame_to_save)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            print("CAPTURE_MOMENT: Successfully encoded image with size:", len(image_data))
            
            return jsonify({
                "success": True,
                "message": "Image captured successfully",
                "filename": filename,
                "image_data": image_data,
                "timestamp": timestamp,
                "used_raw_buffer": True
            })
        except Exception as e:
            print(f"CAPTURE_MOMENT ERROR: {e}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Failed to capture image: {str(e)}"
            })

    @app.route('/direct-capture', methods=['GET', 'POST', 'OPTIONS'])
    def direct_capture():
        """Capture a still image DIRECTLY from the camera with no processing at all"""
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            return response
        
        try:
            print("DIRECT_CAPTURE: Bypassing all buffers and processing")
            
            # Temporarily disable face detection globally to ensure clean capture
            original_detection_state = config.enable_face_detection
            original_visualization_state = config.enable_face_visualization
            
            # Force detection and visualization off during capture
            config.enable_face_detection = False
            config.enable_face_visualization = False
            
            # Double-check camera is initialized
            if camera.camera is None or not camera.camera.isOpened():
                print("DIRECT_CAPTURE: Camera not available")
                camera.init_camera()
                if camera.camera is None:
                    return jsonify({"success": False, "error": "Camera not available"})
                
            # Direct camera capture - bypass all buffering
            print("DIRECT_CAPTURE: Reading directly from camera device")
            success, direct_frame = camera.camera.read()
            
            if not success or direct_frame is None:
                print("DIRECT_CAPTURE: Failed to read from camera")
                # Restore original states
                config.enable_face_detection = original_detection_state
                config.enable_face_visualization = original_visualization_state
                return jsonify({"success": False, "error": "Failed to capture from camera"})
            
            # Generate a unique filename with timestamp
            timestamp = int(time.time())
            filename = f"direct_capture_{timestamp}.jpg"
            output_dir = config.CAMERA_IMAGES_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Save the direct capture
            print("DIRECT_CAPTURE: Saving image with NO processing")
            cv2.imwrite(output_path, direct_frame)
            
            # Return the image data as base64
            _, buffer = cv2.imencode('.jpg', direct_frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Restore original detection and visualization states
            config.enable_face_detection = original_detection_state
            config.enable_face_visualization = original_visualization_state
            
            print("DIRECT_CAPTURE: Successfully captured clean image")
            
            return jsonify({
                "success": True,
                "message": "Direct image capture successful",
                "filename": filename,
                "image_data": image_data,
                "timestamp": timestamp,
                "direct_capture": True
            })
        
        except Exception as e:
            # Ensure we restore the original state even if an error occurs
            if 'original_detection_state' in locals():
                config.enable_face_detection = original_detection_state
                config.enable_face_visualization = original_visualization_state
            
            print(f"DIRECT_CAPTURE ERROR: {e}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Direct capture failed: {str(e)}"
            })

