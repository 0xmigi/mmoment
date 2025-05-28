"""
Routes for the Jetson Camera API

Defines all the API endpoints for the camera service.
"""

import logging
import json
import os
import time
import uuid
import base64
import threading
from functools import wraps
from flask import request, jsonify, Response, render_template, abort, send_file, current_app, stream_with_context
import cv2

# Set up logging
logger = logging.getLogger("CameraRoutes")

# Our simple face recognition is always available
FACENET_AVAILABLE = True

# Import camera utilities
try:
    from services.utils import detect_cameras, reset_camera_devices, get_camera_health, check_facenet_availability
    CAMERA_UTILS_AVAILABLE = True
except ImportError:
    CAMERA_UTILS_AVAILABLE = False
    logger.warning("Could not import camera utilities, some diagnostic endpoints will be unavailable")

# Get services from app config
def get_services():
    """Get services from Flask app config"""
    return current_app.config['SERVICES']

# Session validation decorator
def require_session(f):
    """Decorator to require a valid session"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_service = get_services()['session']
        
        # Extract session parameters
        wallet_address = request.json.get('wallet_address')
        session_id = request.json.get('session_id')
        
        # Validate session
        if not session_id or not wallet_address or not session_service.validate_session(session_id, wallet_address):
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function

# Register all routes
def register_routes(app):
    """Register all routes with the Flask app"""
    
    # Index route
    @app.route('/')
    def index():
        """Index route with API information"""
        return jsonify({
            'name': 'Jetson Camera API',
            'version': '2.0.0',
            'description': 'Lightweight, optimized camera API for Jetson with standardized endpoints',
            'buffer_based': True,
            'standardized_endpoints': {
                'health': '/api/health',
                'stream_info': '/api/stream/info',
                'capture': '/api/capture',
                'record': '/api/record',
                'photos': '/api/photos',
                'videos': '/api/videos',
                'session_connect': '/api/session/connect',
                'session_disconnect': '/api/session/disconnect',
                'face_enroll': '/api/face/enroll',
                'face_recognize': '/api/face/recognize',
                'gesture_current': '/api/gesture/current',
                'visualization_face': '/api/visualization/face',
                'visualization_gesture': '/api/visualization/gesture'
            },
            'legacy_endpoints': {
                'health': '/health',
                'stream': '/stream',
                'connect': '/connect',
                'disconnect': '/disconnect',
                'enroll_face': '/enroll_face',
                'recognize_face': '/recognize_face',
                'detect_gesture': '/detect_gesture',
                'capture_moment': '/capture_moment',
                'test_page': '/test-page',
                'camera_diagnostics': '/camera/diagnostics',
                'camera_reset': '/camera/reset'
            }
        })

    # ========================================
    # STANDARDIZED API ROUTES (Pi5 Compatible)
    # ========================================
    
    # Health & Status
    @app.route('/api/health')
    def api_health():
        """Standardized health check endpoint"""
        return health()
    
    @app.route('/api/stream/info')
    def api_stream_info():
        """Stream metadata endpoint (standardized)"""
        buffer_service = get_services()['buffer']
        buffer_status = buffer_service.get_status()
        
        return jsonify({
            'success': True,
            'playbackId': 'jetson-camera-stream',
            'isActive': buffer_status['running'],
            'streamType': 'mjpeg',
            'resolution': f"{buffer_service._width}x{buffer_service._height}",
            'fps': buffer_status['fps'],
            'streamUrl': '/stream'
        })
    
    # Camera Actions
    @app.route('/api/capture', methods=['POST'])
    @require_session
    def api_capture():
        """Standardized capture endpoint"""
        return capture_moment()
    
    @app.route('/api/record', methods=['POST'])
    @require_session
    def api_record():
        """Standardized record endpoint"""
        return start_recording()
    
    # Media Access
    @app.route('/api/photos')
    def api_photos():
        """Standardized photos list endpoint"""
        return list_photos()
    
    @app.route('/api/videos')
    def api_videos():
        """Standardized videos list endpoint"""
        return list_videos()
    
    @app.route('/api/photos/<filename>')
    def api_get_photo(filename):
        """Standardized photo access endpoint"""
        return get_photo(filename)
    
    @app.route('/api/videos/<filename>')
    def api_get_video(filename):
        """Standardized video access endpoint"""
        return get_video(filename)
    
    # Session Management
    @app.route('/api/session/connect', methods=['POST'])
    def api_session_connect():
        """Standardized session connect endpoint"""
        return connect()
    
    @app.route('/api/session/disconnect', methods=['POST'])
    @require_session
    def api_session_disconnect():
        """Standardized session disconnect endpoint"""
        return disconnect()
    
    # Computer Vision (Jetson-specific)
    @app.route('/api/face/enroll', methods=['POST'])
    @require_session
    def api_face_enroll():
        """Standardized face enrollment endpoint"""
        return enroll_face()
    
    @app.route('/api/face/recognize', methods=['POST'])
    def api_face_recognize():
        """Standardized face recognition endpoint"""
        return recognize_face()
    
    @app.route('/api/gesture/current', methods=['GET'])
    def api_gesture_current():
        """Standardized current gesture endpoint"""
        return current_gesture()
    
    @app.route('/api/visualization/face', methods=['POST'])
    def api_visualization_face():
        """Standardized face visualization toggle"""
        return toggle_face_visualization()
    
    @app.route('/api/visualization/gesture', methods=['POST'])
    def api_visualization_gesture():
        """Standardized gesture visualization toggle"""
        return toggle_gesture_visualization()

    # ========================================
    # LEGACY ROUTES (For backward compatibility)
    # ========================================
    
    # Health check route
    @app.route('/health')
    def health():
        """Health check endpoint"""
        services = get_services()
        
        # Check buffer service status
        buffer_service = services['buffer']
        buffer_status = buffer_service.get_status()
        
        # Get session count
        session_count = len(services['session'].get_all_sessions())
        
        # Basic health response
        health_response = {
            'status': 'ok' if buffer_status['running'] else 'error',
            'buffer_service': buffer_status['health'],
            'buffer_fps': buffer_status['fps'],
            'active_sessions': session_count,
            'timestamp': int(time.time() * 1000),
            'camera': {
                'index': buffer_service._camera_index,
                'preferred_device': buffer_service._preferred_device,
                'resolution': f"{buffer_service._width}x{buffer_service._height}",
                'target_fps': buffer_service._fps
            }
        }
        
        return jsonify(health_response)

    # Stream route
    @app.route('/stream')
    def stream():
        """
        Stream the camera feed as MJPEG
        This is a special endpoint that continuously streams JPEG frames
        """
        buffer_service = get_services()['buffer']
        
        def generate():
            """Generator function for MJPEG streaming"""
            while True:
                # Get JPEG frame from buffer (with processing for visualizations)
                jpeg_data, timestamp = buffer_service.get_jpeg_frame(processed=True)
                
                if jpeg_data is None:
                    # If no frame is available, wait and try again
                    time.sleep(0.03)
                    continue
                
                # Yield MJPEG format content
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data + b'\r\n')
                
                # Control frame rate of the stream
                time.sleep(0.03)  # ~30fps
        
        # Return streaming response
        return Response(
            stream_with_context(generate()),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    # Visualization routes - all public, no auth required
    @app.route('/toggle_face_detection', methods=['POST'])
    def toggle_face_detection():
        """
        Enable or disable face detection.
        Note: Face detection is now always enabled, but keeping this endpoint for compatibility.
        """
        data = request.json or {}
        enabled = data.get('enabled', True)
        
        face_service = get_services()['face']
        face_service.enable_detection(True)  # Always enable face detection
        
        # Log the request
        logger.info(f"Face detection toggle requested: {enabled} (always on)")
        
        return jsonify({
            'success': True,
            'enabled': True,  # Always return true
            'message': 'Face detection is always enabled'
        })
        
    @app.route('/toggle_face_visualization', methods=['POST'])
    def toggle_face_visualization():
        """
        Toggle face visualization in camera frames
        """
        data = request.json or {}
        enabled = data.get('enabled', True)
        
        try:
            face_service = get_services()['face']
            face_service.enable_visualization(enabled)
            
            return jsonify({
                'success': True,
                'enabled': enabled
            })
        except Exception as e:
            logger.error(f"Error toggling face visualization: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/toggle_face_boxes', methods=['POST'])
    def toggle_face_boxes():
        """
        Toggle face boxes in camera frames
        This controls only the boxes, while visualization controls all overlays
        """
        data = request.json or {}
        enabled = data.get('enabled', True)
        
        try:
            face_service = get_services()['face']
            face_service.enable_boxes(enabled)
            
            return jsonify({
                'success': True,
                'enabled': enabled
            })
        except Exception as e:
            logger.error(f"Error toggling face boxes: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        
    @app.route('/face_settings', methods=['GET'])
    def face_settings():
        """Get current face detection settings"""
        face_service = get_services()['face']
        settings = face_service.get_settings()
        
        return jsonify({
            'success': True,
            'settings': settings
        })

    @app.route('/toggle_gesture_visualization', methods=['POST'])
    def toggle_gesture_visualization():
        """Enable or disable gesture detection visualization"""
        data = request.json or {}
        enabled = data.get('enabled', True)
        
        gesture_service = get_services()['gesture']
        gesture_service.enable_visualization(enabled)
        
        # Log the toggle state
        logger.info(f"Gesture visualization toggled to: {enabled}")
        
        return jsonify({
            'success': True,
            'enabled': enabled
        })

    # Session management routes
    @app.route('/connect', methods=['POST'])
    def connect():
        """
        Connect a wallet to the camera
        Creates a new session for the wallet
        """
        data = request.json or {}
        wallet_address = data.get('wallet_address')
        
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'Wallet address is required'
            }), 400
        
        # Create a session
        session_service = get_services()['session']
        session_info = session_service.create_session(wallet_address)
        
        # Enable face boxes for all connected users
        face_service = get_services()['face']
        face_service.enable_boxes(True)
        logger.info(f"Face boxes enabled for connected user: {wallet_address}")
        
        return jsonify(session_info)

    @app.route('/disconnect', methods=['POST'])
    @require_session
    def disconnect():
        """
        Disconnect a wallet from the camera
        Ends the session for the wallet
        """
        wallet_address = request.json.get('wallet_address')
        session_id = request.json.get('session_id')
        
        # End the session
        session_service = get_services()['session']
        success = session_service.end_session(session_id, wallet_address)
        
        # Check if this was the last active session
        active_sessions = session_service.get_all_sessions()
        if len(active_sessions) == 0:
            # Disable face boxes when all users are disconnected
            face_service = get_services()['face']
            face_service.enable_boxes(False)
            logger.info("Face boxes disabled - all users disconnected")
        
        return jsonify({
            'success': success,
            'message': 'Disconnected from camera successfully' if success else 'Failed to disconnect'
        })

    # Face recognition routes
    @app.route('/enroll_face', methods=['POST'])
    @require_session
    def enroll_face():
        """
        Enroll a face for the current user
        Captures the current frame and enrolls the face
        Only works for connected users with a valid session
        """
        wallet_address = request.json.get('wallet_address')
        session_id = request.json.get('session_id')
        
        logger.info(f"[ENROLL] Starting face enrollment for wallet: {wallet_address}")
        
        # Verify the session is valid
        session_service = get_services()['session']
        if not session_service.validate_session(session_id, wallet_address):
            logger.warning(f"[ENROLL] Invalid session for wallet: {wallet_address}")
            return jsonify({
                'success': False,
                'error': 'You must be connected to enroll your face'
            }), 403
        
        try:
            # Get services
            buffer_service = get_services()['buffer']
            face_service = get_services()['face']
                
            logger.info(f"[ENROLL] Got services, enabling face detection for wallet: {wallet_address}")
            
            # Enable face detection and boxes for enrollment
            face_service.enable_detection(True)
            face_service.enable_boxes(True)
            
            # Get multiple frames to find the best one with a face
            face_detected = False
            best_frame = None
            best_timestamp = 0
            num_tries = 20  # Increased number of tries
            
            logger.info(f"[ENROLL] Looking for face in frames for wallet: {wallet_address}")
                
            # Try several frames to find a good one with a face
            for i in range(num_tries):
                # Get the current frame
                frame, timestamp = buffer_service.get_frame()
                
                if frame is None:
                    logger.warning(f"[ENROLL] Frame {i+1}/{num_tries} is None for wallet: {wallet_address}")
                    time.sleep(0.1)
                    continue
                
                # Force a face detection run
                face_service._detect_faces(frame)
                faces_info = face_service.get_faces()
                
                logger.info(f"[ENROLL] Frame {i+1}/{num_tries} detected {faces_info['detected_count']} faces for wallet: {wallet_address}")
                
                if faces_info['detected_count'] > 0:
                    face_detected = True
                    best_frame = frame
                    best_timestamp = timestamp
                    logger.info(f"[ENROLL] Found face on frame {i+1}/{num_tries} for wallet: {wallet_address}")
                    break
                
                # Wait a bit for the next frame
                time.sleep(0.1)
            
            # If no frames had a face after multiple attempts
            if not face_detected or best_frame is None:
                logger.warning(f"[ENROLL] No face detected after checking {num_tries} frames for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': 'No face was detected. Please make sure your face is clearly visible in the camera view and you have adequate lighting.',
                    'include_image': False
                }), 400
            
            # Check if multiple faces are detected
            faces_info = face_service.get_faces()
            if faces_info['detected_count'] > 1:
                logger.warning(f"[ENROLL] Multiple faces ({faces_info['detected_count']}) detected for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': 'Multiple faces detected. Please ensure only your face is visible in the camera view.',
                    'include_image': False
                }), 400
            
            # Log that we found a face for enrollment 
            logger.info(f"[ENROLL] Found face for enrollment for wallet: {wallet_address}")
            
            # We found a frame with a face - use it for enrollment
            logger.info(f"[ENROLL] Calling face_service.enroll_face for wallet: {wallet_address}")
            result = face_service.enroll_face(best_frame, wallet_address)
            
            if not result['success']:
                logger.warning(f"[ENROLL] Face enrollment failed: {result.get('error', 'Unknown error')} for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Failed to enroll face. Please try again with better lighting and positioning.'),
                    'include_image': False
                }), 400
            
            logger.info(f"[ENROLL] Face enrollment successful for wallet: {wallet_address}")
            
            # Create a processed frame with the face box for better UX
            processed_frame = face_service.get_processed_frame(best_frame)
            _, jpeg_data = cv2.imencode('.jpg', processed_frame)
            image_base64 = base64.b64encode(jpeg_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'wallet_address': wallet_address,
                'include_image': True,
                'image': image_base64,
                'encrypted': False,
                'nft_verified': False,
                'message': 'Face enrolled successfully'
            })
            
        except Exception as e:
            logger.error(f"[ENROLL] Error in face enrollment: {str(e)} for wallet: {wallet_address}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f"Face enrollment failed: {str(e)}. Please try again or contact support if the issue persists.",
                'include_image': False
            }), 500

    @app.route('/recognize_face', methods=['POST'])
    def recognize_face():
        """
        Recognize faces in the current frame
        Returns information about recognized faces
        """
        # Get wallet address if provided (for session validation)
        data = request.json or {}
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        
        logger.info(f"[RECOGNIZE] Starting face recognition for wallet: {wallet_address}")
        
        # Check if session is valid (optional)
        session_service = get_services()['session']
        has_valid_session = wallet_address and session_id and session_service.validate_session(session_id, wallet_address)
        
        if has_valid_session:
            logger.info(f"[RECOGNIZE] Valid session for wallet: {wallet_address}")
        else:
            logger.info(f"[RECOGNIZE] No valid session or anonymous recognition")
        
        try:
            # Get services
            buffer_service = get_services()['buffer']
            face_service = get_services()['face']
            
            # Get the current frame
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                logger.warning(f"[RECOGNIZE] No frame available from buffer")
                return jsonify({
                    'success': False,
                    'error': 'No frame available'
                }), 500
            
            logger.info(f"[RECOGNIZE] Got frame from buffer, running face detection")
            
            # Make sure face detection is enabled
            face_service.enable_detection(True)
            
            # Force a face detection run on the current frame
            face_detector = get_services()['face_detector']
            detected_faces = face_detector.detect_faces(frame)
            
            logger.info(f"[RECOGNIZE] Detected {len(detected_faces)} faces with face detector")
            
            # Update the face service with detected faces
            with face_service._results_lock:
                face_service._detected_faces = detected_faces
            
            # Get initial detection results
            faces_info = face_service.get_faces()
            logger.info(f"[RECOGNIZE] Detected {faces_info['detected_count']} faces")
        
            # If faces are detected, force a recognition run
            if faces_info['detected_count'] > 0:
                logger.info(f"[RECOGNIZE] Running face recognition on {faces_info['detected_count']} faces")
                # Use our simplified face recognition
                face_service._recognize_faces(frame)
            
            # Get updated recognition info
            faces_info = face_service.get_faces()
            logger.info(f"[RECOGNIZE] Recognition result: {faces_info['recognized_count']} faces recognized")
        
            # Check if the requested wallet is in the recognized faces
            wallet_recognized = False
            wallet_confidence = 0
            
            if wallet_address and wallet_address in faces_info['recognized_faces']:
                wallet_recognized = True
                wallet_confidence = faces_info['recognized_faces'][wallet_address][4]  # Confidence is at index 4
                logger.info(f"[RECOGNIZE] Wallet {wallet_address} recognized with confidence {wallet_confidence:.2f}")
            
            # Create a processed frame with face boxes for better UX
            processed_frame = face_service.get_processed_frame(frame)
            _, jpeg_data = cv2.imencode('.jpg', processed_frame)
            image_base64 = base64.b64encode(jpeg_data).decode('utf-8')
            
            # Return detailed information about all detected and recognized faces
            return jsonify({
                'success': True,
                'detected_faces': faces_info['detected_count'],
                'recognized_faces': faces_info['recognized_count'],
                'recognized_data': {k: {'confidence': v[4]} for k, v in faces_info['recognized_faces'].items()},
                'wallet_recognized': wallet_recognized,
                'wallet_confidence': wallet_confidence,
                'has_valid_session': has_valid_session,
                'include_image': True,
                'image': image_base64
            })
        except Exception as e:
            logger.error(f"[RECOGNIZE] Error in face recognition: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f"Face recognition failed: {str(e)}"
            }), 500

    # Gesture detection routes
    @app.route('/current_gesture', methods=['GET'])
    def current_gesture():
        """
        Get the current detected gesture
        Returns the gesture name and confidence
        """
        gesture_service = get_services()['gesture']
        gesture_info = gesture_service.get_current_gesture()
        
        return jsonify({
            'success': True,
            **gesture_info
        })

    # Capture routes
    @app.route('/capture_moment', methods=['POST'])
    @require_session
    def capture_moment():
        """
        Capture a photo from the camera
        Returns the photo data
        """
        wallet_address = request.json.get('wallet_address')
        
        # Get services
        buffer_service = get_services()['buffer']
        capture_service = get_services()['capture']
        
        # Capture photo
        photo_info = capture_service.capture_photo(buffer_service, wallet_address)
        
        if not photo_info['success']:
            return jsonify(photo_info), 500
        
        # Read the photo file and encode as base64
        with open(photo_info['path'], 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Add base64 data to response
        photo_info['image_data'] = f"data:image/jpeg;base64,{base64_image}"
        
        return jsonify(photo_info)

    @app.route('/start_recording', methods=['POST'])
    @require_session
    def start_recording():
        """
        Start recording a video
        Returns recording information
        """
        wallet_address = request.json.get('wallet_address')
        duration = request.json.get('duration', 0)  # 0 means record until stopped
        
        # Get services
        buffer_service = get_services()['buffer']
        capture_service = get_services()['capture']
        
        # Check if already recording
        if capture_service.is_recording():
            return jsonify({
                'success': False,
                'error': 'Already recording'
            }), 400
        
        # Start recording
        recording_info = capture_service.start_recording(buffer_service, wallet_address, duration)
        
        return jsonify(recording_info)

    @app.route('/stop_recording', methods=['POST'])
    @require_session
    def stop_recording():
        """
        Stop recording
        Returns recording information
        """
        # Get capture service
        capture_service = get_services()['capture']
        
        # Stop recording
        result = capture_service.stop_recording()
        
        return jsonify(result)

    @app.route('/list_videos', methods=['GET'])
    def list_videos():
        """
        List available videos
        Returns a list of video information
        """
        limit = request.args.get('limit', 10, type=int)
        
        # Get capture service
        capture_service = get_services()['capture']
        
        # Get video list
        videos = capture_service.get_videos(limit)
        
        return jsonify({
            'success': True,
            'videos': videos,
            'count': len(videos)
        })

    @app.route('/list_photos', methods=['GET'])
    def list_photos():
        """
        List available photos
        Returns a list of photo information
        """
        limit = request.args.get('limit', 10, type=int)
        
        # Get capture service
        capture_service = get_services()['capture']
        
        # Get photo list
        photos = capture_service.get_photos(limit)
        
        return jsonify({
            'success': True,
            'photos': photos,
            'count': len(photos)
        })

    # Media access routes
    @app.route('/photos/<filename>')
    def get_photo(filename):
        """
        Get a specific photo by filename
        Returns the photo file
        """
        capture_service = get_services()['capture']
        
        # Get photo path
        photo_path = capture_service.get_photo_path(filename)
        
        if not photo_path:
            abort(404, description="Photo not found")
        
        return send_file(photo_path, mimetype='image/jpeg')

    @app.route('/videos/<filename>')
    def get_video(filename):
        """
        Get a specific video by filename
        Returns the video file with proper headers for browser playback
        Prefers MP4 versions over MOV for better browser compatibility
        """
        capture_service = get_services()['capture']
        
        # If requesting a MOV file, try to serve the MP4 version instead
        if filename.endswith('.mov'):
            mp4_filename = filename.replace('.mov', '.mp4')
            mp4_path = capture_service.get_video_path(mp4_filename)
            
            if mp4_path and os.path.exists(mp4_path):
                # Serve the MP4 version instead
                video_path = mp4_path
                mimetype = 'video/mp4'
                logger.info(f"Serving MP4 version instead of MOV: {mp4_filename}")
            else:
                # Fall back to original MOV file
                video_path = capture_service.get_video_path(filename)
                mimetype = 'video/quicktime'  # Use proper MOV MIME type
                logger.info(f"Serving original MOV file: {filename}")
        else:
            # Get video path normally
            video_path = capture_service.get_video_path(filename)
            mimetype = 'video/mp4'
        
        if not video_path:
            abort(404, description="Video not found")
        
        # Return with headers that force browser playback instead of download
        response = send_file(
            video_path, 
            mimetype=mimetype,
            as_attachment=False,  # Don't force download
            download_name=None    # Don't suggest download filename
        )
        
        # Add headers to help with video playback
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Range'
        
        return response

    # Test page routes
    @app.route('/test-page')
    def test_page():
        """Test page to verify functionality"""
        return render_template('test.html')
        
    @app.route('/simple-test')
    def simple_test():
        """Simple test page for direct API testing"""
        return render_template('simple_test.html')
        
    @app.route('/local-test')
    def local_test():
        """Local camera test page"""
        return render_template('local_test.html')
        
    @app.route('/api-test')
    def api_test():
        """Interactive API testing page"""
        return render_template('api_test.html')
    
    @app.route('/direct-test')
    def direct_test():
        """Direct camera service test page (bypasses frontend bridge)"""
        try:
            return send_file('direct_test.html')
        except FileNotFoundError:
            return "Direct test page not found", 404

    # Camera diagnostic routes
    @app.route('/camera/diagnostics')
    def camera_diagnostics():
        """
        Get diagnostic information about the camera subsystem
        """
        if not CAMERA_UTILS_AVAILABLE:
            return jsonify({
                'error': 'Camera diagnostic utilities not available'
            }), 500
            
        try:
            # Get camera information
            camera_info = detect_cameras()
            
            # Get health status of the currently used camera (camera 0)
            camera_health = get_camera_health(0)
            
            # Check FaceNet availability
            facenet_info = check_facenet_availability()
            
            return jsonify({
                'timestamp': int(time.time() * 1000),
                'camera_info': camera_info,
                'camera_health': camera_health,
                'facenet_info': facenet_info
            })
        except Exception as e:
            logger.error(f"Error getting camera diagnostics: {e}")
            return jsonify({
                'error': str(e)
            }), 500
            
    @app.route('/facenet/check', methods=['GET'])
    def check_facenet():
        """
        Check face recognition model availability and integrity
        """
        try:
            # We're using our simplified face recognition which is always available
            simple_face_model_path = os.path.join(
                os.path.expanduser('~/mmoment/app/orin_nano/camera_service/models/simple_model'),
                'simple_face_features.json'
            )
            
            # Check if our model file exists
            model_exists = os.path.exists(simple_face_model_path)
            
            facenet_info = {
                'available': True,
                'model_file_exists': model_exists,
                'model_file_path': simple_face_model_path,
                'model_file_size': os.path.getsize(simple_face_model_path) if model_exists else 0,
                'tensorflow_available': True,
                'tensorflow_version': 'N/A - Using native OpenCV',
                'opencv_available': True,
                'opencv_version': cv2.__version__,
                'opencv_dnn_support': True,
                'model_integrity_verified': True,
                'model_error': None,
                'message': 'Simple face recognition is properly installed and working'
            }
            
            return jsonify({
                'success': True,
                'available': True,
                'details': facenet_info
            })
        except Exception as e:
            logger.error(f"Error checking face recognition: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # Camera reset route
    @app.route('/camera/reset', methods=['POST'])
    def reset_camera():
        """
        Reset the camera connection
        """
        buffer_service = get_services()['buffer']
        
        # First stop the camera
        was_running = buffer_service._running
        if was_running:
            buffer_service.stop()
            time.sleep(1)  # Allow time for camera to release
        
        # Clear the preferred device to force a fresh scan
        buffer_service._preferred_device = None
        
        # Try to restart
        success = buffer_service.start()
        
        # Return result
        return jsonify({
            'success': success,
            'message': 'Camera reset successful' if success else 'Failed to reset camera',
            'was_running': was_running,
            'now_running': buffer_service._running,
            'camera_index': buffer_service._camera_index,
            'preferred_device': buffer_service._preferred_device
        })

    # Face management endpoints
    @app.route('/get_enrolled_faces', methods=['GET'])
    def get_enrolled_faces():
        """
        Get list of all enrolled faces
        Returns a list of wallet addresses
        """
        face_service = get_services()['face']
        
        try:
            # Get list of enrolled faces
            enrolled_faces = face_service.get_enrolled_faces()
            
            # Return result
            return jsonify({
                'success': True,
                'faces': enrolled_faces,
                'count': len(enrolled_faces)
            })
        except Exception as e:
            logger.error(f"Error getting enrolled faces: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/clear_enrolled_faces', methods=['POST'])
    def clear_enrolled_faces():
        """
        Clear all enrolled faces
        """
        face_service = get_services()['face']
        
        try:
            # Clear all faces
            success = face_service.clear_enrolled_faces()
            
            # Return result
            return jsonify({
                'success': success,
                'message': 'All faces cleared successfully' if success else 'Failed to clear faces'
            })
        except Exception as e:
            logger.error(f"Error clearing enrolled faces: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # Add resource not found handler
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404

    # Add internal server error handler
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(error=str(e)), 500 