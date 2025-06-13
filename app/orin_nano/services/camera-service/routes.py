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
            'version': '3.0.0',
            'description': 'Standardized camera API for Jetson with GPU-accelerated computer vision',
            'buffer_based': True,
            'endpoints': {
                'health': '/api/health',
                'stream_info': '/api/stream/info',
                'stream': '/stream',
                'livepeer_start': '/api/stream/livepeer/start',
                'livepeer_stop': '/api/stream/livepeer/stop',
                'livepeer_status': '/api/stream/livepeer/status',
                'livepeer_reload': '/api/stream/livepeer/reload',
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
            'features': {
                'gpu_acceleration': True,
                'face_recognition': True,
                'gesture_detection': True,
                'real_time_processing': True,
                'session_management': True
            }
        })

    # ========================================
    # STANDARDIZED API ROUTES (Pi5 Compatible)
    # ========================================
    
    # Health & Status
    @app.route('/api/health')
    def api_health():
        """Standardized health check endpoint"""
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
        data = request.json or {}
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        
        try:
            # Get services
            buffer_service = get_services()['buffer']
            
            # Get the current frame
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'No camera frame available'
                }), 400
            
            # Save the photo
            photos_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/photos')
            os.makedirs(photos_dir, exist_ok=True)
            
            # Generate filename with timestamp
            filename = f"photo_{int(timestamp * 1000)}.jpg"
            filepath = os.path.join(photos_dir, filename)
            
            # Save the frame as JPEG
            cv2.imwrite(filepath, frame)
            
            logger.info(f"Photo captured: {filename} for wallet: {wallet_address}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'timestamp': timestamp,
                'wallet_address': wallet_address
            })
            
        except Exception as e:
            logger.error(f"Error capturing photo: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Photo capture failed: {str(e)}"
            }), 500
    
    @app.route('/api/record', methods=['POST'])
    @require_session
    def api_record():
        """Standardized record endpoint"""
        data = request.json or {}
        action = data.get('action', 'start')  # 'start' or 'stop'
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        
        try:
            buffer_service = get_services()['buffer']
            capture_service = get_services()['capture']
            
            if action == 'start':
                # Start recording using the capture service
                result = capture_service.start_recording(buffer_service, user_id=wallet_address)
                
                if result['success']:
                    logger.info(f"Recording started: {result['filename']} for wallet: {wallet_address}")
                    return jsonify({
                        'success': True,
                        'action': 'started',
                        'filename': result['filename'],
                        'wallet_address': wallet_address
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Failed to start recording')
                    }), 500
                    
            elif action == 'stop':
                # Stop recording using the capture service
                result = capture_service.stop_recording()
                
                if result['success']:
                    logger.info(f"Recording stopped for wallet: {wallet_address}")
                    return jsonify({
                        'success': True,
                        'action': 'stopped',
                        'filename': result.get('filename'),
                        'wallet_address': wallet_address
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result.get('error', 'Failed to stop recording')
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid action. Use "start" or "stop"'
                }), 400
                
        except Exception as e:
            logger.error(f"Error in recording: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Recording failed: {str(e)}"
            }), 500
    
    # Media Access
    @app.route('/api/photos')
    def api_photos():
        """Standardized photos list endpoint"""
        try:
            photos_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/photos')
            
            if not os.path.exists(photos_dir):
                return jsonify({
                    'success': True,
                    'photos': [],
                    'count': 0
                })
            
            # Get all photo files
            photo_files = []
            for filename in os.listdir(photos_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(photos_dir, filename)
                    stat = os.stat(filepath)
                    photo_files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created': stat.st_ctime,
                        'url': f'/api/photos/{filename}'
                    })
            
            # Sort by creation time (newest first)
            photo_files.sort(key=lambda x: x['created'], reverse=True)
            
            return jsonify({
                'success': True,
                'photos': photo_files,
                'count': len(photo_files)
            })
            
        except Exception as e:
            logger.error(f"Error listing photos: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to list photos: {str(e)}"
            }), 500
    
    @app.route('/api/videos')
    def api_videos():
        """Standardized videos list endpoint"""
        try:
            videos_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/videos')
            
            if not os.path.exists(videos_dir):
                return jsonify({
                    'success': True,
                    'videos': [],
                    'count': 0
                })
            
            # Get all video files
            video_files = []
            for filename in os.listdir(videos_dir):
                if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                    filepath = os.path.join(videos_dir, filename)
                    stat = os.stat(filepath)
                    video_files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'created': stat.st_ctime,
                        'url': f'/api/videos/{filename}'
                    })
            
            # Sort by creation time (newest first)
            video_files.sort(key=lambda x: x['created'], reverse=True)
            
            return jsonify({
                'success': True,
                'videos': video_files,
                'count': len(video_files)
            })
            
        except Exception as e:
            logger.error(f"Error listing videos: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to list videos: {str(e)}"
            }), 500
    
    @app.route('/api/photos/<filename>')
    def api_get_photo(filename):
        """Standardized photo access endpoint"""
        try:
            photos_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/photos')
            filepath = os.path.join(photos_dir, filename)
            
            if not os.path.exists(filepath):
                return jsonify({'error': 'Photo not found'}), 404
            
            return send_file(filepath, mimetype='image/jpeg')
            
        except Exception as e:
            logger.error(f"Error serving photo {filename}: {str(e)}")
            return jsonify({'error': 'Failed to serve photo'}), 500
    
    @app.route('/api/videos/<filename>')
    def api_get_video(filename):
        """Standardized video access endpoint"""
        try:
            videos_dir = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/videos')
            filepath = os.path.join(videos_dir, filename)
            
            if not os.path.exists(filepath):
                return jsonify({'error': 'Video not found'}), 404
            
            return send_file(filepath, mimetype='video/mp4')
            
        except Exception as e:
            logger.error(f"Error serving video {filename}: {str(e)}")
            return jsonify({'error': 'Failed to serve video'}), 500
    
    # Session Management
    @app.route('/api/session/connect', methods=['POST'])
    def api_session_connect():
        """Standardized session connect endpoint"""
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
    
    @app.route('/api/session/disconnect', methods=['POST'])
    @require_session
    def api_session_disconnect():
        """Standardized session disconnect endpoint"""
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
    
    # Computer Vision (Jetson-specific)
    @app.route('/api/face/enroll', methods=['POST'])
    @require_session
    def api_face_enroll():
        """Standardized face enrollment endpoint"""
        data = request.json or {}
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        user_id = data.get('user_id', wallet_address)  # Use user_id if provided, otherwise wallet_address
        
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
            num_tries = 20
            
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
                faces_status = face_service.get_status()
                
                logger.info(f"[ENROLL] Frame {i+1}/{num_tries} detected {faces_status['detected_faces']} faces for wallet: {wallet_address}")
                
                if faces_status['detected_faces'] > 0:
                    face_detected = True
                    best_frame = frame
                    logger.info(f"[ENROLL] Found face on frame {i+1}/{num_tries} for wallet: {wallet_address}")
                    break
                
                # Wait a bit for the next frame
                time.sleep(0.1)
            
            # If no frames had a face after multiple attempts
            if not face_detected or best_frame is None:
                logger.warning(f"[ENROLL] No face detected after checking {num_tries} frames for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': 'No face was detected. Please make sure your face is clearly visible in the camera view and you have adequate lighting.'
                }), 400
            
            # Check if multiple faces are detected
            faces_status = face_service.get_status()
            if faces_status['detected_faces'] > 1:
                logger.warning(f"[ENROLL] Multiple faces ({faces_status['detected_faces']}) detected for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': 'Multiple faces detected. Please ensure only your face is visible in the camera view.'
                }), 400
            
            # We found a frame with a face - use it for enrollment
            logger.info(f"[ENROLL] Calling face_service.enroll_face for wallet: {wallet_address}")
            result = face_service.enroll_face(best_frame, user_id)
            
            if not result['success']:
                logger.warning(f"[ENROLL] Face enrollment failed: {result.get('error', 'Unknown error')} for wallet: {wallet_address}")
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Failed to enroll face. Please try again with better lighting and positioning.')
                }), 400
            
            logger.info(f"[ENROLL] Face enrollment successful for wallet: {wallet_address}")
            
            return jsonify({
                'success': True,
                'wallet_address': wallet_address,
                'user_id': user_id,
                'message': 'Face enrolled successfully'
            })
            
        except Exception as e:
            logger.error(f"[ENROLL] Error in face enrollment: {str(e)} for wallet: {wallet_address}")
            return jsonify({
                'success': False,
                'error': f"Face enrollment failed: {str(e)}. Please try again or contact support if the issue persists."
            }), 500
    
    @app.route('/api/face/recognize', methods=['POST'])
    def api_face_recognize():
        """Standardized face recognition endpoint"""
        try:
            # Get services
            buffer_service = get_services()['buffer']
            gpu_face_service = get_services().get('gpu_face')
            
            # Get the current frame
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'No camera frame available'
                }), 400
            
            # Use GPU face service for detection and recognition
            if gpu_face_service:
                faces = gpu_face_service.detect_and_recognize_faces(frame)
                return jsonify({
                    'success': True,
                    'faces': faces,
                    'timestamp': timestamp
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'GPU face service not available'
                }), 500
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Face recognition failed: {str(e)}"
            }), 500

    @app.route('/api/face/detect', methods=['POST'])
    def api_face_detect():
        """Face detection endpoint that returns embeddings"""
        try:
            # Get services
            buffer_service = get_services()['buffer']
            gpu_face_service = get_services().get('gpu_face')
            
            if not gpu_face_service:
                return jsonify({
                    'success': False,
                    'error': 'GPU face service not available'
                }), 500
            
            # Get the current frame
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'No camera frame available'
                }), 400
            
            # Detect and extract face embeddings
            faces = gpu_face_service.detect_and_recognize_faces(frame)
            
            # Extract embeddings for each detected face and ensure JSON serialization
            processed_faces = []
            for face in faces:
                processed_face = {}
                for key, value in face.items():
                    # Convert numpy arrays to lists for JSON serialization
                    if hasattr(value, 'tolist'):
                        processed_face[key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        processed_face[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                    else:
                        processed_face[key] = value
                
                # Extract additional embedding if needed
                bbox_key = 'bbox' if 'bbox' in face else 'box'
                if bbox_key in face:
                    x1, y1, x2, y2 = face[bbox_key]
                    face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    if face_img.size > 0:
                        embedding = gpu_face_service.extract_face_embedding(face_img)
                        if embedding is not None:
                            processed_face['embedding'] = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                processed_faces.append(processed_face)
            
            faces = processed_faces
            
            return jsonify({
                'success': True,
                'faces': faces,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Face detection failed: {str(e)}"
            }), 500
    
    @app.route('/api/gesture/current', methods=['GET'])
    def api_gesture_current():
        """Standardized current gesture endpoint"""
        try:
            gesture_service = get_services()['gesture']
            gesture_info = gesture_service.get_current_gesture()
            
            # Extract gesture name from the gesture info
            gesture_name = 'none'
            if gesture_info and isinstance(gesture_info, dict):
                gesture_name = gesture_info.get('gesture', 'none')
            elif gesture_info and isinstance(gesture_info, str):
                gesture_name = gesture_info
            
            return jsonify({
                'success': True,
                'gesture': gesture_name
            })
        except Exception as e:
            logger.error(f"Error getting current gesture: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/visualization/face', methods=['POST'])
    def api_visualization_face():
        """Standardized face visualization toggle"""
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
    
    @app.route('/api/visualization/gesture', methods=['POST'])
    def api_visualization_gesture():
        """Standardized gesture visualization toggle"""
        data = request.json or {}
        enabled = data.get('enabled', True)
        
        try:
            gesture_service = get_services()['gesture']
            gesture_service.enable_visualization(enabled)
            
            logger.info(f"Gesture visualization toggled to: {enabled}")
            
            return jsonify({
                'success': True,
                'enabled': enabled
            })
        except Exception as e:
            logger.error(f"Error toggling gesture visualization: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # ========================================
    # LIVEPEER STREAMING ROUTES
    # ========================================
    
    @app.route('/api/stream/livepeer/start', methods=['POST'])
    def api_livepeer_start():
        """Start Livepeer streaming"""
        try:
            livepeer_service = get_services()['livepeer']
            result = livepeer_service.start_stream()
            
            logger.info(f"Livepeer stream start result: {result}")
            
            return jsonify(result), 200 if result.get('status') == 'success' else 500
            
        except Exception as e:
            logger.error(f"Error starting Livepeer stream: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to start stream: {str(e)}'
            }), 500

    @app.route('/api/stream/livepeer/stop', methods=['POST'])
    def api_livepeer_stop():
        """Stop Livepeer streaming"""
        try:
            livepeer_service = get_services()['livepeer']
            result = livepeer_service.stop_stream()
            
            logger.info(f"Livepeer stream stop result: {result}")
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error stopping Livepeer stream: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to stop stream: {str(e)}'
            }), 500

    @app.route('/api/stream/livepeer/status', methods=['GET'])
    def api_livepeer_status():
        """Get Livepeer streaming status"""
        try:
            livepeer_service = get_services()['livepeer']
            status = livepeer_service.get_stream_status()
            
            return jsonify(status), 200
            
        except Exception as e:
            logger.error(f"Error getting Livepeer stream status: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to get stream status: {str(e)}'
            }), 500

    @app.route('/api/stream/livepeer/reload', methods=['POST'])
    def api_livepeer_reload():
        """Reload Livepeer streaming configuration"""
        try:
            livepeer_service = get_services()['livepeer']
            
            # Stop current stream if running
            if livepeer_service.get_stream_status().get('is_streaming', False):
                livepeer_service.stop_stream()
                time.sleep(2)  # Allow time for cleanup
            
            # Reset the service instance to reload configuration
            from services.livepeer_stream_service import LivepeerStreamService
            LivepeerStreamService.reset_instance()
            
            # Get new instance with fresh config
            new_livepeer_service = LivepeerStreamService()
            buffer_service = get_services()['buffer']
            new_livepeer_service.set_buffer_service(buffer_service)
            
            # Update the services dict
            get_services()['livepeer'] = new_livepeer_service
            
            logger.info("Livepeer service configuration reloaded")
            
            return jsonify({
                'status': 'success',
                'message': 'Livepeer configuration reloaded successfully'
            }), 200
            
        except Exception as e:
            logger.error(f"Error reloading Livepeer configuration: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to reload configuration: {str(e)}'
            }), 500

    # ========================================
    # CORE UTILITY ROUTES
    # ========================================

    # Stream route
    @app.route('/stream')
    def stream():
        """
        Stream the camera feed as MJPEG
        This is a special endpoint that continuously streams JPEG frames
        """
        services = get_services()
        buffer_service = services['buffer']
        face_service = services.get('face')
        gesture_service = services.get('gesture')
        
        def generate():
            """Generator function for MJPEG streaming"""
            while True:
                # Get raw frame from buffer
                frame, timestamp = buffer_service.get_frame()
                
                if frame is None:
                    # If no frame is available, wait and try again
                    time.sleep(0.03)
                    continue
                
                # Apply overlays manually from services
                processed_frame = frame.copy()
                
                # Apply face service overlays if available
                if face_service:
                    try:
                        processed_frame = face_service.get_processed_frame(processed_frame)
                    except Exception as e:
                        logger.error(f"Error applying face overlays: {e}")
                
                # Apply gesture service overlays if available
                if gesture_service:
                    try:
                        processed_frame = gesture_service.get_processed_frame(processed_frame)
                    except Exception as e:
                        logger.error(f"Error applying gesture overlays: {e}")
                
                # Encode the processed frame to JPEG
                _, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                jpeg_data = jpeg.tobytes()
                
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

    # ========================================
    # UTILITY ENDPOINTS (Keep for diagnostics)
    # ========================================

    # Face management endpoints
    @app.route('/get_enrolled_faces', methods=['GET'])
    def get_enrolled_faces():
        """
        Get list of all enrolled faces
        Returns a list of user IDs
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

    # Camera diagnostic routes
    @app.route('/camera/diagnostics')
    def camera_diagnostics():
        """
        Get diagnostic information about the camera subsystem
        """
        try:
            services = get_services()
            buffer_service = services['buffer']
            buffer_status = buffer_service.get_status()
            
            return jsonify({
                'timestamp': int(time.time() * 1000),
                'camera_health': buffer_status,
                'services_available': {
                    'buffer': True,
                    'face': 'face' in services,
                    'gesture': 'gesture' in services,
                    'session': 'session' in services
                }
            })
        except Exception as e:
            logger.error(f"Error getting camera diagnostics: {e}")
            return jsonify({
                'error': str(e)
            }), 500
            
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

    # Web interface routes
    @app.route('/local-test')
    def local_test():
        """Local camera test page"""
        return render_template('local_test.html')

    # Add resource not found handler
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404

    # Add internal server error handler
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(error=str(e)), 500 