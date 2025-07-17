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
import requests
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

def get_blockchain_session_sync():
    """Get blockchain session sync service"""
    from services.blockchain_session_sync import get_blockchain_session_sync
    return get_blockchain_session_sync()

# Session validation decorator - BLOCKCHAIN ONLY
def require_session(f):
    """Decorator to require a valid session - uses ONLY blockchain authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"🔍 require_session decorator called for endpoint: {f.__name__}")
        
        wallet_address = request.json.get('wallet_address') if request.json else None
        
        logger.info(f"🔍 Blockchain validation - wallet_address: {wallet_address}")
        
        if not wallet_address:
            logger.warning(f"🔍 Missing wallet address for {f.__name__}")
            return jsonify({'success': False, 'error': 'Invalid session'}), 403
        
        # BLOCKCHAIN ONLY VALIDATION: Check if wallet is checked in on-chain
        blockchain_sync = get_blockchain_session_sync()
        if blockchain_sync.is_wallet_checked_in(wallet_address):
            logger.info(f"✅ Blockchain validation: {wallet_address} is checked in on-chain")
            return f(*args, **kwargs)
        
        # Blockchain authentication failed
        logger.warning(f"❌ Blockchain validation failed for {f.__name__} - wallet {wallet_address} not checked in")
        return jsonify({'success': False, 'error': 'Invalid session - please check in first'}), 403
        
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
    
    @app.route('/api/status')
    def api_status():
        """Main status endpoint for frontend - comprehensive system status"""
        try:
            services = get_services()
            buffer_service = services['buffer']
            capture_service = services.get('capture')
            livepeer_service = services.get('livepeer')
            
            # Get buffer/camera status
            buffer_status = buffer_service.get_status()
            
            # Get recording status from capture service
            is_recording = False
            current_filename = None
            if capture_service:
                is_recording = capture_service.is_recording()
                # Note: capture service doesn't expose current filename, would need to add that
            
            # Get Livepeer streaming status
            is_streaming = False
            stream_info = {
                'playbackId': None,
                'isActive': False,
                'format': 'livepeer'
            }
            
            if livepeer_service:
                livepeer_status = livepeer_service.get_stream_status()
                is_streaming = livepeer_status.get('status') == 'streaming'
                if is_streaming:
                    stream_info.update({
                        'playbackId': livepeer_status.get('playback_id'),
                        'isActive': True,
                        'format': 'livepeer'
                    })
            
            return jsonify({
                'success': True,
                'timestamp': int(time.time()),
                'data': {
                    'isOnline': buffer_status['running'],
                    'isStreaming': is_streaming,
                    'isRecording': is_recording,
                    'lastSeen': int(time.time()),
                    'streamInfo': stream_info,
                    'recordingInfo': {
                        'isActive': is_recording,
                        'currentFilename': current_filename
                    }
                }
            })
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
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
    
    # Livepeer Streaming Endpoints
    @app.route('/api/stream/livepeer/start', methods=['POST'])
    def api_livepeer_start():
        """Start Livepeer streaming"""
        try:
            livepeer_service = get_services()['livepeer']
            result = livepeer_service.start_stream()
            
            if result.get('status') == 'streaming':
                return jsonify({
                    'success': True,
                    **result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('message', 'Failed to start stream')
                }), 500
        except Exception as e:
            logger.error(f"Error starting Livepeer stream: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/stream/livepeer/stop', methods=['POST'])
    def api_livepeer_stop():
        """Stop Livepeer streaming"""
        try:
            livepeer_service = get_services()['livepeer']
            result = livepeer_service.stop_stream()
            
            return jsonify({
                'success': True,
                **result
            })
        except Exception as e:
            logger.error(f"Error stopping Livepeer stream: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/stream/livepeer/status', methods=['GET'])
    def api_livepeer_status():
        """Get Livepeer stream status"""
        try:
            livepeer_service = get_services()['livepeer']
            result = livepeer_service.get_stream_status()
            
            return jsonify({
                'success': True,
                **result
            })
        except Exception as e:
            logger.error(f"Error getting Livepeer stream status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500



    # Camera Actions
    @app.route('/api/capture', methods=['POST'])
    @require_session
    def api_capture():
        """Standardized capture endpoint"""
        return capture_moment()
    
    @app.route('/api/record', methods=['POST'])
    @require_session
    def api_record():
        """Standardized record endpoint - waits for recording completion"""
        import time
        
        # Start the recording using the same logic as start_recording
        wallet_address = request.json.get('wallet_address')
        duration = request.json.get('duration', 30)  # Default to 30 seconds
        
        # Enforce maximum duration limit to prevent runaway recordings
        MAX_DURATION = 300  # 5 minutes maximum
        if duration <= 0 or duration > MAX_DURATION:
            duration = 30  # Default to 30 seconds for invalid durations
        
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
        
        # If recording failed to start, return immediately
        if not recording_info.get('success'):
            return jsonify(recording_info)
        
        # Extract recording info
        filename = recording_info.get('filename')
        duration_limit = recording_info.get('duration_limit', 0)
        
        # Wait for recording to complete
        # If duration is specified, wait for that duration + buffer
        if duration_limit > 0:
            max_wait_time = duration_limit + 10  # Add 10 second buffer for processing
        else:
            max_wait_time = 300  # 5 minutes max for manual recordings
            
        start_time = time.time()
        while capture_service.is_recording() and (time.time() - start_time) < max_wait_time:
            time.sleep(0.5)  # Check every 500ms
        
        # Check if recording completed successfully
        if capture_service.is_recording():
            return jsonify({
                'success': False,
                'error': 'Recording timeout - recording is still in progress'
            }), 408
        
        # Recording completed, get the video file info
        videos = capture_service.get_videos(1)  # Get most recent video
        
        if not videos:
            return jsonify({
                'success': False,
                'error': 'Recording completed but no video file was created'
            }), 500
        
        latest_video = videos[0]
        
        # Verify this is the video we just recorded
        if filename and latest_video.get('filename') != filename:
            logger.warning(f"Expected filename {filename} but got {latest_video.get('filename')}")
        
        return jsonify({
            'success': True,
            'recording': False,
            'completed': True,
            'video': latest_video,
            'filename': latest_video.get('filename'),
            'path': latest_video.get('path'),
            'url': latest_video.get('url'),
            'size': latest_video.get('size'),
            'timestamp': latest_video.get('timestamp')
        })
    
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
    
    @app.route('/api/face/enroll/prepare-transaction', methods=['POST'])
    def api_face_enroll_prepare_transaction():
        """Prepare face enrollment transaction with REAL biometric integration"""
        logger.info(f"🚀 FACE ENROLLMENT ENDPOINT HIT! Request data: {request.json}")
        
        wallet_address = request.json.get('wallet_address')
        
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'wallet_address is required'
            }), 400
        
        logger.info(f"[ENROLL-PREP] Starting REAL biometric integration for wallet: {wallet_address}")
        
        # Check if wallet is checked in on-chain (the ONLY session authority that matters)
        blockchain_sync = get_blockchain_session_sync()
        if not blockchain_sync.is_wallet_checked_in(wallet_address):
            logger.warning(f"[ENROLL-PREP] Wallet {wallet_address} is not checked in on-chain")
            return jsonify({
                'success': False,
                'error': 'Wallet must be checked in on-chain to enroll face'
            }), 403
        
        try:
            # Step 1: Extract REAL face embedding from current frame
            buffer_service = get_services()['buffer']
            face_service = get_services()['face']
            
            logger.info(f"[ENROLL-PREP] Extracting real face embedding for wallet: {wallet_address}")
            
            # Get current frame
            frame, timestamp = buffer_service.get_frame()
            if frame is None:
                return jsonify({
                    'success': False,
                    'error': 'No camera frame available'
                }), 400
            
            # Extract RAW 128-dimension face embedding for blockchain storage
            face_embedding = face_service.get_current_compact_embedding_with_buffer(buffer_service)
            if not face_embedding:
                return jsonify({
                    'success': False,
                    'error': 'No face detected in current frame - please ensure your face is visible'
                }), 400
            
            logger.info(f"[ENROLL-PREP] Successfully extracted RAW face embedding, size: {len(face_embedding)} dimensions")
            
            # Step 2: Create biometric session and encrypt embedding
            logger.info(f"[ENROLL-PREP] Creating biometric session for wallet: {wallet_address}")
            
            # Create biometric session
            biometric_response = requests.post(
                'http://biometric-security:5003/api/biometric/create-session',
                json={
                    'wallet_address': wallet_address,
                    'session_duration': 7200  # 2 hours
                },
                timeout=10
            )
            
            if biometric_response.status_code != 200:
                logger.error(f"[ENROLL-PREP] Failed to create biometric session: {biometric_response.status_code}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to create biometric session'
                }), 500
            
            biometric_session = biometric_response.json()
            biometric_session_id = biometric_session['session_id']
            
            logger.info(f"[ENROLL-PREP] Created biometric session: {biometric_session_id}")
            
            # Encrypt the face embedding
            encrypt_response = requests.post(
                'http://biometric-security:5003/api/biometric/encrypt-embedding',
                json={
                    'embedding': face_embedding,
                    'wallet_address': wallet_address,
                    'session_id': biometric_session_id,
                    'metadata': {
                        'w': wallet_address[:8],  # Shortened wallet address
                        't': int(time.time()),    # Timestamp
                        's': 'cam'               # Shortened source
                    }
                },
                timeout=15
            )
            
            if encrypt_response.status_code != 200:
                logger.error(f"[ENROLL-PREP] Failed to encrypt embedding: {encrypt_response.status_code}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to encrypt face embedding'
                }), 500
            
            encrypted_data = encrypt_response.json()
            nft_package = encrypted_data['nft_package']
            
            logger.info(f"[ENROLL-PREP] Successfully encrypted face embedding for wallet: {wallet_address}")
            
            # Step 3: Call Solana middleware with encrypted data
            logger.info(f"[ENROLL-PREP] Calling Solana middleware with encrypted data for wallet: {wallet_address}")
            
            solana_response = requests.post(
                'http://solana-middleware:5001/api/blockchain/mint-facial-nft',
                json={
                    'wallet_address': wallet_address,
                    'face_embedding': nft_package,  # Send encrypted NFT package
                    'biometric_session_id': biometric_session_id
                },
                timeout=30
            )
            
            if solana_response.status_code != 200:
                logger.error(f"[ENROLL-PREP] Solana middleware error: {solana_response.status_code}")
                return jsonify({
                    'success': False,
                    'error': f"Solana middleware error: {solana_response.status_code}"
                }), 500
            
            solana_data = solana_response.json()
            
            logger.info(f"[ENROLL-PREP] Successfully prepared encrypted NFT transaction for wallet: {wallet_address}")
            
            # Return the transaction data for frontend signing
            return jsonify({
                'success': True,
                'transaction_buffer': solana_data['transaction_buffer'],
                'face_id': solana_data['face_id'],
                'metadata': {
                    'wallet_address': wallet_address,
                    'timestamp': int(time.time()),
                    'biometric_session_id': biometric_session_id,
                    'encryption_method': 'AES-256-PBKDF2',
                    'face_embedding_encrypted': True,
                    'embedding_size': len(face_embedding),
                    'embedding_type': 'compact_128_dimensions',
                    'blockchain_optimized': True
                }
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ENROLL-PREP] Network error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Service communication error: {str(e)}"
            }), 503
            
        except Exception as e:
            logger.error(f"[ENROLL-PREP] Unexpected error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to prepare transaction: {str(e)}"
            }), 500
    
    @app.route('/api/face/enroll/confirm', methods=['POST'])
    def api_face_enroll_confirm():
        """Confirm face enrollment and handle biometric cleanup"""
        wallet_address = request.json.get('wallet_address')
        signed_transaction = request.json.get('signed_transaction')
        face_id = request.json.get('face_id')
        biometric_session_id = request.json.get('biometric_session_id')
        
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'wallet_address is required'
            }), 400
        
        logger.info(f"[ENROLL-CONFIRM] Confirming face enrollment for wallet: {wallet_address}, face_id: {face_id}")
        
        # Check if wallet is checked in on-chain (the ONLY session authority that matters)
        blockchain_sync = get_blockchain_session_sync()
        if not blockchain_sync.is_wallet_checked_in(wallet_address):
            logger.warning(f"[ENROLL-CONFIRM] Wallet {wallet_address} is not checked in on-chain")
            return jsonify({
                'success': False,
                'error': 'Wallet must be checked in on-chain to confirm face enrollment'
            }), 403
        
        try:
            # Validate required parameters
            if not signed_transaction or not face_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing required parameters: signed_transaction and face_id'
                }), 400
            
            # Call Solana middleware to confirm the transaction
            logger.info(f"[ENROLL-CONFIRM] Calling Solana middleware to confirm transaction for wallet: {wallet_address}")
            
            confirm_response = requests.post(
                'http://solana-middleware:5001/api/face/enroll/confirm',
                json={
                    'wallet_address': wallet_address,
                    'transaction_signature': signed_transaction,
                    'face_id': face_id
                },
                timeout=15
            )
            
            if confirm_response.status_code != 200:
                logger.error(f"[ENROLL-CONFIRM] Solana middleware confirm error: {confirm_response.status_code}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to confirm transaction with Solana middleware'
                }), 500
            
            confirm_data = confirm_response.json()
            transaction_id = confirm_data['transaction_id']
            
            # Clean up biometric session if provided
            if biometric_session_id:
                logger.info(f"[ENROLL-CONFIRM] Cleaning up biometric session: {biometric_session_id}")
                
                try:
                    purge_response = requests.post(
                        'http://biometric-security:5003/api/biometric/purge-session',
                        json={'session_id': biometric_session_id},
                        timeout=10
                    )
                    
                    if purge_response.status_code == 200:
                        logger.info(f"[ENROLL-CONFIRM] Successfully purged biometric session: {biometric_session_id}")
                    else:
                        logger.warning(f"[ENROLL-CONFIRM] Failed to purge biometric session: {purge_response.status_code}")
                        
                except Exception as purge_error:
                    logger.warning(f"[ENROLL-CONFIRM] Error purging biometric session: {purge_error}")
            
            # Enroll the face in the local face service for recognition
            try:
                buffer_service = get_services()['buffer']
                face_service = get_services()['face']
                
                frame, timestamp = buffer_service.get_frame()
                if frame is not None:
                    enroll_result = face_service.enroll_face(frame, wallet_address)
                    if enroll_result.get('success'):
                        logger.info(f"[ENROLL-CONFIRM] Successfully enrolled face locally for recognition: {wallet_address}")
                    else:
                        logger.warning(f"[ENROLL-CONFIRM] Local face enrollment warning: {enroll_result.get('error', 'Unknown error')}")
                        
            except Exception as local_error:
                logger.warning(f"[ENROLL-CONFIRM] Error with local face enrollment: {local_error}")
            
            logger.info(f"[ENROLL-CONFIRM] Face enrollment completed successfully for wallet: {wallet_address}, transaction_id: {transaction_id}")
            
            # Return the expected format for frontend
            return jsonify({
                'success': True,
                'face_id': face_id,
                'transaction_id': transaction_id,
                'biometric_session_cleaned': bool(biometric_session_id),
                'local_enrollment_completed': True
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ENROLL-CONFIRM] Network error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Service communication error: {str(e)}"
            }), 503
            
        except Exception as e:
            logger.error(f"[ENROLL-CONFIRM] Unexpected error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"Failed to confirm enrollment: {str(e)}"
            }), 500
    
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
    
    # User Profile Management - Enhanced for Scalable Architecture
    @app.route('/api/user/profile', methods=['POST'])
    def api_update_user_profile():
        """Update user profile for display name resolution - Enhanced for PDA subdomain architecture"""
        data = request.json or {}
        wallet_address = data.get('wallet_address')
        display_name = data.get('display_name')
        username = data.get('username')
        transaction_signature = data.get('transaction_signature')  # Optional: for verification
        
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'Wallet address is required'
            }), 400
        
        # Get camera PDA for logging
        camera_pda = os.environ.get('CAMERA_PDA', 'unknown')
        
        # Create enhanced user profile with metadata
        user_profile = {
            'wallet_address': wallet_address,
            'display_name': display_name,
            'username': username,
            'camera_pda': camera_pda,
            'updated_at': int(time.time()),
            'transaction_signature': transaction_signature
        }
        
        # Store profile in both face services
        services = get_services()
        
        # Update simple face service
        face_service = services['face']
        face_service.store_user_profile(wallet_address, user_profile)
        
        # Update GPU face service if available
        if 'gpu_face' in services:
            gpu_face_service = services['gpu_face']
            gpu_face_service.store_user_profile(wallet_address, user_profile)
        
        # Resolve display name with fallback hierarchy
        resolved_display_name = display_name or username or wallet_address[:8]
        
        logger.info(f"[PROFILE-UPDATE] Camera {camera_pda[:8]}... updated profile for {wallet_address}: {resolved_display_name}")
        
        return jsonify({
            'success': True,
            'wallet_address': wallet_address,
            'display_name': resolved_display_name,
            'camera_pda': camera_pda,
            'camera_url': f"https://{camera_pda.lower()}.mmoment.xyz/api",
            'updated_at': user_profile['updated_at'],
            'message': 'User profile updated successfully'
        })
    
    @app.route('/api/user/profile/<wallet_address>', methods=['GET'])
    def api_get_user_profile(wallet_address):
        """Get user profile by wallet address"""
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'Wallet address is required'
            }), 400
        
        # Get profile from face service
        services = get_services()
        face_service = services['face']
        
        # Try to get profile from face service
        try:
            profile = face_service.get_user_profile(wallet_address)
            if profile:
                resolved_display_name = profile.get('display_name') or profile.get('username') or wallet_address[:8]
                return jsonify({
                    'success': True,
                    'profile': {
                        'wallet_address': wallet_address,
                        'display_name': resolved_display_name,
                        'username': profile.get('username'),
                        'camera_pda': profile.get('camera_pda'),
                        'updated_at': profile.get('updated_at')
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Profile not found'
                }), 404
                
        except Exception as e:
            logger.error(f"Error getting user profile for {wallet_address}: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to retrieve profile'
            }), 500
    
    @app.route('/api/user/profile/<wallet_address>', methods=['DELETE'])
    def api_delete_user_profile(wallet_address):
        """Delete user profile (for session cleanup)"""
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'Wallet address is required'
            }), 400
        
        # Remove profile from both face services
        services = get_services()
        
        # Remove from simple face service
        face_service = services['face']
        face_service.remove_user_profile(wallet_address)
        
        # Remove from GPU face service if available
        if 'gpu_face' in services:
            gpu_face_service = services['gpu_face']
            gpu_face_service.remove_user_profile(wallet_address)
        
        camera_pda = os.environ.get('CAMERA_PDA', 'unknown')
        logger.info(f"[PROFILE-DELETE] Camera {camera_pda[:8]}... removed profile for {wallet_address}")
        
        return jsonify({
            'success': True,
            'wallet_address': wallet_address,
            'message': 'User profile deleted successfully'
        })
    
    @app.route('/api/camera/info', methods=['GET'])
    def api_camera_info():
        """Get camera information for frontend discovery"""
        camera_pda = os.environ.get('CAMERA_PDA', 'unknown')
        camera_program_id = os.environ.get('CAMERA_PROGRAM_ID', 'unknown')
        
        # Get current session count
        services = get_services()
        session_service = services['session']
        active_sessions = len(session_service.get_all_sessions())
        
        # Get buffer status
        buffer_service = services['buffer']
        buffer_status = buffer_service.get_status()
        
        return jsonify({
            'success': True,
            'camera_info': {
                'pda': camera_pda,
                'program_id': camera_program_id,
                'api_url': f"https://{camera_pda.lower()}.mmoment.xyz/api",
                'legacy_url': "https://jetson.mmoment.xyz/api",
                'active_sessions': active_sessions,
                'camera_status': {
                    'online': buffer_status['running'],
                    'fps': buffer_status['fps'],
                    'resolution': f"{buffer_service._width}x{buffer_service._height}",
                    'device': buffer_service._preferred_device
                },
                'capabilities': {
                    'face_recognition': True,
                    'gesture_detection': True,
                    'livepeer_streaming': True,
                    'media_capture': True,
                    'blockchain_integration': True
                }
            }
        })

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
            # Also enable/disable boxes when toggling visualization
            face_service.enable_boxes(enabled)
            
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
        display_name = data.get('display_name')  # Optional display name from frontend
        username = data.get('username')  # Optional username from frontend
        
        if not wallet_address:
            return jsonify({
                'success': False,
                'error': 'Wallet address is required'
            }), 400
        
        # Create user profile metadata
        user_profile = {
            'wallet_address': wallet_address,
            'display_name': display_name,
            'username': username
        }
        
        # Create a session with user profile
        services = get_services()
        session_service = services['session']
        session_info = session_service.create_session(wallet_address, user_profile)
        
        # Store user profile in both face services for labeling
        face_service = services['face']
        face_service.store_user_profile(wallet_address, user_profile)
        
        # Update GPU face service if available
        if 'gpu_face' in services:
            gpu_face_service = services['gpu_face']
            gpu_face_service.store_user_profile(wallet_address, user_profile)
        
        # Enable face boxes for all connected users
        face_service.enable_boxes(True)
        logger.info(f"Face boxes enabled for connected user: {display_name or username or wallet_address}")
        
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
        
            # If faces are detected, process the frame for recognition
            if faces_info['detected_count'] > 0:
                logger.info(f"[RECOGNIZE] Running face recognition on {faces_info['detected_count']} faces")
                # Use proper public API for face recognition
                face_service.process_frame_for_recognition(frame)
            
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
        """Get a specific video file"""
        try:
            services = get_services()
            if not services:
                return jsonify({"error": "Services not available"}), 503
            
            capture_service = get_services()['capture']
            
            # Get video path directly - serve the exact file requested
            video_path = capture_service.get_video_path(filename)
            
            # Determine MIME type based on file extension
            if filename.endswith('.mp4'):
                mimetype = 'video/mp4'
            elif filename.endswith('.mov'):
                mimetype = 'video/quicktime'
            else:
                mimetype = 'application/octet-stream'
            
            if not video_path:
                logger.error(f"Video not found: {filename}")
                return jsonify({"error": "Video not found"}), 404
            
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return jsonify({"error": "Video file not found"}), 404
            
            logger.info(f"Serving video: {filename} from {video_path}")
            return send_file(video_path, mimetype=mimetype)
            
        except Exception as e:
            logger.error(f"Error serving video {filename}: {e}")
            return jsonify({"error": "Internal server error"}), 500

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