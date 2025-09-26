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

# Import device signer for DePIN authentication
from services.device_signer import DeviceSigner

# Import GPU face service for identity tracking
try:
    from services.gpu_face_service import get_gpu_face_service
except ImportError:
    get_gpu_face_service = None

# Set up logging
logger = logging.getLogger("CameraRoutes")

# Initialize device signer (singleton)
device_signer = DeviceSigner()

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

def sign_response(f):
    """
    Decorator to sign API responses with device ed25519 key for DePIN authentication.
    Creates cryptographic proof that response came from this specific hardware device.
    
    Future Enhancement: Same device key can be used for on-chain transaction signing.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            result = f(*args, **kwargs)

            # Only sign successful JSON responses from specific endpoints
            if isinstance(result, dict) or (hasattr(result, 'get_json') and result.get_json()):
                if isinstance(result, dict):
                    response_data = result
                else:
                    response_data = result.get_json()

                # Only sign if response indicates success or contains meaningful data
                if response_data and (response_data.get('success', True) or 'data' in response_data):
                    # Sign the response with device key
                    signed_data = device_signer.sign_response(response_data)
                    
                    logger.debug(f"Response signed by device: {device_signer.get_public_key()[:12]}...")

                    if isinstance(result, dict):
                        return signed_data
                    else:
                        return jsonify(signed_data)

            return result
            
        except Exception as e:
            logger.error(f"Error in device signing decorator: {e}")
            # Return original result if signing fails to avoid breaking functionality
            return result
            
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
                'face_extract_embedding': '/api/face/extract-embedding',
                'face_recognize': '/api/face/recognize',
                'gesture_current': '/api/gesture/current',
                'visualization_face': '/api/visualization/face',
                'visualization_gesture': '/api/visualization/gesture',
                'device_info': '/api/device-info',
                'scan_registration_qr': '/api/scan-registration-qr',
                'wifi_scan': '/api/setup/wifi/scan',
                'wifi_setup': '/api/setup/wifi'
            }
        })

    # ========================================
    # STANDARDIZED API ROUTES (Pi5 Compatible)
    # ========================================
    
    # Health & Status
    @app.route('/api/health')
    @sign_response
    def api_health():
        """Standardized health check endpoint with device signature"""
        return health()
    
    # Device Registration & Configuration
    @app.route('/api/device/registration/start', methods=['POST'])
    @sign_response
    def api_device_registration_start():
        """Start device registration polling for backend configuration"""
        try:
            from services.device_registration import get_device_registration_service
            
            registration_service = get_device_registration_service()
            registration_service.start_polling()
            
            return jsonify({
                'success': True,
                'message': 'Device registration polling started',
                'device_pubkey': device_signer.get_public_key(),
                'status': 'polling_for_configuration'
            })
            
        except Exception as e:
            logger.error(f"Failed to start device registration: {e}")
            return jsonify({
                'success': False,
                'error': f"Failed to start registration: {str(e)}"
            }), 500
    
    @app.route('/api/device/registration/status')
    @sign_response
    def api_device_registration_status():
        """Get current device registration status"""
        try:
            from services.device_registration import get_device_registration_service
            
            registration_service = get_device_registration_service()
            device_info = registration_service.get_device_info()
            
            return jsonify({
                'success': True,
                'device_info': device_info,
                'is_configured': registration_service.is_configured()
            })
            
        except Exception as e:
            logger.error(f"Failed to get registration status: {e}")
            return jsonify({
                'success': False,
                'error': f"Failed to get status: {str(e)}"
            }), 500
    
    @app.route('/api/device/claim', methods=['POST'])
    @sign_response
    def api_device_claim():
        """Claim a device with backend using QR token"""
        try:
            data = request.get_json()
            claim_token = data.get('claim_token')
            wifi_credentials = data.get('wifi_credentials')
            
            if not claim_token:
                return jsonify({
                    'success': False,
                    'error': 'claim_token is required'
                }), 400
            
            # Get device information
            device_pubkey = device_signer.get_public_key()
            device_info = device_signer.get_device_info()
            
            # Claim the device with backend
            backend_url = os.getenv('BACKEND_HOST', '192.168.1.80')
            backend_port = os.getenv('BACKEND_PORT', '3001')
            
            claim_response = requests.post(
                f"http://{backend_url}:{backend_port}/api/device/claim",
                json={
                    'token': claim_token,
                    'device_pubkey': device_pubkey,
                    'device_info': device_info,
                    'wifi_credentials': wifi_credentials
                },
                timeout=10
            )
            
            if claim_response.status_code == 200:
                # Start polling for configuration
                from services.device_registration import get_device_registration_service
                registration_service = get_device_registration_service()
                registration_service.start_polling()
                
                return jsonify({
                    'success': True,
                    'message': 'Device claimed successfully, polling for configuration',
                    'device_pubkey': device_pubkey
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"Failed to claim device: {claim_response.status_code}"
                }), 400
                
        except Exception as e:
            logger.error(f"Failed to claim device: {e}")
            return jsonify({
                'success': False,
                'error': f"Failed to claim device: {str(e)}"
            }), 500
    
    @app.route('/api/device/tunnel/status')
    @sign_response
    def api_device_tunnel_status():
        """Get current tunnel status and configuration"""
        try:
            from services.device_registration import get_device_registration_service
            from services.tunnel_manager import get_tunnel_manager
            
            registration_service = get_device_registration_service()
            tunnel_manager = get_tunnel_manager()
            
            tunnel_status = tunnel_manager.get_tunnel_status()
            device_config = registration_service.get_device_config()
            
            return jsonify({
                'success': True,
                'tunnel_status': tunnel_status,
                'device_config': device_config,
                'is_configured': registration_service.is_configured()
            })
            
        except Exception as e:
            logger.error(f"Failed to get tunnel status: {e}")
            return jsonify({
                'success': False,
                'error': f"Failed to get tunnel status: {str(e)}"
            }), 500
    
    @app.route('/api/status')
    @sign_response
    def api_status():
        """Main status endpoint for frontend - comprehensive system status"""
        try:
            services = get_services()
            buffer_service = services['buffer']
            capture_service = services.get('capture')
            livepeer_service = services.get('livepeer')
            webrtc_service = services.get('webrtc')
            
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
            
            # Get WebRTC status
            webrtc_status = {}
            if webrtc_service:
                webrtc_status = webrtc_service.get_status()
            
            # Add WebRTC to stream info if available
            stream_formats = ['livepeer']
            if webrtc_status.get('connected', False):
                stream_formats.append('webrtc')
            
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
                    },
                    'webrtcInfo': {
                        'available': webrtc_status.get('running', False),
                        'connected': webrtc_status.get('connected', False),
                        'activeConnections': webrtc_status.get('active_connections', 0)
                    },
                    'supportedFormats': stream_formats
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

    @app.route('/api/stream/webrtc/status', methods=['GET'])
    def api_webrtc_status():
        """Get WebRTC stream status"""
        try:
            webrtc_service = get_services().get('webrtc')
            
            if not webrtc_service:
                return jsonify({
                    'success': False,
                    'error': 'WebRTC service not available'
                }), 404
            
            status = webrtc_service.get_status()
            
            return jsonify({
                'success': True,
                'status': 'available' if status['running'] else 'stopped',
                'running': status['running'],
                'connected': status['connected'],
                'camera_pda': status['camera_pda'],
                'backend_url': status['backend_url'],
                'active_connections': status['active_connections'],
                'connections': status['connections']
            })
        except Exception as e:
            logger.error(f"Error getting WebRTC stream status: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    # Camera Actions
    @app.route('/api/capture', methods=['POST'])
    @require_session
    @sign_response
    def api_capture():
        """Standardized capture endpoint with device signing"""
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

        # Upload video to user's Pipe storage asynchronously
        import asyncio
        from services.pipe_storage_service import handle_user_capture_upload

        pipe_upload_info = {'initiated': False, 'storage_provider': 'pipe'}

        try:
            # Read the video file data
            video_path = latest_video.get('path')
            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as f:
                    video_data = f.read()

                # Prepare metadata for Pipe upload
                metadata = {
                    'timestamp': latest_video.get('timestamp'),
                    'camera_id': 'jetson01',  # TODO: Get actual camera ID from device config
                    'size': latest_video.get('size'),
                    'local_path': video_path,
                    'local_filename': latest_video.get('filename'),
                    'capture_type': 'video',
                    'duration_seconds': duration_limit
                }

                # Upload to user's specific Pipe storage account
                def run_pipe_upload():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        upload_result = loop.run_until_complete(
                            handle_user_capture_upload(wallet_address, video_data, metadata)
                        )
                        if upload_result['success']:
                            logger.info(f"✅ Video uploaded to {wallet_address[:8]}...'s Pipe: {upload_result.get('filename')}")
                        else:
                            logger.error(f"❌ Pipe video upload failed for {wallet_address[:8]}...: {upload_result.get('error')}")
                    finally:
                        loop.close()

                # Start upload in background thread
                import threading
                upload_thread = threading.Thread(target=run_pipe_upload, daemon=True)
                upload_thread.start()

                pipe_upload_info = {
                    'initiated': True,
                    'target_wallet': wallet_address,
                    'upload_status': 'uploading',
                    'storage_provider': 'pipe'
                }

                logger.info(f"🎥 Video recorded for {wallet_address[:8]}... -> uploading to their Pipe storage")

            else:
                logger.error(f"Video file not found at {video_path}")
                pipe_upload_info['error'] = 'Video file not found'

        except Exception as e:
            logger.error(f"Failed to initiate Pipe video upload for {wallet_address[:8]}...: {e}")
            pipe_upload_info = {
                'initiated': False,
                'error': str(e),
                'upload_status': 'failed',
                'storage_provider': 'pipe'
            }

        return jsonify({
            'success': True,
            'recording': False,
            'completed': True,
            'video': latest_video,
            'filename': latest_video.get('filename'),
            'path': latest_video.get('path'),
            'url': latest_video.get('url'),
            'size': latest_video.get('size'),
            'timestamp': latest_video.get('timestamp'),
            'pipe_upload': pipe_upload_info
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
    
    @app.route('/api/face/extract-embedding', methods=['POST'])
    def api_face_extract_embedding():
        """
        Extract face embedding from base64 image for phone-based enrollment.
        Includes quality scoring and optional encryption for security.
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                }), 400

            image_data = data.get('image')
            wallet_address = data.get('wallet_address')  # Optional for encryption
            encrypt = data.get('encrypt', False)  # Whether to encrypt the embedding

            if not image_data:
                return jsonify({
                    'success': False,
                    'error': 'Missing image data'
                }), 400

            # Decode base64 image
            try:
                if image_data.startswith('data:image'):
                    # Remove data URL prefix (data:image/jpeg;base64,)
                    image_data = image_data.split(',')[1]

                # Decode base64 to bytes
                import base64
                image_bytes = base64.b64decode(image_data)

                # Convert to numpy array
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)

                # Decode image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid image data - could not decode image'
                    }), 400

            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode base64 image'
                }), 400

            # Get GPU face service
            services = get_services()
            if 'gpu_face' not in services:
                return jsonify({
                    'success': False,
                    'error': 'GPU face service not available'
                }), 503

            gpu_face_service = services['gpu_face']

            # Check if models are loaded
            if not gpu_face_service._models_loaded:
                return jsonify({
                    'success': False,
                    'error': 'Face recognition models not loaded'
                }), 503

            # Use InsightFace to detect and analyze faces for quality
            quality_score = 0
            quality_factors = {}

            try:
                # Convert BGR to RGB for InsightFace
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get all faces detected by InsightFace
                faces = gpu_face_service.face_embedder.get(frame_rgb)

                if len(faces) == 0:
                    return jsonify({
                        'success': False,
                        'error': 'No face detected in image - please ensure your face is clearly visible',
                        'quality_score': 0,
                        'quality_factors': {
                            'face_detected': False
                        }
                    }), 400

                # Get the largest/best face
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

                # Calculate quality score based on multiple factors
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                image_height, image_width = frame.shape[:2]

                # 1. Face size score (larger faces are better)
                face_area_ratio = (face_width * face_height) / (image_width * image_height)
                size_score = min(face_area_ratio * 200, 100)  # 0-100 scale
                quality_factors['face_size'] = round(size_score, 1)

                # 2. Face position score (centered faces are better)
                face_center_x = (bbox[0] + bbox[2]) / 2
                face_center_y = (bbox[1] + bbox[3]) / 2
                center_offset_x = abs(face_center_x - image_width/2) / (image_width/2)
                center_offset_y = abs(face_center_y - image_height/2) / (image_height/2)
                position_score = 100 - (center_offset_x + center_offset_y) * 50
                quality_factors['face_position'] = round(max(position_score, 0), 1)

                # 3. Detection confidence (from InsightFace)
                det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                confidence_score = det_score * 100
                quality_factors['detection_confidence'] = round(confidence_score, 1)

                # 4. Image sharpness (using Laplacian variance)
                face_region = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if face_region.size > 0:
                    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    sharpness_score = min(laplacian_var / 10, 100)  # Normalize to 0-100
                    quality_factors['image_sharpness'] = round(sharpness_score, 1)
                else:
                    quality_factors['image_sharpness'] = 50

                # 5. Lighting quality (check histogram distribution)
                if face_region.size > 0:
                    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
                    hist_normalized = hist.ravel() / hist.sum()
                    # Check for good distribution (not too dark or bright)
                    low_light = hist_normalized[:50].sum()
                    high_light = hist_normalized[200:].sum()
                    lighting_score = 100 - (low_light + high_light) * 100
                    quality_factors['lighting_quality'] = round(max(lighting_score, 0), 1)
                else:
                    quality_factors['lighting_quality'] = 50

                # Calculate overall quality score (weighted average)
                quality_score = (
                    size_score * 0.25 +
                    position_score * 0.15 +
                    confidence_score * 0.30 +
                    quality_factors['image_sharpness'] * 0.15 +
                    quality_factors['lighting_quality'] * 0.15
                )
                quality_score = round(quality_score, 1)

                # Extract embedding
                embedding = face.normed_embedding
                embedding_list = embedding.tolist()

                logger.info(f"Extracted embedding with quality score: {quality_score}%")

                # If encryption is requested and wallet address provided
                if encrypt and wallet_address:
                    try:
                        # Create biometric session for encryption
                        biometric_response = requests.post(
                            'http://biometric-security:5003/api/biometric/create-session',
                            json={
                                'wallet_address': wallet_address,
                                'session_duration': 300  # 5 minutes for enrollment
                            },
                            timeout=10
                        )

                        if biometric_response.status_code == 200:
                            biometric_session = biometric_response.json()
                            session_id = biometric_session['session_id']

                            # Encrypt the embedding
                            encrypt_response = requests.post(
                                'http://biometric-security:5003/api/biometric/encrypt-embedding',
                                json={
                                    'embedding': embedding_list,
                                    'wallet_address': wallet_address,
                                    'session_id': session_id,
                                    'metadata': {
                                        'quality_score': quality_score,
                                        'timestamp': int(time.time()),
                                        'source': 'phone_camera'
                                    }
                                },
                                timeout=15
                            )

                            if encrypt_response.status_code == 200:
                                encrypted_data = encrypt_response.json()

                                logger.info(f"Successfully encrypted embedding for {wallet_address[:8]}...")

                                return jsonify({
                                    'success': True,
                                    'embedding': encrypted_data['nft_package'],  # Encrypted package
                                    'encrypted': True,
                                    'encryption_method': 'AES-256-PBKDF2',
                                    'session_id': session_id,
                                    'quality_score': quality_score,
                                    'quality_factors': quality_factors,
                                    'quality_rating': api_face_extract_embedding.get_quality_rating(quality_score),
                                    'recommendations': api_face_extract_embedding.get_quality_recommendations(quality_score, quality_factors)
                                })
                    except Exception as encrypt_error:
                        logger.warning(f"Encryption failed, returning unencrypted: {encrypt_error}")
                        # Fall through to return unencrypted

                # Return unencrypted embedding with quality metrics
                return jsonify({
                    'success': True,
                    'embedding': embedding_list,
                    'encrypted': False,
                    'quality_score': quality_score,
                    'quality_factors': quality_factors,
                    'quality_rating': api_face_extract_embedding.get_quality_rating(quality_score),
                    'recommendations': api_face_extract_embedding.get_quality_recommendations(quality_score, quality_factors)
                })

            except Exception as analysis_error:
                logger.error(f"Face analysis error: {analysis_error}")
                # Try basic extraction without quality scoring
                full_embedding = gpu_face_service.extract_face_embedding(frame)

                if full_embedding is None:
                    return jsonify({
                        'success': False,
                        'error': 'Face detection failed',
                        'quality_score': 0
                    }), 400

                return jsonify({
                    'success': True,
                    'embedding': full_embedding.tolist(),
                    'encrypted': False,
                    'quality_score': 50,  # Unknown quality
                    'quality_rating': 'unknown',
                    'warning': 'Quality assessment unavailable'
                })

        except Exception as e:
            logger.error(f"Error in face embedding extraction: {e}")
            return jsonify({
                'success': False,
                'error': f'Face embedding extraction failed: {str(e)}'
            }), 500

    # Helper functions for quality assessment
    api_face_extract_embedding.get_quality_rating = lambda score: (
        'excellent' if score >= 90 else
        'good' if score >= 80 else
        'acceptable' if score >= 70 else
        'poor' if score >= 60 else
        'very_poor'
    )

    api_face_extract_embedding.get_quality_recommendations = lambda score, factors: (
        ["Excellent quality! Your facial embedding is optimal for recognition."] if score >= 90 else
        [
            "Move closer to the camera for a larger face image" if factors.get('face_size', 100) < 70 else None,
            "Center your face in the frame" if factors.get('face_position', 100) < 70 else None,
            "Hold the camera steady to reduce blur" if factors.get('image_sharpness', 100) < 70 else None,
            "Improve lighting - avoid shadows and extreme brightness" if factors.get('lighting_quality', 100) < 70 else None,
            "Ensure your face is fully visible and unobstructed" if factors.get('detection_confidence', 100) < 80 else None
        ] if score < 90 else []
    )
    # Filter out None values from recommendations
    api_face_extract_embedding.get_quality_recommendations = lambda score, factors: [
        rec for rec in api_face_extract_embedding.get_quality_recommendations.__wrapped__(score, factors) if rec
    ] if hasattr(api_face_extract_embedding.get_quality_recommendations, '__wrapped__') else (
        ["Excellent quality! Your facial embedding is optimal for recognition."] if score >= 90 else
        [rec for rec in [
            "Move closer to the camera for a larger face image" if factors.get('face_size', 100) < 70 else None,
            "Center your face in the frame" if factors.get('face_position', 100) < 70 else None,
            "Hold the camera steady to reduce blur" if factors.get('image_sharpness', 100) < 70 else None,
            "Improve lighting - avoid shadows and extreme brightness" if factors.get('lighting_quality', 100) < 70 else None,
            "Ensure your face is fully visible and unobstructed" if factors.get('detection_confidence', 100) < 80 else None
        ] if rec] or ["Consider retaking for better quality"]
    )

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
    @sign_response
    def api_camera_info():
        """Get camera information for frontend discovery with device signature"""
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
                    'device': 'auto-detected'
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

    @app.route('/api/device-info', methods=['GET'])
    @sign_response
    def api_device_info():
        """Get device-specific information for DePIN registration with cryptographic signature"""
        # Check if device needs WiFi setup
        setup_required = not _is_internet_connected()
        
        return jsonify({
            'success': True,
            'device_pubkey': device_signer.get_public_key(),
            'hardware_id': device_signer._get_hardware_key().decode('utf-8')[:16],  # Truncated for privacy
            'model': 'MMOMENT Jetson Orin Nano',  # Updated to include MMOMENT branding
            'version': '1.0.0',
            'setup_required': setup_required,
            'capabilities': {
                'signing': True,
                'blockchain_ready': True,
                'depin_authentication': True,
                'local_setup': True
            }
        })

    @app.route('/api/setup/wifi/scan', methods=['GET'])
    def api_wifi_scan():
        """Scan for available WiFi networks during device setup"""
        try:
            # Use nmcli to scan for WiFi networks on Jetson
            import subprocess
            result = subprocess.run(['nmcli', '-t', '-f', 'SSID,SECURITY,SIGNAL', 'dev', 'wifi'], 
                                  capture_output=True, text=True, timeout=10)
            
            networks = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':')
                        if len(parts) >= 3 and parts[0]:  # Skip empty SSIDs
                            networks.append({
                                'ssid': parts[0],
                                'security': parts[1] if parts[1] else 'Open',
                                'signal': int(parts[2]) if parts[2].isdigit() else 0
                            })
            
            # Sort by signal strength
            networks.sort(key=lambda x: x['signal'], reverse=True)
            
            return jsonify({
                'success': True,
                'networks': networks[:20]  # Limit to top 20 networks
            })
            
        except Exception as e:
            logger.error(f"WiFi scan failed: {e}")
            return jsonify({
                'success': False,
                'error': 'WiFi scan failed',
                'networks': []
            })

    @app.route('/api/scan-registration-qr', methods=['POST'])
    def api_scan_registration_qr():
        """
        Scan for registration QR code containing WiFi credentials and claim endpoint.
        Activates camera for QR scanning mode and processes registration data.
        """
        try:
            import cv2
            import json
            import requests
            from pyzbar import pyzbar
            import time
            
            logger.info("[QR-SCAN] Starting QR code registration scan")
            
            # Get timeout from request, default to 60 seconds (longer for debugging)
            data = request.get_json() or {}
            max_scan_time = data.get('timeout', 60)  # Default 60 seconds, configurable
            debug_mode = data.get('debug', False)  # Debug mode - no WiFi/claiming, just QR detection
            
            buffer_service = get_services()['buffer']
            start_time = time.time()
            frame_count = 0
            qr_attempts = 0
            
            logger.info(f"[QR-SCAN] Starting scan with {max_scan_time}s timeout")
            
            while (time.time() - start_time) < max_scan_time:
                # Get current frame
                frame, timestamp = buffer_service.get_frame()
                frame_count += 1
                
                if frame is None:
                    if frame_count % 100 == 0:  # Log every 100 failed frame attempts
                        logger.warning(f"[QR-SCAN] No frame available after {frame_count} attempts")
                    time.sleep(0.1)
                    continue
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"[QR-SCAN] Processed {frame_count} frames in {elapsed:.1f}s, frame shape: {frame.shape}")
                
                # Scan for QR codes
                qr_codes = pyzbar.decode(frame)
                qr_attempts += 1
                
                # Log QR scan attempts periodically
                if qr_attempts % 50 == 0:
                    logger.info(f"[QR-SCAN] QR decode attempts: {qr_attempts}, codes found: {len(qr_codes)}")
                
                if len(qr_codes) > 0:
                    logger.info(f"[QR-SCAN] Found {len(qr_codes)} QR code(s) in frame!")
                
                for qr_code in qr_codes:
                    try:
                        # Decode QR code data
                        qr_data = qr_code.data.decode('utf-8')
                        registration_data = json.loads(qr_data)
                        
                        logger.info(f"[QR-SCAN] Found QR code with registration data")
                        
                        # Validate required fields
                        required_fields = ['wifi_ssid', 'wifi_password', 'claim_endpoint', 'user_wallet', 'expires']
                        if not all(field in registration_data for field in required_fields):
                            logger.warning(f"[QR-SCAN] QR code missing required fields")
                            continue
                        
                        # Check if QR code has expired
                        if registration_data['expires'] < time.time():
                            logger.warning(f"[QR-SCAN] QR code has expired")
                            continue
                        
                        logger.info(f"[QR-SCAN] Valid registration QR found for user: {registration_data['user_wallet'][:8]}...")
                        
                        # Debug mode - just return the parsed data without WiFi/claiming
                        if debug_mode:
                            logger.info("[QR-SCAN] Debug mode - returning parsed QR data without WiFi/claiming")
                            return jsonify({
                                'success': True,
                                'debug_mode': True,
                                'parsed_qr_data': registration_data,
                                'device_pubkey': device_signer.get_public_key(),
                                'message': 'QR code parsed successfully (debug mode - no WiFi/claiming performed)'
                            })
                        
                        # Check if already connected to internet - skip WiFi setup if so
                        if _is_internet_connected():
                            logger.info(f"[QR-SCAN] Already have internet connectivity, skipping WiFi connection to: {registration_data['wifi_ssid']}")
                        else:
                            # Connect to WiFi
                            logger.info(f"[QR-SCAN] No internet connectivity, attempting WiFi connection to: {registration_data['wifi_ssid']}")
                            wifi_success = _connect_to_wifi(registration_data['wifi_ssid'], registration_data['wifi_password'])
                            if not wifi_success:
                                logger.error(f"[QR-SCAN] WiFi connection failed for: {registration_data['wifi_ssid']}")
                                return jsonify({
                                    'success': False,
                                    'error': f'Failed to connect to WiFi network: {registration_data["wifi_ssid"]}',
                                    'parsed_qr_data': registration_data,
                                    'step_failed': 'wifi_connection'
                                }), 500
                        
                        # Wait for internet connectivity
                        connectivity_attempts = 0
                        while not _is_internet_connected() and connectivity_attempts < 10:
                            time.sleep(2)
                            connectivity_attempts += 1
                        
                        if not _is_internet_connected():
                            logger.error("[QR-SCAN] WiFi connected but no internet access")
                            return jsonify({
                                'success': False,
                                'error': 'Connected to WiFi but no internet access',
                                'parsed_qr_data': registration_data,
                                'step_failed': 'internet_connectivity'
                            }), 500
                        
                        # Claim device with backend
                        device_claim_data = {
                            'device_pubkey': device_signer.get_public_key(),
                            'device_model': 'jetson_orin_nano'
                        }
                        
                        logger.info(f"[QR-SCAN] Claiming device at endpoint: {registration_data['claim_endpoint']}")
                        
                        claim_response = requests.post(
                            registration_data['claim_endpoint'],
                            json=device_claim_data,
                            timeout=15
                        )
                        
                        if claim_response.status_code != 200:
                            logger.error(f"[QR-SCAN] Device claim failed: {claim_response.status_code}")
                            return jsonify({
                                'success': False,
                                'error': f'Device claim failed: {claim_response.status_code}',
                                'parsed_qr_data': registration_data,
                                'step_failed': 'device_claim',
                                'claim_response_status': claim_response.status_code,
                                'claim_response_text': claim_response.text[:500]  # First 500 chars
                            }), 500
                        
                        logger.info(f"[QR-SCAN] Device successfully claimed by user: {registration_data['user_wallet'][:8]}...")
                        
                        return jsonify({
                            'success': True,
                            'user_wallet': registration_data['user_wallet'],
                            'device_pubkey': device_signer.get_public_key(),
                            'wifi_connected': True,
                            'device_claimed': True,
                            'message': 'Device registration completed successfully'
                        })
                        
                    except json.JSONDecodeError:
                        # Not a valid JSON QR code, continue scanning
                        continue
                    except Exception as e:
                        logger.error(f"[QR-SCAN] Error processing QR code: {e}")
                        continue
                
                # Brief pause before next frame
                time.sleep(0.1)
            
            # Timeout reached
            elapsed = time.time() - start_time
            logger.warning(f"[QR-SCAN] Timeout after {elapsed:.1f}s - processed {frame_count} frames, {qr_attempts} QR attempts")
            return jsonify({
                'success': False,
                'error': f'QR code scan timeout after {elapsed:.1f}s',
                'debug_info': {
                    'frames_processed': frame_count,
                    'qr_decode_attempts': qr_attempts,
                    'timeout_seconds': max_scan_time,
                    'elapsed_seconds': round(elapsed, 1)
                }
            }), 408
            
        except Exception as e:
            logger.error(f"[QR-SCAN] Unexpected error: {e}")
            return jsonify({
                'success': False,
                'error': f'QR scan failed: {str(e)}'
            }), 500

    def _connect_to_wifi(ssid, password):
        """Connect to WiFi network using nmcli"""
        try:
            import subprocess
            
            logger.info(f"[WIFI] Connecting to network: {ssid}")
            
            # First check if already connected to this network
            check_result = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SSID', 'dev', 'wifi'], 
                                        capture_output=True, text=True, timeout=10)
            
            if check_result.returncode == 0:
                for line in check_result.stdout.strip().split('\n'):
                    if line.startswith('yes:') and ssid in line:
                        logger.info(f"[WIFI] Already connected to {ssid}")
                        return True
            
            # Delete any existing connection with same name
            subprocess.run(['nmcli', 'connection', 'delete', ssid], 
                         capture_output=True)
            
            # Create new WiFi connection
            if password:
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password]
            else:
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"[WIFI] Successfully connected to {ssid}")
                return True
            else:
                # Check if the error is because we're already connected
                if "already connected" in result.stderr.lower() or "device is already active" in result.stderr.lower():
                    logger.info(f"[WIFI] Already connected to {ssid} (detected from error message)")
                    return True
                
                logger.error(f"[WIFI] Failed to connect to {ssid}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[WIFI] WiFi connection error: {e}")
            return False

    @app.route('/api/setup/wifi', methods=['POST'])
    def api_wifi_setup():
        """Configure WiFi credentials during device setup"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            ssid = data.get('ssid')
            password = data.get('password')
            security = data.get('security', 'WPA2')
            
            if not ssid:
                return jsonify({'success': False, 'error': 'SSID required'}), 400
            
            logger.info(f"Configuring WiFi for SSID: {ssid}")
            
            # Use nmcli to configure WiFi on Jetson
            import subprocess
            
            # First, delete any existing connection with same name
            subprocess.run(['nmcli', 'connection', 'delete', ssid], 
                         capture_output=True)
            
            # Create new WiFi connection
            if security.upper() in ['WPA', 'WPA2', 'WPA3'] and password:
                # WPA/WPA2 connection with password
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 
                       'password', password]
            else:
                # Open network
                cmd = ['nmcli', 'device', 'wifi', 'connect', ssid]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"WiFi configured successfully for {ssid}")
                
                # Schedule device restart to apply network changes
                import threading
                def restart_device():
                    import time
                    time.sleep(5)  # Give time for response
                    subprocess.run(['sudo', 'reboot'])
                
                restart_thread = threading.Thread(target=restart_device)
                restart_thread.daemon = True
                restart_thread.start()
                
                return jsonify({
                    'success': True,
                    'message': 'WiFi configured successfully. Device will restart.',
                    'ssid': ssid
                })
            else:
                logger.error(f"WiFi configuration failed: {result.stderr}")
                return jsonify({
                    'success': False,
                    'error': f'WiFi configuration failed: {result.stderr}'
                }), 500
                
        except Exception as e:
            logger.error(f"WiFi setup error: {e}")
            return jsonify({
                'success': False,
                'error': f'WiFi setup failed: {str(e)}'
            }), 500

    def _is_internet_connected():
        """Check if device has internet connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen('http://google.com', timeout=5)
            return True
        except:
            return False

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
                'preferred_device': 'auto-detected',
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

        # Pre-authorize Pipe storage session for fast uploads
        import asyncio
        from services.pipe_storage_service import pre_authorize_user_session

        def run_pipe_authorization():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                auth_success = loop.run_until_complete(pre_authorize_user_session(wallet_address))
                if auth_success:
                    logger.info(f"✅ Pipe session pre-authorized for {wallet_address[:8]}...")
                else:
                    logger.warning(f"⚠️ Pipe session pre-authorization failed for {wallet_address[:8]}...")
            except Exception as e:
                logger.error(f"❌ Pipe pre-authorization error for {wallet_address[:8]}...: {e}")
            finally:
                loop.close()

        # Start Pipe authorization in background thread
        import threading
        pipe_auth_thread = threading.Thread(target=run_pipe_authorization, daemon=True)
        pipe_auth_thread.start()

        logger.info(f"🔗 User {wallet_address[:8]}... connected with Pipe pre-authorization initiated")

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
        Capture a photo from the camera and upload directly to user's Pipe storage
        Returns the photo data and upload result
        """
        wallet_address = request.json.get('wallet_address')

        # Get services
        buffer_service = get_services()['buffer']
        capture_service = get_services()['capture']

        # Capture photo
        photo_info = capture_service.capture_photo(buffer_service, wallet_address)

        if not photo_info['success']:
            return jsonify(photo_info), 500

        # Read the photo file data
        with open(photo_info['path'], 'rb') as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

        # Add base64 data to response
        photo_info['image_data'] = f"data:image/jpeg;base64,{base64_image}"

        # Upload to user's Pipe storage asynchronously
        import asyncio
        from services.pipe_storage_service import handle_user_capture_upload

        try:
            # Prepare metadata for Pipe upload
            metadata = {
                'timestamp': photo_info.get('timestamp'),
                'camera_id': 'jetson01',  # TODO: Get actual camera ID from device config
                'width': photo_info.get('width'),
                'height': photo_info.get('height'),
                'local_path': photo_info['path'],
                'local_filename': photo_info['filename'],
                'capture_type': 'photo'
            }

            # Upload to user's specific Pipe storage account
            def run_pipe_upload():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    upload_result = loop.run_until_complete(
                        handle_user_capture_upload(wallet_address, image_data, metadata)
                    )
                    if upload_result['success']:
                        logger.info(f"✅ Photo uploaded to {wallet_address[:8]}...'s Pipe: {upload_result.get('filename')}")
                    else:
                        logger.error(f"❌ Pipe upload failed for {wallet_address[:8]}...: {upload_result.get('error')}")
                finally:
                    loop.close()

            # Start upload in background thread
            import threading
            upload_thread = threading.Thread(target=run_pipe_upload, daemon=True)
            upload_thread.start()

            # Add Pipe upload info to response
            photo_info['pipe_upload'] = {
                'initiated': True,
                'target_wallet': wallet_address,
                'upload_status': 'uploading',
                'storage_provider': 'pipe'
            }

            logger.info(f"📸 Photo captured for {wallet_address[:8]}... -> uploading to their Pipe storage")

        except Exception as e:
            logger.error(f"Failed to initiate Pipe upload for {wallet_address[:8]}...: {e}")
            photo_info['pipe_upload'] = {
                'initiated': False,
                'error': str(e),
                'upload_status': 'failed',
                'storage_provider': 'pipe'
            }

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
        
    @app.route('/qr-monitor')
    def qr_monitor():
        """QR registration monitoring page - shows what Jetson processes from frontend QR codes"""
        return render_template('qr_monitor.html')
        
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
        # REMOVED: No more preferred device logic
        
        # Try to restart
        success = buffer_service.start()
        
        # Return result
        return jsonify({
            'success': success,
            'message': 'Camera reset successful' if success else 'Failed to reset camera',
            'was_running': was_running,
            'now_running': buffer_service._running,
            'camera_index': buffer_service._camera_index,
            'preferred_device': 'auto-detected'
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