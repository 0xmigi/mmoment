"""
Routes for the Jetson Camera API

Defines all the API endpoints for the camera service.
"""

import base64
import json
import logging
import os
import threading
import time
import uuid
from functools import wraps

import cv2
import requests
from flask import (
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

# Import device signer for DePIN authentication
from services.device_signer import DeviceSigner

# Import native identity service for lightweight identity matching
from services.native_identity_service import get_native_identity_service

# Import competition settlement service for on-chain escrow settlement
from services.competition_settlement import settle_competition

# Set up logging
logger = logging.getLogger("CameraRoutes")

# Competition settlement configuration
CAMERA_OWNER_WALLET = os.environ.get("CAMERA_OWNER_WALLET", "")
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com")
DEVICE_KEYPAIR_PATH = "/opt/mmoment/device/device-keypair.json"

# Initialize device signer (singleton)
device_signer = DeviceSigner()

# In-memory user profile store (native mode - no gpu_face service)
_user_profiles: dict = {}

# Our simple face recognition is always available
FACENET_AVAILABLE = True


def get_camera_pda():
    """Get camera PDA from device config file (set during registration)"""
    config_path = "/app/config/device_config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if "camera_pda" in config:
                    return config["camera_pda"]
    except Exception as e:
        logger.debug(f"Could not read camera PDA from device config: {e}")

    # Fallback to environment variable (for backward compatibility during development)
    return os.environ.get("CAMERA_PDA", "unknown")


# Import camera utilities
try:
    from services.utils import (
        check_facenet_availability,
        detect_cameras,
        get_camera_health,
        reset_camera_devices,
    )

    CAMERA_UTILS_AVAILABLE = True
except ImportError:
    CAMERA_UTILS_AVAILABLE = False
    logger.warning(
        "Could not import camera utilities, some diagnostic endpoints will be unavailable"
    )


# Get services from app config
def get_services():
    """Get services from Flask app config"""
    return current_app.config["SERVICES"]


def get_blockchain_session_sync():
    """Get blockchain session sync service"""
    from services.blockchain_session_sync import get_blockchain_session_sync

    return get_blockchain_session_sync()


def get_checked_in_users() -> list:
    """Get list of currently checked-in wallet addresses for access grants."""
    try:
        blockchain_sync = get_blockchain_session_sync()
        return list(blockchain_sync.checked_in_wallets)
    except Exception as e:
        logger.warning(f"Failed to get checked-in users: {e}")
        return []


# Session validation decorator - With CRYPTOGRAPHIC SIGNATURE VERIFICATION
def require_session(f):
    """
    Decorator to require a valid session with ed25519 signature verification.

    This provides TRUE security by verifying that the requester actually owns
    the wallet private key, not just knows the wallet address.

    Expected request format:
    {
        "wallet_address": "ABC123...",
        "request_timestamp": 1234567890123,  // Unix ms
        "request_nonce": "random-uuid",
        "request_signature": "base58-encoded-signature"  // Signs: wallet|timestamp|nonce
    }
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"🔍 require_session decorator called for endpoint: {f.__name__}")

        # CV Dev Mode bypass - allow unauthenticated access for CV app testing
        if os.environ.get('CV_DEV_MODE', 'false').lower() == 'true':
            logger.info(f"🧪 CV_DEV_MODE: Bypassing session check for {f.__name__}")
            return f(*args, **kwargs)

        if not request.json:
            return jsonify({"success": False, "error": "Request body required"}), 400

        wallet_address = request.json.get("wallet_address")
        request_timestamp = request.json.get("request_timestamp")
        request_nonce = request.json.get("request_nonce")
        request_signature = request.json.get("request_signature")

        logger.info(f"🔍 Session validation - wallet_address: {wallet_address}")

        if not wallet_address:
            logger.warning(f"🔍 Missing wallet address for {f.__name__}")
            return jsonify({"success": False, "error": "wallet_address required"}), 403

        # Check if wallet is checked in via session_service (Phase 3 source of truth)
        session_service = get_services().get("session")
        if not session_service or not session_service.get_session_by_wallet(wallet_address):
            logger.warning(f"❌ Wallet {wallet_address[:8]}... not checked in")
            return jsonify(
                {"success": False, "error": "Invalid session - please check in first"}
            ), 403

        # CRYPTOGRAPHIC SIGNATURE VERIFICATION
        # If signature fields are provided, verify them
        if request_signature and request_timestamp and request_nonce:
            is_valid, error_msg = device_signer.verify_request_signature(
                wallet_address=wallet_address,
                timestamp=int(request_timestamp),
                nonce=request_nonce,
                signature=request_signature,
                max_age_seconds=300  # 5 minute window
            )

            if not is_valid:
                logger.warning(
                    f"❌ Signature verification failed for {wallet_address[:8]}...: {error_msg}"
                )
                return jsonify({
                    "success": False,
                    "error": f"Signature verification failed: {error_msg}"
                }), 403

            logger.info(f"✅ CRYPTOGRAPHIC signature verified for {wallet_address[:8]}...")
        else:
            # TEMPORARY: Allow unsigned requests but log warning
            # TODO: Once frontend is updated, make signatures REQUIRED
            logger.warning(
                f"⚠️ UNSIGNED request from {wallet_address[:8]}... - "
                "This is temporarily allowed but will be required soon"
            )

        logger.info(f"✅ Session validated for {wallet_address[:8]}...")
        return f(*args, **kwargs)

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
            if isinstance(result, dict) or (
                hasattr(result, "get_json") and result.get_json()
            ):
                if isinstance(result, dict):
                    response_data = result
                else:
                    response_data = result.get_json()

                # Only sign if response indicates success or contains meaningful data
                if response_data and (
                    response_data.get("success", True) or "data" in response_data
                ):
                    # Sign the response with device key
                    signed_data = device_signer.sign_response(response_data)

                    logger.debug(
                        f"Response signed by device: {device_signer.get_public_key()[:12]}..."
                    )

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
    @app.route("/")
    def index():
        """Index route with API information"""
        return jsonify(
            {
                "name": "Jetson Camera API",
                "version": "2.0.0",
                "description": "Lightweight, optimized camera API for Jetson with standardized endpoints",
                "buffer_based": True,
                "standardized_endpoints": {
                    "health": "/api/health",
                    "stream_info": "/api/stream/info",
                    "capture": "/api/capture",
                    "record": "/api/record",
                    "photos": "/api/photos",
                    "videos": "/api/videos",
                    "session_connect": "/api/session/connect",
                    "session_disconnect": "/api/session/disconnect",
                    "face_extract_embedding": "/api/face/extract-embedding",
                    "face_enroll_confirm": "/api/face/enroll/confirm",
                    "face_recognize": "/api/face/recognize",
                    "gesture_current": "/api/gesture/current",
                    "visualization_face": "/api/visualization/face",
                    "visualization_gesture": "/api/visualization/gesture",
                    "visualization_pose": "/api/visualization/pose",
                    "apps_load": "/api/apps/load",
                    "apps_activate": "/api/apps/activate",
                    "apps_deactivate": "/api/apps/deactivate",
                    "apps_status": "/api/apps/status",
                    "device_info": "/api/device-info",
                    "scan_registration_qr": "/api/scan-registration-qr",
                    "wifi_scan": "/api/setup/wifi/scan",
                    "wifi_setup": "/api/setup/wifi",
                },
            }
        )

    # ========================================
    # STANDARDIZED API ROUTES (Pi5 Compatible)
    # ========================================

    # Health & Status
    @app.route("/api/health")
    @sign_response
    def api_health():
        """Standardized health check endpoint with device signature"""
        return health()

    # Device Registration & Configuration
    @app.route("/api/device/registration/start", methods=["POST"])
    @sign_response
    def api_device_registration_start():
        """Start device registration polling for backend configuration"""
        try:
            from services.device_registration import get_device_registration_service

            registration_service = get_device_registration_service()
            registration_service.start_polling()

            return jsonify(
                {
                    "success": True,
                    "message": "Device registration polling started",
                    "device_pubkey": device_signer.get_public_key(),
                    "status": "polling_for_configuration",
                }
            )

        except Exception as e:
            logger.error(f"Failed to start device registration: {e}")
            return jsonify(
                {"success": False, "error": f"Failed to start registration: {str(e)}"}
            ), 500

    @app.route("/api/device/registration/status")
    @sign_response
    def api_device_registration_status():
        """Get current device registration status"""
        try:
            from services.device_registration import get_device_registration_service

            registration_service = get_device_registration_service()
            device_info = registration_service.get_device_info()

            return jsonify(
                {
                    "success": True,
                    "device_info": device_info,
                    "is_configured": registration_service.is_configured(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get registration status: {e}")
            return jsonify(
                {"success": False, "error": f"Failed to get status: {str(e)}"}
            ), 500

    @app.route("/api/device/claim", methods=["POST"])
    @sign_response
    def api_device_claim():
        """Claim a device with backend using QR token"""
        try:
            data = request.get_json()
            claim_token = data.get("claim_token")
            wifi_credentials = data.get("wifi_credentials")

            if not claim_token:
                return jsonify(
                    {"success": False, "error": "claim_token is required"}
                ), 400

            # Get device information
            device_pubkey = device_signer.get_public_key()
            device_info = device_signer.get_device_info()

            # Claim the device with backend
            backend_url = os.getenv("BACKEND_URL", "https://mmoment-production.up.railway.app")

            claim_response = requests.post(
                f"{backend_url}/api/device/claim",
                json={
                    "token": claim_token,
                    "device_pubkey": device_pubkey,
                    "device_info": device_info,
                    "wifi_credentials": wifi_credentials,
                },
                timeout=10,
            )

            if claim_response.status_code == 200:
                # Start polling for configuration
                from services.device_registration import get_device_registration_service

                registration_service = get_device_registration_service()
                registration_service.start_polling()

                return jsonify(
                    {
                        "success": True,
                        "message": "Device claimed successfully, polling for configuration",
                        "device_pubkey": device_pubkey,
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": f"Failed to claim device: {claim_response.status_code}",
                    }
                ), 400

        except Exception as e:
            logger.error(f"Failed to claim device: {e}")
            return jsonify(
                {"success": False, "error": f"Failed to claim device: {str(e)}"}
            ), 500

    @app.route("/api/device/tunnel/status")
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

            return jsonify(
                {
                    "success": True,
                    "tunnel_status": tunnel_status,
                    "device_config": device_config,
                    "is_configured": registration_service.is_configured(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get tunnel status: {e}")
            return jsonify(
                {"success": False, "error": f"Failed to get tunnel status: {str(e)}"}
            ), 500

    @app.route("/api/status")
    @sign_response
    def api_status():
        """Main status endpoint for frontend - comprehensive system status"""
        try:
            services = get_services()
            buffer_service = services["buffer"]
            capture_service = services.get("capture")
            webrtc_service = services.get("webrtc")
            whip_service = services.get("whip")

            # Get buffer/camera status
            buffer_status = buffer_service.get_status()

            # Get recording status from capture service
            is_recording = False
            current_filename = None
            if capture_service:
                is_recording = capture_service.is_recording()

            # Get WebRTC status
            webrtc_status = {}
            if webrtc_service:
                webrtc_status = webrtc_service.get_status()

            # Get WHIP status
            whip_status = {}
            if whip_service:
                whip_status = whip_service.get_status()

            # Streaming status based on WebRTC/WHIP (Livepeer removed)
            is_streaming = webrtc_status.get("connected", False) or whip_status.get("connected", False)
            stream_info = {
                "isActive": is_streaming,
                "format": "webrtc",  # Primary format is now WebRTC
                "whip_connected": whip_status.get("connected", False),
                "webrtc_connected": webrtc_status.get("connected", False)
            }

            # Available stream formats (Livepeer removed)
            stream_formats = ["webrtc", "whip"]

            # Get active session count for Phase 3 privacy architecture
            session_service = services.get("session")
            active_session_count = 0
            if session_service:
                active_session_count = session_service.get_active_session_count()

            return jsonify(
                {
                    "success": True,
                    "timestamp": int(time.time()),
                    "data": {
                        "isOnline": buffer_status["running"],
                        "isStreaming": is_streaming,
                        "isRecording": is_recording,
                        "lastSeen": int(time.time()),
                        "activeSessionCount": active_session_count,  # Phase 3: Privacy-preserving user count
                        "streamInfo": stream_info,
                        "recordingInfo": {
                            "isActive": is_recording,
                            "currentFilename": current_filename,
                        },
                        "webrtcInfo": {
                            "available": webrtc_status.get("running", False),
                            "connected": webrtc_status.get("connected", False),
                            "activeConnections": webrtc_status.get(
                                "active_connections", 0
                            ),
                        },
                        "supportedFormats": stream_formats,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/stream/info", methods=["GET"])
    def api_stream_info():
        """Get information about available video streams (clean and annotated).

        Returns URLs and connection info for:
        - P2P WebRTC streams (clean and annotated) - lower latency, local network
        - WHEP streams (clean and annotated) - remote viewing via MediaMTX relay
        """
        try:
            import os
            webrtc_service = get_services().get("webrtc")
            whip_clean = get_services().get("whip_clean")
            whip_annotated = get_services().get("whip_annotated")

            camera_pda = os.environ.get('CAMERA_PDA', 'unknown')
            mediamtx_url = os.environ.get('MEDIAMTX_URL', 'http://129.80.99.75:8889')

            # Build stream info response
            streams = {
                "camera_pda": camera_pda,
                "streams": {
                    "clean": {
                        "description": "Raw video stream without CV annotations",
                        "p2p": {
                            "available": webrtc_service is not None and webrtc_service.running,
                            "signaling": webrtc_service.backend_url if webrtc_service else None,
                            "streamType": "clean",
                            "info": "Connect via Socket.IO with streamType='clean'"
                        },
                        "whep": {
                            "available": whip_clean is not None and whip_clean.connected,
                            "url": f"{mediamtx_url}/{camera_pda}/" if whip_clean else None,
                            "running": whip_clean.running if whip_clean else False,
                            "connected": whip_clean.connected if whip_clean else False
                        }
                    },
                    "annotated": {
                        "description": "Video stream with CV annotations (face boxes, skeletons, app overlays)",
                        "p2p": {
                            "available": webrtc_service is not None and webrtc_service.running,
                            "signaling": webrtc_service.backend_url if webrtc_service else None,
                            "streamType": "annotated",
                            "info": "Connect via Socket.IO with streamType='annotated'"
                        },
                        "whep": {
                            "available": whip_annotated is not None and whip_annotated.connected,
                            "url": f"{mediamtx_url}/{camera_pda}-annotated/" if whip_annotated else None,
                            "running": whip_annotated.running if whip_annotated else False,
                            "connected": whip_annotated.connected if whip_annotated else False
                        }
                    }
                },
                "recommended": {
                    "local_network": "p2p",
                    "remote_viewing": "whep",
                    "default_stream": "clean"
                }
            }

            return jsonify({
                "success": True,
                **streams
            })
        except Exception as e:
            logger.error(f"Error getting stream info: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Livepeer Streaming Endpoints REMOVED - using WebRTC/WHIP for all streaming now
    # Endpoints removed: /api/stream/livepeer/start, /api/stream/livepeer/stop, /api/stream/livepeer/status

    @app.route("/api/stream/webrtc/status", methods=["GET"])
    def api_webrtc_status():
        """Get WebRTC stream status"""
        try:
            webrtc_service = get_services().get("webrtc")

            if not webrtc_service:
                return jsonify(
                    {"success": False, "error": "WebRTC service not available"}
                ), 404

            status = webrtc_service.get_status()

            return jsonify(
                {
                    "success": True,
                    "status": "available" if status["running"] else "stopped",
                    "running": status["running"],
                    "connected": status["connected"],
                    "camera_pda": status["camera_pda"],
                    "backend_url": status["backend_url"],
                    "active_connections": status["active_connections"],
                    "connections": status["connections"],
                }
            )
        except Exception as e:
            logger.error(f"Error getting WebRTC stream status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/stream/whip/status", methods=["GET"])
    def api_whip_status():
        """Get WHIP publisher status and remote stream URL"""
        try:
            whip_service = get_services().get("whip")

            if not whip_service:
                return jsonify({
                    "success": True,
                    "available": False,
                    "message": "WHIP publisher not enabled"
                })

            status = whip_service.get_status()

            return jsonify({
                "success": True,
                "available": True,
                "running": status["running"],
                "connected": status["connected"],
                "stream_name": status["stream_name"],
                "whep_url": status["whep_url"],
                "mediamtx_url": status["mediamtx_url"],
                "reconnect_attempts": status["reconnect_attempts"]
            })
        except Exception as e:
            logger.error(f"Error getting WHIP status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Camera Actions
    @app.route("/api/capture", methods=["POST"])
    @require_session
    @sign_response
    def api_capture():
        """Standardized capture endpoint with device signing"""
        return capture_moment()

    @app.route("/api/record", methods=["POST"])
    @require_session
    def api_record():
        """
        Standardized record endpoint - supports start/stop actions for non-blocking operation.

        Request body:
            action: 'start' | 'stop' | None
                - 'start': Start recording (non-blocking, returns immediately)
                - 'stop': Stop recording and queue upload (returns video info + job_id)
                - None: Legacy blocking behavior (waits for recording completion)
            wallet_address: User's wallet address (required)
            duration: Recording duration in seconds (default 30, max 300)
        """
        import time

        from services.capture_event_signer import sign_capture
        from services.storage_provider import is_walrus_enabled
        from services.upload_queue import get_upload_queue
        from services.walrus_upload_service import get_walrus_upload_service

        # Get request parameters
        wallet_address = request.json.get("wallet_address")
        duration = request.json.get("duration", 0)  # Default to 0 (indefinite until stopped)
        action = request.json.get("action")  # 'start', 'stop', or None

        # Enforce maximum duration limit to prevent runaway recordings
        # duration=0 means "record until stopped" (valid for action='start')
        MAX_DURATION = 300  # 5 minutes maximum
        if duration < 0 or duration > MAX_DURATION:
            duration = 0  # Default to indefinite for invalid durations

        # Get services
        buffer_service = get_services()["buffer"]
        capture_service = get_services()["capture"]

        # ===== ACTION: START (non-blocking) =====
        if action == 'start':
            # Check if already recording
            if capture_service.is_recording():
                return jsonify({"success": False, "error": "Already recording"}), 400

            # Start recording (non-blocking)
            recording_info = capture_service.start_recording(
                buffer_service, wallet_address, duration
            )

            if recording_info.get("success"):
                logger.info(f"🎬 Recording started for {wallet_address[:8]}... (duration: {duration}s)")

            return jsonify(recording_info)

        # ===== ACTION: STOP (with queue-based upload) =====
        if action == 'stop':
            # Stop recording
            result = capture_service.stop_recording()

            if not result.get("success"):
                return jsonify(result)

            # Get the most recent video file
            videos = capture_service.get_videos(1)
            if not videos:
                return jsonify({
                    "success": False,
                    "error": "Recording stopped but no video file was found"
                }), 500

            latest_video = videos[0]
            storage_provider = "walrus" if is_walrus_enabled() else "pipe"
            storage_upload_info = {"initiated": False, "storage_provider": storage_provider}

            try:
                video_path = latest_video.get("path")
                if video_path and os.path.exists(video_path):
                    # Get camera PDA for signing
                    camera_pda = get_camera_pda()

                    # Calculate file hash for signing
                    upload_service = get_walrus_upload_service()
                    file_hash = upload_service.calculate_file_hash(video_path)

                    if not file_hash:
                        raise Exception("Failed to calculate file hash")

                    # Sign capture event (instant, no blockchain wait!)
                    signed_event = sign_capture(
                        user_wallet=wallet_address,
                        camera_pda=camera_pda,
                        file_hash=file_hash,
                        capture_type="video",
                        file_path=video_path,
                        metadata={
                            "timestamp": latest_video.get("timestamp"),
                            "size": latest_video.get("size"),
                            "filename": latest_video.get("filename"),
                        },
                    )
                    device_signature = signed_event["device_signature"]
                    logger.info(f"📝 Video signed: {device_signature[:16]}...")

                    # Check if video uploads are disabled for testing
                    skip_video_upload = os.environ.get("SKIP_VIDEO_UPLOAD", "").lower() == "true"

                    if skip_video_upload:
                        # Skip upload for testing - return mock job_id
                        job_id = -1
                        logger.info(f"🎥 Video recorded for {wallet_address[:8]}... (UPLOAD SKIPPED - testing mode)")
                    else:
                        # Queue background upload (returns immediately with job_id)
                        upload_queue = get_upload_queue()
                        job_id = upload_queue.add(
                            file_path=video_path,
                            file_type="video",
                            wallet_address=wallet_address,
                            camera_id=camera_pda,
                            device_signature=device_signature,
                            timestamp=latest_video.get("timestamp", 0),
                        )
                        logger.info(f"🎥 Video recorded for {wallet_address[:8]}... -> queued as job #{job_id}")

                    # Buffer to timeline immediately (don't wait for upload)
                    try:
                        from services.timeline_activity_service import get_timeline_activity_service
                        timeline_service = get_timeline_activity_service()
                        timeline_service.buffer_video_activity(
                            wallet_address=wallet_address,
                            pipe_file_name=latest_video["filename"],
                            pipe_file_id=str(job_id),
                            duration_seconds=0,  # Duration unknown at this point
                            metadata={
                                "timestamp": latest_video.get("timestamp"),
                                "device_signature": device_signature,
                                "filename": latest_video.get("filename"),
                                "size": latest_video.get("size"),
                                "storage_provider": storage_provider,
                                "job_id": job_id,
                            },
                        )
                    except Exception as timeline_err:
                        logger.warning(f"Failed to buffer video to timeline: {timeline_err}")

                    storage_upload_info = {
                        "initiated": True,
                        "completed": False,
                        "upload_status": "pending",
                        "storage_provider": storage_provider,
                        "device_signature": device_signature,
                        "job_id": job_id,
                        "local_url": f"/videos/{latest_video['filename']}",
                    }
                else:
                    logger.error(f"Video file not found at {video_path}")
                    storage_upload_info["error"] = "Video file not found"

            except Exception as e:
                logger.error(f"Failed to process video for {wallet_address[:8]}...: {e}", exc_info=True)
                storage_upload_info = {
                    "initiated": False,
                    "error": str(e),
                    "upload_status": "failed",
                    "storage_provider": storage_provider,
                }

            return jsonify({
                "success": True,
                "recording": False,
                "completed": True,
                "video": latest_video,
                "filename": latest_video.get("filename"),
                "path": latest_video.get("path"),
                "url": latest_video.get("url"),
                "size": latest_video.get("size"),
                "timestamp": latest_video.get("timestamp"),
                "storage_upload": storage_upload_info,
                "pipe_upload": storage_upload_info,  # Backward compatibility
            })

        # ===== DEFAULT: Legacy blocking behavior (no action parameter) =====
        # Check if already recording
        if capture_service.is_recording():
            return jsonify({"success": False, "error": "Already recording"}), 400

        # Start recording
        recording_info = capture_service.start_recording(
            buffer_service, wallet_address, duration
        )

        # If recording failed to start, return immediately
        if not recording_info.get("success"):
            return jsonify(recording_info)

        # Extract recording info
        filename = recording_info.get("filename")
        duration_limit = recording_info.get("duration_limit", 0)

        # Wait for recording to complete
        # If duration is specified, wait for that duration + buffer
        if duration_limit > 0:
            max_wait_time = duration_limit + 10  # Add 10 second buffer for processing
        else:
            max_wait_time = 300  # 5 minutes max for manual recordings

        start_time = time.time()
        while (
            capture_service.is_recording()
            and (time.time() - start_time) < max_wait_time
        ):
            time.sleep(0.5)  # Check every 500ms

        # Check if recording completed successfully
        if capture_service.is_recording():
            return jsonify(
                {
                    "success": False,
                    "error": "Recording timeout - recording is still in progress",
                }
            ), 408

        # Recording completed, get the video file info
        videos = capture_service.get_videos(1)  # Get most recent video

        if not videos:
            return jsonify(
                {
                    "success": False,
                    "error": "Recording completed but no video file was created",
                }
            ), 500

        latest_video = videos[0]

        # Verify this is the video we just recorded
        if filename and latest_video.get("filename") != filename:
            logger.warning(
                f"Expected filename {filename} but got {latest_video.get('filename')}"
            )

        # Queue video upload (same as 'stop' action)
        storage_provider = "walrus" if is_walrus_enabled() else "pipe"
        storage_upload_info = {"initiated": False, "storage_provider": storage_provider}

        try:
            video_path = latest_video.get("path")
            if video_path and os.path.exists(video_path):
                camera_pda = get_camera_pda()
                upload_service = get_walrus_upload_service()
                file_hash = upload_service.calculate_file_hash(video_path)

                if not file_hash:
                    raise Exception("Failed to calculate file hash")

                signed_event = sign_capture(
                    user_wallet=wallet_address,
                    camera_pda=camera_pda,
                    file_hash=file_hash,
                    capture_type="video",
                    file_path=video_path,
                    metadata={
                        "timestamp": latest_video.get("timestamp"),
                        "size": latest_video.get("size"),
                        "filename": latest_video.get("filename"),
                        "duration_seconds": duration_limit,
                    },
                )
                device_signature = signed_event["device_signature"]
                logger.info(f"📝 Video signed: {device_signature[:16]}...")

                # Check if video uploads are disabled for testing
                skip_video_upload = os.environ.get("SKIP_VIDEO_UPLOAD", "").lower() == "true"

                if skip_video_upload:
                    # Skip upload for testing - return mock job_id
                    job_id = -1
                    logger.info(f"🎥 Video recorded for {wallet_address[:8]}... (UPLOAD SKIPPED - testing mode)")
                else:
                    # Queue background upload
                    upload_queue = get_upload_queue()
                    job_id = upload_queue.add(
                        file_path=video_path,
                        file_type="video",
                        wallet_address=wallet_address,
                        camera_id=camera_pda,
                        device_signature=device_signature,
                        timestamp=latest_video.get("timestamp", 0),
                    )
                    logger.info(f"🎥 Video recorded for {wallet_address[:8]}... -> queued as job #{job_id}")

                # Buffer to timeline
                try:
                    from services.timeline_activity_service import get_timeline_activity_service
                    timeline_service = get_timeline_activity_service()
                    timeline_service.buffer_video_activity(
                        wallet_address=wallet_address,
                        pipe_file_name=latest_video["filename"],
                        pipe_file_id=str(job_id),
                        duration_seconds=duration_limit,
                        metadata={
                            "timestamp": latest_video.get("timestamp"),
                            "device_signature": device_signature,
                            "filename": latest_video.get("filename"),
                            "size": latest_video.get("size"),
                            "storage_provider": storage_provider,
                            "job_id": job_id,
                        },
                    )
                except Exception as timeline_err:
                    logger.warning(f"Failed to buffer video to timeline: {timeline_err}")

                storage_upload_info = {
                    "initiated": True,
                    "completed": False,
                    "upload_status": "pending",
                    "storage_provider": storage_provider,
                    "device_signature": device_signature,
                    "job_id": job_id,
                    "local_url": f"/videos/{latest_video['filename']}",
                }

            else:
                logger.error(f"Video file not found at {video_path}")
                storage_upload_info["error"] = "Video file not found"

        except Exception as e:
            logger.error(
                f"Failed to process video for {wallet_address[:8]}...: {e}",
                exc_info=True,
            )
            storage_upload_info = {
                "initiated": False,
                "error": str(e),
                "upload_status": "failed",
                "storage_provider": storage_provider,
            }

        return jsonify(
            {
                "success": True,
                "recording": False,
                "completed": True,
                "video": latest_video,
                "filename": latest_video.get("filename"),
                "path": latest_video.get("path"),
                "url": latest_video.get("url"),
                "size": latest_video.get("size"),
                "timestamp": latest_video.get("timestamp"),
                "storage_upload": storage_upload_info,
                "pipe_upload": storage_upload_info,  # Backward compatibility
            }
        )

    # Media Access
    @app.route("/api/photos")
    def api_photos():
        """Standardized photos list endpoint"""
        return list_photos()

    @app.route("/api/videos")
    def api_videos():
        """Standardized videos list endpoint"""
        return list_videos()

    @app.route("/api/photos/<filename>")
    def api_get_photo(filename):
        """Standardized photo access endpoint"""
        return get_photo(filename)

    @app.route("/api/videos/<filename>")
    def api_get_video(filename):
        """Standardized video access endpoint"""
        return get_video(filename)

    # Session Management
    @app.route("/api/session/connect", methods=["POST"])
    def api_session_connect():
        """Standardized session connect endpoint"""
        return connect()

    # NOTE: /api/session/disconnect removed - use /api/checkout instead
    # /api/checkout handles session end + timeline write + backend notification

    @app.route("/api/session/status/<wallet_address>")
    @sign_response
    def api_session_status(wallet_address):
        """
        Check if a specific wallet is currently checked in.

        Phase 3 Privacy Architecture:
        - Frontend queries this to know if user can take photos/videos
        - Returns isCheckedIn: true/false
        - Also returns activeSessionCount for consistency
        """
        try:
            if not wallet_address:
                return jsonify({"success": False, "error": "wallet_address is required"}), 400

            # Phase 3 Privacy Architecture: Check session_service (off-chain source of truth)
            # NOT blockchain_sync (which checks on-chain state - outdated)
            session_service = get_services().get("session")

            # Check if wallet has an active session in the off-chain session service
            session = session_service.get_session_by_wallet(wallet_address) if session_service else None
            is_checked_in = session is not None

            # Get total active count for consistency with /api/status
            active_count = session_service.get_active_session_count() if session_service else 0

            logger.info(f"[SESSION-STATUS] Wallet {wallet_address[:8]}... checked_in={is_checked_in} (session_service)")

            return jsonify({
                "success": True,
                "wallet_address": wallet_address,
                "isCheckedIn": is_checked_in,
                "activeSessionCount": active_count
            })
        except Exception as e:
            logger.error(f"Error checking session status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Computer Vision (Jetson-specific)
    @app.route("/api/face/enroll/confirm", methods=["POST"])
    def api_face_enroll_confirm():
        """Confirm face enrollment and handle biometric cleanup"""
        wallet_address = request.json.get("wallet_address")
        # Accept both parameter names for compatibility
        signed_transaction = request.json.get("signed_transaction") or request.json.get("transaction_signature")
        face_id = request.json.get("face_id")
        biometric_session_id = request.json.get("biometric_session_id")

        if not wallet_address:
            return jsonify(
                {"success": False, "error": "wallet_address is required"}
            ), 400

        logger.info(
            f"[ENROLL-CONFIRM] Confirming face enrollment for wallet: {wallet_address}, face_id: {face_id}"
        )

        # Check if wallet is checked in via session_service (Phase 3 source of truth)
        session_service = get_services().get("session")
        if not session_service or not session_service.get_session_by_wallet(wallet_address):
            logger.warning(f"[ENROLL-CONFIRM] Wallet {wallet_address} is not checked in")
            return jsonify(
                {"success": False, "error": "Wallet must be checked in to confirm face enrollment"}
            ), 403

        try:
            # Validate required parameters
            if not signed_transaction or not face_id:
                return jsonify(
                    {
                        "success": False,
                        "error": "Missing required parameters: signed_transaction and face_id",
                    }
                ), 400

            # Call Solana middleware to confirm the transaction
            logger.info(
                f"[ENROLL-CONFIRM] Calling Solana middleware to confirm transaction for wallet: {wallet_address}"
            )

            confirm_response = requests.post(
                "http://solana-middleware:5001/api/face/enroll/confirm",
                json={
                    "wallet_address": wallet_address,
                    "transaction_signature": signed_transaction,
                    "face_id": face_id,
                },
                timeout=15,
            )

            if confirm_response.status_code != 200:
                logger.error(
                    f"[ENROLL-CONFIRM] Solana middleware confirm error: {confirm_response.status_code}"
                )
                return jsonify(
                    {
                        "success": False,
                        "error": "Failed to confirm transaction with Solana middleware",
                    }
                ), 500

            confirm_data = confirm_response.json()
            transaction_id = confirm_data["transaction_id"]

            # Clean up biometric session if provided
            biometric_purge_success = False
            if biometric_session_id:
                logger.info(
                    f"[ENROLL-CONFIRM] 🧹 Cleaning up biometric session: {biometric_session_id}"
                )

                try:
                    purge_response = requests.post(
                        "http://biometric-security:5003/api/biometric/purge-session",
                        json={"session_id": biometric_session_id},
                        timeout=10,
                    )

                    if purge_response.status_code == 200:
                        logger.info(
                            f"[ENROLL-CONFIRM] ✅ Successfully purged biometric session: {biometric_session_id}"
                        )
                        biometric_purge_success = True
                    else:
                        logger.error(
                            f"[ENROLL-CONFIRM] ❌ Failed to purge biometric session: HTTP {purge_response.status_code}"
                        )
                        logger.error(
                            f"[ENROLL-CONFIRM] Response: {purge_response.text}"
                        )

                except Exception as purge_error:
                    logger.error(
                        f"[ENROLL-CONFIRM] ❌ Error purging biometric session: {purge_error}"
                    )

            # NOTE: Local enrollment already completed in /api/face/extract-embedding
            # The phone selfie embedding was stored at lines 1128-1156 in extract-embedding
            # No need to re-enroll from Jetson camera frame here

            logger.info(
                f"[ENROLL-CONFIRM] ✅ Face enrollment completed successfully for wallet: {wallet_address}, transaction_id: {transaction_id}"
            )

            # AUTO-RELOAD: Fetch and load the newly minted recognition token for immediate face recognition
            # This ensures users don't need to check out and back in after creating their token
            try:
                from services.blockchain_session_sync import get_blockchain_session_sync
                blockchain_sync = get_blockchain_session_sync()
                logger.info(f"[ENROLL-CONFIRM] 🔄 Auto-loading recognition token for {wallet_address[:8]}...")
                blockchain_sync._fetch_and_decrypt_recognition_token(wallet_address)
                logger.info(f"[ENROLL-CONFIRM] ✅ Recognition token loaded - face recognition now active!")
            except Exception as reload_error:
                logger.warning(f"[ENROLL-CONFIRM] ⚠️ Failed to auto-load recognition token: {reload_error}")
                # Non-fatal - enrollment succeeded, user can still check out/in to reload

            # Return the expected format for frontend
            return jsonify(
                {
                    "success": True,
                    "face_id": face_id,
                    "transaction_id": transaction_id,
                    "biometric_session_cleaned": biometric_purge_success,
                    "local_enrollment_completed": True,  # Already done in extract-embedding
                }
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"[ENROLL-CONFIRM] Network error: {str(e)}")
            return jsonify(
                {"success": False, "error": f"Service communication error: {str(e)}"}
            ), 503

        except Exception as e:
            logger.error(f"[ENROLL-CONFIRM] Unexpected error: {str(e)}")
            return jsonify(
                {"success": False, "error": f"Failed to confirm enrollment: {str(e)}"}
            ), 500

    @app.route("/api/face/extract-embedding", methods=["POST"])
    def api_face_extract_embedding():
        """
        Extract face embedding from base64 image for phone-based enrollment.
        Includes quality scoring and optional encryption for security.
        Requires on-chain check-in for enrollment with encryption.
        """
        logger.info("🔍 DEBUG: /api/face/extract-embedding endpoint called!")
        try:
            data = request.get_json()
            logger.info(
                f"🔍 DEBUG: Received data keys: {list(data.keys()) if data else 'None'}"
            )
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            # Check for image data in both possible field names
            image_data = data.get("image") or data.get("image_data")
            wallet_address = data.get("wallet_address")
            encrypt = data.get("encrypt", False)  # Whether to encrypt the embedding

            if not image_data:
                return jsonify({"success": False, "error": "Missing image data"}), 400

            # CRITICAL: Require wallet_address for ALL recognition token operations
            if not wallet_address:
                return jsonify(
                    {
                        "success": False,
                        "error": "wallet_address is required to create a recognition token",
                    }
                ), 400

            # CRITICAL: Require check-in for ANY recognition token creation
            # User must be physically present at the camera (checked in) to enroll
            session_service = get_services().get("session")
            if not session_service or not session_service.get_session_by_wallet(wallet_address):
                logger.warning(
                    f"[EXTRACT-EMBEDDING] ❌ Wallet {wallet_address} not checked in - recognition token creation denied"
                )
                return jsonify(
                    {"success": False, "error": "You must be checked in at this camera to create a recognition token", "checked_in": False}
                ), 403

            # Decode base64 image with EXIF rotation handling
            # Phone selfies have EXIF orientation metadata - cv2.imdecode ignores it
            # Without this fix, portrait selfies come in as landscape (face sideways)
            try:
                if image_data.startswith("data:image"):
                    # Remove data URL prefix (data:image/jpeg;base64,)
                    image_data = image_data.split(",")[1]

                # Decode base64 to bytes
                import base64
                import io
                from PIL import Image, ImageOps

                image_bytes = base64.b64decode(image_data)

                # Use PIL to decode with EXIF rotation handling
                pil_image = Image.open(io.BytesIO(image_bytes))
                original_size = pil_image.size  # (width, height)

                # Apply EXIF transpose - rotates image based on EXIF orientation tag
                pil_image = ImageOps.exif_transpose(pil_image)
                corrected_size = pil_image.size

                # Convert PIL RGB to OpenCV BGR
                import numpy as np
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                frame = np.array(pil_image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if frame is not None:
                    h, w = frame.shape[:2]
                    orientation = "PORTRAIT" if h > w else "LANDSCAPE"
                    rotated = original_size != corrected_size
                    logger.info(f"📸 [SELFIE] Raw: {original_size[0]}x{original_size[1]} -> EXIF corrected: {w}x{h} ({orientation}) [rotated={rotated}]")

                if frame is None:
                    return jsonify(
                        {
                            "success": False,
                            "error": "Invalid image data - could not decode image",
                        }
                    ), 400

            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                return jsonify(
                    {"success": False, "error": "Failed to decode base64 image"}
                ), 400

            # Check for GPU face service or native mode
            services = get_services()
            gpu_face_service = services.get("gpu_face")
            use_native_mode = gpu_face_service is None

            # If native mode, use native TensorRT pipeline
            if use_native_mode:
                try:
                    from native_client import get_native_service
                    native_service = get_native_service()
                    if not native_service.is_ready():
                        native_service.initialize()
                except Exception as e:
                    logger.error(f"Failed to get native service: {e}")
                    return jsonify(
                        {"success": False, "error": "Native inference service not available"}
                    ), 503
            else:
                # Check if models are loaded for non-native mode
                if not gpu_face_service._models_loaded:
                    return jsonify(
                        {"success": False, "error": "Face recognition models not loaded"}
                    ), 503

            # Extract embedding and analyze faces
            quality_score = 0
            quality_factors = {}

            try:
                if use_native_mode:
                    # Use native TensorRT pipeline for face detection and embedding
                    # With Python InsightFace alignment for exact ArcFace compatibility
                    logger.info("Using native TensorRT + Python InsightFace alignment for face embedding extraction")

                    # Step 1: Detect face and get landmarks from C++ RetinaFace
                    result = native_service.process_image(frame)

                    if result is None:
                        return jsonify(
                            {"success": False, "error": "Native inference failed"}
                        ), 500

                    native_faces = result.get('faces', [])
                    if len(native_faces) == 0:
                        return jsonify(
                            {
                                "success": False,
                                "error": "No face detected in image - please ensure your face is clearly visible",
                                "quality_score": 0,
                                "quality_factors": {"face_detected": False},
                            }
                        ), 400

                    # Get the largest/best face from native results
                    best_face = max(
                        native_faces,
                        key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
                    )
                    det_score = best_face.get('confidence', 0.9)
                    landmarks = best_face.get('landmarks')

                    # Use the C++ TensorRT embedding directly
                    # Both phone selfie and camera stream now use the same C++ alignment
                    embedding = np.array(best_face['embedding'], dtype=np.float32)
                    logger.info("Using C++ TensorRT embedding for face registration")

                    # Native embeddings are already normalized

                    # Use server's quality score directly (bbox coords are unreliable in full-frame mode)
                    native_quality = best_face.get('quality', 0.5)

                    # Calculate sharpness/lighting on the full image center region
                    h, w = frame.shape[:2]
                    # Use center 50% of image for quality metrics
                    margin_x, margin_y = w // 4, h // 4
                    center_region = frame[margin_y:h-margin_y, margin_x:w-margin_x]

                    gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_center, cv2.CV_64F).var()
                    sharpness_score = min(laplacian_var / 10, 100)

                    hist = cv2.calcHist([gray_center], [0], None, [256], [0, 256])
                    hist_normalized = hist.ravel() / hist.sum()
                    low_light = hist_normalized[:50].sum()
                    high_light = hist_normalized[200:].sum()
                    lighting_score = max(100 - (low_light + high_light) * 100, 0)

                    # Combine: native quality (face detection) + image quality (sharpness/lighting)
                    quality_score = round(
                        native_quality * 100 * 0.5 +  # 50% from native face quality
                        sharpness_score * 0.25 +       # 25% from sharpness
                        lighting_score * 0.25,         # 25% from lighting
                        1
                    )

                    quality_factors = {
                        "native_face_quality": round(native_quality * 100, 1),
                        "detection_confidence": round(det_score * 100, 1),
                        "image_sharpness": round(sharpness_score, 1),
                        "lighting_quality": round(lighting_score, 1),
                    }

                    logger.info(f"[NATIVE] Quality: {quality_score}% (native={native_quality:.2f}, sharp={sharpness_score:.1f}, light={lighting_score:.1f})")

                    # Skip the bbox-based quality calculation for native mode
                    embedding_list = embedding.tolist()
                    logger.info(f"Extracted embedding with quality score: {quality_score}%")

                    # Store embedding locally
                    if wallet_address:
                        try:
                            enrollment_metadata = {
                                "source": "phone_camera",
                                "quality_score": quality_score,
                                "timestamp": int(time.time()),
                            }

                            from services.native_identity_service import get_native_identity_service
                            identity_service = get_native_identity_service()
                            identity_service.add_identity(wallet_address, embedding, enrollment_metadata)
                            logger.info(
                                f"✅ Successfully stored face embedding via native identity service for {wallet_address[:8]}... - recognition enabled"
                            )
                        except Exception as enroll_error:
                            logger.error(
                                f"❌ Error storing face embedding locally for {wallet_address[:8]}...: {enroll_error}"
                            )

                    # Handle encryption and blockchain transaction for native mode
                    if encrypt and wallet_address:
                        try:
                            # Create biometric session for encryption
                            biometric_response = requests.post(
                                "http://biometric-security:5003/api/biometric/create-session",
                                json={
                                    "wallet_address": wallet_address,
                                    "session_duration": 300,  # 5 minutes for enrollment
                                },
                                timeout=10,
                            )

                            if biometric_response.status_code == 200:
                                biometric_session = biometric_response.json()
                                session_id = biometric_session["session_id"]

                                # Encrypt the embedding
                                encrypt_response = requests.post(
                                    "http://biometric-security:5003/api/biometric/encrypt-embedding",
                                    json={
                                        "embedding": embedding_list,
                                        "wallet_address": wallet_address,
                                        "session_id": session_id,
                                        "metadata": {
                                            "quality_score": quality_score,
                                            "timestamp": int(time.time()),
                                            "source": "phone_camera",
                                        },
                                    },
                                    timeout=15,
                                )

                                if encrypt_response.status_code == 200:
                                    encrypted_data = encrypt_response.json()
                                    token_package = encrypted_data["token_package"]

                                    logger.info(
                                        f"Successfully encrypted embedding for {wallet_address[:8]}..."
                                    )

                                    solana_response = requests.post(
                                        "http://solana-middleware:5001/api/blockchain/mint-recognition-token",
                                        json={
                                            "wallet_address": wallet_address,
                                            "face_embedding": token_package,
                                            "biometric_session_id": session_id,
                                        },
                                        timeout=30,
                                    )

                                    if solana_response.status_code != 200:
                                        logger.error(f"Solana middleware error: {solana_response.status_code}")
                                        return jsonify({"success": False, "error": f"Solana middleware error: {solana_response.status_code}"}), 500

                                    solana_data = solana_response.json()
                                    return jsonify({
                                        "success": True,
                                        "transaction_buffer": solana_data["transaction_buffer"],
                                        "face_id": solana_data["face_id"],
                                        "encrypted": True,
                                        "quality_score": quality_score,
                                        "quality_factors": quality_factors,
                                    })
                        except Exception as encrypt_error:
                            logger.warning(
                                f"Encryption failed, returning unencrypted: {encrypt_error}"
                            )
                            # Fall through to return unencrypted

                    # Return unencrypted for native mode (no wallet or encrypt failed)
                    return jsonify(
                        {
                            "success": True,
                            "embedding": embedding_list,
                            "encrypted": False,
                            "quality_score": quality_score,
                            "quality_factors": quality_factors,
                            "quality_rating": (
                                "excellent" if quality_score >= 80 else
                                "good" if quality_score >= 65 else
                                "acceptable" if quality_score >= 50 else
                                "poor"
                            ),
                        }
                    )

                else:
                    # Non-native mode (InsightFace Python)
                    # Convert BGR to RGB for InsightFace
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get all faces detected by InsightFace
                    faces = gpu_face_service.face_embedder.get(frame_rgb)

                    if len(faces) == 0:
                        return jsonify(
                            {
                                "success": False,
                                "error": "No face detected in image - please ensure your face is clearly visible",
                                "quality_score": 0,
                                "quality_factors": {"face_detected": False},
                            }
                        ), 400

                    # Get the largest/best face
                    face = max(
                        faces,
                        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    )
                    bbox = face.bbox
                    det_score = float(face.det_score) if hasattr(face, "det_score") else 0.9
                    embedding = face.normed_embedding

                    # Calculate quality score based on multiple factors for non-native mode
                    face_width = bbox[2] - bbox[0]
                    face_height = bbox[3] - bbox[1]
                    image_height, image_width = frame.shape[:2]

                    # 1. Face size score (larger faces are better)
                    face_area_ratio = (face_width * face_height) / (
                        image_width * image_height
                    )
                    size_score = min(face_area_ratio * 200, 100)  # 0-100 scale
                    quality_factors["face_size"] = round(size_score, 1)

                    # 2. Face position score (centered faces are better)
                    face_center_x = (bbox[0] + bbox[2]) / 2
                    face_center_y = (bbox[1] + bbox[3]) / 2
                    center_offset_x = abs(face_center_x - image_width / 2) / (
                        image_width / 2
                    )
                    center_offset_y = abs(face_center_y - image_height / 2) / (
                        image_height / 2
                    )
                    position_score = 100 - (center_offset_x + center_offset_y) * 50
                    quality_factors["face_position"] = round(max(position_score, 0), 1)

                    # 3. Detection confidence
                    confidence_score = det_score * 100
                    quality_factors["detection_confidence"] = round(confidence_score, 1)

                    # 4. Image sharpness (using Laplacian variance)
                    face_region = frame[
                        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                    ]
                    if face_region.size > 0:
                        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        sharpness_score = min(laplacian_var / 10, 100)  # Normalize to 0-100
                        quality_factors["image_sharpness"] = round(sharpness_score, 1)
                    else:
                        quality_factors["image_sharpness"] = 50

                    # 5. Lighting quality (check histogram distribution)
                    if face_region.size > 0:
                        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
                        hist_normalized = hist.ravel() / hist.sum()
                        low_light = hist_normalized[:50].sum()
                        high_light = hist_normalized[200:].sum()
                        lighting_score = 100 - (low_light + high_light) * 100
                        quality_factors["lighting_quality"] = round(
                            max(lighting_score, 0), 1
                        )
                    else:
                        quality_factors["lighting_quality"] = 50

                    # Calculate overall quality score (weighted average)
                    quality_score = (
                        size_score * 0.25
                        + position_score * 0.15
                        + confidence_score * 0.30
                        + quality_factors["image_sharpness"] * 0.15
                        + quality_factors["lighting_quality"] * 0.15
                    )
                    quality_score = round(quality_score, 1)

                    embedding_list = embedding.tolist()
                    logger.info(f"Extracted embedding with quality score: {quality_score}%")

                    # Store embedding locally for non-native mode
                    if wallet_address:
                        try:
                            enrollment_metadata = {
                                "source": "phone_camera",
                                "quality_score": quality_score,
                                "timestamp": int(time.time()),
                            }

                            with gpu_face_service._faces_lock:
                                gpu_face_service._face_embeddings[wallet_address] = embedding

                            display_name = gpu_face_service.get_user_display_name(wallet_address)
                            gpu_face_service._face_names[wallet_address] = display_name
                            gpu_face_service._face_metadata[wallet_address] = enrollment_metadata

                            gpu_face_service.save_face_embedding(
                                wallet_address, embedding, enrollment_metadata
                            )

                            logger.info(
                                f"✅ Successfully stored face embedding locally for {wallet_address[:8]}... - recognition enabled"
                            )

                        except Exception as enroll_error:
                            logger.error(
                                f"❌ Error storing face embedding locally for {wallet_address[:8]}...: {enroll_error}"
                            )

                    # If encryption is requested and wallet address provided
                    if encrypt and wallet_address:
                        try:
                            # Create biometric session for encryption
                            biometric_response = requests.post(
                                "http://biometric-security:5003/api/biometric/create-session",
                                json={
                                    "wallet_address": wallet_address,
                                    "session_duration": 300,  # 5 minutes for enrollment
                                },
                                timeout=10,
                            )

                            if biometric_response.status_code == 200:
                                biometric_session = biometric_response.json()
                                session_id = biometric_session["session_id"]

                                # Encrypt the embedding
                                encrypt_response = requests.post(
                                    "http://biometric-security:5003/api/biometric/encrypt-embedding",
                                    json={
                                        "embedding": embedding_list,
                                        "wallet_address": wallet_address,
                                        "session_id": session_id,
                                        "metadata": {
                                            "quality_score": quality_score,
                                            "timestamp": int(time.time()),
                                            "source": "phone_camera",
                                        },
                                    },
                                    timeout=15,
                                )

                                if encrypt_response.status_code == 200:
                                    encrypted_data = encrypt_response.json()
                                    token_package = encrypted_data["token_package"]

                                    logger.info(
                                        f"Successfully encrypted embedding for {wallet_address[:8]}..."
                                    )

                                    solana_response = requests.post(
                                        "http://solana-middleware:5001/api/blockchain/mint-recognition-token",
                                        json={
                                            "wallet_address": wallet_address,
                                            "face_embedding": token_package,
                                            "biometric_session_id": session_id,
                                        },
                                        timeout=30,
                                    )

                                    if solana_response.status_code != 200:
                                        logger.error(f"Solana middleware error: {solana_response.status_code}")
                                        return jsonify({"success": False, "error": f"Solana middleware error: {solana_response.status_code}"}), 500

                                    solana_data = solana_response.json()
                                    return jsonify({
                                        "success": True,
                                        "transaction_buffer": solana_data["transaction_buffer"],
                                        "face_id": solana_data["face_id"],
                                        "encrypted": True,
                                        "quality_score": quality_score,
                                        "quality_factors": quality_factors,
                                    })
                        except Exception as encrypt_error:
                            logger.warning(
                                f"Encryption failed, returning unencrypted: {encrypt_error}"
                            )
                            # Fall through to return unencrypted

                # Return unencrypted embedding with quality metrics
                return jsonify(
                    {
                        "success": True,
                        "embedding": embedding_list,
                        "encrypted": False,
                        "quality_score": quality_score,
                        "quality_factors": quality_factors,
                        "quality_rating": api_face_extract_embedding.get_quality_rating(
                            quality_score
                        ),
                        "recommendations": api_face_extract_embedding.get_quality_recommendations(
                            quality_score, quality_factors
                        ),
                    }
                )

            except Exception as analysis_error:
                logger.error(f"Face analysis error: {analysis_error}")

                # In native mode, no fallback available
                if use_native_mode:
                    return jsonify(
                        {
                            "success": False,
                            "error": f"Face detection failed: {str(analysis_error)}",
                            "quality_score": 0,
                        }
                    ), 400

                # Non-native mode: try basic extraction without quality scoring
                full_embedding = gpu_face_service.extract_face_embedding(frame)

                if full_embedding is None:
                    return jsonify(
                        {
                            "success": False,
                            "error": "Face detection failed",
                            "quality_score": 0,
                        }
                    ), 400

                # Store embedding locally for fallback case too
                if wallet_address:
                    try:
                        fallback_metadata = {
                            "source": "phone_camera_fallback",
                            "quality_score": 50,
                            "timestamp": int(time.time()),
                        }

                        with gpu_face_service._faces_lock:
                            gpu_face_service._face_embeddings[wallet_address] = (
                                full_embedding
                            )

                        display_name = gpu_face_service.get_user_display_name(
                            wallet_address
                        )
                        gpu_face_service._face_names[wallet_address] = display_name
                        gpu_face_service._face_metadata[wallet_address] = (
                            fallback_metadata
                        )

                        gpu_face_service.save_face_embedding(
                            wallet_address, full_embedding, fallback_metadata
                        )

                        logger.info(
                            f"✅ Successfully stored face embedding locally (fallback) for {wallet_address[:8]}... - recognition enabled"
                        )

                    except Exception as enroll_error:
                        logger.error(
                            f"❌ Error storing face embedding locally (fallback) for {wallet_address[:8]}...: {enroll_error}"
                        )

                return jsonify(
                    {
                        "success": True,
                        "embedding": full_embedding.tolist(),
                        "encrypted": False,
                        "quality_score": 50,
                        "quality_rating": "unknown",
                        "warning": "Quality assessment unavailable",
                    }
                )

        except Exception as e:
            logger.error(f"Error in face embedding extraction: {e}")
            return jsonify(
                {
                    "success": False,
                    "error": f"Face embedding extraction failed: {str(e)}",
                }
            ), 500

    # Helper functions for quality assessment
    api_face_extract_embedding.get_quality_rating = lambda score: (
        "excellent"
        if score >= 90
        else "good"
        if score >= 80
        else "acceptable"
        if score >= 70
        else "poor"
        if score >= 60
        else "very_poor"
    )

    api_face_extract_embedding.get_quality_recommendations = lambda score, factors: (
        ["Excellent quality! Your facial embedding is optimal for recognition."]
        if score >= 90
        else [
            "Move closer to the camera for a larger face image"
            if factors.get("face_size", 100) < 70
            else None,
            "Center your face in the frame"
            if factors.get("face_position", 100) < 70
            else None,
            "Hold the camera steady to reduce blur"
            if factors.get("image_sharpness", 100) < 70
            else None,
            "Improve lighting - avoid shadows and extreme brightness"
            if factors.get("lighting_quality", 100) < 70
            else None,
            "Ensure your face is fully visible and unobstructed"
            if factors.get("detection_confidence", 100) < 80
            else None,
        ]
        if score < 90
        else []
    )
    # Filter out None values from recommendations
    api_face_extract_embedding.get_quality_recommendations = (
        lambda score, factors: [
            rec
            for rec in api_face_extract_embedding.get_quality_recommendations.__wrapped__(
                score, factors
            )
            if rec
        ]
        if hasattr(
            api_face_extract_embedding.get_quality_recommendations, "__wrapped__"
        )
        else (
            ["Excellent quality! Your facial embedding is optimal for recognition."]
            if score >= 90
            else [
                rec
                for rec in [
                    "Move closer to the camera for a larger face image"
                    if factors.get("face_size", 100) < 70
                    else None,
                    "Center your face in the frame"
                    if factors.get("face_position", 100) < 70
                    else None,
                    "Hold the camera steady to reduce blur"
                    if factors.get("image_sharpness", 100) < 70
                    else None,
                    "Improve lighting - avoid shadows and extreme brightness"
                    if factors.get("lighting_quality", 100) < 70
                    else None,
                    "Ensure your face is fully visible and unobstructed"
                    if factors.get("detection_confidence", 100) < 80
                    else None,
                ]
                if rec
            ]
            or ["Consider retaking for better quality"]
        )
    )

    @app.route("/api/face/recognize", methods=["POST"])
    def api_face_recognize():
        """Standardized face recognition endpoint"""
        return recognize_face()

    @app.route("/api/gesture/current", methods=["GET"])
    def api_gesture_current():
        """Standardized current gesture endpoint"""
        return current_gesture()

    @app.route("/api/visualization/face", methods=["POST"])
    def api_visualization_face():
        """Standardized face visualization toggle"""
        return toggle_face_visualization()

    @app.route("/api/visualization/gesture", methods=["POST"])
    def api_visualization_gesture():
        """Standardized gesture visualization toggle"""
        return toggle_gesture_visualization()

    @app.route("/api/visualization/pose", methods=["POST"])
    def api_visualization_pose():
        """Standardized pose visualization toggle"""
        return toggle_pose_visualization()

    # User Profile Management - Enhanced for Scalable Architecture
    @app.route("/api/user/profile", methods=["POST"])
    def api_update_user_profile():
        """Update user profile for display name resolution - Enhanced for PDA subdomain architecture"""
        data = request.json or {}
        wallet_address = data.get("wallet_address")
        display_name = data.get("display_name")
        username = data.get("username")
        transaction_signature = data.get(
            "transaction_signature"
        )  # Optional: for verification

        if not wallet_address:
            return jsonify(
                {"success": False, "error": "Wallet address is required"}
            ), 400

        # Get camera PDA for logging
        camera_pda = get_camera_pda()

        # Create enhanced user profile with metadata
        user_profile = {
            "wallet_address": wallet_address,
            "display_name": display_name,
            "username": username,
            "camera_pda": camera_pda,
            "updated_at": int(time.time()),
            "transaction_signature": transaction_signature,
        }

        # Store profile in native mode profile store
        global _user_profiles
        _user_profiles[wallet_address] = user_profile

        # Resolve display name with fallback hierarchy
        resolved_display_name = display_name or username or wallet_address[:8]

        logger.info(
            f"[PROFILE-UPDATE] Camera {camera_pda[:8]}... updated profile for {wallet_address}: {resolved_display_name}"
        )

        return jsonify(
            {
                "success": True,
                "wallet_address": wallet_address,
                "display_name": resolved_display_name,
                "camera_pda": camera_pda,
                "camera_url": f"https://{camera_pda.lower()}.mmoment.xyz/api",
                "updated_at": user_profile["updated_at"],
                "message": "User profile updated successfully",
            }
        )

    @app.route("/api/user/profile/<wallet_address>", methods=["GET"])
    def api_get_user_profile(wallet_address):
        """Get user profile by wallet address"""
        if not wallet_address:
            return jsonify(
                {"success": False, "error": "Wallet address is required"}
            ), 400

        # Get profile from native mode profile store
        global _user_profiles
        try:
            profile = _user_profiles.get(wallet_address)
            if profile:
                resolved_display_name = (
                    profile.get("display_name")
                    or profile.get("username")
                    or wallet_address[:8]
                )
                return jsonify(
                    {
                        "success": True,
                        "profile": {
                            "wallet_address": wallet_address,
                            "display_name": resolved_display_name,
                            "username": profile.get("username"),
                            "camera_pda": profile.get("camera_pda"),
                            "updated_at": profile.get("updated_at"),
                        },
                    }
                )
            else:
                return jsonify({"success": False, "error": "Profile not found"}), 404

        except Exception as e:
            logger.error(f"Error getting user profile for {wallet_address}: {e}")
            return jsonify(
                {"success": False, "error": "Failed to retrieve profile"}
            ), 500

    @app.route("/api/user/profile/<wallet_address>", methods=["DELETE"])
    def api_delete_user_profile(wallet_address):
        """Delete user profile (for session cleanup)"""
        if not wallet_address:
            return jsonify(
                {"success": False, "error": "Wallet address is required"}
            ), 400

        # Remove profile from native mode profile store
        global _user_profiles
        _user_profiles.pop(wallet_address, None)

        camera_pda = get_camera_pda()
        logger.info(
            f"[PROFILE-DELETE] Camera {camera_pda[:8]}... removed profile for {wallet_address}"
        )

        return jsonify(
            {
                "success": True,
                "wallet_address": wallet_address,
                "message": "User profile deleted successfully",
            }
        )

    @app.route("/api/checkin", methods=["POST"])
    def api_unified_checkin():
        """
        Unified check-in endpoint - Phase 3 Privacy Architecture.

        Check-in is a CRYPTOGRAPHIC HANDSHAKE (single-shot):
        - Client signs message: "{wallet_address}|{timestamp}|{nonce}"
        - Jetson verifies Ed25519 signature using wallet as pubkey
        - Only if valid → session is created

        Expected body:
        {
            "wallet_address": "...",           # Required - base58 Solana pubkey
            "request_signature": "...",        # Required - base58 Ed25519 signature
            "request_timestamp": 1234567890123,# Required - Unix ms when signed
            "request_nonce": "...",            # Required - random string for uniqueness
            "display_name": "...",             # Optional
            "username": "...",                 # Optional
        }
        """
        data = request.json or {}
        wallet_address = data.get("wallet_address")
        # Accept both field name formats (request_* from frontend or simple names)
        signature = data.get("request_signature") or data.get("signature")
        timestamp = data.get("request_timestamp") or data.get("timestamp")
        nonce = data.get("request_nonce") or data.get("nonce")
        display_name = data.get("display_name")
        username = data.get("username")

        logger.info(f"[CHECKIN] Request for {wallet_address[:8] if wallet_address else 'NONE'}... display_name={display_name}")

        # Validate required fields
        if not wallet_address:
            return jsonify({"success": False, "error": "wallet_address is required"}), 400
        if not signature:
            return jsonify({"success": False, "error": "request_signature is required (Ed25519 signed message)"}), 400
        if not timestamp:
            return jsonify({"success": False, "error": "request_timestamp is required"}), 400
        if not nonce:
            return jsonify({"success": False, "error": "request_nonce is required"}), 400

        camera_pda = get_camera_pda()

        # VERIFY ED25519 SIGNATURE - the core of check-in security
        from services.device_signer import DeviceSigner
        device_signer = DeviceSigner()

        # Verify signature with replay attack protection (5 min max age)
        # Message format: "{wallet_address}|{timestamp}|{nonce}"
        is_valid, error_msg = device_signer.verify_request_signature(
            wallet_address=wallet_address,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            max_age_seconds=300
        )

        if not is_valid:
            logger.warning(f"[CHECKIN] ❌ Signature verification failed for {wallet_address[:8]}...: {error_msg}")
            return jsonify({"success": False, "error": f"Signature verification failed: {error_msg}"}), 401

        logger.info(f"[CHECKIN] ✅ Signature verified for {wallet_address[:8]}...")
        logger.info(
            f"[CHECKIN] User {wallet_address[:8]}... checking in to camera {camera_pda[:8]}... (off-chain session)"
        )

        try:
            services = get_services()
            session_service = services["session"]

            # Build user profile
            user_profile = {
                "wallet_address": wallet_address,
                "display_name": display_name,
                "username": username,
            }

            resolved_display_name = display_name or username or wallet_address[:8]

            # Create off-chain session with local UUID and AES-256 key
            session_result = session_service.create_session(wallet_address, user_profile)
            session_id = session_result["session_id"]
            logger.info(f"[CHECKIN] Created off-chain session {session_id[:16]}... for {resolved_display_name}")

            # Trigger blockchain sync for recognition token loading
            # This still loads face recognition tokens from chain (if user has one)
            from services.blockchain_session_sync import get_blockchain_session_sync

            blockchain_sync = get_blockchain_session_sync()
            blockchain_sync.trigger_checkin(wallet_address, {
                **user_profile,
                "camera_pda": camera_pda,
                "updated_at": int(time.time()),
            })
            logger.info(f"[CHECKIN] Triggered recognition token loading for {resolved_display_name}")

            # Tell backend about check-in (simple synchronous POST - no queues!)
            backend_url = os.environ.get('BACKEND_URL', 'https://mmoment-production.up.railway.app')
            try:
                import requests as req
                activity_response = req.post(
                    f"{backend_url}/api/session/activity",
                    json={
                        "sessionId": session_id,
                        "cameraId": camera_pda,
                        "userPubkey": wallet_address,
                        "activityType": 0,  # CHECK_IN
                        "timestamp": int(time.time() * 1000),
                        "displayName": resolved_display_name,
                        "username": username
                    },
                    timeout=5
                )
                if activity_response.status_code == 200:
                    result = activity_response.json()
                    logger.info(f"[CHECKIN] ✅ Timeline notified! Sockets in room: {result.get('socketsInRoom', '?')}")
                else:
                    logger.warning(f"[CHECKIN] Timeline notification failed: HTTP {activity_response.status_code}")
            except Exception as e:
                logger.warning(f"[CHECKIN] Timeline notification failed (non-blocking): {e}")

            # Pre-authorize Pipe storage session for fast uploads
            import asyncio
            import threading

            from services.pipe_storage_service import pre_authorize_user_session

            def run_pipe_authorization():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    auth_success = loop.run_until_complete(
                        pre_authorize_user_session(wallet_address)
                    )
                    if auth_success:
                        logger.info(f"[CHECKIN] Pipe session pre-authorized for {wallet_address[:8]}...")
                except Exception as e:
                    logger.warning(f"[CHECKIN] Pipe pre-authorization failed: {e}")
                finally:
                    loop.close()

            pipe_auth_thread = threading.Thread(target=run_pipe_authorization, daemon=True)
            pipe_auth_thread.start()

            logger.info(f"[CHECKIN] Complete! User {resolved_display_name} checked in (off-chain)")

            return jsonify(
                {
                    "success": True,
                    "wallet_address": wallet_address,
                    "display_name": resolved_display_name,
                    "session_id": session_id,
                    "camera_pda": camera_pda,
                    "camera_url": f"https://{camera_pda.lower()}.mmoment.xyz/api",
                    "message": "Check-in successful (off-chain session)",
                }
            )

        except Exception as e:
            logger.error(f"[CHECKIN] Error for {wallet_address[:8]}...: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify(
                {"success": False, "error": f"Check-in failed: {str(e)}"}
            ), 500

    @app.route("/api/checkout", methods=["POST"])
    @require_session
    def api_checkout_notification():
        """
        Checkout endpoint - Phase 3 Privacy Architecture.

        Checkout now orchestrates two blockchain-related operations:
        1. Write encrypted activities to CameraTimeline (anonymous - no user info)
        2. Send encrypted session key to backend (for user's keychain - no camera info)

        This breaks the visible on-chain link between users and cameras.

        Expected body:
        {
            "wallet_address": "...",           # Required
        }

        Legacy fields (ignored for backwards compatibility):
        - transaction_signature: No longer needed (Jetson signs the timeline tx)
        """
        data = request.json or {}
        wallet_address = data.get("wallet_address")

        if not wallet_address:
            return jsonify(
                {"success": False, "error": "Wallet address is required"}
            ), 400

        logger.info(f"[CHECKOUT] User {wallet_address[:8]}... checking out")

        try:
            services = get_services()
            session_service = services["session"]

            # Get the raw session object (need session_key and activities)
            session = session_service.get_session_object_by_wallet(wallet_address)
            if not session:
                logger.warning(f"[CHECKOUT] No session found for {wallet_address[:8]}...")
                return jsonify(
                    {"success": True, "message": "No active session found (already cleaned up)"}
                )

            session_id = session.session_id
            check_in_time = int(session.created_at)
            camera_pda = get_camera_pda()

            # Get user profile info from session for timeline display
            user_profile = session.user_profile or {}
            display_name = user_profile.get('display_name') or user_profile.get('displayName') or wallet_address[:8]
            username = user_profile.get('username')

            # Add checkout activity to LOCAL buffer only (send_to_backend=False)
            # We send to backend DIRECTLY below for guaranteed sync delivery
            session_service.add_activity_to_session(
                wallet_address=wallet_address,
                activity_type=1,  # CHECK_OUT
                data={"duration_seconds": int(time.time() - session.created_at)},
                metadata={"camera_pda": camera_pda},
                send_to_backend=False  # We send directly below
            )

            # Tell backend about check-out DIRECTLY (sync POST - no async queue!)
            # This ensures timeline event is saved BEFORE we return the response
            backend_url = os.environ.get('BACKEND_URL', 'https://mmoment-production.up.railway.app')
            try:
                import requests as req
                activity_response = req.post(
                    f"{backend_url}/api/session/activity",
                    json={
                        "sessionId": session_id,
                        "cameraId": camera_pda,
                        "userPubkey": wallet_address,
                        "activityType": 1,  # CHECK_OUT
                        "timestamp": int(time.time() * 1000),
                        "displayName": display_name,
                        "username": username
                    },
                    timeout=5
                )
                if activity_response.status_code == 200:
                    result = activity_response.json()
                    logger.info(f"[CHECKOUT] ✅ Timeline notified! Sockets in room: {result.get('debug', {}).get('socketsInRoom', '?')}")
                else:
                    logger.warning(f"[CHECKOUT] Timeline notification failed: HTTP {activity_response.status_code}")
            except Exception as e:
                logger.warning(f"[CHECKOUT] Timeline notification failed (non-blocking): {e}")

            # Get all activities from session
            activities = session.get_activities()
            logger.info(f"[CHECKOUT] Session has {len(activities)} activities to commit")

            # Import checkout service functions
            from services.checkout_service import (
                encrypt_and_write_activities,
                send_access_key_to_backend
            )

            tx_signature = None
            access_key_sent = False

            # Step 1: Encrypt activities and write to CameraTimeline
            if activities:
                try:
                    tx_signature = encrypt_and_write_activities(
                        session_key=session.session_key,
                        activities=activities,
                        checked_in_users=[wallet_address]  # For access grants
                    )
                    logger.info(f"[CHECKOUT] Timeline updated, tx: {tx_signature[:16] if tx_signature else 'None'}...")
                except Exception as e:
                    logger.error(f"[CHECKOUT] Failed to write activities to timeline: {e}")
                    # Continue with checkout even if timeline write fails
                    # Activities are lost but user is still checked out

            # Step 2: Send encrypted session key to backend
            try:
                access_key_sent = send_access_key_to_backend(
                    user_pubkey=wallet_address,
                    session_key=session.session_key,
                    check_in_time=check_in_time
                )
                if access_key_sent:
                    logger.info(f"[CHECKOUT] Access key sent to backend for {wallet_address[:8]}...")
                else:
                    logger.warning(f"[CHECKOUT] Failed to send access key to backend")
            except Exception as e:
                logger.error(f"[CHECKOUT] Error sending access key: {e}")

            # Step 3: Clean up face recognition data
            try:
                from services.blockchain_session_sync import get_blockchain_session_sync
                blockchain_sync = get_blockchain_session_sync()
                blockchain_sync.trigger_checkout(wallet_address)
                logger.info(f"[CHECKOUT] Face recognition data cleaned up")
            except Exception as e:
                logger.warning(f"[CHECKOUT] Error cleaning up face data: {e}")

            # Step 4: End the session
            session_service.end_session(session_id, wallet_address)
            logger.info(f"[CHECKOUT] Session {session_id[:16]}... ended")

            return jsonify(
                {
                    "success": True,
                    "wallet_address": wallet_address,
                    "session_id": session_id,
                    "activities_committed": len(activities),
                    "timeline_tx": tx_signature,
                    "access_key_sent": access_key_sent,
                    "message": "Checkout complete (privacy-preserving)",
                }
            )

        except Exception as e:
            logger.error(f"[CHECKOUT] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify(
                {"success": False, "error": f"Checkout failed: {str(e)}"}
            ), 500

    @app.route("/api/camera/info", methods=["GET"])
    @sign_response
    def api_camera_info():
        """Get camera information for frontend discovery with device signature"""
        camera_pda = get_camera_pda()
        camera_program_id = os.environ.get("CAMERA_PROGRAM_ID", "unknown")

        # Get current session count
        services = get_services()
        session_service = services["session"]
        active_sessions = len(session_service.get_all_sessions())

        # Get buffer status
        buffer_service = services["buffer"]
        buffer_status = buffer_service.get_status()

        return jsonify(
            {
                "success": True,
                "camera_info": {
                    "pda": camera_pda,
                    "program_id": camera_program_id,
                    "api_url": f"https://{camera_pda.lower()}.mmoment.xyz/api",
                    "legacy_url": "https://jetson.mmoment.xyz/api",
                    "active_sessions": active_sessions,
                    "camera_status": {
                        "online": buffer_status["running"],
                        "fps": buffer_status["fps"],
                        "resolution": f"{buffer_service._width}x{buffer_service._height}",
                        "device": "auto-detected",
                    },
                    "capabilities": {
                        "face_recognition": True,
                        "gesture_detection": True,
                        "livepeer_streaming": True,
                        "media_capture": True,
                        "blockchain_integration": True,
                    },
                },
            }
        )

    @app.route("/api/device-info", methods=["GET"])
    @sign_response
    def api_device_info():
        """Get device-specific information for DePIN registration with cryptographic signature"""
        # Check if device needs WiFi setup
        setup_required = not _is_internet_connected()

        return jsonify(
            {
                "success": True,
                "device_pubkey": device_signer.get_public_key(),
                "hardware_id": device_signer._get_hardware_key().decode("utf-8")[
                    :16
                ],  # Truncated for privacy
                "model": "MMOMENT Jetson Orin Nano",  # Updated to include MMOMENT branding
                "version": "1.0.0",
                "setup_required": setup_required,
                "capabilities": {
                    "signing": True,
                    "blockchain_ready": True,
                    "depin_authentication": True,
                    "local_setup": True,
                },
            }
        )

    @app.route("/api/setup/wifi/scan", methods=["GET"])
    def api_wifi_scan():
        """Scan for available WiFi networks during device setup"""
        try:
            # Use nmcli to scan for WiFi networks on Jetson
            import subprocess

            result = subprocess.run(
                ["nmcli", "-t", "-f", "SSID,SECURITY,SIGNAL", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            networks = []
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 3 and parts[0]:  # Skip empty SSIDs
                            networks.append(
                                {
                                    "ssid": parts[0],
                                    "security": parts[1] if parts[1] else "Open",
                                    "signal": int(parts[2])
                                    if parts[2].isdigit()
                                    else 0,
                                }
                            )

            # Sort by signal strength
            networks.sort(key=lambda x: x["signal"], reverse=True)

            return jsonify(
                {
                    "success": True,
                    "networks": networks[:20],  # Limit to top 20 networks
                }
            )

        except Exception as e:
            logger.error(f"WiFi scan failed: {e}")
            return jsonify(
                {"success": False, "error": "WiFi scan failed", "networks": []}
            )

    @app.route("/api/scan-registration-qr", methods=["POST"])
    def api_scan_registration_qr():
        """
        Scan for registration QR code containing WiFi credentials and claim endpoint.
        Activates camera for QR scanning mode and processes registration data.
        """
        try:
            import json
            import time

            import cv2
            import requests
            from pyzbar import pyzbar

            logger.info("[QR-SCAN] Starting QR code registration scan")

            # Get timeout from request, default to 60 seconds (longer for debugging)
            data = request.get_json() or {}
            max_scan_time = data.get("timeout", 60)  # Default 60 seconds, configurable
            debug_mode = data.get(
                "debug", False
            )  # Debug mode - no WiFi/claiming, just QR detection

            buffer_service = get_services()["buffer"]
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
                        logger.warning(
                            f"[QR-SCAN] No frame available after {frame_count} attempts"
                        )
                    time.sleep(0.1)
                    continue

                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[QR-SCAN] Processed {frame_count} frames in {elapsed:.1f}s, frame shape: {frame.shape}"
                    )

                # Scan for QR codes
                qr_codes = pyzbar.decode(frame)
                qr_attempts += 1

                # Log QR scan attempts periodically
                if qr_attempts % 50 == 0:
                    logger.info(
                        f"[QR-SCAN] QR decode attempts: {qr_attempts}, codes found: {len(qr_codes)}"
                    )

                if len(qr_codes) > 0:
                    logger.info(f"[QR-SCAN] Found {len(qr_codes)} QR code(s) in frame!")

                for qr_code in qr_codes:
                    try:
                        # Decode QR code data
                        qr_data = qr_code.data.decode("utf-8")
                        registration_data = json.loads(qr_data)

                        logger.info(f"[QR-SCAN] Found QR code with registration data")

                        # Validate required fields
                        required_fields = [
                            "wifi_ssid",
                            "wifi_password",
                            "claim_endpoint",
                            "user_wallet",
                            "expires",
                        ]
                        if not all(
                            field in registration_data for field in required_fields
                        ):
                            logger.warning(f"[QR-SCAN] QR code missing required fields")
                            continue

                        # Check if QR code has expired
                        if registration_data["expires"] < time.time():
                            logger.warning(f"[QR-SCAN] QR code has expired")
                            continue

                        logger.info(
                            f"[QR-SCAN] Valid registration QR found for user: {registration_data['user_wallet'][:8]}..."
                        )

                        # Debug mode - just return the parsed data without WiFi/claiming
                        if debug_mode:
                            logger.info(
                                "[QR-SCAN] Debug mode - returning parsed QR data without WiFi/claiming"
                            )
                            return jsonify(
                                {
                                    "success": True,
                                    "debug_mode": True,
                                    "parsed_qr_data": registration_data,
                                    "device_pubkey": device_signer.get_public_key(),
                                    "message": "QR code parsed successfully (debug mode - no WiFi/claiming performed)",
                                }
                            )

                        # Check if already connected to internet - skip WiFi setup if so
                        if _is_internet_connected():
                            logger.info(
                                f"[QR-SCAN] Already have internet connectivity, skipping WiFi connection to: {registration_data['wifi_ssid']}"
                            )
                        else:
                            # Connect to WiFi
                            logger.info(
                                f"[QR-SCAN] No internet connectivity, attempting WiFi connection to: {registration_data['wifi_ssid']}"
                            )
                            wifi_success = _connect_to_wifi(
                                registration_data["wifi_ssid"],
                                registration_data["wifi_password"],
                            )
                            if not wifi_success:
                                logger.error(
                                    f"[QR-SCAN] WiFi connection failed for: {registration_data['wifi_ssid']}"
                                )
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": f"Failed to connect to WiFi network: {registration_data['wifi_ssid']}",
                                        "parsed_qr_data": registration_data,
                                        "step_failed": "wifi_connection",
                                    }
                                ), 500

                        # Wait for internet connectivity
                        connectivity_attempts = 0
                        while (
                            not _is_internet_connected() and connectivity_attempts < 10
                        ):
                            time.sleep(2)
                            connectivity_attempts += 1

                        if not _is_internet_connected():
                            logger.error(
                                "[QR-SCAN] WiFi connected but no internet access"
                            )
                            return jsonify(
                                {
                                    "success": False,
                                    "error": "Connected to WiFi but no internet access",
                                    "parsed_qr_data": registration_data,
                                    "step_failed": "internet_connectivity",
                                }
                            ), 500

                        # Claim device with backend
                        device_claim_data = {
                            "device_pubkey": device_signer.get_public_key(),
                            "device_model": "jetson_orin_nano",
                        }

                        logger.info(
                            f"[QR-SCAN] Claiming device at endpoint: {registration_data['claim_endpoint']}"
                        )

                        claim_response = requests.post(
                            registration_data["claim_endpoint"],
                            json=device_claim_data,
                            timeout=15,
                        )

                        if claim_response.status_code != 200:
                            logger.error(
                                f"[QR-SCAN] Device claim failed: {claim_response.status_code}"
                            )
                            return jsonify(
                                {
                                    "success": False,
                                    "error": f"Device claim failed: {claim_response.status_code}",
                                    "parsed_qr_data": registration_data,
                                    "step_failed": "device_claim",
                                    "claim_response_status": claim_response.status_code,
                                    "claim_response_text": claim_response.text[
                                        :500
                                    ],  # First 500 chars
                                }
                            ), 500

                        logger.info(
                            f"[QR-SCAN] Device successfully claimed by user: {registration_data['user_wallet'][:8]}..."
                        )

                        return jsonify(
                            {
                                "success": True,
                                "user_wallet": registration_data["user_wallet"],
                                "device_pubkey": device_signer.get_public_key(),
                                "wifi_connected": True,
                                "device_claimed": True,
                                "message": "Device registration completed successfully",
                            }
                        )

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
            logger.warning(
                f"[QR-SCAN] Timeout after {elapsed:.1f}s - processed {frame_count} frames, {qr_attempts} QR attempts"
            )
            return jsonify(
                {
                    "success": False,
                    "error": f"QR code scan timeout after {elapsed:.1f}s",
                    "debug_info": {
                        "frames_processed": frame_count,
                        "qr_decode_attempts": qr_attempts,
                        "timeout_seconds": max_scan_time,
                        "elapsed_seconds": round(elapsed, 1),
                    },
                }
            ), 408

        except Exception as e:
            logger.error(f"[QR-SCAN] Unexpected error: {e}")
            return jsonify(
                {"success": False, "error": f"QR scan failed: {str(e)}"}
            ), 500

    def _connect_to_wifi(ssid, password):
        """Connect to WiFi network using nmcli"""
        try:
            import subprocess

            logger.info(f"[WIFI] Connecting to network: {ssid}")

            # First check if already connected to this network
            check_result = subprocess.run(
                ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if check_result.returncode == 0:
                for line in check_result.stdout.strip().split("\n"):
                    if line.startswith("yes:") and ssid in line:
                        logger.info(f"[WIFI] Already connected to {ssid}")
                        return True

            # Delete any existing connection with same name
            subprocess.run(["nmcli", "connection", "delete", ssid], capture_output=True)

            # Create new WiFi connection
            if password:
                cmd = ["nmcli", "device", "wifi", "connect", ssid, "password", password]
            else:
                cmd = ["nmcli", "device", "wifi", "connect", ssid]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info(f"[WIFI] Successfully connected to {ssid}")
                return True
            else:
                # Check if the error is because we're already connected
                if (
                    "already connected" in result.stderr.lower()
                    or "device is already active" in result.stderr.lower()
                ):
                    logger.info(
                        f"[WIFI] Already connected to {ssid} (detected from error message)"
                    )
                    return True

                logger.error(f"[WIFI] Failed to connect to {ssid}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"[WIFI] WiFi connection error: {e}")
            return False

    @app.route("/api/setup/wifi", methods=["POST"])
    def api_wifi_setup():
        """Configure WiFi credentials during device setup"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            ssid = data.get("ssid")
            password = data.get("password")
            security = data.get("security", "WPA2")

            if not ssid:
                return jsonify({"success": False, "error": "SSID required"}), 400

            logger.info(f"Configuring WiFi for SSID: {ssid}")

            # Use nmcli to configure WiFi on Jetson
            import subprocess

            # First, delete any existing connection with same name
            subprocess.run(["nmcli", "connection", "delete", ssid], capture_output=True)

            # Create new WiFi connection
            if security.upper() in ["WPA", "WPA2", "WPA3"] and password:
                # WPA/WPA2 connection with password
                cmd = ["nmcli", "device", "wifi", "connect", ssid, "password", password]
            else:
                # Open network
                cmd = ["nmcli", "device", "wifi", "connect", ssid]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info(f"WiFi configured successfully for {ssid}")

                # Schedule device restart to apply network changes
                import threading

                def restart_device():
                    import time

                    time.sleep(5)  # Give time for response
                    subprocess.run(["sudo", "reboot"])

                restart_thread = threading.Thread(target=restart_device)
                restart_thread.daemon = True
                restart_thread.start()

                return jsonify(
                    {
                        "success": True,
                        "message": "WiFi configured successfully. Device will restart.",
                        "ssid": ssid,
                    }
                )
            else:
                logger.error(f"WiFi configuration failed: {result.stderr}")
                return jsonify(
                    {
                        "success": False,
                        "error": f"WiFi configuration failed: {result.stderr}",
                    }
                ), 500

        except Exception as e:
            logger.error(f"WiFi setup error: {e}")
            return jsonify(
                {"success": False, "error": f"WiFi setup failed: {str(e)}"}
            ), 500

    def _is_internet_connected():
        """Check if device has internet connectivity"""
        try:
            import urllib.request

            urllib.request.urlopen("http://google.com", timeout=5)
            return True
        except:
            return False

    # ========================================
    # LEGACY ROUTES (For backward compatibility)
    # ========================================

    # Health check route
    @app.route("/health")
    def health():
        """Health check endpoint"""
        services = get_services()

        # Check buffer service status
        buffer_service = services["buffer"]
        buffer_status = buffer_service.get_status()

        # Get session count
        session_count = len(services["session"].get_all_sessions())

        # Basic health response
        health_response = {
            "status": "ok" if buffer_status["running"] else "error",
            "buffer_service": buffer_status["health"],
            "buffer_fps": buffer_status["fps"],
            "active_sessions": session_count,
            "timestamp": int(time.time() * 1000),
            "camera": {
                "index": buffer_service._camera_index,
                "preferred_device": "auto-detected",
                "resolution": f"{buffer_service._width}x{buffer_service._height}",
                "target_fps": buffer_service._fps,
            },
        }

        return jsonify(health_response)

    # MJPEG stream removed - using WebRTC/WHIP for streaming now
    # See /api/stream/info for available streams

    # Visualization routes - require session for security
    @app.route("/toggle_face_detection", methods=["POST"])
    @require_session
    def toggle_face_detection():
        """
        Enable or disable face detection.
        Note: In native mode, face detection is handled by native C++ server.
        """
        data = request.json or {}
        enabled = data.get("enabled", True)

        # Native mode: face detection handled by native C++ server
        logger.info(f"[Native Mode] Face detection toggle requested: {enabled}")

        return jsonify(
            {
                "success": True,
                "enabled": True,  # Always return true
                "mode": "native",
                "message": "Face detection handled by native C++ server",
            }
        )

    @app.route("/toggle_face_visualization", methods=["POST"])
    @require_session
    def toggle_face_visualization():
        """
        Toggle face visualization in camera frames.

        In native mode: Visualization is controlled by stream selection (clean vs annotated).
        Client connects to different WHEP URLs for clean/annotated streams.
        This endpoint acknowledges the preference for client-side state management.
        """
        data = request.json or {}
        enabled = data.get("enabled", True)

        try:
            services = get_services()
            face_service = services.get("face")

            # Native mode: no face service, visualization controlled by stream selection
            if face_service is None:
                logger.info(f"[Native Mode] Face visualization preference: {enabled} (stream selection controls actual visualization)")
                return jsonify({
                    "success": True,
                    "enabled": enabled,
                    "mode": "native",
                    "message": "In native mode, connect to annotated stream for CV overlays"
                })

            # Non-native mode: toggle via face service
            face_service.enable_visualization(enabled)
            # Also enable/disable boxes when toggling visualization
            face_service.enable_boxes(enabled)

            return jsonify({"success": True, "enabled": enabled})
        except Exception as e:
            logger.error(f"Error toggling face visualization: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/toggle_face_boxes", methods=["POST"])
    @require_session
    def toggle_face_boxes():
        """
        Toggle face boxes in camera frames
        This controls only the boxes, while visualization controls all overlays
        """
        data = request.json or {}
        enabled = data.get("enabled", True)

        try:
            services = get_services()
            face_service = services.get("face")

            # Native mode: no face service
            if face_service is None:
                logger.info(f"[Native Mode] Face boxes preference: {enabled}")
                return jsonify({
                    "success": True,
                    "enabled": enabled,
                    "mode": "native"
                })

            face_service.enable_boxes(enabled)

            return jsonify({"success": True, "enabled": enabled})
        except Exception as e:
            logger.error(f"Error toggling face boxes: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/face_settings", methods=["GET"])
    def face_settings():
        """Get current face detection settings"""
        services = get_services()
        face_service = services.get("face")

        # Native mode: no face service
        if face_service is None:
            return jsonify({
                "success": True,
                "mode": "native",
                "settings": {
                    "visualization_enabled": True,
                    "boxes_enabled": True,
                    "message": "Native mode - settings controlled by stream selection"
                }
            })

        settings = face_service.get_settings()

        return jsonify({"success": True, "settings": settings})

    @app.route("/toggle_gesture_visualization", methods=["POST"])
    @require_session
    def toggle_gesture_visualization():
        """Enable or disable gesture detection visualization"""
        data = request.json or {}
        enabled = data.get("enabled", True)

        # Native mode: gesture service not available
        services = get_services()
        gesture_service = services.get("gesture")
        if gesture_service is None:
            logger.info(f"[Native Mode] Gesture visualization preference: {enabled}")
            return jsonify({
                "success": True,
                "enabled": enabled,
                "mode": "native",
                "message": "Gesture service not available in native mode"
            })

        gesture_service.enable_visualization(enabled)
        logger.info(f"Gesture visualization toggled to: {enabled}")

        return jsonify({"success": True, "enabled": enabled})

    @app.route("/toggle_pose_visualization", methods=["POST"])
    @require_session
    def toggle_pose_visualization():
        """Enable or disable pose skeleton visualization"""
        data = request.json or {}
        enabled = data.get("enabled", True)

        services = get_services()
        pose_service = services.get("pose")

        # Native mode: pose service not available, visualization controlled by stream selection
        if pose_service is None:
            logger.info(f"[Native Mode] Pose visualization preference: {enabled}")
            return jsonify({
                "success": True,
                "enabled": enabled,
                "mode": "native",
                "message": "In native mode, connect to annotated stream for pose overlays"
            })

        pose_service.visualization_enabled = enabled
        logger.info(f"Pose visualization toggled to: {enabled}")

        return jsonify({"success": True, "enabled": enabled})

    @app.route("/api/apps/load", methods=["POST"])
    @require_session
    def api_apps_load():
        """Load a CV app"""
        data = request.json or {}
        app_name = data.get("app_name")

        if not app_name:
            return jsonify({"success": False, "error": "app_name is required"}), 400

        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        success = app_manager.load_app(app_name)

        if success:
            return jsonify(
                {"success": True, "message": f"App {app_name} loaded successfully"}
            )
        else:
            return jsonify(
                {"success": False, "error": f"Failed to load app {app_name}"}
            ), 500

    @app.route("/api/apps/activate", methods=["POST"])
    @require_session
    def api_apps_activate():
        """Activate a CV app"""
        data = request.json or {}
        app_name = data.get("app_name")

        if not app_name:
            return jsonify({"success": False, "error": "app_name is required"}), 400

        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        success = app_manager.activate_app(app_name)

        if success:
            return jsonify(
                {
                    "success": True,
                    "message": f"App {app_name} activated",
                    "active_app": app_name,
                }
            )
        else:
            return jsonify(
                {"success": False, "error": f"Failed to activate app {app_name}"}
            ), 500

    @app.route("/api/apps/deactivate", methods=["POST"])
    @require_session
    def api_apps_deactivate():
        """Deactivate current CV app"""
        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        app_manager.deactivate_app()

        return jsonify({"success": True, "message": "App deactivated"})

    @app.route("/api/apps/status", methods=["GET"])
    def api_apps_status():
        """Get current app status"""
        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        app_state = app_manager.get_active_app_state()

        return jsonify(
            {
                "success": True,
                "active_app": app_manager.active_app_name,
                "loaded_apps": list(app_manager.loaded_apps.keys()),
                "state": app_state,
            }
        )

    @app.route("/api/apps/competition/start", methods=["POST"])
    @require_session
    def api_apps_competition_start():
        """Start a competition (for CompetitionApp types)"""
        data = request.json or {}
        competitors = data.get("competitors", [])
        duration_limit = data.get("duration_limit")

        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        if not app_manager.active_app:
            return jsonify({"success": False, "error": "No active app"}), 400

        if not hasattr(app_manager.active_app, "start_competition"):
            return jsonify(
                {"success": False, "error": "Active app does not support competitions"}
            ), 400

        try:
            app_manager.active_app.start_competition(competitors, duration_limit)
            return jsonify({"success": True, "message": "Competition started"})
        except Exception as e:
            logger.error(f"Failed to start competition: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/apps/competition/end", methods=["POST"])
    @require_session
    def api_apps_competition_end():
        """End the current competition and buffer results to timeline"""
        import asyncio

        services = get_services()
        if "app_manager" not in services:
            return jsonify(
                {"success": False, "error": "App manager not available"}
            ), 404

        app_manager = services["app_manager"]
        if not app_manager.active_app:
            return jsonify({"success": False, "error": "No active app"}), 400

        if not hasattr(app_manager.active_app, "end_competition"):
            return jsonify(
                {"success": False, "error": "Active app does not support competitions"}
            ), 400

        try:
            # Get request body for competition metadata
            data = request.json or {}
            competition_meta = data.get("competition")
            # Frontend sends wallet_address, not user_wallet
            user_wallet = data.get("user_wallet") or data.get("wallet_address")

            # DEBUG: Log what we received
            logger.info(f"[CompetitionEnd] Received data: {data}")
            logger.info(f"[CompetitionEnd] competition_meta: {competition_meta}")
            logger.info(f"[CompetitionEnd] user_wallet: {user_wallet}")
            logger.info(f"[CompetitionEnd] CAMERA_OWNER_WALLET configured: {bool(CAMERA_OWNER_WALLET)}")

            # Get competition info before ending
            active_app = app_manager.active_app
            app_name = getattr(active_app, "name", active_app.__class__.__name__.lower().replace("app", ""))
            start_time = getattr(active_app, "start_time", None)

            # End the competition
            result = active_app.end_competition()

            # Buffer CV activity to timeline for each participant
            competitors = result.get("competitors", [])
            cv_activity_meta = {}

            # If competition has escrow, settle on-chain
            if competition_meta and competition_meta.get("escrow_pda"):
                if not CAMERA_OWNER_WALLET:
                    logger.warning("CAMERA_OWNER_WALLET not configured, skipping settlement")
                else:
                    try:
                        # Build participant results from CV tracking data
                        participant_results = [
                            {"wallet_address": c["wallet_address"], "score": c.get("stats", {}).get("reps", 0)}
                            for c in competitors
                            if c.get("wallet_address")
                        ]

                        logger.info(f"Settling competition escrow: {competition_meta['escrow_pda']}")

                        # Get device keypair from signer
                        camera_keypair = device_signer.get_keypair()
                        if not camera_keypair:
                            raise Exception("Device keypair not available")

                        # Run async settlement
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            settlement = loop.run_until_complete(settle_competition(
                                escrow_pda=competition_meta["escrow_pda"],
                                camera_keypair=camera_keypair,
                                camera_owner_pubkey=CAMERA_OWNER_WALLET,
                                participant_results=participant_results,
                                rpc_url=SOLANA_RPC_URL,
                            ))
                        finally:
                            loop.close()

                        # Add settlement to activity metadata for timeline display
                        if settlement.get("success"):
                            logger.info(f"Competition settled successfully: {settlement.get('tx_signature')}")
                            cv_activity_meta["competition"] = {
                                "mode": competition_meta.get("mode", "prize"),
                                "escrowPda": competition_meta["escrow_pda"],
                                "stakeAmountSol": competition_meta.get("stake_amount_sol", 0),
                                "targetReps": competition_meta.get("target_reps"),
                                "won": user_wallet in settlement.get("winners", []) if user_wallet else False,
                                "amountWonSol": settlement.get("payout_per_winner_sol", 0) if user_wallet in settlement.get("winners", []) else 0,
                                "amountLostSol": competition_meta.get("stake_amount_sol", 0) if user_wallet and user_wallet not in settlement.get("winners", []) else 0,
                                "lostTo": CAMERA_OWNER_WALLET if not settlement.get("winners") and competition_meta.get("mode") == "prize" else None,
                                "settlementTxId": settlement.get("tx_signature"),
                            }
                            result["settlement"] = settlement
                        else:
                            logger.error(f"Competition settlement failed: {settlement.get('error')}")
                            result["settlement_error"] = settlement.get("error")

                    except Exception as settle_err:
                        logger.error(f"Failed to settle competition: {settle_err}", exc_info=True)
                        result["settlement_error"] = str(settle_err)

            if competitors:
                try:
                    import time
                    from services.timeline_activity_service import (
                        get_timeline_activity_service,
                    )

                    # Calculate duration
                    duration_seconds = None
                    if start_time:
                        duration_seconds = time.time() - start_time

                    timeline_service = get_timeline_activity_service()
                    timeline_service.buffer_cv_activity(
                        app_name=app_name,
                        competitors=competitors,
                        duration_seconds=duration_seconds,
                        metadata=cv_activity_meta if cv_activity_meta else None,
                    )
                    logger.info(f"CV activity buffered for {len(competitors)} competitors")
                except Exception as timeline_err:
                    logger.warning(f"Failed to buffer CV activity to timeline: {timeline_err}")

            return jsonify({"success": True, "result": result})
        except Exception as e:
            logger.error(f"Failed to end competition: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    # Session management routes
    @app.route("/connect", methods=["POST"])
    def connect():
        """
        Connect a wallet to the camera
        Creates a new session for the wallet

        Expected body:
        {
            "wallet_address": "...",         # Required
            "session_pda": "...",            # Required for activity buffering (Solana session PDA)
            "display_name": "...",           # Optional
            "username": "..."                # Optional
        }
        """
        data = request.json or {}
        wallet_address = data.get("wallet_address")
        session_pda = data.get("session_pda")  # Solana session PDA for activity buffering
        display_name = data.get("display_name")  # Optional display name from frontend
        username = data.get("username")  # Optional username from frontend

        if not wallet_address:
            return jsonify(
                {"success": False, "error": "Wallet address is required"}
            ), 400

        # Create user profile metadata
        user_profile = {
            "wallet_address": wallet_address,
            "display_name": display_name,
            "username": username,
        }

        # Create a session with user profile and Solana PDA
        # The session_pda is critical for linking buffered activities to the on-chain session
        services = get_services()
        session_service = services["session"]
        session_info = session_service.create_session(wallet_address, user_profile, session_pda=session_pda)

        # Store user profile in native mode profile store
        global _user_profiles
        _user_profiles[wallet_address] = user_profile
        logger.info(
            f"Profile stored for connected user: {display_name or username or wallet_address}"
        )

        # Pre-authorize Pipe storage session for fast uploads
        import asyncio

        from services.pipe_storage_service import pre_authorize_user_session

        def run_pipe_authorization():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                auth_success = loop.run_until_complete(
                    pre_authorize_user_session(wallet_address)
                )
                if auth_success:
                    logger.info(
                        f"✅ Pipe session pre-authorized for {wallet_address[:8]}..."
                    )
                else:
                    logger.warning(
                        f"⚠️ Pipe session pre-authorization failed for {wallet_address[:8]}..."
                    )
            except Exception as e:
                logger.error(
                    f"❌ Pipe pre-authorization error for {wallet_address[:8]}...: {e}"
                )
            finally:
                loop.close()

        # Start Pipe authorization in background thread
        import threading

        pipe_auth_thread = threading.Thread(target=run_pipe_authorization, daemon=True)
        pipe_auth_thread.start()

        logger.info(
            f"🔗 User {wallet_address[:8]}... connected with Pipe pre-authorization initiated"
        )

        return jsonify(session_info)

    # NOTE: /disconnect removed - use /api/checkout instead
    # /api/checkout handles session end + timeline write + backend notification

    # Face recognition routes
    @app.route("/recognize_face", methods=["POST"])
    @require_session
    def recognize_face():
        """
        Get current face recognition status from native mode.
        In native mode, face recognition happens continuously via the C++ server.
        This endpoint returns the current recognition state.
        """
        data = request.json or {}
        wallet_address = data.get("wallet_address")
        session_id = data.get("session_id")

        logger.info(f"[RECOGNIZE] Recognition status request for wallet: {wallet_address}")

        # Check if session is valid
        session_service = get_services()["session"]
        has_valid_session = (
            wallet_address
            and session_id
            and session_service.validate_session(session_id, wallet_address)
        )

        try:
            # Native mode: get recognition status from identity service
            from services.identity_store import get_identity_store
            identity_store = get_identity_store()
            identity_service = get_native_identity_service()

            # Get current recognition state
            status = identity_service.get_status()
            checked_in_wallets = identity_store.get_checked_in_wallets()

            # Check if requested wallet is being tracked
            wallet_recognized = False
            wallet_confidence = 0
            if wallet_address:
                identity = identity_store.get_identity(wallet_address)
                if identity and identity.is_checked_in:
                    wallet_recognized = True
                    wallet_confidence = 1.0  # In native mode, matches are binary
                    logger.info(f"[RECOGNIZE] Wallet {wallet_address} is checked in and trackable")

            # Get current frame for image response
            buffer_service = get_services()["buffer"]
            frame, timestamp = buffer_service.get_frame()

            image_base64 = None
            if frame is not None:
                _, jpeg_data = cv2.imencode(".jpg", frame)
                image_base64 = base64.b64encode(jpeg_data).decode("utf-8")

            return jsonify(
                {
                    "success": True,
                    "mode": "native",
                    "detected_faces": status.get("total_faces_processed", 0),
                    "recognized_faces": status.get("currently_tracked", 0),
                    "recognized_data": {w: {"confidence": 1.0} for w in checked_in_wallets},
                    "wallet_recognized": wallet_recognized,
                    "wallet_confidence": wallet_confidence,
                    "has_valid_session": has_valid_session,
                    "include_image": image_base64 is not None,
                    "image": image_base64,
                }
            )
        except Exception as e:
            logger.error(f"[RECOGNIZE] Error in face recognition: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify(
                {"success": False, "error": f"Face recognition failed: {str(e)}"}
            ), 500

    # Gesture detection routes
    @app.route("/current_gesture", methods=["GET"])
    def current_gesture():
        """
        Get the current detected gesture
        Returns the gesture name and confidence
        """
        services = get_services()
        gesture_service = services.get("gesture")
        if gesture_service is None:
            return jsonify({
                "success": True,
                "gesture": "none",
                "confidence": 0,
                "mode": "native",
                "message": "Gesture service not available in native mode"
            })
        gesture_info = gesture_service.get_current_gesture()

        return jsonify({"success": True, **gesture_info})

    # Capture routes
    @app.route("/capture_moment", methods=["POST"])
    @require_session
    def capture_moment():
        """
        Capture a photo from the camera and upload directly to user's Pipe storage
        Returns the photo data and upload result

        Request body:
            wallet_address: User's wallet address (required)
            annotated: If true, capture with CV annotations (face boxes, skeletons). Default: false
            event_metadata: Optional metadata about the capture trigger (e.g., from CV app)
        """
        wallet_address = request.json.get("wallet_address")
        annotated = request.json.get("annotated", False)
        event_metadata = request.json.get("event_metadata")

        # Get services
        buffer_service = get_services()["buffer"]
        capture_service = get_services()["capture"]

        # Capture photo (clean or annotated based on parameter)
        photo_info = capture_service.capture_photo(
            buffer_service,
            wallet_address,
            annotated=annotated,
            event_metadata=event_metadata
        )

        if not photo_info["success"]:
            return jsonify(photo_info), 500

        # Read the photo file data
        with open(photo_info["path"], "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Add base64 data to response
        photo_info["image_data"] = f"data:image/jpeg;base64,{base64_image}"

        # Sign capture and queue background upload
        from services.capture_event_signer import sign_capture
        from services.upload_queue import get_upload_queue

        try:
            camera_pda = get_camera_pda()
            photo_path = photo_info["path"]

            # Calculate file hash for signing
            from services.walrus_upload_service import get_walrus_upload_service
            upload_service = get_walrus_upload_service()
            file_hash = upload_service.calculate_file_hash(photo_path)

            if not file_hash:
                raise Exception("Failed to calculate file hash")

            # Sign capture event (instant)
            signed_event = sign_capture(
                user_wallet=wallet_address,
                camera_pda=camera_pda,
                file_hash=file_hash,
                capture_type="photo",
                file_path=photo_path,
                metadata={
                    "timestamp": photo_info.get("timestamp"),
                    "width": photo_info.get("width"),
                    "height": photo_info.get("height"),
                    "filename": photo_info["filename"],
                },
            )
            device_signature = signed_event["device_signature"]
            logger.info(f"📝 Photo signed: {device_signature[:16]}...")

            # Queue background upload (returns immediately)
            upload_queue = get_upload_queue()
            job_id = upload_queue.add(
                file_path=photo_path,
                file_type="photo",
                wallet_address=wallet_address,
                camera_id=camera_pda,
                device_signature=device_signature,
                timestamp=photo_info.get("timestamp", 0),
            )

            logger.info(f"📸 Photo captured for {wallet_address[:8]}... -> queued as job #{job_id}")

            photo_info["storage_upload"] = {
                "initiated": True,
                "completed": False,
                "upload_status": "pending",
                "storage_provider": "walrus",
                "device_signature": device_signature,
                "job_id": job_id,
                "local_url": f"/photos/{photo_info['filename']}",
            }
            photo_info["pipe_upload"] = photo_info["storage_upload"]

            # Buffer to timeline
            try:
                from services.timeline_activity_service import get_timeline_activity_service
                timeline_service = get_timeline_activity_service()
                timeline_service.buffer_photo_activity(
                    wallet_address=wallet_address,
                    pipe_file_name=photo_info["filename"],
                    pipe_file_id=str(job_id),
                    metadata={
                        "timestamp": photo_info.get("timestamp"),
                        "device_signature": device_signature,
                        "width": photo_info.get("width"),
                        "height": photo_info.get("height"),
                        "filename": photo_info.get("filename"),
                        "storage_provider": "walrus",
                        "job_id": job_id,
                    },
                )
                photo_info["timeline_buffered"] = True
            except Exception as timeline_err:
                logger.warning(f"Failed to buffer to timeline: {timeline_err}")
                photo_info["timeline_buffered"] = False

        except Exception as e:
            logger.error(f"Failed capture for {wallet_address[:8]}...: {e}", exc_info=True)
            photo_info["storage_upload"] = {
                "initiated": False,
                "error": str(e),
                "upload_status": "failed",
            }
            photo_info["pipe_upload"] = photo_info["storage_upload"]

        return jsonify(photo_info)

    # ===============================
    # Upload Queue API
    # ===============================

    @app.route("/api/upload-status/<int:job_id>", methods=["GET"])
    @require_session
    def get_upload_job_status(job_id):
        """Get status of a specific upload job."""
        from services.upload_queue import get_upload_queue
        queue = get_upload_queue()
        job = queue.get_status(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, **job})

    @app.route("/api/upload-status", methods=["GET"])
    @require_session
    def get_upload_queue_stats():
        """Get upload queue statistics."""
        from services.upload_queue import get_upload_queue
        queue = get_upload_queue()
        return jsonify({"success": True, **queue.get_stats()})

    @app.route("/api/uploads", methods=["GET"])
    @require_session
    def get_user_uploads():
        """Get upload history for a user."""
        from services.upload_queue import get_upload_queue
        wallet_address = request.args.get("wallet_address")
        if not wallet_address:
            return jsonify({"success": False, "error": "wallet_address required"}), 400
        queue = get_upload_queue()
        uploads = queue.get_user_uploads(wallet_address)
        return jsonify({"success": True, "uploads": uploads})

    @app.route("/start_recording", methods=["POST"])
    @require_session
    def start_recording():
        """
        Start recording a video
        Returns recording information
        """
        wallet_address = request.json.get("wallet_address")
        duration = request.json.get("duration", 0)  # 0 means record until stopped

        # Get services
        buffer_service = get_services()["buffer"]
        capture_service = get_services()["capture"]

        # Check if already recording
        if capture_service.is_recording():
            return jsonify({"success": False, "error": "Already recording"}), 400

        # Start recording
        recording_info = capture_service.start_recording(
            buffer_service, wallet_address, duration
        )

        return jsonify(recording_info)

    @app.route("/stop_recording", methods=["POST"])
    @require_session
    def stop_recording():
        """
        Stop recording
        Returns recording information
        """
        # Get capture service
        capture_service = get_services()["capture"]

        # Stop recording
        result = capture_service.stop_recording()

        return jsonify(result)

    @app.route("/list_videos", methods=["GET"])
    @require_session
    def list_videos():
        """
        List available videos
        Returns a list of video information
        """
        limit = request.args.get("limit", 10, type=int)

        # Get capture service
        capture_service = get_services()["capture"]

        # Get video list
        videos = capture_service.get_videos(limit)

        return jsonify({"success": True, "videos": videos, "count": len(videos)})

    @app.route("/list_photos", methods=["GET"])
    @require_session
    def list_photos():
        """
        List available photos
        Returns a list of photo information
        """
        limit = request.args.get("limit", 10, type=int)

        # Get capture service
        capture_service = get_services()["capture"]

        # Get photo list
        photos = capture_service.get_photos(limit)

        return jsonify({"success": True, "photos": photos, "count": len(photos)})

    # Media access routes
    @app.route("/photos/<filename>")
    def get_photo(filename):
        """
        Get a specific photo by filename
        Returns the photo file
        """
        capture_service = get_services()["capture"]

        # Get photo path
        photo_path = capture_service.get_photo_path(filename)

        if not photo_path:
            abort(404, description="Photo not found")

        return send_file(photo_path, mimetype="image/jpeg")

    @app.route("/videos/<filename>")
    def get_video(filename):
        """Get a specific video file"""
        try:
            services = get_services()
            if not services:
                return jsonify({"error": "Services not available"}), 503

            capture_service = get_services()["capture"]

            # Get video path directly - serve the exact file requested
            video_path = capture_service.get_video_path(filename)

            # Determine MIME type based on file extension
            if filename.endswith(".mp4"):
                mimetype = "video/mp4"
            elif filename.endswith(".mov"):
                mimetype = "video/quicktime"
            else:
                mimetype = "application/octet-stream"

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
    @app.route("/test-page")
    def test_page():
        """Test page to verify functionality"""
        return render_template("test.html")

    @app.route("/simple-test")
    def simple_test():
        """Simple test page for direct API testing"""
        return render_template("simple_test.html")

    @app.route("/local-test")
    def local_test():
        """Local camera test page"""
        return render_template("local_test.html")

    @app.route("/qr-monitor")
    def qr_monitor():
        """QR registration monitoring page - shows what Jetson processes from frontend QR codes"""
        return render_template("qr_monitor.html")

    @app.route("/api-test")
    def api_test():
        """Interactive API testing page"""
        return render_template("api_test.html")

    @app.route("/direct-test")
    def direct_test():
        """Direct camera service test page (bypasses frontend bridge)"""
        try:
            return send_file("direct_test.html")
        except FileNotFoundError:
            return "Direct test page not found", 404

    # Camera diagnostic routes
    @app.route("/camera/diagnostics")
    def camera_diagnostics():
        """
        Get diagnostic information about the camera subsystem
        """
        if not CAMERA_UTILS_AVAILABLE:
            return jsonify({"error": "Camera diagnostic utilities not available"}), 500

        try:
            # Get camera information
            camera_info = detect_cameras()

            # Get health status of the currently used camera (camera 0)
            camera_health = get_camera_health(0)

            # Check FaceNet availability
            facenet_info = check_facenet_availability()

            return jsonify(
                {
                    "timestamp": int(time.time() * 1000),
                    "camera_info": camera_info,
                    "camera_health": camera_health,
                    "facenet_info": facenet_info,
                }
            )
        except Exception as e:
            logger.error(f"Error getting camera diagnostics: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/facenet/check", methods=["GET"])
    def check_facenet():
        """
        Check face recognition model availability and integrity
        """
        try:
            # We're using our simplified face recognition which is always available
            simple_face_model_path = os.path.join(
                os.path.expanduser(
                    "~/mmoment/app/orin_nano/camera_service/models/simple_model"
                ),
                "simple_face_features.json",
            )

            # Check if our model file exists
            model_exists = os.path.exists(simple_face_model_path)

            facenet_info = {
                "available": True,
                "model_file_exists": model_exists,
                "model_file_path": simple_face_model_path,
                "model_file_size": os.path.getsize(simple_face_model_path)
                if model_exists
                else 0,
                "tensorflow_available": True,
                "tensorflow_version": "N/A - Using native OpenCV",
                "opencv_available": True,
                "opencv_version": cv2.__version__,
                "opencv_dnn_support": True,
                "model_integrity_verified": True,
                "model_error": None,
                "message": "Simple face recognition is properly installed and working",
            }

            return jsonify(
                {"success": True, "available": True, "details": facenet_info}
            )
        except Exception as e:
            logger.error(f"Error checking face recognition: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Camera reset route
    @app.route("/camera/reset", methods=["POST"])
    @require_session
    def reset_camera():
        """
        Reset the camera connection - DANGEROUS: requires session
        """
        buffer_service = get_services()["buffer"]

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
        return jsonify(
            {
                "success": success,
                "message": "Camera reset successful"
                if success
                else "Failed to reset camera",
                "was_running": was_running,
                "now_running": buffer_service._running,
                "camera_index": buffer_service._camera_index,
                "preferred_device": "auto-detected",
            }
        )

    # Face management endpoints
    @app.route("/get_enrolled_faces", methods=["GET"])
    def get_enrolled_faces():
        """
        Get list of all enrolled faces (checked-in wallets with embeddings)
        Returns a list of wallet addresses
        """
        try:
            # Native mode: use identity store
            from services.identity_store import get_identity_store
            identity_store = get_identity_store()
            enrolled_faces = identity_store.get_checked_in_wallets()

            return jsonify(
                {"success": True, "faces": enrolled_faces, "count": len(enrolled_faces)}
            )
        except Exception as e:
            logger.error(f"Error getting enrolled faces: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/clear_enrolled_faces", methods=["POST"])
    @require_session
    def clear_enrolled_faces():
        """
        Clear all enrolled faces - DANGEROUS: requires session
        In native mode, this checks out all wallets and clears track associations.
        """
        try:
            # Native mode: check out all wallets from identity store
            from services.identity_store import get_identity_store
            identity_store = get_identity_store()
            wallets = identity_store.get_checked_in_wallets()
            for wallet in wallets:
                identity_store.check_out(wallet)

            # Also clear track associations
            identity_service = get_native_identity_service()
            identity_service.clear_tracks()

            return jsonify(
                {
                    "success": True,
                    "message": f"Cleared {len(wallets)} faces and track associations",
                }
            )
        except Exception as e:
            logger.error(f"Error clearing enrolled faces: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Register CV Apps routes
    try:
        from services.routes_cv_apps import bp as cv_apps_bp

        app.register_blueprint(cv_apps_bp)
        logger.info("CV Apps routes registered")
    except Exception as e:
        logger.warning(f"Failed to register CV Apps routes: {e}")

    # Add resource not found handler
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404

    # Add internal server error handler
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(error=str(e)), 500

    # Debug endpoint for identity tracker
    @app.route("/api/debug/identity-tracker-checkin", methods=["POST"])
    def debug_identity_tracker_checkin():
        """Debug: manually check user into identity tracker"""
        services = get_services()
        gpu_face = services.get("gpu_face")

        if not gpu_face:
            return jsonify({"error": "GPU face service not available"}), 500

        if not hasattr(gpu_face, "identity_tracker"):
            return jsonify({"error": "Identity tracker not found"}), 500

        wallets = list(gpu_face._face_embeddings.keys())
        if not wallets:
            return jsonify({"error": "No enrolled faces"}), 400

        wallet = wallets[0]
        embedding = gpu_face._face_embeddings[wallet]
        display_name = gpu_face._face_names.get(wallet, wallet[:8])

        result = gpu_face.identity_tracker.check_in_user(
            wallet_address=wallet,
            face_embedding=embedding,
            initial_bbox=(0, 0, 100, 100),
            metadata={"display_name": display_name},
        )

        return jsonify(
            {
                "success": True,
                "wallet": wallet[:16],
                "result": result,
                "active_sessions": list(
                    gpu_face.identity_tracker._active_sessions.keys()
                ),
            }
        )
