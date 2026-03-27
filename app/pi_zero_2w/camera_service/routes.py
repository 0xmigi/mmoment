"""
Routes — Pi Zero 2W Camera Service

Implements the same API surface the backend expects from any MMOMENT camera.
No CV/ML routes. Simpler session management. Same ed25519 security model.

Backend → device calls:
  POST /api/capture          → { filename, timestamp, width, height }
  GET  /api/photos/<name>    → JPEG bytes
  POST /api/record           → { success, filename, duration }
  GET  /api/videos/<name>    → video bytes
  GET  /api/camera/info      → capabilities
  GET  /api/health           → { status }
  GET  /api/device-info      → { device_pubkey, setup_required, ... }
  POST /api/checkin          → { success, session_id }
  POST /api/checkout         → { success }
  POST /api/scan-registration-qr → setup flow
  POST /api/device/claim     → claim flow
"""

import hashlib
import io
import logging
import os
import time
import threading
from typing import Optional

import cv2
import requests as req
from flask import Flask, jsonify, request, send_file

from .config.settings import Settings
from .services.buffer_service import get_buffer_service
from .services.capture_event_signer import sign_capture
from .services.device_registration import get_registration_service
from .services.device_signer import get_device_signer
from .services.whip_publisher import get_whip_publisher

logger = logging.getLogger(__name__)

# In-memory session store: wallet_address → { session_id, checked_in_at, display_name }
_sessions: dict = {}
_sessions_lock = threading.Lock()

# In-memory recording state
_recording_state: dict = {"active": False, "start_time": None, "wallet": None, "duration": None}
_recording_lock = threading.Lock()


def _get_camera_pda() -> Optional[str]:
    pda = os.getenv("CAMERA_PDA")
    if not pda:
        reg = get_registration_service()
        pda = reg.get_camera_pda()
    return pda


def _require_auth(wallet_address: str, signature: str, timestamp: int, nonce: str) -> tuple[bool, str]:
    if Settings.SKIP_AUTH:
        return True, "OK"
    signer = get_device_signer()
    return signer.verify_request_signature(wallet_address, timestamp, nonce, signature)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def register_routes(app: Flask):

    # ── Health ─────────────────────────────────────────────────────────────

    @app.route("/api/health")
    @app.route("/health")
    def api_health():
        buf = get_buffer_service()
        return jsonify({
            "status": "ok",
            "camera_pda": _get_camera_pda(),
            "buffer": buf.get_status(),
            "whip": get_whip_publisher().get_status(),
        })

    # ── Device info / discovery ─────────────────────────────────────────────

    @app.route("/api/device-info")
    def api_device_info():
        reg = get_registration_service()
        return jsonify(reg.get_device_info())

    @app.route("/api/camera/info")
    def api_camera_info():
        buf = get_buffer_service()
        status = buf.get_status()
        return jsonify({
            "camera_pda": _get_camera_pda(),
            "device_type": "pi_zero_2w",
            "capabilities": {
                "photo": True,
                "video": True,
                "live_stream": True,
                "face_recognition": False,
                "pose_detection": False,
            },
            "stream": {
                "whep_url": get_whip_publisher().whep_url if get_whip_publisher().stream_name else None,
                "resolution": status["resolution"],
                "fps": status["fps"],
            },
            "online": True,
        })

    @app.route("/api/status")
    def api_status():
        buf = get_buffer_service()
        whip = get_whip_publisher()
        reg = get_registration_service()
        with _sessions_lock:
            session_count = len(_sessions)
        return jsonify({
            "camera_pda": _get_camera_pda(),
            "registered": reg.is_registered(),
            "buffer": buf.get_status(),
            "whip": whip.get_status(),
            "active_sessions": session_count,
        })

    @app.route("/api/stream/whip/status")
    def api_whip_status():
        return jsonify(get_whip_publisher().get_status())

    # ── Registration / setup ────────────────────────────────────────────────

    @app.route("/api/scan-registration-qr", methods=["POST"])
    def api_scan_registration_qr():
        """
        Scans a QR code from the camera and extracts registration data.
        The web app displays a QR containing WiFi + claim endpoint JSON.
        Device camera scans it, parses the JSON, connects to WiFi, claims identity.
        """
        import base64
        import json as _json

        timeout = request.json.get("timeout", 60) if request.json else 60
        buf = get_buffer_service()

        deadline = time.time() + timeout
        while time.time() < deadline:
            jpeg = buf.get_jpeg_frame()
            if jpeg:
                frame = cv2.imdecode(
                    __import__("numpy").frombuffer(jpeg, dtype=__import__("numpy").uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is not None:
                    detector = cv2.QRCodeDetector()
                    data, _, _ = detector.detectAndDecode(frame)
                    if data:
                        try:
                            qr_data = _json.loads(data)
                            logger.info(f"QR scanned: {list(qr_data.keys())}")

                            required = ["claim_endpoint", "user_wallet"]
                            for field in required:
                                if field not in qr_data:
                                    return jsonify({"success": False, "error": f"Missing QR field: {field}"}), 400

                            # Check expiry
                            expires = qr_data.get("expires", 0)
                            if expires and time.time() * 1000 > expires:
                                return jsonify({"success": False, "error": "QR code expired"}), 400

                            # Connect to WiFi if credentials provided
                            ssid = qr_data.get("wifi_ssid")
                            password = qr_data.get("wifi_password")
                            if ssid:
                                _connect_wifi(ssid, password)

                            # POST to claim endpoint
                            signer = get_device_signer()
                            claim_resp = req.post(
                                qr_data["claim_endpoint"],
                                json={
                                    "device_pubkey": signer.get_public_key(),
                                    "device_model": "pi_zero_2w",
                                    "user_wallet": qr_data["user_wallet"],
                                },
                                timeout=10,
                            )

                            if claim_resp.status_code in (200, 201):
                                # Start polling for PDA assignment
                                reg = get_registration_service()
                                reg.start_polling()
                                return jsonify({
                                    "success": True,
                                    "message": "Claimed successfully, polling for PDA",
                                    "device_pubkey": signer.get_public_key(),
                                })
                            else:
                                return jsonify({
                                    "success": False,
                                    "error": f"Claim failed: {claim_resp.status_code}",
                                }), 400

                        except _json.JSONDecodeError:
                            # Not a registration QR, keep scanning
                            pass

            time.sleep(0.2)

        return jsonify({"success": False, "error": "No QR code detected within timeout"}), 408

    @app.route("/api/device/claim", methods=["POST"])
    def api_device_claim():
        """Backend calls this to assign the camera PDA after claim."""
        data = request.json or {}
        camera_pda = data.get("camera_pda")
        full_domain = data.get("full_domain")
        if not camera_pda or not full_domain:
            return jsonify({"success": False, "error": "camera_pda and full_domain required"}), 400

        reg = get_registration_service()
        reg._apply_config({
            "device_pubkey": get_device_signer().get_public_key(),
            "camera_pda": camera_pda,
            "subdomain": full_domain.split(".")[0],
            "full_domain": full_domain,
        })

        # Update WHIP stream name
        whip = get_whip_publisher()
        whip.set_stream_name(camera_pda)
        if not whip.running:
            whip.start()

        return jsonify({"success": True, "camera_pda": camera_pda})

    @app.route("/api/device/registration/status")
    def api_registration_status():
        reg = get_registration_service()
        return jsonify({
            "registered": reg.is_registered(),
            "camera_pda": reg.get_camera_pda(),
            "device_pubkey": get_device_signer().get_public_key(),
        })

    # ── WiFi setup ──────────────────────────────────────────────────────────

    @app.route("/api/setup/wifi/scan")
    def api_wifi_scan():
        import subprocess
        try:
            result = subprocess.run(
                ["sudo", "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"],
                capture_output=True, text=True, timeout=15,
            )
            networks = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(":")
                if len(parts) >= 2 and parts[0]:
                    networks.append({
                        "ssid": parts[0],
                        "signal": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                        "security": parts[2] if len(parts) > 2 else "",
                    })
            return jsonify({"success": True, "networks": networks})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/setup/wifi", methods=["POST"])
    def api_wifi_setup():
        data = request.json or {}
        ssid = data.get("ssid")
        password = data.get("password", "")
        if not ssid:
            return jsonify({"success": False, "error": "ssid required"}), 400
        success = _connect_wifi(ssid, password)
        return jsonify({"success": success})

    # ── Check-in / checkout ─────────────────────────────────────────────────

    @app.route("/api/checkin", methods=["POST"])
    def api_checkin():
        data = request.json or {}
        wallet = data.get("wallet_address")
        signature = data.get("request_signature") or data.get("signature")
        timestamp = data.get("request_timestamp") or data.get("timestamp")
        nonce = data.get("request_nonce") or data.get("nonce")
        display_name = data.get("display_name") or data.get("username") or (wallet[:8] if wallet else None)

        if not all([wallet, signature, timestamp, nonce]):
            return jsonify({"success": False, "error": "wallet_address, signature, timestamp, nonce required"}), 400

        valid, error = _require_auth(wallet, signature, timestamp, nonce)
        if not valid:
            return jsonify({"success": False, "error": error}), 401

        import uuid
        session_id = str(uuid.uuid4())
        with _sessions_lock:
            _sessions[wallet] = {
                "session_id": session_id,
                "checked_in_at": int(time.time()),
                "display_name": display_name,
            }

        camera_pda = _get_camera_pda()

        # Notify backend (non-blocking)
        def notify():
            try:
                req.post(
                    f"{Settings.BACKEND_URL}/api/session/activity",
                    json={
                        "sessionId": session_id,
                        "cameraId": camera_pda,
                        "userPubkey": wallet,
                        "activityType": 0,  # CHECK_IN
                        "timestamp": int(time.time() * 1000),
                        "displayName": display_name,
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.warning(f"Checkin backend notify failed: {e}")

        threading.Thread(target=notify, daemon=True).start()

        logger.info(f"Checkin: {display_name} ({wallet[:8]}...) session={session_id[:8]}...")
        return jsonify({
            "success": True,
            "wallet_address": wallet,
            "session_id": session_id,
            "camera_pda": camera_pda,
            "display_name": display_name,
        })

    @app.route("/api/checkout", methods=["POST"])
    def api_checkout():
        data = request.json or {}
        wallet = data.get("wallet_address")
        if not wallet:
            return jsonify({"success": False, "error": "wallet_address required"}), 400

        with _sessions_lock:
            session = _sessions.pop(wallet, None)

        camera_pda = _get_camera_pda()

        if session:
            def notify():
                try:
                    req.post(
                        f"{Settings.BACKEND_URL}/api/session/activity",
                        json={
                            "sessionId": session["session_id"],
                            "cameraId": camera_pda,
                            "userPubkey": wallet,
                            "activityType": 1,  # CHECK_OUT
                            "timestamp": int(time.time() * 1000),
                        },
                        timeout=5,
                    )
                except Exception as e:
                    logger.warning(f"Checkout backend notify failed: {e}")

            threading.Thread(target=notify, daemon=True).start()

        logger.info(f"Checkout: {wallet[:8]}...")
        return jsonify({"success": True})

    @app.route("/api/session/status/<wallet_address>")
    def api_session_status(wallet_address):
        with _sessions_lock:
            session = _sessions.get(wallet_address)
        return jsonify({
            "checked_in": session is not None,
            "session": session,
        })

    # ── Photo capture ───────────────────────────────────────────────────────

    @app.route("/api/capture", methods=["POST"])
    def api_capture():
        data = request.json or {}
        wallet = data.get("wallet_address")

        if wallet and not Settings.SKIP_AUTH:
            with _sessions_lock:
                if wallet not in _sessions:
                    return jsonify({"success": False, "error": "Not checked in"}), 403

        buf = get_buffer_service()
        jpeg = buf.get_jpeg_frame()
        if not jpeg:
            return jsonify({"success": False, "error": "No frame available"}), 503

        # Save to disk so backend can fetch via /api/photos/<filename>
        timestamp = int(time.time())
        filename = f"photo_{timestamp}.jpg"
        photo_path = os.path.join(Settings.VIDEOS_DIR, "..", "photos", filename)
        os.makedirs(os.path.dirname(photo_path), exist_ok=True)
        with open(photo_path, "wb") as f:
            f.write(jpeg)

        camera_pda = _get_camera_pda()
        file_hash = hashlib.sha256(jpeg).hexdigest()

        # Sign the capture event
        signed = {}
        if wallet and camera_pda:
            signed = sign_capture(
                user_wallet=wallet,
                camera_pda=camera_pda,
                file_hash=file_hash,
                capture_type="photo",
                metadata={"filename": filename},
            )

        logger.info(f"Photo captured: {filename} for {wallet[:8] if wallet else 'anon'}...")
        return jsonify({
            "success": True,
            "filename": filename,
            "timestamp": timestamp,
            "width": 720,
            "height": 1280,
            "file_hash": file_hash,
            "device_signature": signed.get("device_signature"),
            "device_public_key": signed.get("device_public_key"),
            "camera_pda": camera_pda,
        })

    @app.route("/api/photos/<filename>")
    def api_get_photo(filename):
        photo_path = os.path.join(Settings.BASE_DIR, "photos", filename)
        if not os.path.exists(photo_path):
            return jsonify({"error": "Not found"}), 404
        return send_file(photo_path, mimetype="image/jpeg")

    # ── Video recording ─────────────────────────────────────────────────────

    @app.route("/api/record", methods=["POST"])
    def api_record():
        data = request.json or {}
        wallet = data.get("wallet_address")
        duration = min(int(data.get("duration", 10)), 60)  # cap at 60s for Zero 2W
        action = data.get("action")

        if wallet and not Settings.SKIP_AUTH:
            with _sessions_lock:
                if wallet not in _sessions:
                    return jsonify({"success": False, "error": "Not checked in"}), 403

        buf = get_buffer_service()

        # Non-blocking start
        if action == "start":
            with _recording_lock:
                if _recording_state["active"]:
                    return jsonify({"success": False, "error": "Already recording"}), 400
                _recording_state.update({
                    "active": True,
                    "start_time": time.time(),
                    "wallet": wallet,
                    "duration": duration,
                })
            return jsonify({"success": True, "recording": True, "duration": duration})

        # Stop and extract
        if action == "stop":
            with _recording_lock:
                if not _recording_state["active"]:
                    return jsonify({"success": False, "error": "Not recording"}), 400
                elapsed = int(time.time() - _recording_state["start_time"])
                _recording_state["active"] = False
                actual_duration = min(elapsed, duration)
                wallet = wallet or _recording_state["wallet"]

            try:
                result = buf.get_video_segment(actual_duration)
                return _handle_video_result(result, wallet)
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

        # Legacy blocking mode
        try:
            result = buf.get_video_segment(duration)
            return _handle_video_result(result, wallet)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    def _handle_video_result(result: dict, wallet: Optional[str]) -> object:
        filename = result["filename"]
        path = result["path"]
        camera_pda = _get_camera_pda()
        file_hash = _sha256_file(path)

        signed = {}
        if wallet and camera_pda:
            signed = sign_capture(
                user_wallet=wallet,
                camera_pda=camera_pda,
                file_hash=file_hash,
                capture_type="video",
                metadata={"filename": filename, "duration": result["duration"]},
            )

        logger.info(f"Video recorded: {filename} ({result['duration']}s)")

        # Notify backend to upload to Walrus (non-blocking)
        if wallet and camera_pda:
            def upload_to_backend():
                try:
                    with open(path, "rb") as f:
                        video_data = f.read()
                    req.post(
                        f"{Settings.BACKEND_URL}/api/walrus/upload",
                        data=video_data,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "x-wallet-address": wallet,
                            "x-camera-id": camera_pda,
                            "x-device-signature": signed.get("device_signature", ""),
                            "x-file-type": "video",
                            "x-timestamp": str(int(time.time())),
                            "x-original-size": str(os.path.getsize(path)),
                            "x-nonce": os.urandom(16).hex(),
                        },
                        timeout=60,
                    )
                    logger.info(f"Video uploaded to Walrus: {filename}")
                except Exception as e:
                    logger.warning(f"Walrus upload failed (non-blocking): {e}")

            threading.Thread(target=upload_to_backend, daemon=True).start()

        return jsonify({
            "success": True,
            "filename": filename,
            "duration": result["duration"],
            "frame_count": result["frame_count"],
            "file_hash": file_hash,
            "device_signature": signed.get("device_signature"),
            "camera_pda": camera_pda,
        })

    @app.route("/api/videos/<filename>")
    def api_get_video(filename):
        video_path = os.path.join(Settings.VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            return jsonify({"error": "Not found"}), 404
        return send_file(video_path, mimetype="video/quicktime")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _connect_wifi(ssid: str, password: str) -> bool:
    import subprocess
    try:
        cmd = ["sudo", "nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        success = result.returncode == 0
        if success:
            logger.info(f"Connected to WiFi: {ssid}")
        else:
            logger.error(f"WiFi connect failed: {result.stderr}")
        return success
    except Exception as e:
        logger.error(f"WiFi connect error: {e}")
        return False
