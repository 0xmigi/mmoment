"""
Camera Service — Pi Zero 2W

Two-mode startup:
  ONBOARDING (no PDA): rpicam-still for fast QR scanning, no streaming.
  STREAMING  (has PDA): rpicam-vid + ffmpeg H.264 pipeline, no QR scanning.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import requests as req
from flask import Flask
from flask_cors import CORS

from .config.settings import Settings
from .routes import register_routes
from .services.device_registration import get_registration_service
from .services.device_signer import get_device_signer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# rpicam-still capture settings for onboarding mode
_STILL_WIDTH = 720
_STILL_HEIGHT = 1280
_STILL_PATH = "/tmp/mmoment_onboard.jpg"


def create_app() -> Flask:
    Settings.setup()

    app = Flask(__name__)
    CORS(app)
    register_routes(app)
    return app


def main():
    logger.info("=== MMOMENT Camera Service — Pi Zero 2W ===")
    Settings.setup()

    # Device identity
    signer = get_device_signer()
    logger.info(f"Device pubkey: {signer.get_public_key()}")

    # Registration — load saved config or start polling
    reg = get_registration_service()
    camera_pda = reg.get_camera_pda() or os.getenv("CAMERA_PDA")

    if camera_pda:
        _start_streaming_mode(camera_pda)
    else:
        _start_onboarding_mode(signer)

    # Flask API (both modes)
    app = create_app()
    logger.info(f"API listening on {Settings.HOST}:{Settings.PORT}")
    app.run(host=Settings.HOST, port=Settings.PORT, threaded=True)


# ── Streaming mode ────────────────────────────────────────────────────────────

def _start_streaming_mode(camera_pda: str):
    """Registered device: full H.264 pipeline + segment buffer."""
    from .services.buffer_service import get_buffer_service
    from .services.stream_manager import get_stream_manager

    logger.info(f"Camera PDA: {camera_pda} — starting streaming mode")

    stream = get_stream_manager()
    stream.start(camera_pda)
    logger.info(f"Streaming to {stream.rtsp_url}")

    # Give segments time to start
    time.sleep(3)

    buf = get_buffer_service()
    status = buf.get_status()
    logger.info(f"Buffer: {status['resolution']}, temp={status['temperature']:.1f}°C")


# ── Onboarding mode ──────────────────────────────────────────────────────────

def _start_onboarding_mode(signer):
    """Unregistered device: lightweight QR scanning via rpicam-still."""
    reg = get_registration_service()

    logger.info("No PDA configured — entering onboarding mode")
    logger.info("Using rpicam-still for fast QR scanning (no streaming pipeline)")
    reg.start_polling()

    thread = threading.Thread(
        target=_qr_scan_loop, args=(signer,), daemon=True, name="QRScanner"
    )
    thread.start()


def _capture_still() -> np.ndarray | None:
    """Capture a single JPEG via rpicam-still and return as numpy array."""
    try:
        result = subprocess.run(
            [
                "rpicam-still",
                "--width", str(_STILL_WIDTH),
                "--height", str(_STILL_HEIGHT),
                "--quality", "60",
                "--immediate",
                "--nopreview",
                "-o", _STILL_PATH,
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(f"rpicam-still failed: {result.stderr.decode(errors='replace')[:200]}")
            return None

        if not os.path.exists(_STILL_PATH) or os.path.getsize(_STILL_PATH) == 0:
            return None

        frame = cv2.imread(_STILL_PATH)
        return frame
    except Exception as e:
        logger.error(f"Still capture error: {e}")
        return None


def _qr_scan_loop(signer):
    """Scan for registration QR codes using rpicam-still."""
    detector = cv2.QRCodeDetector()
    logger.info("QR scanner started — rpicam-still mode")

    while True:
        reg = get_registration_service()
        if reg.is_registered():
            logger.info("QR scanner stopped — device is registered")
            return

        try:
            frame = _capture_still()
            if frame is None:
                time.sleep(1)
                continue

            # Use grayscale at full resolution — downscaling kills QR detail
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, _, _ = detector.detectAndDecode(gray)
            if not data:
                time.sleep(0.5)
                continue

            try:
                qr_data = json.loads(data)
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue

            if "claim_endpoint" not in qr_data or "user_wallet" not in qr_data:
                time.sleep(0.5)
                continue

            expires = qr_data.get("expires", 0)
            if expires and time.time() * 1000 > expires:
                logger.warning("QR code expired, ignoring")
                time.sleep(1)
                continue

            logger.info("Registration QR detected! Claiming...")

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
                logger.info("Claim successful — polling for PDA assignment")
                reg.start_polling()
                return
            else:
                logger.warning(f"Claim failed: {claim_resp.status_code} {claim_resp.text}")
                time.sleep(5)

        except Exception as e:
            logger.error(f"QR scanner error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
