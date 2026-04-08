"""
Camera Service — Pi Zero 2W
Entry point. Starts buffer, WHIP publisher, registration, and Flask API.
"""

import json
import logging
import os
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
from .services.buffer_service import get_buffer_service
from .services.device_registration import get_registration_service
from .services.device_signer import get_device_signer
from .services.whip_publisher import get_whip_publisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


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
        logger.info(f"Camera PDA: {camera_pda}")
    else:
        logger.info("No PDA configured — waiting for organizer to scan setup QR")
        reg.start_polling()

    # Frame buffer (starts camera immediately)
    logger.info("Starting frame buffer...")
    buf = get_buffer_service()
    # Give the camera a moment to produce frames
    time.sleep(2)
    status = buf.get_status()
    logger.info(f"Buffer: {status['resolution']} @ {status['fps']}fps, temp={status['temperature']:.1f}°C")

    # WHIP publisher
    if camera_pda:
        whip = get_whip_publisher()
        whip.set_stream_name(camera_pda)
        whip.set_buffer_service(buf)
        whip.start()
        logger.info(f"WHIP publishing: {whip.whip_url}")
    else:
        logger.info("WHIP publisher deferred until PDA assigned")
        # Auto-scan for registration QR codes when unregistered
        _start_qr_scanner(buf, signer)

    # Flask API
    app = create_app()
    logger.info(f"API listening on {Settings.HOST}:{Settings.PORT}")
    app.run(host=Settings.HOST, port=Settings.PORT, threaded=True)


def _start_qr_scanner(buf, signer):
    """Background thread that scans camera frames for a registration QR code."""

    def scan_loop():
        detector = cv2.QRCodeDetector()
        logger.info("QR scanner started — waiting for registration QR code")

        while True:
            # Stop scanning once registered
            reg = get_registration_service()
            if reg.is_registered():
                logger.info("QR scanner stopped — device is registered")
                return

            try:
                time.sleep(2)  # Scan every 2 seconds to save CPU

                jpeg = buf.get_jpeg_frame()
                if not jpeg:
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,  # Grayscale is faster for QR detection
                )
                if frame is None:
                    continue

                # Downscale to 360x640 for faster QR detection
                small = cv2.resize(frame, (360, 640))
                data, _, _ = detector.detectAndDecode(small)
                if not data:
                    continue

                # Try to parse as registration QR
                try:
                    qr_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Validate required fields
                if "claim_endpoint" not in qr_data or "user_wallet" not in qr_data:
                    continue

                # Check expiry
                expires = qr_data.get("expires", 0)
                if expires and time.time() * 1000 > expires:
                    logger.warning("QR code expired, ignoring")
                    continue

                logger.info(f"Registration QR detected! Claiming...")

                # POST to claim endpoint
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

    thread = threading.Thread(target=scan_loop, daemon=True, name="QRScanner")
    thread.start()


if __name__ == "__main__":
    main()
