"""
Camera Service — Pi Zero 2W
Entry point. Starts buffer, WHIP publisher, registration, and Flask API.
"""

import logging
import os
import sys
import time

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

    # Flask API
    app = create_app()
    logger.info(f"API listening on {Settings.HOST}:{Settings.PORT}")
    app.run(host=Settings.HOST, port=Settings.PORT, threaded=True)


if __name__ == "__main__":
    main()
