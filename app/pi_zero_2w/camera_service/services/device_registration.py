"""
Device Registration — Pi Zero 2W

Polls backend for device config after the organizer scans the setup QR.
On receiving config: writes camera PDA to env file, configures Cloudflare tunnel,
then restarts itself via systemctl so the new PDA takes effect.
"""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .device_signer import get_device_signer
from .tunnel_manager import get_tunnel_manager

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.path.expanduser("~/.mmoment/device_config.json"))
ENV_PATH = Path(os.path.expanduser("~/.mmoment/device.env"))


class DeviceRegistrationService:
    def __init__(self):
        self.device_signer = get_device_signer()
        self.tunnel_manager = get_tunnel_manager()
        self.backend_url = os.getenv("BACKEND_URL", "https://mmoment-backend-production.up.railway.app")
        self.device_config: Optional[Dict[str, Any]] = None
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        # Load persisted config if available
        self._load_saved_config()

    def _load_saved_config(self):
        if CONFIG_PATH.exists():
            try:
                self.device_config = json.loads(CONFIG_PATH.read_text())
                logger.info(f"Loaded saved config — PDA: {self.device_config.get('camera_pda')}")
            except Exception as e:
                logger.warning(f"Could not load saved config: {e}")

    def start_polling(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._poll, daemon=True, name="DeviceRegistration")
        self._thread.start()
        logger.info("Device registration polling started")

    def stop_polling(self):
        self._stop = True

    def _poll(self):
        pubkey = self.device_signer.get_public_key()
        url = f"{self.backend_url}/api/device/{pubkey}/config"
        interval = 10
        max_attempts = 360  # 1 hour

        for attempt in range(max_attempts):
            if self._stop:
                break
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    config = resp.json()
                    if self._validate_config(config):
                        logger.info(f"Received config: PDA={config['camera_pda']}")
                        self.device_config = config
                        self._apply_config(config)
                        return
                elif resp.status_code != 404:
                    logger.warning(f"Unexpected poll response: {resp.status_code}")
            except Exception as e:
                logger.debug(f"Poll error (attempt {attempt + 1}): {e}")

            time.sleep(interval)

        logger.warning("Registration polling timed out after 1 hour")

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        required = ["device_pubkey", "camera_pda", "subdomain", "full_domain"]
        for field in required:
            if field not in config:
                logger.error(f"Missing field in config: {field}")
                return False
        if config["device_pubkey"] != self.device_signer.get_public_key():
            logger.error("device_pubkey mismatch")
            return False
        return True

    def _apply_config(self, config: Dict[str, Any]):
        # Save config to disk
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
        logger.info(f"Config saved to {CONFIG_PATH}")

        # Write env file so systemd picks up CAMERA_PDA on restart
        env_content = f"CAMERA_PDA={config['camera_pda']}\n"
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENV_PATH.write_text(env_content)
        logger.info(f"Env file written: {ENV_PATH}")

        # Configure Cloudflare tunnel
        if self.tunnel_manager.configure_tunnel(config["full_domain"]):
            logger.info("Tunnel configured successfully")
            time.sleep(5)
            if self.tunnel_manager.test_connectivity(config["full_domain"]):
                logger.info(f"Camera live at https://{config['full_domain']}")
            else:
                logger.info("Tunnel configured — DNS may still be propagating")
        else:
            logger.error("Tunnel configuration failed")

        # Restart service to pick up new CAMERA_PDA env var
        self._restart_self()

    def _restart_self(self):
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "restart", "camera-service"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("camera-service restarted via systemctl")
            else:
                logger.error(f"systemctl restart failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to restart camera-service: {e}")

    def is_registered(self) -> bool:
        return self.device_config is not None and "camera_pda" in self.device_config

    def get_camera_pda(self) -> Optional[str]:
        return self.device_config.get("camera_pda") if self.device_config else None

    def get_device_info(self) -> Dict[str, Any]:
        info = self.device_signer.get_device_info()
        info["setup_required"] = not self.is_registered()
        if self.device_config:
            info["camera_pda"] = self.device_config.get("camera_pda")
            info["full_domain"] = self.device_config.get("full_domain")
        info["tunnel_status"] = self.tunnel_manager.get_status()
        return info


_service: Optional[DeviceRegistrationService] = None


def get_registration_service() -> DeviceRegistrationService:
    global _service
    if _service is None:
        _service = DeviceRegistrationService()
    return _service
