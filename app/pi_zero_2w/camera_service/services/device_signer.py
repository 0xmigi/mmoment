"""
Device Signer — Pi Zero 2W

Hardware-bound ed25519 keypair tied to the Pi's CPU serial.
Same signing interface as Orin Nano — backend is agnostic to device type.
"""

import os
import json
import logging
import time
import base64
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

KEYPAIR_PATH_PROD = Path("/opt/mmoment/device/device-keypair.enc")
KEYPAIR_PATH_DEV = Path(os.path.expanduser("~/.mmoment/device-keypair.enc"))


class DeviceSigner:
    def __init__(self):
        self.keypair = None
        self._load_or_create_keypair()

    def _get_hardware_key(self) -> bytes:
        """Derive encryption key from Pi CPU serial (deterministic per device)."""
        serial = None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("Serial"):
                        serial = line.split(":")[1].strip()
                        break
        except Exception:
            pass

        if not serial or serial == "0000000000000000":
            try:
                serial = open("/sys/class/net/eth0/address").read().strip()
            except Exception:
                pass

        if not serial:
            try:
                serial = open("/etc/machine-id").read().strip()
            except Exception:
                serial = "mmoment_dev_fallback"

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"mmoment_depin_device_v1",
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(serial.encode()))
        logger.info(f"Hardware key derived from: {serial[:8]}...")
        return key

    def _load_or_create_keypair(self):
        try:
            keypair_path = KEYPAIR_PATH_PROD
            keypair_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            keypair_path = KEYPAIR_PATH_DEV
            keypair_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using dev keypair path: {keypair_path}")

        cipher = Fernet(self._get_hardware_key())

        if keypair_path.exists():
            try:
                encrypted = keypair_path.read_bytes()
                data = json.loads(cipher.decrypt(encrypted).decode())
                from solders.keypair import Keypair
                self.keypair = Keypair.from_bytes(bytes(data["private_key"]))
                logger.info(f"Loaded device keypair: {self.keypair.pubkey()}")
                return
            except Exception as e:
                logger.warning(f"Failed to load keypair, regenerating: {e}")

        from solders.keypair import Keypair
        self.keypair = Keypair()

        data = {
            "private_key": list(bytes(self.keypair)),
            "public_key": str(self.keypair.pubkey()),
            "created_at": int(time.time()),
            "version": "1.0",
        }
        encrypted = cipher.encrypt(json.dumps(data).encode())
        keypair_path.write_bytes(encrypted)
        os.chmod(keypair_path, 0o600)
        logger.info(f"Generated new device keypair: {self.keypair.pubkey()}")

    def sign_response(self, response_data: dict) -> dict:
        signed = response_data.copy()
        signed["device_signature"] = {
            "device_pubkey": str(self.keypair.pubkey()),
            "timestamp": int(time.time() * 1000),
            "version": "1.0",
            "algorithm": "ed25519",
        }
        payload = json.dumps(
            {k: v for k, v in signed.items() if k != "device_signature"},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        meta = json.dumps(signed["device_signature"], sort_keys=True).encode()
        sig = self.keypair.sign_message(payload + b"|" + meta)
        signed["device_signature"]["signature"] = base64.b64encode(bytes(sig)).decode()
        return signed

    def get_public_key(self) -> str:
        return str(self.keypair.pubkey()) if self.keypair else None

    def get_keypair(self):
        return self.keypair

    def get_device_info(self) -> dict:
        return {
            "device_pubkey": self.get_public_key(),
            "device_type": "pi_zero_2w",
            "capabilities": ["camera_capture", "video_streaming"],
            "version": "1.0",
        }

    def verify_request_signature(
        self,
        wallet_address: str,
        timestamp: int,
        nonce: str,
        signature: str,
        max_age_seconds: int = 300,
    ) -> tuple[bool, str]:
        current_ms = int(time.time() * 1000)
        age_ms = current_ms - timestamp
        if age_ms < 0:
            return False, "Timestamp is in the future"
        if age_ms > max_age_seconds * 1000:
            return False, f"Request too old ({age_ms // 1000}s)"

        message = f"{wallet_address}|{timestamp}|{nonce}"
        try:
            import base58
            from nacl.signing import VerifyKey
            from nacl.exceptions import BadSignatureError

            verify_key = VerifyKey(base58.b58decode(wallet_address))
            verify_key.verify(message.encode(), base58.b58decode(signature))
            return True, "OK"
        except Exception as e:
            return False, f"Invalid signature: {e}"


_signer: DeviceSigner = None


def get_device_signer() -> DeviceSigner:
    global _signer
    if _signer is None:
        _signer = DeviceSigner()
    return _signer
