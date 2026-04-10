"""
Capture Event Signer — Pi Zero 2W

Signs photo/video capture events with the device ed25519 key.
Proves a capture happened at this physical device, for this wallet, at this time.
No per-capture blockchain transaction — batch verified via Merkle root at checkout.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from .device_signer import get_device_signer, DeviceSigner

logger = logging.getLogger(__name__)


class CaptureEventSigner:
    def __init__(self, device_signer: Optional[DeviceSigner] = None):
        self.device_signer = device_signer or get_device_signer()
        logger.info(f"CaptureEventSigner ready — device key: {self.device_signer.get_public_key()[:16]}...")

    def sign_capture_event(
        self,
        user_wallet: str,
        camera_pda: str,
        file_hash: str,
        capture_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = time.time()
        event = {
            "user_wallet": user_wallet,
            "camera_pda": camera_pda,
            "file_hash": file_hash,
            "capture_type": capture_type,
            "timestamp": timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "metadata": metadata or {},
        }

        signed = self.device_signer.sign_response(event)
        sig_dict = signed["device_signature"]
        sig_str = sig_dict["signature"]

        logger.info(f"Signed {capture_type} for {user_wallet[:8]}... sig={sig_str[:16]}...")
        return {
            "success": True,
            "capture_event": event,
            "device_signature": sig_str,
            "device_signature_full": sig_dict,
            "device_public_key": sig_dict["device_pubkey"],
            "timestamp": timestamp,
        }

    def create_session_merkle_root(self, signatures: list) -> str:
        if not signatures:
            return hashlib.sha256(b"").hexdigest()
        combined = "".join(sorted(signatures)).encode()
        return hashlib.sha256(combined).hexdigest()

    def get_device_public_key(self) -> str:
        return self.device_signer.get_public_key()


_signer: Optional[CaptureEventSigner] = None


def get_capture_event_signer() -> CaptureEventSigner:
    global _signer
    if _signer is None:
        _signer = CaptureEventSigner()
    return _signer


def sign_capture(
    user_wallet: str,
    camera_pda: str,
    file_hash: str,
    capture_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return get_capture_event_signer().sign_capture_event(
        user_wallet=user_wallet,
        camera_pda=camera_pda,
        file_hash=file_hash,
        capture_type=capture_type,
        metadata=metadata,
    )
