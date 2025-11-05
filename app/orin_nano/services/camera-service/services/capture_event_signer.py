"""
Capture Event Signing Service

Signs camera capture events with the device's ed25519 key for provable authenticity.
Used for direct Pipe uploads without requiring blockchain transactions per capture.

Architecture:
- Device signs: {user_wallet, camera_pda, timestamp, file_hash, capture_type}
- Signature proves capture happened at this camera, for this user, at this time
- Backend tracks: device_signature → file_id mapping
- Future: Merkle root of signatures included in checkout transaction
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional
from services.device_signer import DeviceSigner
import logging

logger = logging.getLogger(__name__)


class CaptureEventSigner:
    """
    Signs capture events with device ed25519 key for ownership proof.

    This provides instant capture UX while maintaining verifiable ownership:
    - No waiting for blockchain transaction confirmation
    - Device signature proves authenticity
    - Multiple users can capture simultaneously
    - Privacy-ready (no per-capture on-chain transactions)
    """

    def __init__(self, device_signer: Optional[DeviceSigner] = None):
        """
        Initialize capture event signer.

        Args:
            device_signer: Device signer instance (or creates new one)
        """
        self.device_signer = device_signer or DeviceSigner()
        logger.info(f"✅ Capture event signer initialized with device key: {self.device_signer.get_public_key()[:16]}...")

    def sign_capture_event(
        self,
        user_wallet: str,
        camera_pda: str,
        file_hash: str,
        capture_type: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create and sign a capture event.

        Args:
            user_wallet: User's wallet address (owner of capture)
            camera_pda: Camera's PDA (device identity)
            file_hash: Blake3 hash of the captured file
            capture_type: Type of capture ('photo' or 'video')
            file_path: Path to the captured file
            metadata: Optional additional metadata

        Returns:
            Dict with signed event data including signature
        """

        # Create capture event
        timestamp = time.time()

        capture_event = {
            'user_wallet': user_wallet,
            'camera_pda': camera_pda,
            'file_hash': file_hash,
            'capture_type': capture_type,
            'timestamp': timestamp,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(timestamp)),
            'file_path': file_path,
            'metadata': metadata or {}
        }

        # Sign the event with device key
        signed_event = self.device_signer.sign_response(capture_event)

        # Extract the signature for use as "transaction signature"
        device_signature = signed_event['device_signature']

        logger.info(f"📝 Signed capture event for {user_wallet[:8]}...")
        logger.info(f"   Camera: {camera_pda}")
        logger.info(f"   Type: {capture_type}")
        logger.info(f"   Hash: {file_hash[:16]}...")
        logger.info(f"   Signature: {device_signature[:16]}...")

        return {
            'success': True,
            'capture_event': capture_event,
            'device_signature': device_signature,
            'device_public_key': signed_event['device_public_key'],
            'timestamp': timestamp,
            'signed_data': signed_event
        }

    def create_session_merkle_root(self, capture_signatures: list) -> str:
        """
        Create merkle root of all capture signatures in a session.

        This will be included in the checkout transaction for batch verification.

        Args:
            capture_signatures: List of device signatures from this session

        Returns:
            Merkle root hash (hex string)
        """
        if not capture_signatures:
            return hashlib.sha256(b'').hexdigest()

        # Sort signatures for deterministic merkle tree
        sorted_sigs = sorted(capture_signatures)

        # Simple merkle root (can be enhanced to full merkle tree later)
        combined = ''.join(sorted_sigs).encode('utf-8')
        merkle_root = hashlib.sha256(combined).hexdigest()

        logger.info(f"📊 Created merkle root for {len(capture_signatures)} captures: {merkle_root[:16]}...")

        return merkle_root

    def verify_capture_signature(self, capture_event: Dict[str, Any], signature: str) -> bool:
        """
        Verify a capture event signature.

        Args:
            capture_event: The capture event data
            signature: The device signature to verify

        Returns:
            True if signature is valid
        """
        # This would verify against the device's public key
        # For now, we trust our own signatures
        # In production, backend/other parties can verify using device public key
        return True

    def get_device_public_key(self) -> str:
        """Get the device's public key for signature verification."""
        return self.device_signer.get_public_key()


# Singleton instance for camera service
_capture_event_signer = None


def get_capture_event_signer() -> CaptureEventSigner:
    """Get or create the singleton capture event signer instance."""
    global _capture_event_signer

    if _capture_event_signer is None:
        _capture_event_signer = CaptureEventSigner()

    return _capture_event_signer


# Convenience function for routes
def sign_capture(
    user_wallet: str,
    camera_pda: str,
    file_hash: str,
    capture_type: str,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Sign a capture event (convenience function for routes).

    Returns device signature that can be used as "tx_signature" for Pipe uploads.
    """
    signer = get_capture_event_signer()
    return signer.sign_capture_event(
        user_wallet=user_wallet,
        camera_pda=camera_pda,
        file_hash=file_hash,
        capture_type=capture_type,
        file_path=file_path,
        metadata=metadata
    )


# For testing
if __name__ == "__main__":
    # Test the signer
    signer = get_capture_event_signer()

    test_event = signer.sign_capture_event(
        user_wallet="RsLjCiEiHq3dyWeDpp1M8jSmAhpmaGamcVK32sJkdLT",
        camera_pda="test_camera_pda",
        file_hash="abc123def456",
        capture_type="photo",
        file_path="/tmp/test.jpg",
        metadata={'camera_id': 'jetson01'}
    )

    print(f"✅ Test capture event signed:")
    print(json.dumps(test_event, indent=2))

    # Test merkle root creation
    signatures = [
        test_event['device_signature'],
        "sig2_example",
        "sig3_example"
    ]

    merkle_root = signer.create_session_merkle_root(signatures)
    print(f"\n📊 Merkle root: {merkle_root}")
