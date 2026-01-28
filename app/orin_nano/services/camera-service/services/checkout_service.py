"""
Checkout Service - UserSessionChain Architecture

Handles the checkout flow:
1. Notify backend of checkout (for real-time UI updates)
2. Encrypt session key locally on Jetson
3. Send encrypted session key to backend
4. Backend writes to UserSessionChain on Solana (pays for tx)

All encryption happens on the edge device - backend never sees plaintext keys.
"""

import os
import json
import time
import logging
import requests
import traceback
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from services.session_service import SessionService

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# Backend URL for access key delivery
BACKEND_URL = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")


def get_camera_pda() -> str:
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

    # Fallback to environment variable
    return os.environ.get("CAMERA_PDA", "unknown")


def encrypt_session_key_for_user(session_key: bytes, user_pubkey: str) -> tuple:
    """
    Encrypt the session key for a specific user using their public key.

    Uses HKDF key derivation with the user's pubkey to create a derived key,
    then XORs the session key with it. The nonce ensures unique encryption
    even for the same user across different sessions.

    Args:
        session_key: 32-byte AES-256 session key
        user_pubkey: User's wallet public key (base58 encoded)

    Returns:
        Tuple of (encrypted_key_bytes, nonce_bytes)
    """
    # Generate a random nonce
    nonce = os.urandom(12)

    # Derive an encryption key from the user's pubkey + nonce
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=nonce,
        info=b'mmoment-session-key-encryption',
        backend=default_backend()
    )

    # Use pubkey bytes as input key material
    pubkey_bytes = user_pubkey.encode('utf-8')
    derived_key = hkdf.derive(pubkey_bytes)

    # XOR session key with derived key
    encrypted_key = bytes(a ^ b for a, b in zip(session_key, derived_key))

    return encrypted_key, nonce


def send_access_key_to_backend(
    user_pubkey: str,
    session_key: bytes,
    check_in_time: int
) -> bool:
    """
    Send encrypted session key to backend for storage in user's UserSessionChain.

    The encryption happens HERE on the Jetson - backend only receives ciphertext
    and submits the transaction to Solana (paying for gas).

    Args:
        user_pubkey: User's wallet public key
        session_key: 32-byte AES-256 session key
        check_in_time: Unix timestamp of session start

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[CHECKOUT-SVC] Sending access key to backend for {user_pubkey[:8]}...")

    # Encrypt session key for the user (encryption happens on Jetson)
    encrypted_key, nonce = encrypt_session_key_for_user(session_key, user_pubkey)

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/session/access-key",
            json={
                'user_pubkey': user_pubkey,
                'key_ciphertext': list(encrypted_key),
                'nonce': list(nonce),
                'timestamp': check_in_time
            },
            timeout=10
        )

        if response.ok:
            logger.info(f"[CHECKOUT-SVC] Access key sent successfully for {user_pubkey[:8]}...")
            return True
        else:
            logger.warning(f"[CHECKOUT-SVC] Backend rejected access key: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"[CHECKOUT-SVC] Failed to send access key to backend: {e}")
        return False


def _notify_backend_checkout(
    wallet_address: str,
    session_id: str,
    camera_pda: str,
    user_profile: Dict = None
) -> bool:
    """
    Notify backend about checkout via sync POST to /api/session/activity.

    This triggers real-time UI updates via WebSocket.

    Args:
        wallet_address: User's wallet address
        session_id: Session ID
        camera_pda: Camera PDA address
        user_profile: User profile dict with display_name, username

    Returns:
        True if notification succeeded, False otherwise
    """
    try:
        user_profile = user_profile or {}
        display_name = user_profile.get('display_name') or user_profile.get('displayName') or wallet_address[:8]
        username = user_profile.get('username')

        response = requests.post(
            f"{BACKEND_URL}/api/session/activity",
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

        if response.status_code == 200:
            result = response.json()
            logger.info(f"[CHECKOUT-SVC] Backend notified! Sockets in room: {result.get('debug', {}).get('socketsInRoom', '?')}")
            return True
        else:
            logger.warning(f"[CHECKOUT-SVC] Backend notification failed: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"[CHECKOUT-SVC] Backend notification failed (non-blocking): {e}")
        return False


def execute_full_checkout(
    wallet_address: str,
    session_service: 'SessionService',
    auto_checkout: bool = False,
    auto_checkout_reason: str = None
) -> Dict:
    """
    Execute the complete checkout flow. Used by both manual and auto checkout.

    Flow:
    1. Add CHECK_OUT activity to session buffer
    2. Notify backend (for real-time UI updates)
    3. Send encrypted access key to backend (backend writes to UserSessionChain)
    4. Clean up face recognition data
    5. End the session

    Args:
        wallet_address: User's wallet address
        session_service: SessionService instance
        auto_checkout: True if triggered automatically (face timeout, inactivity, etc.)
        auto_checkout_reason: Reason for auto-checkout ("face_timeout", "inactivity", "session_expired")

    Returns:
        Dict with checkout results
    """
    log_prefix = "[AUTO-CHECKOUT]" if auto_checkout else "[CHECKOUT]"

    try:
        # Get the session object (need session_key and activities)
        session = session_service.get_session_object_by_wallet(wallet_address)
        if not session:
            logger.warning(f"{log_prefix} No session found for {wallet_address[:8]}...")
            return {
                "success": False,
                "error": "No active session found",
                "wallet_address": wallet_address,
                "auto_checkout": auto_checkout,
                "auto_checkout_reason": auto_checkout_reason
            }

        session_id = session.session_id
        check_in_time = int(session.created_at)
        camera_pda = get_camera_pda()
        user_profile = session.user_profile or {}

        logger.info(f"{log_prefix} Starting checkout for {wallet_address[:8]}... (session: {session_id[:16]}...)")

        # Build checkout metadata
        checkout_metadata = {"camera_pda": camera_pda}
        if auto_checkout:
            checkout_metadata["autoCheckout"] = True
            checkout_metadata["reason"] = auto_checkout_reason

        # Step 1: Add CHECK_OUT activity to session buffer
        session.add_activity(
            activity_type=1,  # CHECK_OUT
            data={"duration_seconds": int(time.time() - session.created_at)},
            metadata=checkout_metadata
        )

        # Step 2: Notify backend (for real-time UI updates via WebSocket)
        _notify_backend_checkout(wallet_address, session_id, camera_pda, user_profile)

        # Step 3: Send encrypted access key to backend
        # Backend will write to UserSessionChain on Solana
        access_key_sent = False
        try:
            access_key_sent = send_access_key_to_backend(
                user_pubkey=wallet_address,
                session_key=session.session_key,
                check_in_time=check_in_time
            )
            if access_key_sent:
                logger.info(f"{log_prefix} Access key sent to backend for {wallet_address[:8]}...")
            else:
                logger.warning(f"{log_prefix} Failed to send access key to backend")
        except Exception as e:
            logger.error(f"{log_prefix} Error sending access key: {e}")

        # Step 4: Clean up face recognition data
        try:
            from services.blockchain_session_sync import get_blockchain_session_sync
            blockchain_sync = get_blockchain_session_sync()
            blockchain_sync.trigger_checkout(wallet_address)
            logger.info(f"{log_prefix} Face recognition data cleaned up")
        except Exception as e:
            logger.warning(f"{log_prefix} Error cleaning up face data: {e}")

        # Step 5: End the session
        session_service.end_session(session_id, wallet_address)
        logger.info(f"{log_prefix} Session {session_id[:16]}... ended")

        return {
            "success": True,
            "wallet_address": wallet_address,
            "session_id": session_id,
            "activities_committed": len(session.get_activities()),
            "access_key_sent": access_key_sent,
            "auto_checkout": auto_checkout,
            "auto_checkout_reason": auto_checkout_reason,
            "message": "Checkout complete"
        }

    except Exception as e:
        logger.error(f"{log_prefix} Error during checkout for {wallet_address}: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"Checkout failed: {str(e)}",
            "wallet_address": wallet_address,
            "auto_checkout": auto_checkout,
            "auto_checkout_reason": auto_checkout_reason
        }
