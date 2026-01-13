"""
Checkout Service - Phase 3 Privacy Architecture

Handles the checkout flow:
1. Encrypt activities with session key
2. Write encrypted activities to CameraTimeline (via Solana middleware)
3. Send encrypted session key to backend

This breaks the visible on-chain link between users and cameras.
"""

import os
import json
import time
import logging
import requests
from typing import List, Dict, Optional

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

# Backend URL for access key delivery
BACKEND_URL = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")

# Solana Middleware URL
SOLANA_MIDDLEWARE_URL = os.environ.get("SOLANA_MIDDLEWARE_URL", "http://localhost:5001")


def encrypt_activity(activity: Dict, session_key: bytes) -> Dict:
    """
    Encrypt a single activity using AES-256-GCM.

    Args:
        activity: Activity dict with timestamp, activity_type, data, metadata
        session_key: 32-byte AES-256 key

    Returns:
        Encrypted activity in format expected by CameraTimeline
    """
    # Generate random nonce (12 bytes for GCM)
    nonce = os.urandom(12)

    # Serialize activity content (data + metadata)
    plaintext = json.dumps({
        'data': activity.get('data', {}),
        'metadata': activity.get('metadata', {})
    }).encode('utf-8')

    # Encrypt using AES-256-GCM
    cipher = Cipher(
        algorithms.AES(session_key),
        modes.GCM(nonce),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    # Append auth tag to ciphertext (GCM produces 16-byte tag)
    encrypted_content = ciphertext + encryptor.tag

    return {
        'timestamp': activity['timestamp'],
        'activity_type': activity['activity_type'],
        'encrypted_content': list(encrypted_content),  # Convert to list for JSON
        'nonce': list(nonce),
        'access_grants': []  # Populated for multi-user sessions
    }


def encrypt_session_key_for_user(session_key: bytes, user_pubkey: str) -> tuple:
    """
    Encrypt the session key for a specific user using their public key.

    For now, we use a simple XOR-based encryption with HKDF.
    In production, this should use X25519 key exchange.

    Args:
        session_key: 32-byte AES-256 session key
        user_pubkey: User's wallet public key (base58 encoded)

    Returns:
        Tuple of (encrypted_key_bytes, nonce_bytes)
    """
    import hashlib
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    # Generate a random nonce
    nonce = os.urandom(12)

    # Derive an encryption key from the user's pubkey + nonce
    # Note: This is a simplified approach. Production should use X25519.
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


def create_access_grant(session_key: bytes, user_pubkey: str) -> bytes:
    """
    Create an access grant (encrypted session key) for a user.

    Args:
        session_key: 32-byte AES-256 session key
        user_pubkey: User's wallet public key

    Returns:
        Encrypted session key bytes
    """
    encrypted_key, nonce = encrypt_session_key_for_user(session_key, user_pubkey)
    # Combine nonce + encrypted key for the access grant
    return nonce + encrypted_key


def encrypt_and_write_activities(
    session_key: bytes,
    activities: List[Dict],
    checked_in_users: List[str]
) -> Optional[str]:
    """
    Encrypt all activities and write them to the CameraTimeline.

    Args:
        session_key: 32-byte AES-256 session key
        activities: List of activity dicts
        checked_in_users: List of user wallet addresses for access grants

    Returns:
        Transaction signature if successful, None otherwise
    """
    if not activities:
        logger.info("[CHECKOUT-SVC] No activities to write")
        return None

    logger.info(f"[CHECKOUT-SVC] Encrypting {len(activities)} activities for {len(checked_in_users)} user(s)")

    # Encrypt each activity
    encrypted_activities = []
    for activity in activities:
        encrypted = encrypt_activity(activity, session_key)

        # Add access grants for each checked-in user
        for user_pubkey in checked_in_users:
            access_grant = create_access_grant(session_key, user_pubkey)
            encrypted['access_grants'].append(list(access_grant))

        encrypted_activities.append(encrypted)

    logger.info(f"[CHECKOUT-SVC] Encrypted {len(encrypted_activities)} activities")

    # Send to Solana middleware to write to CameraTimeline
    try:
        response = requests.post(
            f"{SOLANA_MIDDLEWARE_URL}/api/blockchain/write-camera-timeline",
            json={'activities': encrypted_activities},
            timeout=30
        )

        if response.ok:
            result = response.json()
            tx_signature = result.get('signature')
            logger.info(f"[CHECKOUT-SVC] Timeline write successful: {tx_signature[:16] if tx_signature else 'unknown'}...")
            return tx_signature
        else:
            logger.error(f"[CHECKOUT-SVC] Timeline write failed: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"[CHECKOUT-SVC] Failed to reach Solana middleware: {e}")
        return None


def send_access_key_to_backend(
    user_pubkey: str,
    session_key: bytes,
    check_in_time: int
) -> bool:
    """
    Send encrypted session key to backend for storage in user's keychain.

    Args:
        user_pubkey: User's wallet public key
        session_key: 32-byte AES-256 session key
        check_in_time: Unix timestamp of session start

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[CHECKOUT-SVC] Sending access key to backend for {user_pubkey[:8]}...")

    # Encrypt session key for the user
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
        # TODO: Queue for retry
        return False


def perform_checkout(
    wallet_address: str,
    session_key: bytes,
    activities: List[Dict],
    check_in_time: int,
    checked_in_users: List[str] = None
) -> Dict:
    """
    Perform the complete checkout flow.

    Args:
        wallet_address: User's wallet address
        session_key: 32-byte AES-256 session key
        activities: List of activity dicts from session
        check_in_time: Unix timestamp of session start
        checked_in_users: List of all users present (for multi-user access grants)

    Returns:
        Dict with checkout results
    """
    if checked_in_users is None:
        checked_in_users = [wallet_address]

    logger.info(f"[CHECKOUT-SVC] Starting checkout for {wallet_address[:8]}... with {len(activities)} activities")

    results = {
        'wallet_address': wallet_address,
        'activities_count': len(activities),
        'timeline_tx': None,
        'access_key_sent': False,
        'errors': []
    }

    # Step 1: Encrypt and write activities to CameraTimeline
    if activities:
        try:
            tx_signature = encrypt_and_write_activities(
                session_key=session_key,
                activities=activities,
                checked_in_users=checked_in_users
            )
            results['timeline_tx'] = tx_signature
        except Exception as e:
            error_msg = f"Failed to write activities: {str(e)}"
            logger.error(f"[CHECKOUT-SVC] {error_msg}")
            results['errors'].append(error_msg)

    # Step 2: Send access key to backend
    try:
        results['access_key_sent'] = send_access_key_to_backend(
            user_pubkey=wallet_address,
            session_key=session_key,
            check_in_time=check_in_time
        )
    except Exception as e:
        error_msg = f"Failed to send access key: {str(e)}"
        logger.error(f"[CHECKOUT-SVC] {error_msg}")
        results['errors'].append(error_msg)

    logger.info(f"[CHECKOUT-SVC] Checkout complete for {wallet_address[:8]}...: tx={results['timeline_tx'] is not None}, key_sent={results['access_key_sent']}")

    return results
