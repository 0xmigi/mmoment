"""
CameraTimeline Writer Service (Compressed Accounts via Light Protocol)

Writes encrypted session activities to compressed TimelineEntry accounts on Solana.
Uses a two-phase relay through the backend:
  Phase 1: POST encrypted data -> backend builds tx, signs with payer, returns serialized tx
  Phase 2: Jetson signs tx with device key -> POST back -> backend submits

The device private key NEVER leaves the Jetson.

Encryption model:
- All session activities are combined into a single JSON blob
- Encrypted once with AES-256-GCM (one key, one nonce)
- Access grants: each checked-in user gets a sealed-box-encrypted copy of the AES key
- Access grants blob format:
    [2 bytes: grant count (u16 LE)]
    For each grant:
      [32 bytes: user Solana pubkey]
      [2 bytes: encrypted key length (u16 LE)]
      [N bytes: sealed-box-encrypted AES key]
"""

import os
import struct
import json
import base64
import logging
import requests
from typing import List, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")


def _encrypt_key_for_user(activity_key: bytes, user_pubkey: str) -> bytes:
    """
    Encrypt the AES key for a user via Ed25519 -> X25519 sealed box.
    Falls back to HKDF-based wrapping if PyNaCl is unavailable.
    """
    try:
        from nacl.public import SealedBox
        from nacl.signing import VerifyKey
        import base58

        pubkey_bytes = base58.b58decode(user_pubkey)
        verify_key = VerifyKey(pubkey_bytes)
        x25519_pubkey = verify_key.to_curve25519_public_key()
        sealed_box = SealedBox(x25519_pubkey)
        return sealed_box.encrypt(activity_key)

    except ImportError:
        logger.warning("PyNaCl not installed, using HKDF fallback for key wrapping")
    except Exception as e:
        logger.warning(f"NaCl encryption failed for {user_pubkey[:8]}...: {e}, using fallback")

    # Fallback: HKDF-based key wrapping
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'mmoment-activity-encryption-v1',
        info=b'activity-key-wrap'
    )
    wrapping_key = hkdf.derive(user_pubkey.encode('utf-8'))
    return bytes(a ^ b for a, b in zip(activity_key, wrapping_key))


def _build_access_grants_blob(activity_key: bytes, users_present: List[str]) -> bytes:
    """
    Build the access grants blob: encrypted AES key for each user.

    Format:
        [2 bytes: grant count (u16 LE)]
        For each grant:
          [32 bytes: user Solana pubkey raw bytes]
          [2 bytes: encrypted key length (u16 LE)]
          [N bytes: sealed-box-encrypted AES key]
    """
    import base58

    grants_data = struct.pack('<H', len(users_present))

    for user_pubkey in users_present:
        try:
            pubkey_bytes = base58.b58decode(user_pubkey)
            encrypted_key = _encrypt_key_for_user(activity_key, user_pubkey)

            grants_data += pubkey_bytes  # 32 bytes
            grants_data += struct.pack('<H', len(encrypted_key))
            grants_data += encrypted_key
        except Exception as e:
            logger.warning(f"Failed to create access grant for {user_pubkey[:8]}...: {e}")

    return grants_data


def _encrypt_session_blob(
    raw_activities: List[Dict],
    users_present: List[str],
) -> tuple:
    """
    Encrypt all session activities into a single AES-256-GCM blob.

    Returns:
        (encrypted_payload bytes, nonce bytes, access_grants_blob bytes, activity_count int)
    """
    # Generate random AES-256 key and nonce
    activity_key = os.urandom(32)
    nonce = os.urandom(12)

    # Serialize all activities into one JSON blob
    plaintext = json.dumps(raw_activities, separators=(',', ':')).encode('utf-8')

    # Encrypt with AES-256-GCM
    aesgcm = AESGCM(activity_key)
    encrypted_payload = aesgcm.encrypt(nonce, plaintext, None)

    # Build access grants for all present users
    access_grants_blob = _build_access_grants_blob(activity_key, users_present)

    return encrypted_payload, nonce, access_grants_blob, len(raw_activities)


def write_session_to_timeline(
    device_keypair,
    camera_pda: str,
    raw_activities: List[Dict],
    users_present: List[str],
) -> Optional[str]:
    """
    Encrypt session activities and write to compressed CameraTimeline via two-phase relay.

    Args:
        device_keypair: Device's solders Keypair (private key stays local)
        camera_pda: Camera PDA (base58 string)
        raw_activities: List of raw activity dicts from the session
        users_present: List of wallet addresses who were checked in during this session

    Returns:
        Transaction signature on success, None on failure
    """
    if not raw_activities:
        logger.info("[TIMELINE] No activities to write")
        return None

    if not users_present:
        logger.warning("[TIMELINE] No users present for access grants, skipping")
        return None

    try:
        # Step 1: Encrypt all activities into a single blob
        encrypted_payload, nonce, access_grants_blob, activity_count = _encrypt_session_blob(
            raw_activities, users_present
        )

        logger.info(
            f"[TIMELINE] Encrypted {activity_count} activities into {len(encrypted_payload)} bytes "
            f"for {len(users_present)} users"
        )

        # Step 2: Phase 1 — send encrypted data to backend, get back payer-signed tx
        device_pubkey_str = str(device_keypair.pubkey())

        prepare_resp = requests.post(
            f"{BACKEND_URL}/relay/prepare-timeline-entry",
            json={
                'camera_address': camera_pda,
                'device_pubkey': device_pubkey_str,
                'encrypted_payload': base64.b64encode(encrypted_payload).decode('utf-8'),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'access_grants_blob': base64.b64encode(access_grants_blob).decode('utf-8'),
                'activity_count': activity_count,
            },
            timeout=30
        )

        if not prepare_resp.ok:
            logger.error(f"[TIMELINE] Prepare failed: {prepare_resp.status_code} - {prepare_resp.text}")
            return None

        prepare_data = prepare_resp.json()
        tx_base64 = prepare_data['transaction']
        device_signer_index = prepare_data['device_signer_index']
        entry_index = prepare_data.get('entry_index', '?')

        logger.info(f"[TIMELINE] Prepared tx for entry_index={entry_index}, device_signer_index={device_signer_index}")

        # Step 3: Phase 2 — sign the transaction message with device key, send back
        tx_bytes = base64.b64decode(tx_base64)

        # Deserialize the versioned transaction to get the message for signing
        from solders.transaction import VersionedTransaction

        tx = VersionedTransaction.from_bytes(tx_bytes)

        # Sign the message with the device key
        message_bytes = bytes(tx.message)
        device_signature = device_keypair.sign_message(message_bytes)

        # Insert device signature at the correct index
        signatures = list(tx.signatures)
        from solders.signature import Signature
        signatures[device_signer_index] = Signature(bytes(device_signature))

        # Reconstruct the transaction with both signatures
        signed_tx = VersionedTransaction.populate(tx.message, signatures)
        signed_tx_bytes = bytes(signed_tx)

        signed_tx_base64 = base64.b64encode(signed_tx_bytes).decode('utf-8')

        # Submit the fully-signed transaction
        submit_resp = requests.post(
            f"{BACKEND_URL}/relay/submit-timeline-entry",
            json={
                'signed_transaction': signed_tx_base64,
            },
            timeout=30
        )

        if not submit_resp.ok:
            logger.error(f"[TIMELINE] Submit failed: {submit_resp.status_code} - {submit_resp.text}")
            return None

        result = submit_resp.json()
        signature = result.get('signature', '?')
        logger.info(f"[TIMELINE] Timeline entry written. entry_index={entry_index}, tx={signature[:16]}...")
        return signature

    except Exception as e:
        logger.error(f"[TIMELINE] Failed to write timeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
