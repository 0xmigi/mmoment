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

Chunking:
- If the encrypted blob fits in a single Solana transaction (~450 bytes payload),
  it is sent as chunk_index=0, total_chunks=1.
- If larger, the ciphertext is split across multiple transactions.
  Chunk 0 carries nonce + access_grants_blob. Subsequent chunks carry empty grants/nonce.
  Client reassembles all chunks (ordered by entry_index), concatenates ciphertext, decrypts once.
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

# Max encrypted payload bytes per chunk.
# Transaction budget: 1232 total - ~600 overhead (sigs, accounts, blockhash)
#   - ~170 ix overhead (discriminator, proof, tree info, nonce, vec headers, chunk fields)
#   - ~116 access grants (1 user; scales ~82 per additional user)
# Conservatively allow ~450 bytes for payload in chunk 0, ~550 in subsequent chunks.
MAX_PAYLOAD_CHUNK0 = 450
MAX_PAYLOAD_CONTINUATION = 550


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


def _relay_chunk(
    device_keypair,
    camera_pda: str,
    encrypted_chunk: bytes,
    nonce: bytes,
    access_grants_blob: bytes,
    activity_count: int,
    chunk_index: int,
    total_chunks: int,
) -> Optional[str]:
    """
    Send a single chunk through the two-phase relay.

    Returns transaction signature on success, None on failure.
    """
    device_pubkey_str = str(device_keypair.pubkey())

    # Phase 1: Prepare — backend builds tx, signs with payer
    prepare_resp = requests.post(
        f"{BACKEND_URL}/relay/prepare-timeline-entry",
        json={
            'camera_address': camera_pda,
            'device_pubkey': device_pubkey_str,
            'encrypted_payload': base64.b64encode(encrypted_chunk).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'access_grants_blob': base64.b64encode(access_grants_blob).decode('utf-8'),
            'activity_count': activity_count,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
        },
        timeout=30
    )

    if not prepare_resp.ok:
        logger.error(f"[TIMELINE] Prepare failed (chunk {chunk_index}/{total_chunks}): "
                      f"{prepare_resp.status_code} - {prepare_resp.text}")
        return None

    prepare_data = prepare_resp.json()
    tx_base64 = prepare_data['transaction']
    device_signer_index = prepare_data['device_signer_index']
    entry_index = prepare_data.get('entry_index', '?')

    logger.info(f"[TIMELINE] Prepared chunk {chunk_index}/{total_chunks}, "
                f"entry_index={entry_index}, device_signer_index={device_signer_index}")

    # Phase 2: Sign with device key, submit
    from solders.transaction import VersionedTransaction
    from solders.signature import Signature

    tx_bytes = base64.b64decode(tx_base64)
    tx = VersionedTransaction.from_bytes(tx_bytes)

    # Use the exact message bytes from the backend to avoid re-serialization mismatches
    # that would invalidate the payer's signature
    message_bytes = base64.b64decode(prepare_data['message_bytes'])
    device_signature = device_keypair.sign_message(message_bytes)

    signatures = list(tx.signatures)
    signatures[device_signer_index] = Signature(bytes(device_signature))

    signed_tx = VersionedTransaction.populate(tx.message, signatures)
    signed_tx_base64 = base64.b64encode(bytes(signed_tx)).decode('utf-8')

    submit_resp = requests.post(
        f"{BACKEND_URL}/relay/submit-timeline-entry",
        json={'signed_transaction': signed_tx_base64},
        timeout=30
    )

    if not submit_resp.ok:
        logger.error(f"[TIMELINE] Submit failed (chunk {chunk_index}/{total_chunks}): "
                      f"{submit_resp.status_code} - {submit_resp.text}")
        return None

    result = submit_resp.json()
    signature = result.get('signature', '?')
    logger.info(f"[TIMELINE] Chunk {chunk_index}/{total_chunks} written. "
                f"entry_index={entry_index}, tx={signature[:16]}...")
    return signature


def write_session_to_timeline(
    device_keypair,
    camera_pda: str,
    raw_activities: List[Dict],
    users_present: List[str],
) -> Optional[str]:
    """
    Encrypt session activities and write to compressed CameraTimeline via two-phase relay.
    Automatically chunks large payloads across multiple transactions.

    Args:
        device_keypair: Device's solders Keypair (private key stays local)
        camera_pda: Camera PDA (base58 string)
        raw_activities: List of raw activity dicts from the session
        users_present: List of wallet addresses who were checked in during this session

    Returns:
        Transaction signature of the last chunk on success, None on failure
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

        # Step 2: Determine chunking
        if len(encrypted_payload) <= MAX_PAYLOAD_CHUNK0:
            # Single chunk — everything fits in one transaction
            return _relay_chunk(
                device_keypair, camera_pda,
                encrypted_payload, nonce, access_grants_blob,
                activity_count, chunk_index=0, total_chunks=1,
            )

        # Multi-chunk: split ciphertext across transactions
        # Chunk 0 carries nonce + access_grants, subsequent chunks carry empty nonce/grants
        chunks = []
        offset = 0

        # First chunk has smaller payload budget (nonce + grants take space)
        first_size = MAX_PAYLOAD_CHUNK0
        chunks.append(encrypted_payload[:first_size])
        offset = first_size

        # Remaining chunks get more space (empty nonce/grants)
        while offset < len(encrypted_payload):
            end = min(offset + MAX_PAYLOAD_CONTINUATION, len(encrypted_payload))
            chunks.append(encrypted_payload[offset:end])
            offset = end

        total_chunks = len(chunks)
        logger.info(f"[TIMELINE] Splitting {len(encrypted_payload)} bytes into {total_chunks} chunks")

        # Empty nonce/grants for continuation chunks
        empty_nonce = b'\x00' * 12
        empty_grants = struct.pack('<H', 0)  # 0 grants

        last_signature = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                sig = _relay_chunk(
                    device_keypair, camera_pda,
                    chunk, nonce, access_grants_blob,
                    activity_count, chunk_index=i, total_chunks=total_chunks,
                )
            else:
                sig = _relay_chunk(
                    device_keypair, camera_pda,
                    chunk, empty_nonce, empty_grants,
                    activity_count, chunk_index=i, total_chunks=total_chunks,
                )

            if sig is None:
                logger.error(f"[TIMELINE] Chunk {i}/{total_chunks} failed, aborting remaining chunks")
                return None

            last_signature = sig

        logger.info(f"[TIMELINE] All {total_chunks} chunks written successfully")
        return last_signature

    except Exception as e:
        logger.error(f"[TIMELINE] Failed to write timeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
