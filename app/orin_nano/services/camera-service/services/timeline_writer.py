"""
CameraTimeline Writer Service (Compressed Accounts via Light Protocol)

Writes encrypted camera activities to compressed TimelineEntry accounts on Solana.
Uses a two-phase relay through the backend:
  Phase 1: POST encrypted data -> backend builds tx, signs with payer, returns serialized tx
  Phase 2: Jetson signs tx with device key -> POST back -> backend submits

The device private key NEVER leaves the Jetson.

Encryption model:
- All camera activities since the last write are serialized into a compact binary blob
- Encrypted once with AES-256-GCM (one key, one nonce)
- Access grants are NOT stored in the timeline entry (no wallet linkage on-chain)
- Users get decryption keys via their UserSessionChain at checkout

Binary blob format (v1):
  Header:
    [1 byte]  version (0x01)
    [1 byte]  num_participants
    [N x 32]  participant Solana pubkeys (inside encrypted blob, invisible on-chain)
  Activities:
    [2 bytes] num_activities (u16 LE)
    Per activity:
      [4 bytes] timestamp (u32 LE, unix seconds)
      [1 byte]  activity_type
      [1 byte]  participant_index (index into header table)
      [2 bytes] data_length (u16 LE)
      [N bytes] type-specific binary data

  Type-specific data:
    Check-in  (0): empty
    Check-out (1): [u32 LE duration_seconds]
    Photo     (2): [u16 LE len][file_id utf8]
    Video     (3): [u32 LE duration][u16 LE len][file_id utf8]
    Stream    (4): [u16 LE len][stream_id utf8][u16 LE len][playback_id utf8]
    CV        (50): [u16 LE len][app_name utf8][u16 LE len][results_json utf8]
    Other     (255): [u16 LE len][json utf8]
"""

import os
import struct
import base64
import json
import logging
import requests
from typing import List, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")

# Max encrypted payload bytes per chunk.
# Transaction budget: 1232 total bytes.
# Fixed overhead: sigs(128) + header/blockhash(40) + accounts(~448) + ix_overhead(~174) = ~790
# Remaining: ~442 bytes for encrypted_payload (access_grants_blob is empty = 2 bytes)
MAX_PAYLOAD_CHUNK = 420


def _encode_string(s: str) -> bytes:
    """Encode a string as [u16 LE length][utf8 bytes]."""
    b = s.encode('utf-8')
    return struct.pack('<H', len(b)) + b


def _encode_activity_data(activity_type: int, data: Dict) -> bytes:
    """
    Encode type-specific activity data into minimal binary.
    """
    if activity_type == 0:  # Check-in
        return b''

    elif activity_type == 1:  # Check-out
        duration = int(data.get('duration_seconds', 0))
        return struct.pack('<I', duration)

    elif activity_type == 2:  # Photo
        file_id = data.get('pipe_file_id', '') or data.get('file_id', '') or ''
        return _encode_string(file_id)

    elif activity_type == 3:  # Video
        duration = int(data.get('duration_seconds', 0))
        file_id = data.get('pipe_file_id', '') or data.get('file_id', '') or ''
        return struct.pack('<I', duration) + _encode_string(file_id)

    elif activity_type == 4:  # Stream
        stream_id = data.get('stream_id', '') or ''
        playback_id = data.get('playback_id', '') or ''
        return _encode_string(stream_id) + _encode_string(playback_id)

    elif activity_type == 50:  # CV Activity
        app_name = data.get('app_name', '') or ''
        results = data.get('results', [])
        results_json = json.dumps(results, separators=(',', ':'))
        return _encode_string(app_name) + _encode_string(results_json)

    else:  # Unknown/Other — dump as compact JSON
        return _encode_string(json.dumps(data, separators=(',', ':')))


def _serialize_activities_binary(raw_activities: List[Dict]) -> bytes:
    """
    Serialize activities into compact binary format with participant table.

    The participant table stores full wallet pubkeys once. Activities reference
    participants by 1-byte index, saving 31 bytes per activity vs repeating the pubkey.
    """
    import base58

    # Build participant table from all activities
    participant_order = []
    participant_index = {}

    for activity in raw_activities:
        data = activity.get('data', {})
        # Extract the wallet from whichever field carries it
        wallet = (data.get('user') or data.get('captured_by') or
                  data.get('recorded_by') or data.get('started_by') or
                  data.get('ended_by') or '')
        if wallet and wallet not in participant_index:
            participant_index[wallet] = len(participant_order)
            participant_order.append(wallet)

    # Header: version + num_participants + pubkeys
    buf = struct.pack('<B', 1)  # version
    buf += struct.pack('<B', len(participant_order))
    for wallet in participant_order:
        try:
            buf += base58.b58decode(wallet)  # 32 bytes
        except Exception:
            buf += b'\x00' * 32  # fallback

    # Activities
    buf += struct.pack('<H', len(raw_activities))

    for activity in raw_activities:
        timestamp = activity.get('timestamp', 0)
        if timestamp > 2_000_000_000:
            timestamp = timestamp // 1000

        activity_type = activity.get('activity_type', 255)
        data = activity.get('data', {})

        # Resolve participant index
        wallet = (data.get('user') or data.get('captured_by') or
                  data.get('recorded_by') or data.get('started_by') or
                  data.get('ended_by') or '')
        p_idx = participant_index.get(wallet, 0)

        # Encode type-specific data
        type_data = _encode_activity_data(activity_type, data)

        buf += struct.pack('<I', timestamp)
        buf += struct.pack('<B', activity_type)
        buf += struct.pack('<B', p_idx)
        buf += struct.pack('<H', len(type_data))
        buf += type_data

    return buf


def _encrypt_activities(raw_activities: List[Dict]) -> tuple:
    """
    Serialize and encrypt all camera activities into a single AES-256-GCM blob.

    Returns:
        (encrypted_payload bytes, nonce bytes, activity_count int)
    """
    activity_key = os.urandom(32)
    nonce = os.urandom(12)

    plaintext = _serialize_activities_binary(raw_activities)
    logger.info(f"[TIMELINE] Binary serialized {len(raw_activities)} activities into {len(plaintext)} bytes plaintext")

    aesgcm = AESGCM(activity_key)
    encrypted_payload = aesgcm.encrypt(nonce, plaintext, None)

    return encrypted_payload, nonce, len(raw_activities)


def _relay_chunk(
    device_keypair,
    camera_pda: str,
    encrypted_chunk: bytes,
    nonce: bytes,
    activity_count: int,
    chunk_index: int,
    total_chunks: int,
) -> Optional[str]:
    """
    Send a single chunk through the two-phase relay.
    Returns transaction signature on success, None on failure.
    """
    device_pubkey_str = str(device_keypair.pubkey())

    # Empty access grants — keys distributed via UserSessionChain, not timeline
    empty_grants = struct.pack('<H', 0)

    # Phase 1: Prepare — backend builds tx, signs with payer
    prepare_resp = requests.post(
        f"{BACKEND_URL}/relay/prepare-timeline-entry",
        json={
            'camera_address': camera_pda,
            'device_pubkey': device_pubkey_str,
            'encrypted_payload': base64.b64encode(encrypted_chunk).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'access_grants_blob': base64.b64encode(empty_grants).decode('utf-8'),
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
) -> Optional[str]:
    """
    Encrypt camera activities and write to compressed CameraTimeline via two-phase relay.
    Automatically chunks large payloads across multiple transactions.

    Args:
        device_keypair: Device's solders Keypair (private key stays local)
        camera_pda: Camera PDA (base58 string)
        raw_activities: All camera activities since last write (from ALL users)

    Returns:
        Transaction signature of the last chunk on success, None on failure
    """
    if not raw_activities:
        logger.info("[TIMELINE] No activities to write")
        return None

    try:
        # Step 1: Encrypt all activities into a single blob
        encrypted_payload, nonce, activity_count = _encrypt_activities(raw_activities)

        logger.info(
            f"[TIMELINE] Encrypted {activity_count} activities into {len(encrypted_payload)} bytes"
        )

        # Step 2: Determine chunking
        if len(encrypted_payload) <= MAX_PAYLOAD_CHUNK:
            return _relay_chunk(
                device_keypair, camera_pda,
                encrypted_payload, nonce,
                activity_count, chunk_index=0, total_chunks=1,
            )

        # Multi-chunk: split ciphertext across transactions
        # Chunk 0 carries the real nonce; continuation chunks carry zero nonce.
        # Client reassembles chunks by entry_index, concatenates ciphertext, decrypts once.
        chunks = []
        offset = 0
        while offset < len(encrypted_payload):
            end = min(offset + MAX_PAYLOAD_CHUNK, len(encrypted_payload))
            chunks.append(encrypted_payload[offset:end])
            offset = end

        total_chunks = len(chunks)
        logger.info(f"[TIMELINE] Splitting {len(encrypted_payload)} bytes into {total_chunks} chunks")

        empty_nonce = b'\x00' * 12
        last_signature = None

        for i, chunk in enumerate(chunks):
            sig = _relay_chunk(
                device_keypair, camera_pda,
                chunk, nonce if i == 0 else empty_nonce,
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
