"""
CameraTimeline Writer Service

Writes encrypted activities to the CameraTimeline PDA on Solana at checkout.
The device key signs to authenticate the data (proving it came from this camera),
and the backend co-signs to pay gas fees.

Flow:
1. At checkout, collect all encrypted activities from the session
2. Build writeToCameraTimeline transaction
3. Device signs (authenticates the data)
4. Send to backend for fee payer signature + submission
5. Backend clears session_activity_buffers on confirmation

Dependencies:
- solana (pip install solana)
- solders (pip install solders)
- requests
"""

import os
import struct
import hashlib
import base64
import logging
import requests
from typing import List, Dict, Optional

from solders.message import Message
from solders.transaction import Transaction
from solders.hash import Hash
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYSTEM_PROGRAM_ID

logger = logging.getLogger(__name__)

# Camera Network Program ID
PROGRAM_ID = Pubkey.from_string("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL")

# Backend fee payer public key (same one used for competition settlement)
BACKEND_PAYER_PUBKEY = Pubkey.from_string("9k5MGiM9Xqx8f2362M1B2rH5uMKFFVNuXaCDKyTsFXep")

# Anchor discriminator: sha256("global:write_to_camera_timeline")[:8]
WRITE_TIMELINE_DISCRIMINATOR = hashlib.sha256(b"global:write_to_camera_timeline").digest()[:8]

BACKEND_URL = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")


def _borsh_serialize_activities(activities: List[Dict]) -> bytes:
    """
    Borsh-serialize Vec<ActivityData> for the writeToCameraTimeline instruction.

    ActivityData layout:
        timestamp: i64 (8 bytes, little-endian)
        activity_type: u8 (1 byte)
        encrypted_content: Vec<u8> (4-byte length prefix + data)
        nonce: [u8; 12] (fixed 12 bytes, no length prefix)
        access_grants: Vec<Vec<u8>> (4-byte outer length + for each: 4-byte length + data)
    """
    data = struct.pack('<I', len(activities))  # Vec length (u32 LE)

    for activity in activities:
        # timestamp: i64
        data += struct.pack('<q', activity['timestamp'])

        # activity_type: u8
        data += struct.pack('<B', activity['activity_type'])

        # encrypted_content: Vec<u8>
        content = activity['encrypted_content']
        data += struct.pack('<I', len(content)) + content

        # nonce: [u8; 12] (fixed-size array, no length prefix)
        nonce = activity['nonce']
        assert len(nonce) == 12, f"Nonce must be 12 bytes, got {len(nonce)}"
        data += nonce

        # access_grants: Vec<Vec<u8>>
        grants = activity['access_grants']
        data += struct.pack('<I', len(grants))
        for grant in grants:
            data += struct.pack('<I', len(grant)) + grant

    return data


def write_session_to_timeline(
    device_keypair: Keypair,
    camera_pda: str,
    encrypted_activities: List[Dict],
    session_id: str
) -> Optional[str]:
    """
    Build writeToCameraTimeline transaction, sign with device key, relay through backend.

    Args:
        device_keypair: Device's Solana ed25519 keypair
        camera_pda: Camera PDA (base58 string)
        encrypted_activities: List of encrypted activity dicts from the session.
            Each dict has: encryptedContent (b64), nonce (b64), accessGrants, activityType, timestamp
        session_id: Session ID (for backend buffer cleanup after on-chain write)

    Returns:
        Transaction signature on success, None on failure
    """
    if not encrypted_activities:
        logger.info("[TIMELINE] No encrypted activities to write")
        return None

    try:
        camera_pubkey = Pubkey.from_string(camera_pda)

        # Derive CameraTimeline PDA
        camera_timeline_pda, _bump = Pubkey.find_program_address(
            [b"camera-timeline", bytes(camera_pubkey)],
            PROGRAM_ID
        )

        # Convert encrypted activities to raw bytes for Borsh serialization
        activities_for_tx = []
        for ea in encrypted_activities:
            # Normalize timestamp to seconds (blockchain uses i64 seconds)
            ts = ea['timestamp']
            if ts > 10_000_000_000:
                ts = ts // 1000

            activities_for_tx.append({
                'timestamp': ts,
                'activity_type': ea['activityType'],
                'encrypted_content': base64.b64decode(ea['encryptedContent']),
                'nonce': base64.b64decode(ea['nonce']),
                'access_grants': [
                    base64.b64decode(grant['encryptedKey'])
                    for grant in ea.get('accessGrants', [])
                ],
            })

        # Build instruction data
        instruction_data = WRITE_TIMELINE_DISCRIMINATOR + _borsh_serialize_activities(activities_for_tx)

        # Build accounts list (order must match Rust struct)
        accounts = [
            AccountMeta(pubkey=BACKEND_PAYER_PUBKEY, is_signer=True, is_writable=True),   # payer
            AccountMeta(pubkey=device_keypair.pubkey(), is_signer=True, is_writable=False),  # device
            AccountMeta(pubkey=camera_pubkey, is_signer=False, is_writable=True),          # camera
            AccountMeta(pubkey=camera_timeline_pda, is_signer=False, is_writable=True),    # camera_timeline
            AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),     # system_program
        ]

        instruction = Instruction(
            program_id=PROGRAM_ID,
            accounts=accounts,
            data=instruction_data,
        )

        # Get recent blockhash from Solana via backend relay
        # (avoids needing a direct RPC connection on Jetson)
        try:
            info_resp = requests.get(f"{BACKEND_URL}/api/relay-timeline-info", timeout=10)
            if info_resp.ok:
                recent_blockhash = info_resp.json()['recent_blockhash']
            else:
                logger.error(f"[TIMELINE] Failed to get blockhash from backend: {info_resp.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[TIMELINE] Failed to reach backend for blockhash: {e}")
            return None

        # Build message with backend as fee payer
        blockhash_obj = Hash.from_string(recent_blockhash)
        message = Message.new_with_blockhash(
            [instruction],
            BACKEND_PAYER_PUBKEY,
            blockhash_obj
        )

        # Create transaction and partially sign with device key
        tx = Transaction.new_unsigned(message)
        tx.partial_sign([device_keypair], blockhash_obj)

        logger.info(f"[TIMELINE] Transaction built and signed by device: {device_keypair.pubkey()}")

        # Serialize and send to backend for fee payer signature + submission
        tx_bytes = bytes(tx)
        tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

        response = requests.post(
            f"{BACKEND_URL}/api/relay-timeline-write",
            json={
                'transaction': tx_base64,
                'session_id': session_id,
                'camera_id': camera_pda,
                'device_pubkey': str(device_keypair.pubkey()),
                'activity_count': len(activities_for_tx),
            },
            timeout=30
        )

        if response.ok:
            result = response.json()
            signature = result.get('signature', '?')
            logger.info(f"[TIMELINE] Wrote {len(activities_for_tx)} activities to CameraTimeline. Tx: {signature[:16]}...")
            return signature
        else:
            logger.error(f"[TIMELINE] Backend relay failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"[TIMELINE] Failed to write timeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
