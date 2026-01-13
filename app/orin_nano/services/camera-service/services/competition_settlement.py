"""
Competition Settlement Service

Standalone module for settling competition escrows on-chain.
Call settle_competition() when ending a CV activity that has an escrow.

The settlement flow:
1. Camera service builds the settle_competition transaction
2. Camera signs as authority (proves results are authentic)
3. Transaction is sent to backend
4. Backend adds fee payer signature and submits to Solana

Dependencies:
- solana (pip install solana)
- solders (pip install solders)
- httpx (pip install httpx)

Usage:
    from services.competition_settlement import settle_competition

    result = await settle_competition(
        escrow_pda="...",
        camera_keypair=device_signer.get_keypair(),
        camera_owner_pubkey="...",
        participant_results=[
            {"wallet_address": "...", "score": 25},
            {"wallet_address": "...", "score": 18},
        ],
    )

    if result["success"]:
        tx_signature = result["tx_signature"]
        winners = result["winners"]
        payout_per_winner = result["payout_per_winner_sol"]
"""

import os
import json
import struct
import asyncio
import logging
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYSTEM_PROGRAM_ID

logger = logging.getLogger(__name__)

# Competition Escrow Program ID (devnet)
PROGRAM_ID = Pubkey.from_string("32jXEKF2GDjbezk4x8SkgddeVNMYkFjEh5PiAJijxqLJ")

# Backend fee payer public key (the backend wallet that pays for settlement transactions)
BACKEND_PAYER_PUBKEY = Pubkey.from_string("9k5MGiM9Xqx8f2362M1B2rH5uMKFFVNuXaCDKyTsFXep")

# Anchor instruction discriminator for settle_competition
# Generated from: sha256("global:settle_competition")[0:8]
SETTLE_COMPETITION_DISCRIMINATOR = bytes([83, 121, 9, 141, 170, 133, 230, 151])


@dataclass
class ParticipantResult:
    """Result for a single participant"""
    participant: Pubkey
    score: int  # e.g., push-up reps


@dataclass
class SettlementResult:
    """Result of settlement attempt"""
    success: bool
    tx_signature: Optional[str] = None
    winners: List[str] = None
    payout_per_winner_lamports: int = 0
    payout_per_winner_sol: float = 0.0
    total_distributed_lamports: int = 0
    total_distributed_sol: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tx_signature": self.tx_signature,
            "winners": self.winners or [],
            "payout_per_winner_lamports": self.payout_per_winner_lamports,
            "payout_per_winner_sol": self.payout_per_winner_sol,
            "total_distributed_lamports": self.total_distributed_lamports,
            "total_distributed_sol": self.total_distributed_sol,
            "error": self.error,
        }


def load_keypair(keypair_path: str) -> Keypair:
    """Load a Solana keypair from a JSON file"""
    with open(keypair_path, 'r') as f:
        secret_key = json.load(f)
    return Keypair.from_bytes(bytes(secret_key))


def serialize_participant_results(results: List[ParticipantResult]) -> bytes:
    """
    Serialize participant results for the settle_competition instruction.

    Format (Borsh/Anchor):
    - 4 bytes: vec length (u32 LE)
    - For each result:
      - 32 bytes: participant pubkey
      - 4 bytes: score (u32 LE)
    """
    data = struct.pack('<I', len(results))  # vec length

    for result in results:
        data += bytes(result.participant)  # 32 bytes pubkey
        data += struct.pack('<I', result.score)  # u32 score

    return data


async def fetch_escrow_account(
    client: AsyncClient,
    escrow_pda: Pubkey
) -> Optional[Dict[str, Any]]:
    """
    Fetch and parse the escrow account data.
    Returns parsed account data or None if not found.
    """
    try:
        response = await client.get_account_info(escrow_pda, commitment=Confirmed)
        if response.value is None:
            return None

        data = response.value.data
        if len(data) < 8:
            return None

        # Skip 8-byte discriminator
        offset = 8

        # Parse fields (see state.rs for layout)
        initiator = Pubkey.from_bytes(data[offset:offset+32])
        offset += 32

        camera = Pubkey.from_bytes(data[offset:offset+32])
        offset += 32

        stake_per_person = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8

        # Parse participants vec (variable-length, NOT fixed 10 slots)
        participants_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        participants = []
        for _ in range(participants_len):
            participants.append(Pubkey.from_bytes(data[offset:offset+32]))
            offset += 32
        # NOTE: Anchor Vec is variable-length, no padding to skip

        # Parse pending_invites vec (variable-length, NOT fixed 10 slots)
        pending_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        # Skip only the actual pending invites, not fixed 10 slots
        offset += 32 * pending_len

        # Parse total_pool
        total_pool = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8

        # Parse status (1 byte enum)
        status = data[offset]
        offset += 1

        # Parse payout_rule (1 byte enum + optional u32)
        payout_rule_type = data[offset]
        offset += 1
        min_reps = None
        if payout_rule_type == 1:  # ThresholdSplit
            min_reps = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        # Parse created_at
        created_at = struct.unpack('<q', data[offset:offset+8])[0]
        offset += 8

        return {
            "initiator": str(initiator),
            "camera": str(camera),
            "stake_per_person": stake_per_person,
            "participants": [str(p) for p in participants],
            "total_pool": total_pool,
            "status": status,  # 0=Pending, 1=Active, 2=Settled, 3=Cancelled
            "payout_rule_type": payout_rule_type,  # 0=WinnerTakesAll, 1=ThresholdSplit
            "min_reps": min_reps,
            "created_at": created_at,
        }
    except Exception as e:
        logger.error(f"[SettlementService] Error fetching escrow: {e}")
        return None


async def settle_competition(
    escrow_pda: str,
    camera_keypair,  # Can be Keypair object or path string
    camera_owner_pubkey: str,
    participant_results: List[Dict[str, Any]],
    rpc_url: str = "https://api.devnet.solana.com",
) -> Dict[str, Any]:
    """
    Settle a competition escrow on-chain.

    Args:
        escrow_pda: The escrow PDA address (base58)
        camera_keypair: Either a Keypair object or path to keypair JSON file
        camera_owner_pubkey: The camera owner's wallet address (receives funds if no winners in prize mode)
        participant_results: List of {"wallet_address": str, "score": int}
        rpc_url: Solana RPC URL

    Returns:
        Dict with success status, tx_signature, winners, payouts, or error
    """
    try:
        logger.info(f"[SettlementService] Settling escrow: {escrow_pda}")

        # Handle keypair - can be object or path
        if isinstance(camera_keypair, str):
            camera_keypair = load_keypair(camera_keypair)
        camera_pubkey = camera_keypair.pubkey()

        # Parse pubkeys
        escrow_pubkey = Pubkey.from_string(escrow_pda)
        owner_pubkey = Pubkey.from_string(camera_owner_pubkey)

        # Convert participant results
        results = []
        for r in participant_results:
            results.append(ParticipantResult(
                participant=Pubkey.from_string(r["wallet_address"]),
                score=int(r.get("score", r.get("reps", 0)))
            ))

        # Create RPC client
        client = AsyncClient(rpc_url)

        try:
            # Fetch escrow to get current state and participants list
            escrow_data = await fetch_escrow_account(client, escrow_pubkey)
            if escrow_data is None:
                return SettlementResult(
                    success=False,
                    error="Escrow account not found"
                ).to_dict()

            # Verify status is Active (1)
            if escrow_data["status"] != 1:
                status_names = {0: "Pending", 1: "Active", 2: "Settled", 3: "Cancelled"}
                return SettlementResult(
                    success=False,
                    error=f"Escrow not active (status: {status_names.get(escrow_data['status'], 'Unknown')})"
                ).to_dict()

            # Determine winners based on payout rule and results
            winners = []
            payout_rule_type = escrow_data["payout_rule_type"]

            if payout_rule_type == 0:  # WinnerTakesAll
                max_score = max(r.score for r in results) if results else 0
                if max_score > 0:
                    winners = [r.participant for r in results if r.score == max_score]
            else:  # ThresholdSplit
                min_reps = escrow_data.get("min_reps", 0)
                winners = [r.participant for r in results if r.score >= min_reps]

            # Build remaining accounts (recipients for payouts)
            remaining_accounts = []
            if winners:
                # Winners get paid
                for winner in winners:
                    remaining_accounts.append(AccountMeta(
                        pubkey=winner,
                        is_signer=False,
                        is_writable=True,
                    ))
            elif payout_rule_type == 0:
                # WinnerTakesAll with no winners: refund all participants
                for participant_addr in escrow_data["participants"]:
                    remaining_accounts.append(AccountMeta(
                        pubkey=Pubkey.from_string(participant_addr),
                        is_signer=False,
                        is_writable=True,
                    ))
            # ThresholdSplit with no winners: camera_owner gets funds (already in accounts)

            # Build instruction data
            instruction_data = SETTLE_COMPETITION_DISCRIMINATOR + serialize_participant_results(results)

            # Build accounts list (payer first, then camera, then rest)
            accounts = [
                AccountMeta(pubkey=BACKEND_PAYER_PUBKEY, is_signer=True, is_writable=True),  # payer (backend)
                AccountMeta(pubkey=camera_pubkey, is_signer=True, is_writable=False),  # camera (authority)
                AccountMeta(pubkey=escrow_pubkey, is_signer=False, is_writable=True),  # escrow
                AccountMeta(pubkey=owner_pubkey, is_signer=False, is_writable=True),  # camera_owner
                AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),  # system_program
            ] + remaining_accounts

            # Create instruction
            instruction = Instruction(
                program_id=PROGRAM_ID,
                accounts=accounts,
                data=instruction_data,
            )

            # Get recent blockhash
            blockhash_response = await client.get_latest_blockhash(commitment=Confirmed)
            recent_blockhash = blockhash_response.value.blockhash
            logger.info(f"[SettlementService] Got blockhash: {recent_blockhash}")

            # Build transaction with backend as fee payer
            from solders.hash import Hash
            blockhash_obj = Hash.from_string(str(recent_blockhash))
            tx = Transaction(fee_payer=BACKEND_PAYER_PUBKEY, recent_blockhash=blockhash_obj)
            tx.add(instruction)

            # Partial sign with camera keypair only (backend will add payer signature)
            tx.partial_sign(camera_keypair)
            logger.info(f"[SettlementService] Transaction signed by camera: {camera_pubkey}")

            # Serialize transaction for backend
            tx_bytes = tx.serialize()
            tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

            # POST to backend for fee payer signature and submission
            backend_url = os.environ.get("BACKEND_URL", "https://mmoment-production.up.railway.app")
            logger.info(f"[SettlementService] Sending to backend: {backend_url}/api/competition/settle")

            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.post(
                    f"{backend_url}/api/competition/settle",
                    json={
                        "transaction": tx_base64,
                        "cameraPubkey": str(camera_pubkey),
                        "escrowPda": str(escrow_pubkey),
                    }
                )

                if response.status_code != 200:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"error": response.text}
                    logger.error(f"[SettlementService] Backend error: {error_data}")
                    return SettlementResult(
                        success=False,
                        error=f"Backend settlement failed: {error_data.get('error', 'Unknown error')}"
                    ).to_dict()

                result_data = response.json()
                if not result_data.get("success"):
                    return SettlementResult(
                        success=False,
                        error=result_data.get("error", "Backend settlement failed")
                    ).to_dict()

                signature = result_data.get("signature")
                logger.info(f"[SettlementService] Settlement confirmed: {signature}")

            # Calculate payouts (approximate - on-chain calculation is authoritative)
            total_pool = escrow_data["total_pool"]
            num_recipients = len(winners) if winners else (
                len(escrow_data["participants"]) if payout_rule_type == 0 else 1
            )
            payout_per_recipient = total_pool // num_recipients if num_recipients > 0 else 0

            return SettlementResult(
                success=True,
                tx_signature=signature,
                winners=[str(w) for w in winners],
                payout_per_winner_lamports=payout_per_recipient,
                payout_per_winner_sol=payout_per_recipient / 1_000_000_000,
                total_distributed_lamports=payout_per_recipient * num_recipients,
                total_distributed_sol=(payout_per_recipient * num_recipients) / 1_000_000_000,
            ).to_dict()

        finally:
            await client.close()

    except Exception as e:
        logger.error(f"[SettlementService] Settlement failed: {e}", exc_info=True)
        return SettlementResult(
            success=False,
            error=str(e)
        ).to_dict()


# Convenience function for sync contexts
def settle_competition_sync(
    escrow_pda: str,
    camera_keypair,  # Keypair object or path string
    camera_owner_pubkey: str,
    participant_results: List[Dict[str, Any]],
    rpc_url: str = "https://api.devnet.solana.com",
) -> Dict[str, Any]:
    """Synchronous wrapper for settle_competition"""
    return asyncio.run(settle_competition(
        escrow_pda=escrow_pda,
        camera_keypair=camera_keypair,
        camera_owner_pubkey=camera_owner_pubkey,
        participant_results=participant_results,
        rpc_url=rpc_url,
    ))


# Example integration into routes.py end_competition handler:
"""
# In your end_competition route handler, after getting CV results:

from services.competition_settlement import settle_competition
from services.device_signer import get_device_signer

# If competition metadata present, settle on-chain via backend
if competition_meta and competition_meta.get("escrow_pda"):
    # Build participant results from CV tracking
    participant_results = [
        {"wallet_address": p["wallet_address"], "score": p["stats"]["reps"]}
        for p in cv_results["participants"]
    ]

    # Get device keypair for signing
    device_signer = get_device_signer()

    # Settle competition - camera signs, backend pays and submits
    settlement = await settle_competition(
        escrow_pda=competition_meta["escrow_pda"],
        camera_keypair=device_signer.get_keypair(),
        camera_owner_pubkey=CAMERA_OWNER_WALLET,  # From config
        participant_results=participant_results,
    )

    # Include settlement in timeline entry metadata
    if settlement["success"]:
        cv_activity_meta["competition"] = {
            "mode": competition_meta.get("mode", "bet"),
            "escrow_pda": competition_meta["escrow_pda"],
            "stake_amount_sol": competition_meta.get("stake_amount_sol", 0),
            "target_reps": competition_meta.get("target_reps"),
            "won": user_wallet in settlement["winners"],
            "amount_won_sol": settlement["payout_per_winner_sol"] if user_wallet in settlement["winners"] else 0,
            "amount_lost_sol": competition_meta.get("stake_amount_sol", 0) if user_wallet not in settlement["winners"] else 0,
            "lost_to": CAMERA_OWNER_WALLET if not settlement["winners"] and competition_meta.get("mode") == "prize" else None,
            "settlement_tx_id": settlement["tx_signature"],
        }
"""
