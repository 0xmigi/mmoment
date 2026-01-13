"""
Competition Settlement Service

Standalone module for settling competition escrows on-chain.
Call settle_competition() when ending a CV activity that has an escrow.

Dependencies:
- solana (pip install solana)
- solders (pip install solders)

Usage:
    from services.competition_settlement import settle_competition

    result = await settle_competition(
        escrow_pda="...",
        camera_keypair_path="/path/to/camera-keypair.json",
        camera_owner_pubkey="...",
        participant_results=[
            {"wallet_address": "...", "score": 25},
            {"wallet_address": "...", "score": 18},
        ],
        rpc_url="https://api.devnet.solana.com"
    )

    if result["success"]:
        tx_signature = result["tx_signature"]
        winners = result["winners"]
        payout_per_winner = result["payout_per_winner_sol"]
"""

import json
import struct
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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

# Anchor instruction discriminator for settle_competition
# Generated from: sha256("global:settle_competition")[0:8]
SETTLE_COMPETITION_DISCRIMINATOR = bytes([233, 202, 62, 111, 78, 98, 218, 208])


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

        # Parse participants vec
        participants_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        participants = []
        for _ in range(participants_len):
            participants.append(Pubkey.from_bytes(data[offset:offset+32]))
            offset += 32
        # Skip remaining space for max participants
        offset += 32 * (10 - participants_len)

        # Parse pending_invites vec
        pending_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        offset += 32 * 10  # Skip all pending invites space

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

            # Build accounts list
            accounts = [
                AccountMeta(pubkey=camera_pubkey, is_signer=True, is_writable=True),  # camera
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

            # Build and sign transaction
            tx = Transaction.new_signed_with_payer(
                [instruction],
                camera_pubkey,
                [camera_keypair],
                recent_blockhash,
            )

            # Send transaction
            logger.info(f"[SettlementService] Sending settle transaction...")
            response = await client.send_transaction(tx, camera_keypair)

            signature = str(response.value)
            logger.info(f"[SettlementService] Transaction sent: {signature}")

            # Wait for confirmation
            await client.confirm_transaction(signature, commitment=Confirmed)
            logger.info(f"[SettlementService] Transaction confirmed: {signature}")

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
    camera_keypair_path: str,
    camera_owner_pubkey: str,
    participant_results: List[Dict[str, Any]],
    rpc_url: str = "https://api.devnet.solana.com",
) -> Dict[str, Any]:
    """Synchronous wrapper for settle_competition"""
    return asyncio.run(settle_competition(
        escrow_pda=escrow_pda,
        camera_keypair_path=camera_keypair_path,
        camera_owner_pubkey=camera_owner_pubkey,
        participant_results=participant_results,
        rpc_url=rpc_url,
    ))


# Example integration into routes.py end_competition handler:
"""
# In your end_competition route handler, after getting CV results:

from services.competition_settlement import settle_competition

# If competition metadata present, settle on-chain
if competition_meta and competition_meta.get("escrow_pda"):
    # Build participant results from CV tracking
    participant_results = [
        {"wallet_address": p["wallet_address"], "score": p["stats"]["reps"]}
        for p in cv_results["participants"]
    ]

    settlement = await settle_competition(
        escrow_pda=competition_meta["escrow_pda"],
        camera_keypair_path="/path/to/camera-keypair.json",
        camera_owner_pubkey=CAMERA_OWNER_WALLET,  # From config
        participant_results=participant_results,
        rpc_url=RPC_URL,
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
