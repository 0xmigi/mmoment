"""
Blockchain Session Sync Service

Phase 3 Privacy Architecture:
- Check-ins are now fully OFF-CHAIN (no blockchain polling for sessions)
- Still loads recognition tokens from blockchain when users check in
- Provides trigger_checkin() and trigger_checkout() for the new flow
- Maintains face-based auto-checkout monitoring

Key methods:
- trigger_checkin(wallet, profile): Load recognition token, enable face boxes
- trigger_checkout(wallet): Clean up face data, disable boxes if no users
- update_user_seen(wallet): Track when user's face is seen for auto-checkout
"""

import time
import logging
import threading
import requests
import os
from typing import Dict, Set, Optional

from .identity_store import get_identity_store

logger = logging.getLogger(__name__)

class BlockchainSessionSync:
    """
    Service that manages camera sessions and recognition token fetching.

    Phase 3 Privacy Architecture:
    - Check-ins are fully OFF-CHAIN (via /api/checkin with Ed25519 signature)
    - Fetches recognition tokens (encrypted face embeddings) from blockchain
    - Monitors for face-based auto-checkout
    """

    def __init__(self, solana_middleware_url: str = None):
        # Camera service uses host networking, so connect via localhost
        self.solana_middleware_url = solana_middleware_url or os.environ.get('SOLANA_MIDDLEWARE_URL', 'http://localhost:5001')
        logger.info(f"🔧 BlockchainSessionSync initialized with URL: {self.solana_middleware_url}")
        self.is_running = False
        self.sync_thread = None
        self.sync_interval = 10  # Check for face-based auto-checkout every 10 seconds
        self.stop_event = threading.Event()

        # Track current state (populated via trigger_checkin/trigger_checkout API calls)
        self.checked_in_wallets: Set[str] = set()

        # Track when users were last seen in frame (for face-based auto-checkout)
        self.last_seen_at: Dict[str, float] = {}  # wallet_address -> timestamp
        self.auto_checkout_threshold = 1800  # 30 minutes in seconds

        # Track when users checked in via API
        self._api_checkin_times: Dict[str, float] = {}  # wallet_address -> timestamp
        self._api_checkin_grace_period = 30  # seconds grace period

        # Track when checkout activity was already buffered via API (to prevent duplicate checkout events)
        self._api_checkout_buffered: Dict[str, float] = {}  # wallet_address -> timestamp
        self._api_checkout_grace_period = 60  # seconds to skip re-buffering checkout activity

        # Services (will be injected)
        self.session_service = None
        self.face_service = None

        # Get reference to unified IdentityStore
        self.identity_store = get_identity_store()

    def set_services(self, session_service, face_service):
        """Inject required services"""
        self.session_service = session_service
        self.face_service = face_service
        
    def start(self):
        """Start the blockchain sync service"""
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="BlockchainSessionSync"
        )
        self.sync_thread.start()
        
        logger.info("🔗 Blockchain session sync started - camera will auto-enable for checked-in users")
        
    def stop(self):
        """Stop the blockchain sync service"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
            
        logger.info("🔗 Blockchain session sync stopped")
        
    def _sync_loop(self):
        """Main sync loop that monitors face-based auto-checkout"""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._monitor_face_based_checkout()
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(self.sync_interval)
                
    def _handle_check_out(self, wallet_address: str):
        """Handle wallet checkout - clean up session and identity data"""
        try:
            # Check if this wallet just checked in via API - prevent race condition
            api_checkin_time = self._api_checkin_times.get(wallet_address, 0)
            if time.time() - api_checkin_time < self._api_checkin_grace_period:
                logger.info(f"⏳ Ignoring checkout for {wallet_address[:8]}... (within {self._api_checkin_grace_period}s grace period after API check-in)")
                return

            logger.info(f"👋 Wallet {wallet_address} checked out - disabling camera session")

            # Clear API check-in time tracking
            self._api_checkin_times.pop(wallet_address, None)

            # CRITICAL: Clear ALL identity data atomically via IdentityStore
            stats = self.identity_store.check_out(wallet_address)
            if stats:
                logger.info(f"✅ Checkout complete for {wallet_address[:8]}... (duration: {stats['duration']:.1f}s)")
            else:
                logger.warning(f"⚠️  User {wallet_address[:8]}... was not in IdentityStore")

            # End camera session automatically
            if self.session_service:
                # Find session for this wallet
                session = self.session_service.get_session_by_wallet(wallet_address)
                if session:
                    session_id = session['session_id']

                    # Buffer CHECK_OUT activity BEFORE ending the session
                    # BUT skip if already buffered via API (prevents duplicate checkout events)
                    api_checkout_time = self._api_checkout_buffered.get(wallet_address, 0)
                    if time.time() - api_checkout_time < self._api_checkout_grace_period:
                        logger.info(f"⏭️  Skipping checkout activity buffer for {wallet_address[:8]}... (already buffered via API)")
                        # Clean up the tracking
                        self._api_checkout_buffered.pop(wallet_address, None)
                    else:
                        # This creates an encrypted activity for the privacy-preserving timeline
                        try:
                            from services.timeline_activity_service import (
                                get_timeline_activity_service,
                            )

                            timeline_service = get_timeline_activity_service()
                            duration_seconds = stats.get('duration') if stats else None
                            timeline_service.buffer_checkout_activity(
                                wallet_address=wallet_address,
                                session_id=session_id,
                                duration_seconds=duration_seconds,
                                metadata={
                                    "timestamp": int(time.time() * 1000),
                                },
                            )
                        except Exception as e:
                            # Non-fatal - checkout continues even if activity buffering fails
                            logger.warning(f"⚠️  Failed to buffer check-out activity: {e}")

                    success = self.session_service.end_session(session_id, wallet_address)
                    if success:
                        logger.info(f"✅ Ended camera session for {wallet_address}")

            # ✅ NEW: Remove from last_seen tracking
            self.last_seen_at.pop(wallet_address, None)

            # Check if this was the last user - disable face boxes and clear all faces if so
            if self.session_service and self.face_service:
                active_sessions = self.session_service.get_all_sessions()
                if len(active_sessions) == 0:
                    self.face_service.enable_boxes(False)

                    # Clear all facial embeddings for security when no users are present
                    try:
                        if hasattr(self.face_service, 'clear_enrolled_faces'):
                            clear_success = self.face_service.clear_enrolled_faces()
                            if clear_success:
                                logger.info("🗑️  Cleared all facial embeddings - no users checked in")
                            else:
                                logger.warning("⚠️  Failed to clear all facial embeddings")
                    except Exception as clear_error:
                        logger.error(f"❌ Error clearing all facial data: {clear_error}")

                    logger.info("✅ Face boxes disabled - no users checked in")

            # Log for monitoring
            logger.info(f"📹 Camera session ended for user: {wallet_address}")

        except Exception as e:
            logger.error(f"Error handling check-out for {wallet_address}: {e}")

    def _fetch_and_decrypt_recognition_token(self, wallet_address: str, profile: dict = None):
        """
        Fetch the user's recognition token (encrypted facial embedding) from on-chain
        and decrypt it for local recognition use.

        Args:
            wallet_address: User's wallet address
            profile: Optional profile dict with display_name, username (passed from /api/checkin)
        """
        try:
            logger.info(f"🔐 Fetching recognition token for {wallet_address} from blockchain...")

            # Step 1: Get the recognition token (encrypted facial embedding) from Solana middleware
            response = requests.get(
                f"{self.solana_middleware_url}/api/blockchain/get-recognition-token",
                params={'wallet_address': wallet_address},
                timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"⚠️  No recognition token found on-chain for {wallet_address} (HTTP {response.status_code})")
                logger.info(f"ℹ️  User can still use camera without face recognition")
                return

            token_data = response.json()
            if not token_data.get('success'):
                logger.warning(f"⚠️  No recognition token found for {wallet_address}: {token_data.get('error')}")
                logger.info(f"ℹ️  User can still use camera without face recognition")
                return

            encrypted_token_package = token_data.get('token_package')
            if not encrypted_token_package:
                logger.warning(f"⚠️  Recognition token data missing for {wallet_address}")
                return

            logger.info(f"✅ Found recognition token on-chain for {wallet_address}")

            # Step 2: Decrypt the facial embedding via biometric security service
            # Create a temporary biometric session for decryption
            bio_session_response = requests.post(
                'http://biometric-security:5003/api/biometric/create-session',
                json={
                    'wallet_address': wallet_address,
                    'session_duration': 300  # 5 min temporary session
                },
                timeout=10
            )

            if bio_session_response.status_code != 200:
                logger.error(f"❌ Failed to create biometric session for decryption: HTTP {bio_session_response.status_code}")
                return

            bio_session = bio_session_response.json()
            session_id = bio_session['session_id']

            # Decrypt the embedding
            decrypt_response = requests.post(
                'http://biometric-security:5003/api/biometric/decrypt-for-session',
                json={
                    'token_package': encrypted_token_package,
                    'wallet_address': wallet_address,
                    'session_id': session_id
                },
                timeout=15
            )

            if decrypt_response.status_code != 200:
                logger.error(f"❌ Failed to decrypt recognition token: HTTP {decrypt_response.status_code}")
                # Clean up biometric session
                requests.post(
                    'http://biometric-security:5003/api/biometric/purge-session',
                    json={'session_id': session_id},
                    timeout=10
                )
                return

            decrypt_data = decrypt_response.json()
            embedding = decrypt_data.get('embedding')

            if not embedding:
                logger.error(f"❌ Decryption returned no embedding for {wallet_address}")
                return

            logger.info(f"✅ Successfully decrypted recognition token for {wallet_address}, embedding size: {len(embedding)}")

            # Step 3: Store the embedding via IdentityStore (single source of truth)
            import numpy as np
            embedding_array = np.array(embedding, dtype=np.float32)

            # ALWAYS preserve existing profile data if not provided in this call
            existing_identity = self.identity_store.get_identity(wallet_address)
            if existing_identity:
                # Merge: use passed values if present, otherwise keep existing
                if not profile:
                    profile = {}
                profile = {
                    'display_name': profile.get('display_name') or existing_identity.display_name,
                    'username': profile.get('username') or existing_identity.username
                }
                logger.info(f"🔍 [DEBUG] Merged profile (preserving existing): {profile}")
            elif not profile:
                profile = {}

            logger.info(f"🔍 [DEBUG] check_in profile: {profile}")

            # Check user in via IdentityStore with profile
            identity = self.identity_store.check_in(wallet_address, embedding_array, profile)

            logger.info(f"✅ Recognition token activated for {wallet_address[:8]}... ({identity.get_display_name()}) - face recognition enabled!")
            logger.info(f"🔍 [DEBUG] Identity display_name={identity.display_name}, username={identity.username}")

            # Clean up biometric session after successful decryption
            requests.post(
                'http://biometric-security:5003/api/biometric/purge-session',
                json={'session_id': session_id},
                timeout=10
            )

        except Exception as e:
            logger.error(f"❌ Error fetching/decrypting recognition token for {wallet_address}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def update_user_seen(self, wallet_address: str):
        """
        Update the last_seen timestamp for a user.
        Call this when a user's face is recognized in frame.
        """
        if wallet_address in self.checked_in_wallets:
            self.last_seen_at[wallet_address] = time.time()
            # Optional: log only periodically to avoid spam
            # logger.debug(f"👁️  User {wallet_address[:8]}... seen in frame")

    def _monitor_face_based_checkout(self):
        """
        Monitor checked-in users with recognition tokens for auto-checkout.
        If a user hasn't been seen in frame for 30 minutes, auto check them out.
        """
        try:
            if not self.face_service:
                return  # Can't monitor without face service

            now = time.time()

            for wallet_address in list(self.checked_in_wallets):
                # Only auto-checkout users who have a local face embedding
                # (users with recognition tokens)
                if not self._has_local_face_embedding(wallet_address):
                    continue  # Skip users without recognition tokens

                last_seen = self.last_seen_at.get(wallet_address, now)
                time_not_seen = now - last_seen

                # Check if user hasn't been seen for the threshold (30 minutes = 1800 seconds)
                if time_not_seen > self.auto_checkout_threshold:
                    logger.info(f"👋 User {wallet_address[:8]}... not seen for {int(time_not_seen/60)} minutes - initiating auto check-out")
                    self._send_checkout_transaction(wallet_address)

        except Exception as e:
            logger.error(f"❌ Error monitoring face-based checkout: {e}")

    def _has_local_face_embedding(self, wallet_address: str) -> bool:
        """Check if a user has a face embedding stored locally"""
        # Check IdentityStore (unified storage for all identities)
        identity = self.identity_store.get_identity(wallet_address)
        if identity and identity.face_embedding is not None:
            return True

        return False

    def _send_checkout_transaction(self, wallet_address: str):
        """
        Send a check-out transaction to the blockchain for a user.
        This is called when auto-checkout is triggered.
        """
        try:
            # For now, just call the existing check-out handler
            # The actual blockchain transaction is handled by the frontend
            # But we can still clean up the local session
            self._handle_check_out(wallet_address)

            # TODO: In the future, we might want to send an actual blockchain transaction here
            # This would require the Jetson to have a wallet/keypair to sign transactions
            # For now, rely on the 2-hour on-chain timeout as a fallback

        except Exception as e:
            logger.error(f"❌ Error sending checkout transaction for {wallet_address}: {e}")

    def trigger_checkin(self, wallet_address: str, profile: dict = None):
        """
        Trigger check-in for a specific wallet with profile data.
        Called directly by /api/checkin endpoint to ensure profile is passed through.

        Args:
            wallet_address: User's wallet address
            profile: Profile dict with display_name, username, etc.
        """
        logger.info(f"🎯 Triggered check-in for {wallet_address[:8]}... with profile: {profile}")

        # Add to tracked wallets
        self.checked_in_wallets.add(wallet_address)
        self.last_seen_at[wallet_address] = time.time()

        # Record API check-in time
        self._api_checkin_times[wallet_address] = time.time()

        # Create session if not already done
        if self.session_service:
            existing = self.session_service.get_session_by_wallet(wallet_address)
            if not existing:
                self.session_service.create_session(wallet_address, profile)

        # Enable face boxes
        if self.face_service:
            self.face_service.enable_boxes(True)

        # Fetch and decrypt recognition token with profile
        self._fetch_and_decrypt_recognition_token(wallet_address, profile)

    def mark_checkout_activity_buffered(self, wallet_address: str):
        """
        Mark that a checkout activity was already buffered via API.
        This prevents duplicate checkout events when blockchain sync runs.

        Called from /api/checkout-notify after buffering the checkout activity.
        """
        self._api_checkout_buffered[wallet_address] = time.time()
        logger.info(f"[CHECKOUT] Marked checkout activity as buffered for {wallet_address[:8]}...")

    def trigger_checkout(self, wallet_address: str):
        """
        Trigger checkout cleanup for a specific wallet.
        Phase 3 Privacy Architecture: Called from /api/checkout endpoint.

        This handles:
        - Removing face recognition data from IdentityStore
        - Clearing tracking state
        - Disabling face boxes if no users remain

        Args:
            wallet_address: User's wallet address
        """
        logger.info(f"[CHECKOUT] Triggered cleanup for {wallet_address[:8]}...")

        # Remove from tracked wallets
        self.checked_in_wallets.discard(wallet_address)
        self.last_seen_at.pop(wallet_address, None)
        self._api_checkin_times.pop(wallet_address, None)
        self._api_checkout_buffered.pop(wallet_address, None)

        # Remove face data from IdentityStore
        if self.identity_store:
            try:
                stats = self.identity_store.check_out(wallet_address)
                if stats:
                    logger.info(f"[CHECKOUT] Removed identity data for {wallet_address[:8]}... (session duration: {stats.get('duration', 'unknown')}s)")
                else:
                    logger.debug(f"[CHECKOUT] No identity data found for {wallet_address[:8]}...")
            except Exception as e:
                logger.warning(f"[CHECKOUT] Error removing identity data: {e}")

        # Check if this was the last user - disable face boxes if so
        if self.session_service and self.face_service:
            active_sessions = self.session_service.get_all_sessions()
            if len(active_sessions) == 0:
                self.face_service.enable_boxes(False)
                logger.info("[CHECKOUT] Face boxes disabled - no users checked in")

                # Clear all facial embeddings for security when no users present
                try:
                    if hasattr(self.face_service, 'clear_enrolled_faces'):
                        self.face_service.clear_enrolled_faces()
                        logger.info("[CHECKOUT] Cleared all facial embeddings")
                except Exception as e:
                    logger.warning(f"[CHECKOUT] Error clearing facial data: {e}")

    def get_status(self) -> Dict:
        """Get current session status"""
        return {
            'running': self.is_running,
            'sync_interval': self.sync_interval,
            'checked_in_wallets': list(self.checked_in_wallets),
            'total_checked_in': len(self.checked_in_wallets)
        }

    def is_wallet_checked_in(self, wallet_address: str) -> bool:
        """
        Check if a specific wallet is currently checked in (local state).
        Phase 3: Check-ins are off-chain, so we use local tracked wallets.
        """
        return wallet_address in self.checked_in_wallets

# Global service instance
_blockchain_session_sync = None

def get_blockchain_session_sync() -> BlockchainSessionSync:
    """Get the blockchain session sync singleton instance"""
    global _blockchain_session_sync
    if _blockchain_session_sync is None:
        _blockchain_session_sync = BlockchainSessionSync()
    return _blockchain_session_sync

def reset_blockchain_session_sync():
    """Reset the blockchain session sync singleton instance"""
    global _blockchain_session_sync
    logger.info("🔄 Resetting blockchain session sync singleton...")
    if _blockchain_session_sync is not None:
        logger.info("🛑 Stopping existing blockchain session sync instance...")
        _blockchain_session_sync.stop()
        _blockchain_session_sync = None
        logger.info("✅ Blockchain session sync singleton reset complete")
    else:
        logger.info("ℹ️ No existing blockchain session sync instance to reset") 