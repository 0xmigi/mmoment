"""
Blockchain Session Sync Service

Automatically syncs on-chain check-in/check-out state with camera sessions.
Enables visual effects for users who are checked in via the blockchain.
The camera becomes a stateless PDA validator.
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
    Service that syncs blockchain check-in state with camera sessions.
    Automatically enables/disables visual effects based on on-chain state.
    """
    
    def __init__(self, solana_middleware_url: str = None):
        # Camera service uses host networking, so connect via localhost
        self.solana_middleware_url = solana_middleware_url or os.environ.get('SOLANA_MIDDLEWARE_URL', 'http://localhost:5001')
        logger.info(f"🔧 BlockchainSessionSync initialized with URL: {self.solana_middleware_url}")
        self.is_running = False
        self.sync_thread = None
        self.sync_interval = 10  # Check blockchain state every 10 seconds for faster recognition token activation
        self.stop_event = threading.Event()
        
        # Track current state
        self.checked_in_wallets: Set[str] = set()
        self.last_sync = 0

        # ✅ NEW: Track when users were last seen in frame (for face-based auto-checkout)
        self.last_seen_at: Dict[str, float] = {}  # wallet_address -> timestamp
        self.auto_checkout_threshold = 1800  # 30 minutes in seconds

        # Track when users checked in via API (to prevent race condition with blockchain polling)
        self._api_checkin_times: Dict[str, float] = {}  # wallet_address -> timestamp
        self._api_checkin_grace_period = 30  # seconds to ignore blockchain checkout after API check-in

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
        """Main sync loop that checks blockchain state and monitors face-based auto-checkout"""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._sync_blockchain_state()
                # ✅ NEW: Monitor for face-based auto-checkout
                self._monitor_face_based_checkout()
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Error in blockchain sync loop: {e}")
                time.sleep(self.sync_interval)
                
    def _sync_blockchain_state(self):
        """Sync current blockchain state with camera sessions"""
        try:
            # Get current checked-in wallets from blockchain via Solana middleware
            checked_in_wallets = self._get_checked_in_wallets()

            if checked_in_wallets is None:
                return  # Error getting blockchain state

            # ✅ CLEANUP: On first sync (startup/restart), remove stale data for users who checked out while camera was offline
            is_first_sync = (self.last_sync == 0)
            if is_first_sync:
                self._cleanup_stale_identities(checked_in_wallets)

            # Compare with current state
            newly_checked_in = checked_in_wallets - self.checked_in_wallets
            newly_checked_out = self.checked_in_wallets - checked_in_wallets

            # Handle new check-ins
            for wallet in newly_checked_in:
                self._handle_check_in(wallet)

            # Handle check-outs
            for wallet in newly_checked_out:
                self._handle_check_out(wallet)

            # ✅ FIX: On first sync (startup/restart), fetch recognition tokens for all already-checked-in users
            # This ensures users who were already checked-in before camera restart get their tokens loaded
            if is_first_sync and len(checked_in_wallets) > 0:
                logger.info(f"🔄 First sync detected - fetching recognition tokens for {len(checked_in_wallets)} already checked-in users")
                for wallet in checked_in_wallets:
                    # Only fetch token, don't re-create session (already handled by newly_checked_in)
                    if wallet not in newly_checked_in:
                        logger.info(f"🔐 Fetching recognition token for existing user: {wallet[:8]}...")
                        self._fetch_and_decrypt_recognition_token(wallet)
                        # Initialize last_seen tracking
                        self.last_seen_at[wallet] = time.time()

            # Update current state
            self.checked_in_wallets = checked_in_wallets
            self.last_sync = time.time()

            # Log status periodically
            if len(checked_in_wallets) > 0:
                logger.debug(f"🔗 {len(checked_in_wallets)} wallets checked in on-chain: {list(checked_in_wallets)}")
                
        except Exception as e:
            logger.error(f"Error syncing blockchain state: {e}")

    def _cleanup_stale_identities(self, checked_in_wallets: Set[str]):
        """
        Remove stale identity data for users who checked out while camera was offline.

        This ensures no user data persists on the camera after checkout, even if the
        camera missed the checkout event due to being offline.

        Args:
            checked_in_wallets: Set of wallet addresses currently checked in on-chain
        """
        try:
            # Get all wallets that have data stored locally (in memory or on disk)
            local_wallets = set(self.identity_store.get_checked_in_wallets())

            # Find wallets that are stored locally but NOT checked in on-chain
            stale_wallets = local_wallets - checked_in_wallets

            if stale_wallets:
                logger.info(f"🧹 Found {len(stale_wallets)} stale identities - cleaning up data from missed checkouts")
                for wallet in stale_wallets:
                    logger.info(f"🗑️  Removing stale data for {wallet[:8]}... (checked out while camera was offline)")
                    self._handle_check_out(wallet)
            else:
                logger.debug("✅ No stale identity data found on startup")

        except Exception as e:
            logger.error(f"Error cleaning up stale identities: {e}")

    def _get_checked_in_wallets(self) -> Optional[Set[str]]:
        """Get list of currently checked-in wallets from blockchain"""
        try:
            # Query the Solana middleware for checked-in users
            response = requests.get(
                f"{self.solana_middleware_url}/api/blockchain/checked-in-users",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    checked_in_users = data.get('checked_in_users', [])
                    
                    # Extract wallet addresses from the response
                    checked_in_wallets = set()
                    for user_info in checked_in_users:
                        wallet_address = user_info.get('user')
                        if wallet_address:
                            checked_in_wallets.add(wallet_address)
            
                    logger.info(f"🔗 Blockchain sync: Found {len(checked_in_wallets)} checked-in wallets from blockchain")
                    if len(checked_in_wallets) > 0:
                        logger.info(f"🔗 Checked-in wallets: {list(checked_in_wallets)}")
                    
                    return checked_in_wallets
                else:
                    error_msg = data.get('error', 'Unknown error from Solana middleware')
                    logger.error(f"🔗 Solana middleware returned error: {error_msg}")
                    return set()  # Return empty set on error, no fallbacks
                    
            else:
                logger.error(f"🔗 Failed to get checked-in users from Solana middleware: HTTP {response.status_code}")
                return set()  # Return empty set on error, no fallbacks
                
        except requests.exceptions.RequestException as e:
            logger.error(f"🔗 Network error connecting to Solana middleware: {e}")
            return set()  # Return empty set on error, no fallbacks
        except Exception as e:
            logger.error(f"Error getting checked-in wallets from blockchain: {e}")
            return set()  # Return empty set on error, no fallbacks
            
    def _get_hardcoded_wallets(self) -> Set[str]:
        """REMOVED - No more hardcoded fallbacks"""
        return set()
            
    def _handle_check_in(self, wallet_address: str):
        """Handle a wallet checking in on-chain"""
        try:
            logger.info(f"🎉 Wallet {wallet_address} checked in on-chain - enabling camera session")

            # Create camera session automatically
            if self.session_service:
                session_info = self.session_service.create_session(wallet_address)
                logger.info(f"✅ Created camera session {session_info['session_id']} for {wallet_address}")

            # Check if user already has identity with profile (from /api/checkin)
            existing_identity = self.identity_store.get_identity(wallet_address)
            logger.info(f"🔍 [HANDLE-CHECKIN] existing_identity={existing_identity is not None}, display_name={existing_identity.display_name if existing_identity else 'N/A'}")
            if existing_identity and existing_identity.display_name:
                # User already checked in via /api/checkin with profile - just fetch token if needed
                if existing_identity.face_embedding is None:
                    logger.info(f"🔐 Fetching token for existing identity: {existing_identity.get_display_name()}")
                    # Pass existing profile to preserve it
                    profile = {
                        'display_name': existing_identity.display_name,
                        'username': existing_identity.username
                    }
                    self._fetch_and_decrypt_recognition_token(wallet_address, profile)
                else:
                    logger.info(f"✅ User {existing_identity.get_display_name()} already has face embedding")
            else:
                # No existing identity or no profile - this is from blockchain polling (user checked in while camera was offline)
                self._fetch_and_decrypt_recognition_token(wallet_address)

            # Initialize last_seen tracking for face-based auto-checkout
            self.last_seen_at[wallet_address] = time.time()

            # Log for monitoring
            logger.info(f"📹 Camera now active for user: {wallet_address}")

        except Exception as e:
            logger.error(f"Error handling check-in for {wallet_address}: {e}")
            
    def _handle_check_out(self, wallet_address: str):
        """Handle a wallet checking out on-chain"""
        try:
            # Check if this wallet just checked in via API - prevent race condition
            api_checkin_time = self._api_checkin_times.get(wallet_address, 0)
            if time.time() - api_checkin_time < self._api_checkin_grace_period:
                logger.info(f"⏳ Ignoring blockchain checkout for {wallet_address[:8]}... (within {self._api_checkin_grace_period}s grace period after API check-in)")
                return

            logger.info(f"👋 Wallet {wallet_address} checked out on-chain - disabling camera session")

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
                    continue  # Skip, will rely on on-chain timeout

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

    def force_sync(self):
        """Force an immediate sync with blockchain state"""
        logger.info("🔄 Force syncing blockchain state...")
        self._sync_blockchain_state()

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

        # Record API check-in time to prevent race condition with blockchain polling
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
        
    def get_status(self) -> Dict:
        """Get current sync status"""
        return {
            'running': self.is_running,
            'sync_interval': self.sync_interval,
            'last_sync': self.last_sync,
            'checked_in_wallets': list(self.checked_in_wallets),
            'total_checked_in': len(self.checked_in_wallets)
        }

    def is_wallet_checked_in(self, wallet_address: str) -> bool:
        """
        Check if a specific wallet is currently checked in on-chain.
        Uses cached data with optimistic assumption - if we haven't synced recently,
        assume the user is still checked in to avoid blocking actions.
        """
        # If we have recent data, use it
        if time.time() - self.last_sync < 300:  # 5 minutes grace period
            return wallet_address in self.checked_in_wallets
        
        # If data is stale, be optimistic - assume user is still checked in
        # This prevents blocking actions due to network issues or sync delays
        logger.info(f"🔄 Blockchain data is stale ({time.time() - self.last_sync:.0f}s old), being optimistic for {wallet_address}")
        return True

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