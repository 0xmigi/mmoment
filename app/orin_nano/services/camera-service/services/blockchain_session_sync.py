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

logger = logging.getLogger(__name__)

class BlockchainSessionSync:
    """
    Service that syncs blockchain check-in state with camera sessions.
    Automatically enables/disables visual effects based on on-chain state.
    """
    
    def __init__(self, solana_middleware_url: str = None):
        self.solana_middleware_url = solana_middleware_url or os.environ.get('SOLANA_MIDDLEWARE_URL', 'http://solana-middleware:5001')
        logger.info(f"🔧 BlockchainSessionSync initialized with URL: {self.solana_middleware_url}")
        self.is_running = False
        self.sync_thread = None
        self.sync_interval = 60  # Check blockchain state every 60 seconds (much less aggressive)
        self.stop_event = threading.Event()
        
        # Track current state
        self.checked_in_wallets: Set[str] = set()
        self.last_sync = 0
        
        # Services (will be injected)
        self.session_service = None
        self.face_service = None
        
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
        """Main sync loop that checks blockchain state"""
        while self.is_running and not self.stop_event.is_set():
            try:
                self._sync_blockchain_state()
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
                
            # Compare with current state
            newly_checked_in = checked_in_wallets - self.checked_in_wallets
            newly_checked_out = self.checked_in_wallets - checked_in_wallets
            
            # Handle new check-ins
            for wallet in newly_checked_in:
                self._handle_check_in(wallet)
                
            # Handle check-outs
            for wallet in newly_checked_out:
                self._handle_check_out(wallet)
                
            # Update current state
            self.checked_in_wallets = checked_in_wallets
            self.last_sync = time.time()
            
            # Log status periodically
            if len(checked_in_wallets) > 0:
                logger.debug(f"🔗 {len(checked_in_wallets)} wallets checked in on-chain: {list(checked_in_wallets)}")
                
        except Exception as e:
            logger.error(f"Error syncing blockchain state: {e}")
            
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
                
            # Enable face boxes automatically
            if self.face_service:
                self.face_service.enable_boxes(True)
                logger.info(f"✅ Face boxes enabled for checked-in user: {wallet_address}")
                
            # Log for monitoring
            logger.info(f"📹 Camera now active for user: {wallet_address}")
            
        except Exception as e:
            logger.error(f"Error handling check-in for {wallet_address}: {e}")
            
    def _handle_check_out(self, wallet_address: str):
        """Handle a wallet checking out on-chain"""
        try:
            logger.info(f"👋 Wallet {wallet_address} checked out on-chain - disabling camera session")

            # CRITICAL: Clear facial embedding data for security
            if self.face_service:
                try:
                    # Remove this user's facial embedding from memory and disk
                    if hasattr(self.face_service, 'remove_face_embedding'):
                        success = self.face_service.remove_face_embedding(wallet_address)
                        if success:
                            logger.info(f"🗑️  Cleared facial embedding for {wallet_address[:8]}...")
                        else:
                            logger.warning(f"⚠️  Failed to clear facial embedding for {wallet_address[:8]}...")
                    else:
                        logger.warning("⚠️  Face service doesn't support individual face removal")

                except Exception as face_error:
                    logger.error(f"❌ Error clearing facial data for {wallet_address}: {face_error}")

            # End camera session automatically
            if self.session_service:
                # Find session for this wallet
                session = self.session_service.get_session_by_wallet(wallet_address)
                if session:
                    session_id = session['session_id']
                    success = self.session_service.end_session(session_id, wallet_address)
                    if success:
                        logger.info(f"✅ Ended camera session for {wallet_address}")

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
            
    def force_sync(self):
        """Force an immediate sync with blockchain state"""
        logger.info("🔄 Force syncing blockchain state...")
        self._sync_blockchain_state()
        
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