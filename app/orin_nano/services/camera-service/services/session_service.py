"""
Session Service

Manages user sessions and authentication for the camera service.
Maintains a registry of active users and their permissions.
"""

import time
import uuid
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SessionService")

class Session:
    """
    Represents a user session with the camera.
    """
    def __init__(self, wallet_address: str, session_id: str = None, user_profile: Dict = None):
        self.wallet_address = wallet_address
        self.session_id = session_id or uuid.uuid4().hex[:16]
        self.created_at = time.time()
        self.last_active = time.time()
        self.user_profile = user_profile or {}
        self.permissions = {
            "can_capture": True,
            "can_record": True,
            "can_recognize": True,
            "can_enroll": True
        }
    
    def touch(self):
        """Update the last active timestamp"""
        self.last_active = time.time()
    
    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if the session has expired"""
        return (time.time() - self.last_active) > timeout_seconds
    
    def to_dict(self) -> Dict:
        """Convert session to a dictionary"""
        return {
            "session_id": self.session_id,
            "wallet_address": self.wallet_address,
            "created_at": int(self.created_at * 1000),
            "last_active": int(self.last_active * 1000),
            "age_seconds": int(time.time() - self.created_at),
            "permissions": self.permissions,
            "user_profile": self.user_profile
        }

class SessionService:
    """
    Service for managing user sessions and authentication.
    Singleton pattern ensures only one instance exists.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        self._sessions_lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}  # session_id -> Session
        self._wallet_sessions: Dict[str, str] = {}  # wallet_address -> session_id
        
        # Session settings
        self._session_timeout = 3600  # 1 hour in seconds
        self._cleanup_interval = 300  # 5 minutes in seconds
        self._max_sessions = 10  # Maximum number of concurrent sessions
        
        # Start cleanup thread
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SessionCleanupThread"
        )
        self._cleanup_thread.start()
        
        logger.info("SessionService initialized")
    
    def create_session(self, wallet_address: str, user_profile: Dict = None, session_pda: str = None) -> Dict:
        """
        Create a new session for the specified wallet address.
        If a session already exists for this wallet, it will be replaced.

        Args:
            wallet_address: The wallet address to create a session for
            user_profile: Optional user profile information (display_name, username, etc.)
            session_pda: Optional Solana session PDA - REQUIRED for activity buffering to work correctly

        Returns:
            Dict with session information
        """
        with self._sessions_lock:
            # Check if wallet already has a session
            existing_profile = None
            if wallet_address in self._wallet_sessions:
                existing_session_id = self._wallet_sessions[wallet_address]

                # Preserve existing profile data before removing session
                if existing_session_id in self._sessions:
                    existing_session = self._sessions[existing_session_id]
                    existing_profile = getattr(existing_session, 'user_profile', None)
                    logger.info(f"Replacing existing session for wallet {wallet_address}")
                    del self._sessions[existing_session_id]

            # Merge profiles: preserve existing display_name if new one doesn't have it
            if existing_profile and user_profile:
                # New profile takes precedence, but fill in missing fields from existing
                if not user_profile.get('display_name') and existing_profile.get('display_name'):
                    user_profile['display_name'] = existing_profile['display_name']
                if not user_profile.get('username') and existing_profile.get('username'):
                    user_profile['username'] = existing_profile['username']
            elif existing_profile and not user_profile:
                # No new profile provided, use existing
                user_profile = existing_profile
            
            # Clean up if we've reached the maximum number of sessions
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_expired_sessions()
                
                # If we still have too many sessions, remove the oldest one
                if len(self._sessions) >= self._max_sessions:
                    oldest_session_id = None
                    oldest_time = float('inf')
                    
                    for session_id, session in self._sessions.items():
                        if session.created_at < oldest_time:
                            oldest_time = session.created_at
                            oldest_session_id = session_id
                    
                    if oldest_session_id:
                        oldest_session = self._sessions[oldest_session_id]
                        logger.info(f"Removing oldest session for wallet {oldest_session.wallet_address}")
                        del self._sessions[oldest_session_id]
                        del self._wallet_sessions[oldest_session.wallet_address]
            
            # Create a new session with Solana PDA if provided
            # If no session_pda provided, falls back to random UUID (but activity buffering won't work!)
            if not session_pda:
                logger.warning(f"⚠️  No session_pda provided for {wallet_address[:8]}... - activity buffering may not work correctly")
            session = Session(wallet_address, session_id=session_pda, user_profile=user_profile)

            # Store the session
            self._sessions[session.session_id] = session
            self._wallet_sessions[wallet_address] = session.session_id

            display_name = user_profile.get('display_name') if user_profile else None
            pda_info = f"PDA={session_pda[:16]}..." if session_pda else "NO PDA (random UUID)"
            logger.info(f"Created new session {session.session_id[:16]}... for wallet {wallet_address[:8]}... ({display_name or 'no display name'}) - {pda_info}")
            
            return {
                "success": True,
                "session_id": session.session_id,
                "wallet_address": wallet_address,
                "created_at": int(session.created_at * 1000),
                "expires_at": int((session.created_at + self._session_timeout) * 1000)
            }
    
    def validate_session(self, session_id: str, wallet_address: str) -> bool:
        """
        Validate a session ID for the given wallet address.
        
        Args:
            session_id: The session ID to validate
            wallet_address: The wallet address to validate against
            
        Returns:
            True if the session is valid, False otherwise
        """
        with self._sessions_lock:
            # Check if the session exists
            if session_id not in self._sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self._sessions[session_id]
            
            # Check if the wallet address matches
            if session.wallet_address != wallet_address:
                logger.warning(f"Session {session_id} belongs to {session.wallet_address}, not {wallet_address}")
                return False
            
            # Check if the session has expired
            if session.is_expired(self._session_timeout):
                logger.warning(f"Session {session_id} for {wallet_address} has expired")
                return False
            
            # Update the last active timestamp
            session.touch()
            
            return True
    
    def check_permission(self, session_id: str, wallet_address: str, permission: str) -> bool:
        """
        Check if a session has a specific permission.
        
        Args:
            session_id: The session ID to check
            wallet_address: The wallet address to validate against
            permission: The permission to check
            
        Returns:
            True if the session has the permission, False otherwise
        """
        # First validate the session
        if not self.validate_session(session_id, wallet_address):
            return False
        
        with self._sessions_lock:
            session = self._sessions[session_id]
            return session.permissions.get(permission, False)
    
    def end_session(self, session_id: str, wallet_address: str) -> bool:
        """
        End a session for the given wallet address.
        
        Args:
            session_id: The session ID to end
            wallet_address: The wallet address to validate against
            
        Returns:
            True if the session was ended, False otherwise
        """
        with self._sessions_lock:
            # Check if the session exists
            if session_id not in self._sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self._sessions[session_id]
            
            # Check if the wallet address matches
            if session.wallet_address != wallet_address:
                logger.warning(f"Session {session_id} belongs to {session.wallet_address}, not {wallet_address}")
                return False
            
            # Remove the session
            del self._sessions[session_id]
            del self._wallet_sessions[wallet_address]
            
            logger.info(f"Ended session {session_id} for wallet {wallet_address}")
            
            return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a session.
        
        Args:
            session_id: The session ID to look up
            
        Returns:
            Session information dict if found, None otherwise
        """
        with self._sessions_lock:
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            return session.to_dict()
    
    def get_session_by_wallet(self, wallet_address: str) -> Optional[Dict]:
        """
        Get information about a session for a specific wallet address.
        
        Args:
            wallet_address: The wallet address to look up
            
        Returns:
            Session information dict if found, None otherwise
        """
        with self._sessions_lock:
            if wallet_address not in self._wallet_sessions:
                return None
            
            session_id = self._wallet_sessions[wallet_address]
            
            if session_id not in self._sessions:
                # Clean up inconsistent state
                del self._wallet_sessions[wallet_address]
                return None
            
            session = self._sessions[session_id]
            return session.to_dict()
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get information about all active sessions.
        
        Returns:
            List of session information dicts
        """
        with self._sessions_lock:
            return [session.to_dict() for session in self._sessions.values()]
    
    def get_active_session_count(self) -> int:
        """
        Get the count of active sessions.
        
        Returns:
            Number of active sessions
        """
        with self._sessions_lock:
            return len(self._sessions)
    
    def _cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._sessions_lock:
            expired_session_ids = []
            expired_wallet_addresses = []
            
            # Find expired sessions
            for session_id, session in self._sessions.items():
                if session.is_expired(self._session_timeout):
                    expired_session_ids.append(session_id)
                    expired_wallet_addresses.append(session.wallet_address)
            
            # Remove expired sessions
            for session_id in expired_session_ids:
                del self._sessions[session_id]
            
            # Remove expired wallet references
            for wallet_address in expired_wallet_addresses:
                if wallet_address in self._wallet_sessions:
                    del self._wallet_sessions[wallet_address]
            
            if expired_session_ids:
                logger.info(f"Cleaned up {len(expired_session_ids)} expired sessions")
            
            return len(expired_session_ids)
    
    def _cleanup_loop(self):
        """
        Continuous loop that cleans up expired sessions.
        """
        logger.info("Session cleanup loop started")
        
        try:
            while not self._stop_cleanup.is_set():
                # Wait for cleanup interval
                if self._stop_cleanup.wait(self._cleanup_interval):
                    break
                
                # Clean up expired sessions
                self._cleanup_expired_sessions()
                
        except Exception as e:
            logger.error(f"Error in session cleanup loop: {e}")
        finally:
            logger.info("Session cleanup loop stopped")
    
    def stop(self):
        """
        Stop the session service and its cleanup thread.
        """
        logger.info("Stopping SessionService")
        
        # Signal the cleanup thread to stop
        self._stop_cleanup.set()
        
        # Wait for the cleanup thread to stop
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        
        logger.info("SessionService stopped")

# Global function to get the session service instance
def get_session_service() -> SessionService:
    """
    Get the singleton SessionService instance.
    """
    return SessionService() 