#!/usr/bin/env python3
"""
Biometric Session Manager

Handles session lifecycle for biometric operations:
- Session creation and validation
- Temporary storage of encrypted embeddings
- Session cleanup and secure purging
"""

import os
import time
import uuid
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from .secure_storage import SecureStorage

logger = logging.getLogger("BiometricSessionManager")

class SessionManager:
    """
    Manages biometric sessions and temporary encrypted data storage
    """
    
    def __init__(self):
        # Session storage (in-memory for security)
        self._sessions = {}
        self._session_data = {}
        self._session_lock = threading.Lock()
        
        # Initialize secure storage
        self.secure_storage = SecureStorage()
        
        # Session configuration
        self.default_session_duration = 7200  # 2 hours in seconds
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("Biometric session manager initialized")
    
    def create_session(self, wallet_address: str, duration: int = None) -> Dict:
        """
        Create a new biometric session
        
        Args:
            wallet_address: User's wallet address
            duration: Session duration in seconds (default: 2 hours)
            
        Returns:
            Session information including session_id and expiration
        """
        try:
            session_id = str(uuid.uuid4())
            current_time = time.time()
            session_duration = duration or self.default_session_duration
            expires_at = current_time + session_duration
            
            session_info = {
                'session_id': session_id,
                'wallet_address': wallet_address,
                'created_at': current_time,
                'expires_at': expires_at,
                'duration': session_duration,
                'status': 'active'
            }
            
            with self._session_lock:
                self._sessions[session_id] = session_info
                self._session_data[session_id] = {}
            
            logger.info(f"Created session {session_id} for wallet: {wallet_address[:8]}...")
            return session_info
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def validate_session(self, session_id: str, wallet_address: str) -> bool:
        """
        Validate that a session exists and is active
        
        Args:
            session_id: Session ID to validate
            wallet_address: Expected wallet address
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            with self._session_lock:
                session = self._sessions.get(session_id)
                
                if not session:
                    logger.debug(f"Session not found: {session_id}")
                    return False
                
                # Check if session has expired
                if time.time() > session['expires_at']:
                    logger.debug(f"Session expired: {session_id}")
                    self._cleanup_session(session_id)
                    return False
                
                # Check wallet address matches
                if session['wallet_address'] != wallet_address:
                    logger.warning(f"Wallet address mismatch for session: {session_id}")
                    return False
                
                # Check session status
                if session['status'] != 'active':
                    logger.debug(f"Session not active: {session_id}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return False
    
    def store_encrypted_embedding(self, session_id: str, wallet_address: str, encrypted_data: Dict) -> bool:
        """
        Store encrypted embedding data for a session
        
        Args:
            session_id: Session ID
            wallet_address: User's wallet address
            encrypted_data: Encrypted NFT package
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if not self.validate_session(session_id, wallet_address):
                logger.warning(f"Invalid session for storing embedding: {session_id}")
                return False
            
            with self._session_lock:
                if session_id not in self._session_data:
                    self._session_data[session_id] = {}
                
                self._session_data[session_id]['encrypted_embedding'] = encrypted_data
                self._session_data[session_id]['stored_at'] = time.time()
            
            logger.info(f"Stored encrypted embedding for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing encrypted embedding: {e}")
            return False
    
    def get_nft_package(self, session_id: str, wallet_address: str) -> Optional[Dict]:
        """
        Get NFT package for a session
        
        Args:
            session_id: Session ID
            wallet_address: User's wallet address
            
        Returns:
            NFT package if found, None otherwise
        """
        try:
            if not self.validate_session(session_id, wallet_address):
                logger.warning(f"Invalid session for getting NFT package: {session_id}")
                return None
            
            with self._session_lock:
                session_data = self._session_data.get(session_id, {})
                return session_data.get('encrypted_embedding')
                
        except Exception as e:
            logger.error(f"Error getting NFT package: {e}")
            return None
    
    def purge_session(self, session_id: str) -> bool:
        """
        Securely purge all data for a session
        
        Args:
            session_id: Session ID to purge
            
        Returns:
            True if purged successfully, False otherwise
        """
        try:
            with self._session_lock:
                # Get session info before deletion
                session = self._sessions.get(session_id)
                if session:
                    wallet_address = session['wallet_address']
                    logger.info(f"Purging session {session_id} for wallet: {wallet_address[:8]}...")
                
                # Remove from memory
                if session_id in self._sessions:
                    del self._sessions[session_id]
                
                if session_id in self._session_data:
                    del self._session_data[session_id]
            
            # Secure deletion of any temporary files
            self.secure_storage.purge_session_files(session_id)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"Successfully purged session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error purging session {session_id}: {e}")
            return False
    
    def _cleanup_session(self, session_id: str):
        """Internal method to cleanup a specific session"""
        try:
            with self._session_lock:
                if session_id in self._sessions:
                    del self._sessions[session_id]
                if session_id in self._session_data:
                    del self._session_data[session_id]
            
            logger.debug(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            with self._session_lock:
                for session_id, session in list(self._sessions.items()):
                    if current_time > session['expires_at']:
                        expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                self.purge_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    time.sleep(60)  # Wait before retrying
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started session cleanup thread")
    
    def get_active_sessions(self) -> List[Dict]:
        """
        Get list of all active sessions
        
        Returns:
            List of active session information (without sensitive data)
        """
        try:
            current_time = time.time()
            active_sessions = []
            
            with self._session_lock:
                for session_id, session in self._sessions.items():
                    if current_time <= session['expires_at'] and session['status'] == 'active':
                        active_sessions.append({
                            'session_id': session_id,
                            'wallet_address': session['wallet_address'][:8] + '...',  # Truncate for privacy
                            'created_at': session['created_at'],
                            'expires_at': session['expires_at'],
                            'time_remaining': session['expires_at'] - current_time
                        })
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def get_status(self) -> Dict:
        """Get session manager status"""
        try:
            with self._session_lock:
                total_sessions = len(self._sessions)
                active_sessions = len([s for s in self._sessions.values() 
                                     if time.time() <= s['expires_at'] and s['status'] == 'active'])
                sessions_with_data = len([d for d in self._session_data.values() 
                                        if 'encrypted_embedding' in d])
            
            return {
                'service': 'biometric-session-manager',
                'version': '1.0.0',
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'sessions_with_embeddings': sessions_with_data,
                'default_duration': self.default_session_duration,
                'cleanup_interval': self.cleanup_interval,
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting session manager status: {e}")
            return {'status': 'error', 'error': str(e)} 