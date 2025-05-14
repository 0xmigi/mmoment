"""
Session Management Module

Handles session creation, validation, and management for the camera API.
"""

import time
import uuid
import threading

# Dictionary to store active sessions
active_sessions = {}
session_lock = threading.Lock()

def create_session(wallet_address, display_name=None):
    """Create a new session for the wallet"""
    print(f"Creating session for wallet {wallet_address}, display_name={display_name}")
    try:
        with session_lock:
            # Check if there's already an active session for this wallet
            existing_session, _ = get_session_by_wallet(wallet_address)
            if existing_session:
                # Update last activity time and return existing session
                print(f"Found existing session {existing_session} for wallet {wallet_address}")
                active_sessions[existing_session]["last_activity"] = time.time()
                if display_name and not active_sessions[existing_session].get("display_name"):
                    active_sessions[existing_session]["display_name"] = display_name
                return existing_session
            
            # Generate a unique session ID
            session_id = str(uuid.uuid4().hex)[:16]
            
            # Store session data with proper display name
            active_sessions[session_id] = {
                "wallet_address": wallet_address,
                "created_at": time.time(),
                "last_activity": time.time(),
                "display_name": display_name or "User",  # Store display name with session
                "gesture_polling": False  # Whether gesture polling is enabled for this session
            }
            
            print(f"Created new session {session_id} for wallet {wallet_address}, display_name={display_name}")
            
            # Only do cleanup occasionally to avoid slowing down responses
            if len(active_sessions) % 10 == 0:  # Every 10 sessions
                cleanup_old_sessions()
            
            return session_id
    except Exception as e:
        print(f"Error creating session: {e}")
        # Return a fallback session ID rather than failing
        fallback_session_id = str(uuid.uuid4().hex)[:16]
        try:
            active_sessions[fallback_session_id] = {
                "wallet_address": wallet_address,
                "created_at": time.time(),
                "last_activity": time.time(),
                "display_name": display_name or "User",
                "gesture_polling": False,
                "fallback": True  # Mark as fallback session
            }
        except:
            pass  # Ignore errors in fallback handling
        
        print(f"Created fallback session {fallback_session_id} for wallet {wallet_address}")
        return fallback_session_id

def create_temp_session():
    """Create a temporary session without a real wallet"""
    temp_wallet = f"temp_{uuid.uuid4().hex[:8]}"
    session_id = create_session(temp_wallet, "Temporary User")
    return session_id, temp_wallet

def is_session_valid(session_id, wallet_address=None):
    """Check if a session is valid"""
    with session_lock:
        if session_id not in active_sessions:
            return False
        
        # If wallet_address is provided, verify it matches
        if wallet_address and active_sessions[session_id]["wallet_address"] != wallet_address:
            return False
        
        # Update last activity time
        active_sessions[session_id]["last_activity"] = time.time()
        
        return True

def close_session(session_id):
    """Mark session as inactive"""
    with session_lock:
        if session_id in active_sessions:
            active_sessions.pop(session_id, None)
            return True
        return False

def cleanup_old_sessions():
    """Remove sessions that have been inactive for more than 1 hour"""
    try:
        with session_lock:
            if len(active_sessions) == 0:
                return  # No sessions to clean up
            
            current_time = time.time()
            session_ids = list(active_sessions.keys())
            
            # Only process up to 100 sessions at a time to avoid blocking
            for sid in session_ids[:100]:
                if sid in active_sessions and current_time - active_sessions[sid]["last_activity"] > 3600:  # 1 hour in seconds
                    active_sessions.pop(sid, None)
    except Exception as e:
        print(f"Error in cleanup_old_sessions: {e}")
        # Continue execution even if cleanup fails

def count_active_sessions():
    """Count the number of active sessions"""
    with session_lock:
        return len(active_sessions)

def get_session_by_wallet(wallet_address):
    """Find a session ID by wallet address"""
    with session_lock:
        for session_id, data in active_sessions.items():
            if data["wallet_address"] == wallet_address:
                return session_id, data
        
        return None, None

def get_active_sessions():
    """Return a copy of all active sessions"""
    with session_lock:
        return dict(active_sessions)

def get_session_data(session_id):
    """Get session data for a specific session ID"""
    with session_lock:
        return active_sessions.get(session_id, {}) 