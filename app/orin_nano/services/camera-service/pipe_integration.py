#!/usr/bin/env python3
"""
Pipe SDK Integration for MMOMENT Jetson Cameras

Fast, direct uploads to user Pipe accounts with pre-authorization on check-in.
Designed to be expandable for future on-chain integration.
"""

import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import hashlib
import os

logger = logging.getLogger(__name__)

# Pipe Network API endpoints
PIPE_API_BASE = "https://us-west-00-firestarter.pipenetwork.com"

class UserSession:
    """Represents an authorized user session with Pipe credentials"""
    def __init__(self, wallet_address: str, credentials: Dict[str, str], expires_at: datetime):
        self.wallet_address = wallet_address
        self.user_id = credentials.get("userId")
        self.user_app_key = credentials.get("userAppKey")
        self.expires_at = expires_at
        self.upload_count = 0

    def is_valid(self) -> bool:
        return datetime.now() < self.expires_at

class PipeStorageManager:
    """
    Manages direct uploads to Pipe Network for camera captures.

    Architecture:
    1. User checks in with face -> creates session with Pipe credentials
    2. Camera captures -> uploads directly to user's Pipe account
    3. Upload events logged for future on-chain recording
    """

    def __init__(self, backend_url: str = "http://localhost:3001"):
        self.backend_url = backend_url
        self.active_sessions: Dict[str, UserSession] = {}
        self.upload_events = []  # For future on-chain logging

    async def create_user_session(self, wallet_address: str) -> Optional[UserSession]:
        """
        Create a pre-authorized session when user checks in.
        Fetches Pipe credentials from backend (temporary until fully decentralized).
        """
        try:
            # Get or create Pipe account via backend
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/api/pipe/create-account",
                    json={"walletAddress": wallet_address}
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to get Pipe account for {wallet_address[:8]}")
                        return None

                    data = await resp.json()

                    # Create session valid for 1 hour (configurable)
                    expires_at = datetime.now() + timedelta(hours=1)
                    user_session = UserSession(wallet_address, data, expires_at)

                    # Cache the session
                    self.active_sessions[wallet_address] = user_session

                    logger.info(f"✅ Created Pipe session for {wallet_address[:8]}... (expires: {expires_at})")
                    return user_session

        except Exception as e:
            logger.error(f"Failed to create session for {wallet_address}: {e}")
            return None

    async def upload_capture(self,
                            wallet_address: str,
                            image_data: bytes,
                            capture_type: str = "photo",
                            metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload camera capture directly to user's Pipe storage.
        Fast path - uses pre-authorized session from check-in.
        """

        # Get active session
        session = self.active_sessions.get(wallet_address)

        if not session or not session.is_valid():
            # Try to create new session (shouldn't happen often)
            session = await self.create_user_session(wallet_address)
            if not session:
                return {
                    "success": False,
                    "error": "No valid session",
                    "wallet": wallet_address
                }

        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_id = metadata.get('camera_id', 'jetson01') if metadata else 'jetson01'
            filename = f"mmoment_{capture_type}_{camera_id}_{timestamp}.jpg"

            # Upload via backend proxy (handles JWT auth automatically)
            backend_url = f"{self.backend_url}/api/pipe/upload"

            # Convert image data to base64 for JSON transport
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            upload_payload = {
                'walletAddress': wallet_address,
                'imageData': image_b64,
                'filename': filename,
                'metadata': metadata
            }

            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    backend_url,
                    json=upload_payload,
                    headers={'Content-Type': 'application/json'}
                ) as resp:
                    if resp.status == 200:
                        result_data = await resp.json()
                        result_filename = result_data.get('filename', filename)

                        # Log upload event (for future on-chain recording)
                        upload_event = {
                            "wallet": wallet_address,
                            "filename": result_filename,
                            "timestamp": timestamp,
                            "camera_id": camera_id,
                            "size": len(image_data),
                            "encrypted": False,  # Add encryption later
                            "capture_type": metadata.get('capture_type', 'photo'),
                            "local_filename": metadata.get('local_filename'),
                            "upload_timestamp": datetime.now().isoformat(),
                            "event_type": "pipe_upload",
                            "blockchain_recorded": False  # Mark for future on-chain logging
                        }
                        self.upload_events.append(upload_event)
                        session.upload_count += 1

                        # Log structured event for monitoring
                        logger.info(f"📝 Upload event logged: {wallet_address[:8]}.../{result_filename} -> ready for on-chain recording")

                        logger.info(f"✅ Uploaded {filename} for {wallet_address[:8]}... (#{session.upload_count})")

                        return {
                            "success": True,
                            "filename": result_filename,
                            "storage_url": f"pipe://{session.user_id}/{result_filename}",
                            "size": result_data.get('size', len(image_data)),
                            "upload_count": session.upload_count,
                            "backend_response": result_data
                        }
                    else:
                        error_text = await resp.text()
                        logger.error(f"Upload failed: {error_text}")
                        return {
                            "success": False,
                            "error": error_text
                        }

        except Exception as e:
            logger.error(f"Upload exception: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_session_stats(self, wallet_address: str) -> Dict[str, Any]:
        """Get stats for a user's current session"""
        session = self.active_sessions.get(wallet_address)
        if not session:
            return {"active": False}

        return {
            "active": session.is_valid(),
            "upload_count": session.upload_count,
            "expires_at": session.expires_at.isoformat(),
            "remaining_minutes": max(0, (session.expires_at - datetime.now()).seconds // 60)
        }

    def get_pending_events(self, mark_as_retrieved: bool = True) -> list:
        """
        Get upload events that should be recorded on-chain.
        In the future, these would be batched and written to Solana.

        Args:
            mark_as_retrieved: If True, clears events after retrieval (default).
                             If False, keeps events for future retrieval.
        """
        events = self.upload_events.copy()
        if mark_as_retrieved:
            self.upload_events.clear()  # Clear after retrieval
        return events

    def get_upload_stats(self) -> dict:
        """Get statistics about pending upload events"""
        return {
            "pending_events": len(self.upload_events),
            "total_size_bytes": sum(event.get('size', 0) for event in self.upload_events),
            "wallets_with_uploads": len(set(event.get('wallet') for event in self.upload_events)),
            "capture_types": {
                "photo": len([e for e in self.upload_events if e.get('capture_type') == 'photo']),
                "video": len([e for e in self.upload_events if e.get('capture_type') == 'video'])
            }
        }

    def mark_events_as_recorded_on_chain(self, event_hashes: list) -> int:
        """
        Mark upload events as recorded on-chain (for future use).

        Args:
            event_hashes: List of event identifiers that were successfully recorded

        Returns:
            Number of events marked as recorded
        """
        # This is a placeholder for future on-chain integration
        # For now, just remove events by their content hash
        marked_count = 0

        # Create simple hash for each event for identification
        for i, event in enumerate(self.upload_events[:]):
            event_hash = hashlib.md5(
                f"{event.get('wallet')}{event.get('filename')}{event.get('timestamp')}".encode()
            ).hexdigest()

            if event_hash in event_hashes:
                self.upload_events[i]['blockchain_recorded'] = True
                marked_count += 1

        # Remove recorded events
        self.upload_events = [e for e in self.upload_events if not e.get('blockchain_recorded', False)]

        logger.info(f"📝 Marked {marked_count} upload events as recorded on-chain")
        return marked_count

    async def cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        now = datetime.now()
        expired = [
            wallet for wallet, session in self.active_sessions.items()
            if not session.is_valid()
        ]
        for wallet in expired:
            logger.info(f"🧹 Cleaning up expired session for {wallet[:8]}...")
            del self.active_sessions[wallet]

        return len(expired)

# Global instance for camera service
pipe_manager = PipeStorageManager()

# Integration function for camera service
async def handle_camera_capture(wallet_address: str,
                               image_data: bytes,
                               metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for camera service when a photo is captured.
    Called after successful face recognition.
    """
    return await pipe_manager.upload_capture(
        wallet_address=wallet_address,
        image_data=image_data,
        capture_type="photo",
        metadata=metadata
    )

# Function called on user check-in
async def on_user_checkin(wallet_address: str) -> bool:
    """
    Called when user checks in with face recognition.
    Pre-creates session for fast uploads.
    """
    session = await pipe_manager.create_user_session(wallet_address)
    return session is not None

# Cleanup task (run periodically)
async def cleanup_task():
    """Run every 10 minutes to clean expired sessions"""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        expired_count = await pipe_manager.cleanup_expired_sessions()
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")

# For testing
if __name__ == "__main__":
    async def test():
        # Test wallet
        test_wallet = "RsLjCiEiHq3dyWeDpp1M8jSmAhpmaGamcVK32sJkdLT"

        # Simulate check-in
        print("🔐 User checking in...")
        success = await on_user_checkin(test_wallet)
        if not success:
            print("❌ Failed to create session")
            return

        # Simulate photo capture
        print("📸 Capturing photo...")
        fake_image = b"fake_image_data_here"
        result = await handle_camera_capture(
            test_wallet,
            fake_image,
            {"camera_id": "jetson01"}
        )

        print(f"📤 Upload result: {result}")

        # Check stats
        stats = await pipe_manager.get_session_stats(test_wallet)
        print(f"📊 Session stats: {stats}")

        # Get events for on-chain logging
        events = pipe_manager.get_pending_events()
        print(f"📝 Events to log on-chain: {events}")

    asyncio.run(test())