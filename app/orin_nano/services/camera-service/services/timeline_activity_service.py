"""
Timeline Activity Service

Unified service for buffering all timeline activities (photos, videos, streams).
This is the primary integration point for the privacy-preserving timeline architecture.

Activities are:
1. Encrypted with AES-256-GCM (key per activity)
2. Access grants created for all checked-in users (Ed25519 → X25519 sealed box)
3. Buffered to backend during session
4. Committed to blockchain at checkout (bundled into single tx)

Activity Types (from state.rs):
- Photo = 10
- Video = 20
- StreamStart = 30
- StreamEnd = 31
- CVAppActivity = 50
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TimelineActivityService")

# Activity type enums (matching Solana program state.rs)
ACTIVITY_TYPE_PHOTO = 10
ACTIVITY_TYPE_VIDEO = 20
ACTIVITY_TYPE_STREAM_START = 30
ACTIVITY_TYPE_STREAM_END = 31
ACTIVITY_TYPE_CV_APP = 50


def _get_camera_pda() -> str:
    """Get camera PDA from device config file (same logic as routes.py)."""
    config_path = "/app/config/device_config.json"
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                if "camera_pda" in config:
                    return config["camera_pda"]
    except Exception as e:
        logger.debug(f"Could not read camera PDA from device config: {e}")

    # Fallback to environment variable
    return os.environ.get("CAMERA_PDA", "unknown-camera")


class TimelineActivityService:
    """
    Unified service for buffering all timeline activities.

    Provides simple API for photo/video/stream activities that:
    - Encrypts content for privacy
    - Creates access grants for temporal access control
    - Buffers to backend for checkout bundling

    Usage:
        service = get_timeline_activity_service()
        service.buffer_photo_activity(wallet, pipe_url, file_id, metadata)
    """

    def __init__(self):
        """Initialize the timeline activity service."""
        # Lazy-loaded services
        self._encryption_service = None
        self._buffer_client = None
        self._blockchain_sync = None
        self._session_service = None

        # Camera configuration - read from device config file
        self._camera_id = _get_camera_pda()

        # Statistics
        self._buffered_photos = 0
        self._buffered_videos = 0
        self._buffered_streams = 0
        self._errors = 0

        logger.info(
            f"Timeline activity service initialized for camera: {self._camera_id[:16]}..."
        )

    def _get_encryption_service(self):
        """Lazy load encryption service."""
        if self._encryption_service is None:
            from .activity_encryption_service import get_activity_encryption_service

            self._encryption_service = get_activity_encryption_service()
        return self._encryption_service

    def _get_buffer_client(self):
        """Lazy load buffer client."""
        if self._buffer_client is None:
            from .activity_buffer_client import get_activity_buffer_client

            self._buffer_client = get_activity_buffer_client()
        return self._buffer_client

    def _get_blockchain_sync(self):
        """Lazy load blockchain session sync."""
        if self._blockchain_sync is None:
            from .blockchain_session_sync import get_blockchain_session_sync

            self._blockchain_sync = get_blockchain_session_sync()
        return self._blockchain_sync

    def _get_session_service(self):
        """Lazy load session service."""
        if self._session_service is None:
            from .session_service import get_session_service

            self._session_service = get_session_service()
        return self._session_service

    def _get_checked_in_users(self) -> List[str]:
        """Get list of currently checked-in wallet addresses."""
        try:
            blockchain_sync = self._get_blockchain_sync()
            return list(blockchain_sync.checked_in_wallets)
        except Exception as e:
            logger.warning(f"Failed to get checked-in users: {e}")
            return []

    def _get_session_id(self, wallet_address: str) -> str:
        """
        Get session ID for a user, with fallback to camera-based ID.

        Args:
            wallet_address: User's wallet address

        Returns:
            Session ID string
        """
        try:
            session_service = self._get_session_service()
            session = session_service.get_session_by_wallet(wallet_address)
            if session:
                return session["session_id"]
        except Exception as e:
            logger.debug(f"Could not get session for {wallet_address[:8]}...: {e}")

        # Fallback: camera ID + date as session identifier
        # This ensures activities are still grouped even without explicit session
        return f"{self._camera_id}-{int(time.time() // 86400)}"

    def _buffer_activity(
        self, wallet_address: str, activity_content: Dict[str, Any], activity_type: int
    ) -> bool:
        """
        Internal method to encrypt and buffer an activity.

        Args:
            wallet_address: User who triggered the activity
            activity_content: Activity data to encrypt
            activity_type: Activity type enum

        Returns:
            True if buffered successfully
        """
        try:
            # Get all checked-in users for access grants
            checked_in_users = self._get_checked_in_users()

            if not checked_in_users:
                logger.debug("No users checked in, skipping activity buffer")
                return False

            # Ensure the triggering user is included in access grants
            if wallet_address not in checked_in_users:
                checked_in_users.append(wallet_address)

            # Encrypt activity for all checked-in users
            encryption_service = self._get_encryption_service()
            encrypted = encryption_service.encrypt_activity(
                activity_content=activity_content,
                users_present=checked_in_users,
                activity_type=activity_type,
            )

            # Get session ID for this user
            session_id = self._get_session_id(wallet_address)

            # Buffer to backend (async, non-blocking)
            buffer_client = self._get_buffer_client()
            buffer_client.buffer_activity(
                session_id=session_id,
                camera_id=self._camera_id,
                user_pubkey=wallet_address,
                encrypted_activity=encrypted,
            )

            logger.info(
                f"📤 Buffered activity type={activity_type} for {len(checked_in_users)} users"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to buffer activity: {e}", exc_info=True)
            self._errors += 1
            return False

    def buffer_photo_activity(
        self,
        wallet_address: str,
        pipe_file_name: str,
        pipe_file_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a photo capture activity.

        Called after successful photo capture and Pipe upload.

        Args:
            wallet_address: User who captured the photo
            pipe_file_name: Pipe storage file name/URL
            pipe_file_id: Pipe file ID for retrieval
            metadata: Optional additional metadata (timestamp, device_signature, etc.)

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        activity_content = {
            "type": "photo",
            "pipe_file_name": pipe_file_name,
            "pipe_file_id": pipe_file_id,
            "camera_id": self._camera_id,
            "captured_by": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "device_signature": metadata.get("device_signature"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "filename": metadata.get("filename"),
        }

        success = self._buffer_activity(
            wallet_address, activity_content, ACTIVITY_TYPE_PHOTO
        )

        if success:
            self._buffered_photos += 1
            logger.info(
                f"📸 Photo activity buffered for {wallet_address[:8]}... -> {pipe_file_name}"
            )

        return success

    def buffer_video_activity(
        self,
        wallet_address: str,
        pipe_file_name: str,
        pipe_file_id: str,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a video recording activity.

        Called after successful video recording and Pipe upload.

        Args:
            wallet_address: User who recorded the video
            pipe_file_name: Pipe storage file name/URL
            pipe_file_id: Pipe file ID for retrieval
            duration_seconds: Video duration in seconds
            metadata: Optional additional metadata

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        activity_content = {
            "type": "video",
            "pipe_file_name": pipe_file_name,
            "pipe_file_id": pipe_file_id,
            "camera_id": self._camera_id,
            "recorded_by": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "duration_seconds": duration_seconds,
            "device_signature": metadata.get("device_signature"),
            "filename": metadata.get("filename"),
            "size": metadata.get("size"),
        }

        success = self._buffer_activity(
            wallet_address, activity_content, ACTIVITY_TYPE_VIDEO
        )

        if success:
            self._buffered_videos += 1
            logger.info(
                f"🎥 Video activity buffered for {wallet_address[:8]}... -> {pipe_file_name} ({duration_seconds}s)"
            )

        return success

    def buffer_stream_start_activity(
        self,
        wallet_address: str,
        stream_id: str,
        playback_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a livestream start activity.

        Called when a user starts a livestream.

        Args:
            wallet_address: User who started the stream
            stream_id: Livepeer stream ID
            playback_id: Livepeer playback ID for viewing
            metadata: Optional additional metadata

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        activity_content = {
            "type": "stream_start",
            "stream_id": stream_id,
            "playback_id": playback_id,
            "camera_id": self._camera_id,
            "started_by": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "resolution": metadata.get("resolution"),
            "fps": metadata.get("fps"),
        }

        success = self._buffer_activity(
            wallet_address, activity_content, ACTIVITY_TYPE_STREAM_START
        )

        if success:
            self._buffered_streams += 1
            logger.info(
                f"📡 Stream start activity buffered for {wallet_address[:8]}... -> {playback_id}"
            )

        return success

    def buffer_stream_end_activity(
        self,
        wallet_address: str,
        stream_id: str,
        playback_id: str,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a livestream end activity.

        Called when a livestream ends.

        Args:
            wallet_address: User who was streaming
            stream_id: Livepeer stream ID
            playback_id: Livepeer playback ID
            duration_seconds: Total stream duration
            metadata: Optional additional metadata

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        activity_content = {
            "type": "stream_end",
            "stream_id": stream_id,
            "playback_id": playback_id,
            "camera_id": self._camera_id,
            "ended_by": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "duration_seconds": duration_seconds,
            "frames_sent": metadata.get("frames_sent"),
            "avg_fps": metadata.get("avg_fps"),
        }

        success = self._buffer_activity(
            wallet_address, activity_content, ACTIVITY_TYPE_STREAM_END
        )

        if success:
            logger.info(
                f"📡 Stream end activity buffered for {wallet_address[:8]}... -> {playback_id} ({duration_seconds}s)"
            )

        return success

    def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        buffer_status = {}
        try:
            buffer_client = self._get_buffer_client()
            buffer_status = buffer_client.get_status()
        except Exception:
            pass

        return {
            "service": "timeline-activity",
            "camera_id": self._camera_id,
            "statistics": {
                "buffered_photos": self._buffered_photos,
                "buffered_videos": self._buffered_videos,
                "buffered_streams": self._buffered_streams,
                "errors": self._errors,
                "total_buffered": self._buffered_photos
                + self._buffered_videos
                + self._buffered_streams,
            },
            "buffer_client": buffer_status,
            "checked_in_users": len(self._get_checked_in_users()),
        }


# Global singleton instance
_timeline_activity_service: Optional[TimelineActivityService] = None


def get_timeline_activity_service() -> TimelineActivityService:
    """Get the timeline activity service singleton."""
    global _timeline_activity_service
    if _timeline_activity_service is None:
        _timeline_activity_service = TimelineActivityService()
    return _timeline_activity_service
