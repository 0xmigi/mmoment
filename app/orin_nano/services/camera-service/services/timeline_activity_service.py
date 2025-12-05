"""
Timeline Activity Service

Unified service for buffering all timeline activities (photos, videos, streams).
This is the primary integration point for the privacy-preserving timeline architecture.

Activities are:
1. Encrypted with AES-256-GCM (key per activity)
2. Access grants created for all checked-in users (Ed25519 → X25519 sealed box)
3. Buffered to backend during session
4. Committed to blockchain at checkout (bundled into single tx)

Activity Types (from Solana state.rs - MUST match exactly):
- CheckIn = 0
- CheckOut = 1
- PhotoCapture = 2
- VideoRecord = 3
- LiveStream = 4
- FaceRecognition = 5
- CVAppActivity = 50
- Other = 255
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TimelineActivityService")

# Activity type enums - MUST match Solana program state.rs ActivityType enum
ACTIVITY_TYPE_CHECK_IN = 0
ACTIVITY_TYPE_CHECK_OUT = 1
ACTIVITY_TYPE_PHOTO = 2
ACTIVITY_TYPE_VIDEO = 3
ACTIVITY_TYPE_STREAM = 4  # LiveStream
ACTIVITY_TYPE_FACE_RECOGNITION = 5
ACTIVITY_TYPE_CV_APP = 50
ACTIVITY_TYPE_OTHER = 255


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
        self,
        wallet_address: str,
        activity_content: Dict[str, Any],
        activity_type: int,
        transaction_signature: Optional[str] = None,
        cv_activity_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Internal method to encrypt and buffer an activity.

        Args:
            wallet_address: User who triggered the activity
            activity_content: Activity data to encrypt
            activity_type: Activity type enum
            transaction_signature: Optional Solana tx signature for Solscan link
            cv_activity_meta: Optional CV activity metadata for timeline display

        Returns:
            True if buffered successfully
        """
        try:
            # Get all checked-in users for access grants
            checked_in_users = self._get_checked_in_users()

            # Ensure the triggering user is included in access grants
            # (Important: Do this BEFORE the empty check so their own activities are always buffered)
            if wallet_address not in checked_in_users:
                checked_in_users.append(wallet_address)

            if not checked_in_users:
                logger.debug("No users checked in, skipping activity buffer")
                return False

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
                transaction_signature=transaction_signature,
                cv_activity_meta=cv_activity_meta,
            )

            logger.info(
                f"📤 Buffered activity type={activity_type} for {len(checked_in_users)} users"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to buffer activity: {e}", exc_info=True)
            self._errors += 1
            return False

    def buffer_checkin_activity(
        self,
        wallet_address: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a check-in activity.

        Called when a user successfully checks in at this camera.

        Args:
            wallet_address: User's wallet address
            session_id: Session ID from check-in
            metadata: Optional additional metadata (tx_signature, etc.)

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        # Extract transaction signature for Solscan link (passed separately, not encrypted)
        tx_signature = metadata.get("tx_signature")

        activity_content = {
            "type": "check_in",
            "session_id": session_id,
            "camera_id": self._camera_id,
            "user": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "tx_signature": tx_signature,
        }

        success = self._buffer_activity(
            wallet_address,
            activity_content,
            ACTIVITY_TYPE_CHECK_IN,
            transaction_signature=tx_signature,  # Pass separately for timeline event
        )

        if success:
            logger.info(
                f"✅ Check-in activity buffered for {wallet_address[:8]}... session={session_id[:16]}..."
            )

        return success

    def buffer_checkout_activity(
        self,
        wallet_address: str,
        session_id: str,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a check-out activity.

        Called when a user checks out from this camera.

        Args:
            wallet_address: User's wallet address
            session_id: Session ID from check-in
            duration_seconds: Session duration in seconds
            metadata: Optional additional metadata (tx_signature, activity_count, etc.)

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        # Extract transaction signature for Solscan link (passed separately, not encrypted)
        tx_signature = metadata.get("tx_signature")

        activity_content = {
            "type": "check_out",
            "session_id": session_id,
            "camera_id": self._camera_id,
            "user": wallet_address,
            "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
            "duration_seconds": duration_seconds,
            "tx_signature": tx_signature,
            "activity_count": metadata.get("activity_count"),
        }

        success = self._buffer_activity(
            wallet_address,
            activity_content,
            ACTIVITY_TYPE_CHECK_OUT,
            transaction_signature=tx_signature,  # Pass separately for timeline event
        )

        if success:
            logger.info(
                f"👋 Check-out activity buffered for {wallet_address[:8]}... session={session_id[:16]}... ({duration_seconds}s)"
            )

        return success

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
            wallet_address, activity_content, ACTIVITY_TYPE_STREAM
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
            wallet_address, activity_content, ACTIVITY_TYPE_STREAM
        )

        if success:
            logger.info(
                f"📡 Stream end activity buffered for {wallet_address[:8]}... -> {playback_id} ({duration_seconds}s)"
            )

        return success

    def buffer_cv_activity(
        self,
        app_name: str,
        competitors: List[Dict[str, Any]],
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Buffer a CV app activity (competition result, exercise completion, etc.)

        Called when a CV app session completes.

        Args:
            app_name: Name of the CV app (e.g., 'pushup', 'pullup', 'squat')
            competitors: List of competitor results with stats
                [{'wallet_address': str, 'display_name': str, 'stats': {...}}, ...]
            duration_seconds: Duration of the activity/competition
            metadata: Optional additional metadata

        Returns:
            True if buffered successfully
        """
        metadata = metadata or {}

        # Sort competitors by reps (descending) to determine rankings
        sorted_competitors = sorted(
            competitors,
            key=lambda c: c.get("stats", {}).get("reps", 0),
            reverse=True,
        )

        # Build results for each participant
        results = []
        for rank, comp in enumerate(sorted_competitors, 1):
            results.append({
                "wallet_address": comp.get("wallet_address"),
                "display_name": comp.get("display_name"),
                "rank": rank,
                "stats": comp.get("stats", {}),
            })

        # Create activity for each participant
        success_count = 0
        for comp in competitors:
            wallet_address = comp.get("wallet_address")
            if not wallet_address:
                continue

            user_stats = comp.get("stats", {})

            activity_content = {
                "type": "cv_activity",
                "app_name": app_name,
                "camera_id": self._camera_id,
                "user": wallet_address,
                "timestamp": metadata.get("timestamp", int(time.time() * 1000)),
                "duration_seconds": duration_seconds,
                "results": results,  # All participants' results
                "user_stats": user_stats,  # This user's stats
                "participant_count": len(competitors),
            }

            # CV activity metadata for timeline display (unencrypted summary)
            cv_activity_meta = {
                "app_name": app_name,
                "duration_seconds": duration_seconds,
                "participant_count": len(competitors),
                "results": results,
                "user_stats": user_stats,
            }

            success = self._buffer_activity(
                wallet_address,
                activity_content,
                ACTIVITY_TYPE_CV_APP,
                cv_activity_meta=cv_activity_meta,
            )

            if success:
                success_count += 1

        if success_count > 0:
            self._buffered_cv_activities = getattr(self, "_buffered_cv_activities", 0) + 1
            logger.info(
                f"🏋️ CV activity '{app_name}' buffered for {success_count}/{len(competitors)} participants"
            )

        return success_count > 0

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
