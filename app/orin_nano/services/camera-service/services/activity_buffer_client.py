"""
Activity Buffer Client

HTTP client to POST encrypted activities to the backend buffer.
Activities are buffered during active sessions and committed to blockchain at checkout.

This implements the privacy-preserving timeline architecture where:
1. Jetson encrypts activities locally
2. Backend buffers encrypted activities during session
3. Auto-checkout bot commits bundled activities to blockchain
"""

import os
import json
import logging
import requests
import threading
from typing import Dict, List, Optional, Any
from queue import Queue, Empty

logger = logging.getLogger("ActivityBufferClient")


class ActivityBufferClient:
    """
    Client for sending encrypted activities to the backend buffer.

    Activities are queued and sent asynchronously to avoid blocking
    the main frame processing loop.
    """

    def __init__(self, backend_url: str = None):
        """
        Initialize the activity buffer client.

        Args:
            backend_url: Backend API URL (defaults to Railway production URL)
        """
        self.backend_url = backend_url or os.environ.get(
            'BACKEND_URL',
            'https://mmoment-production.up.railway.app'
        )

        # Activity queue for async sending
        self._queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._sender_thread: Optional[threading.Thread] = None

        # Statistics
        self._sent_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

        # Start the sender thread
        self._start_sender()

        logger.info(f"Activity buffer client initialized, backend: {self.backend_url}")

    def _start_sender(self):
        """Start the background sender thread."""
        self._stop_event.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="ActivityBufferSender"
        )
        self._sender_thread.start()
        logger.info("Activity buffer sender thread started")

    def _sender_loop(self):
        """Background loop that sends queued activities to backend."""
        while not self._stop_event.is_set():
            try:
                # Wait for an activity with timeout
                activity = self._queue.get(timeout=1.0)

                # Send to backend
                self._send_activity(activity)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in sender loop: {e}")

    def _send_activity(self, activity: Dict[str, Any]) -> bool:
        """
        Send a single activity to the backend buffer.

        Args:
            activity: Activity dict with sessionId, encryptedContent, etc.

        Returns:
            True if sent successfully
        """
        try:
            response = requests.post(
                f"{self.backend_url}/api/session/activity",
                json=activity,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self._sent_count += 1
                    logger.info(f"✅ Activity buffered: {activity.get('activityType', 'unknown')} for session {activity.get('sessionId', 'unknown')[:8]}...")
                    return True
                else:
                    self._error_count += 1
                    self._last_error = result.get('error', 'Unknown error')
                    logger.error(f"❌ Backend rejected activity: {self._last_error}")
                    return False
            else:
                self._error_count += 1
                self._last_error = f"HTTP {response.status_code}"
                logger.error(f"❌ Backend returned HTTP {response.status_code}: {response.text[:200]}")
                return False

        except requests.exceptions.RequestException as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"❌ Network error sending activity: {e}")
            return False
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"❌ Error sending activity: {e}")
            return False

    def buffer_activity(
        self,
        session_id: str,
        camera_id: str,
        user_pubkey: str,
        encrypted_activity: Dict[str, Any],
        transaction_signature: Optional[str] = None
    ) -> bool:
        """
        Queue an encrypted activity for buffering.

        This method is non-blocking - the activity is queued and sent
        by the background thread.

        Args:
            session_id: Current session ID
            camera_id: Camera PDA address
            user_pubkey: User's wallet address who triggered the activity
            encrypted_activity: Dict from ActivityEncryptionService.encrypt_activity()
                - encryptedContent: base64 encrypted content
                - nonce: base64 nonce
                - accessGrants: list of {pubkey, encryptedKey}
                - activityType: activity type enum
                - timestamp: millisecond timestamp
            transaction_signature: Optional Solana transaction signature for Solscan link

        Returns:
            True if queued successfully
        """
        try:
            # Build the activity payload for backend
            activity = {
                'sessionId': session_id,
                'cameraId': camera_id,
                'userPubkey': user_pubkey,
                'timestamp': encrypted_activity.get('timestamp', 0),
                'activityType': encrypted_activity.get('activityType', 50),
                'encryptedContent': encrypted_activity.get('encryptedContent', ''),
                'nonce': encrypted_activity.get('nonce', ''),
                'accessGrants': encrypted_activity.get('accessGrants', [])
            }

            # Include transaction signature if provided (for check_in/check_out Solscan links)
            if transaction_signature:
                activity['transactionSignature'] = transaction_signature

            # Queue for async sending
            self._queue.put(activity)

            logger.debug(f"Queued activity for session {session_id[:8]}..., queue size: {self._queue.qsize()}")
            return True

        except Exception as e:
            logger.error(f"Failed to queue activity: {e}")
            return False

    def buffer_activity_sync(
        self,
        session_id: str,
        camera_id: str,
        user_pubkey: str,
        encrypted_activity: Dict[str, Any]
    ) -> bool:
        """
        Synchronously send an encrypted activity to the buffer.

        Use this when you need to ensure the activity is sent before continuing.
        Blocks until the request completes.

        Args:
            Same as buffer_activity()

        Returns:
            True if sent successfully
        """
        activity = {
            'sessionId': session_id,
            'cameraId': camera_id,
            'userPubkey': user_pubkey,
            'timestamp': encrypted_activity.get('timestamp', 0),
            'activityType': encrypted_activity.get('activityType', 50),
            'encryptedContent': encrypted_activity.get('encryptedContent', ''),
            'nonce': encrypted_activity.get('nonce', ''),
            'accessGrants': encrypted_activity.get('accessGrants', [])
        }

        return self._send_activity(activity)

    def get_session_activities(self, session_id: str) -> Optional[List[Dict]]:
        """
        Fetch buffered activities for a session.

        This is typically called by the auto-checkout bot, not by Jetson.

        Args:
            session_id: Session ID to fetch activities for

        Returns:
            List of activity dicts, or None on error
        """
        try:
            response = requests.get(
                f"{self.backend_url}/api/session/activities/{session_id}",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('activities', [])

            logger.error(f"Failed to fetch activities for session {session_id}: HTTP {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error fetching session activities: {e}")
            return None

    def clear_session_activities(self, session_id: str) -> bool:
        """
        Clear buffered activities after successful checkout.

        This is typically called by the auto-checkout bot after
        committing activities to blockchain.

        Args:
            session_id: Session ID to clear

        Returns:
            True if cleared successfully
        """
        try:
            response = requests.delete(
                f"{self.backend_url}/api/session/activities/{session_id}",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.info(f"✅ Cleared {result.get('deleted', 0)} activities for session {session_id[:8]}...")
                    return True

            logger.error(f"Failed to clear activities for session {session_id}: HTTP {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Error clearing session activities: {e}")
            return False

    def get_status(self) -> Dict:
        """Get client status and statistics."""
        return {
            'backend_url': self.backend_url,
            'queue_size': self._queue.qsize(),
            'sent_count': self._sent_count,
            'error_count': self._error_count,
            'last_error': self._last_error,
            'sender_running': self._sender_thread.is_alive() if self._sender_thread else False
        }

    def stop(self):
        """Stop the sender thread."""
        logger.info("Stopping activity buffer client...")
        self._stop_event.set()

        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2.0)

        logger.info("Activity buffer client stopped")


# Global instance
_activity_buffer_client = None


def get_activity_buffer_client() -> ActivityBufferClient:
    """Get the activity buffer client singleton."""
    global _activity_buffer_client
    if _activity_buffer_client is None:
        _activity_buffer_client = ActivityBufferClient()
    return _activity_buffer_client
