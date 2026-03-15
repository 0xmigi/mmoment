"""
CV App Manager

Loads and manages CV apps as Python modules.
Apps run in the same process as camera service for real-time performance.

Integrates with privacy-preserving timeline architecture:
- Encrypts significant CV app results with AES-256-GCM
- Buffers encrypted activities to backend
- All checked-in users receive decryption access grants
"""

import cv2
import logging
import importlib
import sys
import os
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

logger = logging.getLogger("AppManager")

# Activity type enum from state.rs
ACTIVITY_TYPE_CV_APP = 50


class AppManager:
    """
    Manages CV apps loaded as Python modules.
    Single app active at a time.

    Integrates with privacy-preserving timeline:
    - Detects significant events (score changes, reps completed)
    - Encrypts activities for all checked-in users
    - Buffers to backend for blockchain commit at checkout
    """

    def __init__(self):
        self.active_app = None
        self.active_app_name = None
        self.loaded_apps = {}

        # Storage for latest app result (for visualization)
        self._results_lock = threading.Lock()
        self._latest_result = None

        # Previous state for detecting significant changes
        self._previous_state: Dict[str, Any] = {}
        self._last_buffer_time: float = 0
        self._min_buffer_interval: float = 1.0  # Minimum 1 second between buffers

        # Privacy-preserving timeline services (lazy loaded)
        self._encryption_service = None
        self._buffer_client = None
        self._blockchain_sync = None

        # Camera configuration
        self._camera_id = os.environ.get('CAMERA_PDA', 'unknown-camera')

        # Add apps directory to Python path
        apps_path = Path("/opt/mmoment/apps")
        if str(apps_path) not in sys.path:
            sys.path.insert(0, str(apps_path))

        logger.info(f"AppManager initialized, apps path: {apps_path}, camera: {self._camera_id[:16]}...")

    def load_app(self, app_name: str) -> bool:
        """
        Load an app module.

        Args:
            app_name: Name of app (e.g., 'pushup')

        Returns:
            True if loaded successfully
        """
        try:
            # Import the app module
            if app_name in self.loaded_apps:
                logger.info(f"App {app_name} already loaded")
                return True

            # Try to import app.main or app.app
            try:
                module = importlib.import_module(f"{app_name}.app")
            except ImportError:
                module = importlib.import_module(f"{app_name}.main")

            # Get the app class (should have a get_app() function or App class)
            if hasattr(module, 'get_app'):
                app_instance = module.get_app()
            elif hasattr(module, 'App'):
                app_instance = module.App()
            else:
                logger.error(f"App {app_name} has no get_app() or App class")
                return False

            self.loaded_apps[app_name] = app_instance
            logger.info(f"Loaded app: {app_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load app {app_name}: {e}", exc_info=True)
            return False

    def activate_app(self, app_name: str) -> bool:
        """
        Activate an app (deactivates any currently active app).

        Args:
            app_name: Name of app to activate

        Returns:
            True if activated successfully
        """
        # Load if not already loaded
        if app_name not in self.loaded_apps:
            if not self.load_app(app_name):
                return False

        # Deactivate current app
        if self.active_app:
            try:
                if hasattr(self.active_app, 'on_deactivate'):
                    self.active_app.on_deactivate()
            except Exception as e:
                logger.error(f"Error deactivating {self.active_app_name}: {e}")

        # Activate new app
        self.active_app = self.loaded_apps[app_name]
        self.active_app_name = app_name

        try:
            if hasattr(self.active_app, 'on_activate'):
                self.active_app.on_activate()
        except Exception as e:
            logger.error(f"Error activating {app_name}: {e}")

        logger.info(f"Activated app: {app_name}")
        return True

    def deactivate_app(self):
        """Deactivate current app"""
        if self.active_app:
            try:
                if hasattr(self.active_app, 'on_deactivate'):
                    self.active_app.on_deactivate()
            except Exception as e:
                logger.error(f"Error deactivating {self.active_app_name}: {e}")

            logger.info(f"Deactivated app: {self.active_app_name}")

        self.active_app = None
        self.active_app_name = None

    def process_frame(self, frame_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Process frame with active app.

        Args:
            frame_data: Dict with 'detections', 'keypoints', 'timestamp', etc.

        Returns:
            App result with 'state' and 'visualization' keys, or None
        """
        if not self.active_app:
            return None

        try:
            if hasattr(self.active_app, 'process'):
                result = self.active_app.process(frame_data)
                # Store result for visualization
                with self._results_lock:
                    self._latest_result = result

                # Check if this is a significant event to buffer
                if result and self._should_buffer_activity(result):
                    self._buffer_activity(result, frame_data)

                return result
            else:
                logger.warning(f"Active app {self.active_app_name} has no process() method")
                return None

        except Exception as e:
            logger.error(f"App {self.active_app_name} processing failed: {e}", exc_info=True)
            return None

    def _should_buffer_activity(self, result: Dict[str, Any]) -> bool:
        """
        Determine if an app result should be buffered to the timeline.

        Significant events include:
        - Explicit 'should_buffer: true' flag from app
        - Score/count increases (rep completed)
        - Game/competition end events
        - Achievement unlocked

        Rate limited to avoid spam (min 1 second between buffers).
        """
        # Rate limiting
        now = time.time()
        if now - self._last_buffer_time < self._min_buffer_interval:
            return False

        # Check for explicit buffer flag
        if result.get('should_buffer'):
            return True

        # Check for discrete events
        if result.get('event'):
            return True

        # Check for state changes (score/count increases)
        state = result.get('state', {})
        if state:
            # Detect count increases (e.g., pushup reps)
            current_count = state.get('count', state.get('reps', state.get('score')))
            previous_count = self._previous_state.get('count', self._previous_state.get('reps', self._previous_state.get('score')))

            if current_count is not None and previous_count is not None:
                if current_count > previous_count:
                    # Update previous state
                    self._previous_state = state.copy()
                    return True

            # Check for game end
            if state.get('game_ended') or state.get('competition_ended'):
                return True

        return False

    def _buffer_activity(self, result: Dict[str, Any], frame_data: Dict[str, Any]):
        """
        Encrypt and buffer an activity to the backend.

        Args:
            result: App result containing state/event to buffer
            frame_data: Original frame data with detections
        """
        try:
            # Lazy load services
            if self._encryption_service is None:
                from .activity_encryption_service import get_activity_encryption_service
                self._encryption_service = get_activity_encryption_service()

            if self._buffer_client is None:
                from .activity_buffer_client import get_activity_buffer_client
                self._buffer_client = get_activity_buffer_client()

            if self._blockchain_sync is None:
                from .blockchain_session_sync import get_blockchain_session_sync
                self._blockchain_sync = get_blockchain_session_sync()

            # Get all currently checked-in users
            checked_in_wallets = list(self._blockchain_sync.checked_in_wallets)
            if not checked_in_wallets:
                logger.debug("No users checked in, skipping activity buffer")
                return

            # Build activity content
            activity_content = {
                'app': self.active_app_name,
                'type': 'cv_app_result',
                'state': result.get('state', {}),
                'event': result.get('event'),
                'timestamp': int(time.time() * 1000)
            }

            # Add user attribution if available from frame_data
            if 'user' in frame_data:
                activity_content['triggered_by'] = frame_data['user']

            # Encrypt activity for all checked-in users
            encrypted = self._encryption_service.encrypt_activity(
                activity_content=activity_content,
                users_present=checked_in_wallets,
                activity_type=ACTIVITY_TYPE_CV_APP
            )

            # Determine session_id (use first checked-in user's session or camera-based)
            session_id = self._get_session_id(checked_in_wallets)

            # Get primary user (who triggered the activity)
            user_pubkey = frame_data.get('user', checked_in_wallets[0] if checked_in_wallets else 'unknown')

            # Buffer to backend (async)
            self._buffer_client.buffer_activity(
                session_id=session_id,
                camera_id=self._camera_id,
                user_pubkey=user_pubkey,
                encrypted_activity=encrypted
            )

            # Update rate limiting
            self._last_buffer_time = time.time()

            logger.info(f"📤 Buffered activity: {self.active_app_name} for {len(checked_in_wallets)} users")

        except Exception as e:
            logger.error(f"Failed to buffer activity: {e}", exc_info=True)

    def _get_session_id(self, checked_in_wallets: List[str]) -> str:
        """
        Get a session ID for buffering activities.

        Uses the first checked-in user's session ID, or falls back to camera-based ID.
        """
        try:
            from .session_service import get_session_service
            session_service = get_session_service()

            # Try to get session ID from first checked-in user
            for wallet in checked_in_wallets:
                session = session_service.get_session_by_wallet(wallet)
                if session:
                    return session['session_id']

            # Fallback: use camera ID + date as session identifier
            return f"{self._camera_id}-{int(time.time() // 86400)}"

        except Exception as e:
            logger.warning(f"Could not get session ID: {e}")
            return f"{self._camera_id}-{int(time.time() // 86400)}"

    def get_active_app_state(self) -> Optional[Dict]:
        """Get state of active app"""
        if not self.active_app:
            return None

        try:
            if hasattr(self.active_app, 'get_state'):
                return self.active_app.get_state()
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get state from {self.active_app_name}: {e}")
            return None

    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw app overlays on frame.
        Reads stored app result and draws visualization.
        """
        if frame is None or not self.active_app:
            return frame

        with self._results_lock:
            result = self._latest_result

        if not result or 'visualization' not in result:
            return frame

        viz = result['visualization']

        # Draw text overlays
        if 'text' in viz:
            for text_item in viz['text']:
                content = text_item.get('content', '')
                pos = text_item.get('pos', (10, 50))
                color = text_item.get('color', (255, 255, 255))
                font_scale = text_item.get('font_scale', 1.5)
                thickness = text_item.get('thickness', 3)

                cv2.putText(frame, str(content), pos, cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, color, thickness, cv2.LINE_AA)

        # Draw boxes if specified
        if 'boxes' in viz:
            for box_item in viz['boxes']:
                bbox = box_item.get('bbox')
                color = box_item.get('color', (0, 255, 0))
                thickness = box_item.get('thickness', 2)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                color, thickness)

        return frame

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw app overlays on frame (same as get_processed_frame for app_manager).
        App manager doesn't have a visualization toggle, so this is identical.
        Used for the annotated stream where CV overlays are always shown.
        """
        return self.get_processed_frame(frame)


# Global instance
_app_manager = None

def get_app_manager() -> AppManager:
    """Get the app manager singleton instance"""
    global _app_manager
    if _app_manager is None:
        _app_manager = AppManager()
    return _app_manager
