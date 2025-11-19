"""
CV Apps Client - Communicates with cv-apps service

Allows camera service to:
- Forward frames + detections to CV apps
- Get competition state/stats
- Optional: render overlays
"""

import logging
import requests
import cv2
import numpy as np
from typing import Dict, List, Optional
import time

logger = logging.getLogger("CVAppsClient")


class CVAppsClient:
    """Client for communicating with cv-apps service"""

    def __init__(self, base_url: str = "http://localhost:5004"):
        self.base_url = base_url
        self.active_app: Optional[str] = None
        self.enabled = True
        self.last_request_time = 0
        self.request_interval = 0.033  # ~30fps max

    def check_health(self) -> bool:
        """Check if cv-apps service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"CV apps service not available: {e}")
            return False

    def activate_app(self, app_name: str) -> bool:
        """
        Activate a specific CV app.

        Args:
            app_name: Name of app to activate (e.g., 'pushup')

        Returns:
            True if activated successfully
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/apps/{app_name}/activate",
                timeout=5
            )

            if response.status_code == 200:
                self.active_app = app_name
                logger.info(f"Activated CV app: {app_name}")
                return True
            else:
                logger.error(f"Failed to activate app: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to activate app: {e}")
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        throttle: bool = True
    ) -> Optional[Dict]:
        """
        Send frame and detections to active CV app for processing.

        Args:
            frame: Video frame
            detections: List of detection dicts with track_id, wallet, bbox
            throttle: If True, throttle requests to avoid overload

        Returns:
            Competition state dict or None if failed
        """
        if not self.enabled or not self.active_app:
            return None

        # Throttle requests
        if throttle:
            now = time.time()
            if now - self.last_request_time < self.request_interval:
                return None
            self.last_request_time = now

        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Send as multipart/form-data for efficiency
            files = {
                'frame': ('frame.jpg', buffer.tobytes(), 'image/jpeg')
            }

            import json
            data = {
                'detections': json.dumps(detections)
            }

            response = requests.post(
                f"{self.base_url}/api/process",
                files=files,
                data=data,
                timeout=1.0  # Fast timeout - don't block video loop
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('state')
            else:
                logger.debug(f"CV apps request failed: {response.status_code}")
                return None

        except requests.Timeout:
            logger.debug("CV apps request timeout")
            return None
        except Exception as e:
            logger.debug(f"CV apps request failed: {e}")
            return None

    def get_state(self, app_name: str = None) -> Optional[Dict]:
        """
        Get current state from CV app.

        Args:
            app_name: Specific app to query, or use active app

        Returns:
            State dict or None
        """
        app = app_name or self.active_app
        if not app:
            return None

        try:
            response = requests.get(
                f"{self.base_url}/api/apps/{app}/state",
                timeout=2
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.debug(f"Failed to get state: {e}")
            return None

    def render_overlay(
        self,
        frame: np.ndarray,
        app_name: str = None
    ) -> np.ndarray:
        """
        Get frame with CV app overlay rendered.

        Args:
            frame: Video frame
            app_name: Specific app, or use active app

        Returns:
            Frame with overlay (or original frame if failed)
        """
        app = app_name or self.active_app
        if not app:
            return frame

        try:
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)

            files = {
                'frame': ('frame.jpg', buffer.tobytes(), 'image/jpeg')
            }

            response = requests.post(
                f"{self.base_url}/api/apps/{app}/overlay",
                files=files,
                timeout=1.0
            )

            if response.status_code == 200:
                # Decode response image
                img_array = np.frombuffer(response.content, np.uint8)
                overlay_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return overlay_frame if overlay_frame is not None else frame

            return frame

        except Exception as e:
            logger.debug(f"Failed to render overlay: {e}")
            return frame

    def start_competition(self, app_name: str, config: Dict) -> Optional[Dict]:
        """
        Start a competition via direct API call.

        Args:
            app_name: App to use (e.g., 'pushup')
            config: Competition config (competitors, duration, bet_amount, etc.)

        Returns:
            Start result dict or None
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/apps/{app_name}/start",
                json=config,
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to start competition: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to start competition: {e}")
            return None

    def end_competition(self, app_name: str = None) -> Optional[Dict]:
        """End active competition"""
        app = app_name or self.active_app
        if not app:
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/apps/{app}/end",
                timeout=5
            )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            logger.error(f"Failed to end competition: {e}")
            return None


# Global singleton instance
_cv_apps_client: Optional[CVAppsClient] = None


def get_cv_apps_client() -> CVAppsClient:
    """Get global CV apps client instance"""
    global _cv_apps_client
    if _cv_apps_client is None:
        _cv_apps_client = CVAppsClient()
        # Auto-activate pushup app on first use
        try:
            if _cv_apps_client.check_health():
                _cv_apps_client.activate_app('pushup')
                logger.info("CV apps client initialized with pushup app")
            else:
                logger.warning("CV apps service not available")
        except Exception as e:
            logger.warning(f"Failed to initialize CV apps: {e}")
    return _cv_apps_client
