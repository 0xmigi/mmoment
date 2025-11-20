"""
CV App Manager

Loads and manages CV apps as Python modules.
Apps run in the same process as camera service for real-time performance.
"""

import cv2
import logging
import importlib
import sys
import threading
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger("AppManager")


class AppManager:
    """
    Manages CV apps loaded as Python modules.
    Single app active at a time.
    """

    def __init__(self):
        self.active_app = None
        self.active_app_name = None
        self.loaded_apps = {}

        # Storage for latest app result (for visualization)
        self._results_lock = threading.Lock()
        self._latest_result = None

        # Add apps directory to Python path
        apps_path = Path("/opt/mmoment/apps")
        if str(apps_path) not in sys.path:
            sys.path.insert(0, str(apps_path))

        logger.info(f"AppManager initialized, apps path: {apps_path}")

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
                return result
            else:
                logger.warning(f"Active app {self.active_app_name} has no process() method")
                return None

        except Exception as e:
            logger.error(f"App {self.active_app_name} processing failed: {e}", exc_info=True)
            return None

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


# Global instance
_app_manager = None

def get_app_manager() -> AppManager:
    """Get the app manager singleton instance"""
    global _app_manager
    if _app_manager is None:
        _app_manager = AppManager()
    return _app_manager
