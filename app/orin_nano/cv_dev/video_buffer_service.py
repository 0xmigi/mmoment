"""
Video Buffer Service

Drop-in replacement for BufferService that reads from video files instead of camera.
Provides playback controls for CV app development and testing.
"""

import cv2
import time
import threading
import numpy as np
import logging
import os
from collections import deque
from typing import Optional, Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoBufferService")

# Check for GPU rotation support (same as BufferService)
GPU_ROTATION_AVAILABLE = False
try:
    import cupy as cp
    GPU_ROTATION_AVAILABLE = True
    logger.info("CuPy available - GPU frame rotation enabled")
except ImportError:
    logger.info("CuPy not available - using CPU frame rotation")


class VideoBufferService:
    """
    Video file-based frame source for CV development.

    Drop-in replacement for BufferService that reads from video files
    instead of camera hardware. Provides playback controls for testing.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, video_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoBufferService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, video_path: Optional[str] = None):
        if self._initialized:
            # If already initialized and new video path provided, load it
            if video_path and video_path != self._current_video_path:
                self.load_video(video_path)
            return

        # Core components
        self._initialized = True
        self._video = None
        self._current_video_path = None
        self._running = False
        self._buffer_thread = None
        self._stop_event = threading.Event()

        # Frame buffer (matches BufferService)
        self._buffer_lock = threading.Lock()
        self._buffer_size = 15
        self._frame_buffer = deque(maxlen=self._buffer_size)
        self._latest_frame = None
        self._latest_timestamp = 0
        self._frame_counter = 0

        # Video properties
        self._width = 720   # After rotation: portrait width
        self._height = 1280 # After rotation: portrait height
        self._fps = 30
        self._video_fps = 30
        self._total_frames = 0
        self._duration = 0

        # Playback state
        self._playback_lock = threading.Lock()
        self._playing = False
        self._current_frame_num = 0
        self._speed = 1.0
        self._loop = True
        self._step_requested = 0  # -1 for back, 1 for forward, 0 for none

        # Rotation control (YouTube videos don't need rotation, camera does)
        self._skip_rotation = True  # Default: skip rotation for pre-recorded videos

        # Statistics
        self._stats_lock = threading.Lock()
        self._fps_actual = 0
        self._last_fps_calc_time = time.time()
        self._frames_since_last_calc = 0

        # Health
        self._health_status = "Not started"

        # Compatibility with BufferService (used by routes.py health endpoint)
        self._camera_index = "video"  # Indicates video source instead of /dev/videoX

        # Injected services (for processed frames)
        self._gpu_face_service = None
        self._gesture_service = None
        self._pose_service = None
        self._app_manager = None

        logger.info("VideoBufferService initialized")

        # Load initial video if provided
        if video_path:
            self.load_video(video_path)

    def _rotate_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame 90 degrees counter-clockwise (landscape video → portrait display)."""
        if GPU_ROTATION_AVAILABLE:
            try:
                gpu_frame = cp.asarray(frame)
                # Rotate 90 degrees counter-clockwise: transpose + flip horizontally
                rotated_gpu = cp.flip(gpu_frame.transpose(1, 0, 2), axis=1)
                return cp.asnumpy(rotated_gpu)
            except Exception as e:
                logger.warning(f"GPU rotation failed, using CPU: {e}")
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def load_video(self, video_path: str) -> bool:
        """
        Load a video file for playback.

        Args:
            video_path: Path to the video file

        Returns:
            True if successful, False otherwise
        """
        # Stop current playback if running
        was_running = self._running
        if was_running:
            self.stop()

        try:
            # Expand path
            video_path = os.path.expanduser(video_path)

            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False

            # Open video
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False

            # Get video properties
            self._video_fps = video.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self._duration = self._total_frames / self._video_fps if self._video_fps > 0 else 0

            video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Close old video if exists
            if self._video is not None:
                self._video.release()

            self._video = video
            self._current_video_path = video_path
            self._current_frame_num = 0

            logger.info(f"Loaded video: {video_path}")
            logger.info(f"  Resolution: {video_width}x{video_height}")
            logger.info(f"  FPS: {self._video_fps}")
            logger.info(f"  Frames: {self._total_frames}")
            logger.info(f"  Duration: {self._duration:.2f}s")

            # Start playback thread (or restart if was running)
            self._start_playback_thread()

            return True

        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False

    def _start_playback_thread(self) -> None:
        """Start or restart the playback thread."""
        # Stop existing thread if running
        if self._buffer_thread and self._buffer_thread.is_alive():
            self._stop_event.set()
            self._buffer_thread.join(timeout=2.0)

        # Reset and start new thread
        self._stop_event.clear()
        self._running = True
        self._playing = True
        self._health_status = "Running"
        self._buffer_thread = threading.Thread(
            target=self._buffer_frames_loop,
            daemon=True,
            name="VideoBufferThread"
        )
        self._buffer_thread.start()
        logger.info("Video playback thread started")

    def start(self) -> bool:
        """Start the video playback buffer thread."""
        with self._lock:
            if self._running:
                logger.info("VideoBufferService already running")
                return True

            # Allow starting without a video - will wait for load_video() call
            if self._video is None or not self._video.isOpened():
                logger.info("VideoBufferService starting (no video loaded yet - use /api/dev/load)")
                self._running = True
                self._health_status = "Waiting for video"
                return True

            try:
                # Reset stop event
                self._stop_event.clear()

                # Start buffer thread
                self._running = True
                self._playing = True  # Auto-play on start
                self._buffer_thread = threading.Thread(
                    target=self._buffer_frames_loop,
                    daemon=True,
                    name="VideoBufferThread"
                )
                self._buffer_thread.start()

                logger.info("VideoBufferService started")
                self._health_status = "Running"
                return True

            except Exception as e:
                logger.error(f"Failed to start VideoBufferService: {e}")
                self.stop()
                return False

    def _buffer_frames_loop(self) -> None:
        """
        Continuous loop that reads frames from video and adds to buffer.
        Handles playback controls (play/pause/seek/speed).
        """
        logger.info("Video buffer loop started")

        try:
            while not self._stop_event.is_set():
                with self._playback_lock:
                    playing = self._playing
                    speed = self._speed
                    loop = self._loop
                    step = self._step_requested
                    self._step_requested = 0

                # Handle step request
                if step != 0:
                    self._do_step(step)
                    continue

                # If paused, just wait
                if not playing:
                    time.sleep(0.05)
                    continue

                # Read next frame
                ret, frame = self._video.read()

                if not ret:
                    # End of video
                    if loop:
                        # Loop back to start
                        self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._current_frame_num = 0
                        logger.info("Video looped to start")
                        continue
                    else:
                        # Pause at end
                        with self._playback_lock:
                            self._playing = False
                        logger.info("Video ended")
                        continue

                # Update frame number
                self._current_frame_num = int(self._video.get(cv2.CAP_PROP_POS_FRAMES))

                # Check if frame needs rotation (if landscape, rotate to portrait)
                # Skip rotation for pre-recorded videos (they're already correctly oriented)
                if not self._skip_rotation and frame.shape[1] > frame.shape[0]:
                    frame = self._rotate_frame_gpu(frame)
                    # Only resize to target when rotating (camera simulation)
                    if frame.shape[0] != self._height or frame.shape[1] != self._width:
                        frame = cv2.resize(frame, (self._width, self._height))
                # When skipping rotation, preserve original aspect ratio (no forced resize)

                current_time = time.time()

                # Update buffer
                with self._buffer_lock:
                    self._frame_buffer.append((frame, current_time))
                    self._latest_frame = frame
                    self._latest_timestamp = current_time
                    self._frame_counter += 1

                # Calculate FPS
                self._frames_since_last_calc += 1
                if current_time - self._last_fps_calc_time >= 1.0:
                    with self._stats_lock:
                        elapsed = current_time - self._last_fps_calc_time
                        self._fps_actual = self._frames_since_last_calc / elapsed
                        self._frames_since_last_calc = 0
                        self._last_fps_calc_time = current_time

                # Frame timing based on speed
                # At speed 1.0, match video FPS. At speed 2.0, go twice as fast.
                if speed > 0:
                    frame_delay = (1.0 / self._video_fps) / speed
                    time.sleep(max(0, frame_delay - 0.001))  # Small buffer for processing

        except Exception as e:
            logger.error(f"Error in video buffer loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Video buffer loop stopped")
            with self._lock:
                self._running = False

    def _do_step(self, direction: int) -> None:
        """Step forward or backward one frame."""
        with self._playback_lock:
            self._playing = False  # Pause when stepping

        target_frame = self._current_frame_num + direction
        target_frame = max(0, min(target_frame, self._total_frames - 1))

        self._video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self._video.read()

        if ret:
            self._current_frame_num = target_frame

            # Rotate/resize as needed (skip rotation for pre-recorded videos)
            if not self._skip_rotation and frame.shape[1] > frame.shape[0]:
                frame = self._rotate_frame_gpu(frame)
                # Only resize to target when rotating (camera simulation)
                if frame.shape[0] != self._height or frame.shape[1] != self._width:
                    frame = cv2.resize(frame, (self._width, self._height))
            # When skipping rotation, preserve original aspect ratio (no forced resize)

            current_time = time.time()
            with self._buffer_lock:
                self._frame_buffer.append((frame, current_time))
                self._latest_frame = frame
                self._latest_timestamp = current_time
                self._frame_counter += 1

            logger.info(f"Stepped to frame {target_frame}/{self._total_frames}")

    # =========================================================================
    # Playback Controls (NEW - dev mode specific)
    # =========================================================================

    def play(self) -> None:
        """Resume playback."""
        with self._playback_lock:
            self._playing = True
        logger.info("Playback resumed")

    def pause(self) -> None:
        """Pause playback."""
        with self._playback_lock:
            self._playing = False
        logger.info("Playback paused")

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful
        """
        if self._video is None:
            return False

        frame_number = max(0, min(frame_number, self._total_frames - 1))

        with self._playback_lock:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame_num = frame_number

        logger.info(f"Seeked to frame {frame_number}")
        return True

    def seek_time(self, seconds: float) -> bool:
        """
        Seek to a specific time in seconds.

        Args:
            seconds: Time in seconds to seek to

        Returns:
            True if successful
        """
        frame_number = int(seconds * self._video_fps)
        return self.seek(frame_number)

    def set_speed(self, multiplier: float) -> None:
        """
        Set playback speed.

        Args:
            multiplier: Speed multiplier (0.5 = half speed, 2.0 = double speed)
                       Use 0 or negative for max speed (no frame delay)
        """
        with self._playback_lock:
            self._speed = max(0, multiplier)
        logger.info(f"Playback speed set to {multiplier}x")

    def set_loop(self, enabled: bool) -> None:
        """Enable or disable looping."""
        with self._playback_lock:
            self._loop = enabled
        logger.info(f"Loop {'enabled' if enabled else 'disabled'}")

    def step_forward(self) -> None:
        """Step forward one frame."""
        with self._playback_lock:
            self._step_requested = 1

    def step_backward(self) -> None:
        """Step backward one frame."""
        with self._playback_lock:
            self._step_requested = -1

    def set_rotation(self, enabled: bool) -> None:
        """Enable or disable frame rotation (disable for pre-recorded videos)."""
        self._skip_rotation = not enabled
        logger.info(f"Frame rotation {'enabled' if enabled else 'disabled'}")

    def get_playback_state(self) -> Dict:
        """Get current playback state."""
        with self._playback_lock:
            return {
                "video_path": self._current_video_path,
                "playing": self._playing,
                "current_frame": self._current_frame_num,
                "total_frames": self._total_frames,
                "current_time": self._current_frame_num / self._video_fps if self._video_fps > 0 else 0,
                "duration": self._duration,
                "speed": self._speed,
                "loop": self._loop,
                "video_fps": self._video_fps,
                "progress": self._current_frame_num / self._total_frames if self._total_frames > 0 else 0,
                "rotation_enabled": not self._skip_rotation
            }

    # =========================================================================
    # BufferService Interface (must match exactly)
    # =========================================================================

    def get_frame(self, processed=False) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame from the buffer."""
        with self._buffer_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._latest_timestamp
            return None, 0

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (reference, no copy for performance)."""
        with self._buffer_lock:
            return self._latest_frame

    def get_processed_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame with CV processing applied."""
        with self._buffer_lock:
            if self._latest_frame is None:
                return None, 0
            frame = self._latest_frame.copy()
            timestamp = self._latest_timestamp

        # Apply processing from injected services
        if self._gpu_face_service:
            frame = self._gpu_face_service.get_processed_frame(frame)
        if self._gesture_service:
            frame = self._gesture_service.get_processed_frame(frame)
        if self._pose_service:
            frame = self._pose_service.get_processed_frame(frame)
        if self._app_manager:
            frame = self._app_manager.get_processed_frame(frame)

        return frame, timestamp

    def get_clean_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get raw frame without annotations."""
        with self._buffer_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._latest_timestamp
            return None, 0

    def get_annotated_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame with CV annotations always applied."""
        with self._buffer_lock:
            if self._latest_frame is None:
                return None, 0
            frame = self._latest_frame.copy()
            timestamp = self._latest_timestamp

        # Apply all annotations
        if self._gpu_face_service:
            frame = self._gpu_face_service.draw_annotations(frame)
        if self._pose_service:
            frame = self._pose_service.draw_annotations(frame)
        if self._app_manager:
            frame = self._app_manager.draw_annotations(frame)

        return frame, timestamp

    def inject_services(self, gesture_service=None, gpu_face_service=None,
                       pose_service=None, app_manager=None):
        """Inject services for frame processing."""
        if gesture_service:
            self._gesture_service = gesture_service
            logger.info(f"Injected gesture service")
        if gpu_face_service:
            self._gpu_face_service = gpu_face_service
            logger.info(f"Injected GPU face service")
        if pose_service:
            self._pose_service = pose_service
            logger.info(f"Injected pose service")
        if app_manager:
            self._app_manager = app_manager
            logger.info(f"Injected app manager")

    def get_buffer_frames(self, count: int = 30) -> List[Tuple[np.ndarray, float]]:
        """Get recent frames from buffer."""
        with self._buffer_lock:
            count = min(count, 10)
            frames_count = min(count, len(self._frame_buffer))
            recent_frames = list(self._frame_buffer)[-frames_count:]
            return [(frame.copy(), ts) for frame, ts in recent_frames]

    def get_status(self) -> Dict:
        """Get service status."""
        with self._stats_lock:
            fps = self._fps_actual

        with self._buffer_lock:
            buffer_size = len(self._frame_buffer)
            buffer_capacity = self._frame_buffer.maxlen
            frame_count = self._frame_counter

        playback = self.get_playback_state()

        return {
            "running": self._running,
            "health": self._health_status,
            "fps": round(fps, 2),
            "buffer_size": buffer_size,
            "buffer_capacity": buffer_capacity,
            "frame_count": frame_count,
            "resolution": f"{self._width}x{self._height}",
            "mode": "video",
            "playback": playback
        }

    def stop(self) -> None:
        """Stop the video buffer service."""
        logger.info("Stopping VideoBufferService")

        self._stop_event.set()

        if self._buffer_thread and self._buffer_thread.is_alive():
            self._buffer_thread.join(timeout=2.0)

        with self._lock:
            self._running = False
            self._playing = False

        # Don't release video - keep it loaded for seeking while paused

        logger.info("VideoBufferService stopped")


# Global function to get the video buffer service instance
_video_buffer_instance = None

def get_video_buffer_service(video_path: Optional[str] = None) -> VideoBufferService:
    """Get the singleton VideoBufferService instance."""
    global _video_buffer_instance
    if _video_buffer_instance is None:
        _video_buffer_instance = VideoBufferService(video_path)
    elif video_path:
        _video_buffer_instance.load_video(video_path)
    return _video_buffer_instance
