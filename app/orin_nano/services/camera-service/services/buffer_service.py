"""
Buffer Service

A high-performance, thread-safe buffer service that provides a constant feed of camera frames
as the single source of truth for all camera operations.
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
logger = logging.getLogger("BufferService")

class BufferService:
    """
    Singleton service that maintains a circular buffer of raw camera frames.
    This buffer is the source of truth for all camera operations and cannot be
    modified by external processes.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BufferService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Core components
        self._initialized = True
        self._camera = None
        self._running = False
        self._buffer_thread = None
        self._stop_event = threading.Event()
        
        # Frame buffer (raw frames only)
        self._buffer_lock = threading.Lock()
        self._buffer_size = 15  # Reduced from 30 to 15 to save even more memory
        self._frame_buffer = deque(maxlen=self._buffer_size)
        self._latest_frame = None
        self._latest_timestamp = 0
        self._frame_counter = 0
        
        # SIMPLE HARDCODED SETTINGS - NO MORE CONFIG FILE MESS
        # NOTE: Camera captures at 1280x720, then rotated to 720x1280 for mobile
        self._width = 720   # After rotation: portrait width
        self._height = 1280 # After rotation: portrait height  
        self._fps = 30
        self._buffersize = 1
        self._reconnect_attempts = 3
        self._reconnect_delay = 0.3
        logger.info(f"Camera settings: {self._width}x{self._height}@{self._fps}fps")
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._fps_actual = 0
        self._last_fps_calc_time = time.time()
        self._frames_since_last_calc = 0
        
        # Health check
        self._last_health_check = 0
        self._health_status = "Not started"
        
        logger.info(f"BufferService initialized with settings: {self._width}x{self._height}@{self._fps}fps")

    def _load_camera_config(self):
        """Load camera configuration from file"""
        config_path = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/config/camera_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
                    logger.info(f"Loaded camera config from {config_path}")
                    return config
            else:
                logger.warning(f"Camera config file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
        
        # Default config
        return {
            'camera': {
                # REMOVED: No more hardcoded device preference
                'width': 1280,
                'height': 720,
                'fps': 30,
                'buffersize': 3,
                'reconnect_attempts': 5,
                'reconnect_delay': 0.5
            }
        }

    def start(self) -> bool:
        """
        Start the camera and frame buffer thread.
        Returns True if successful, False otherwise.
        """
        with self._lock:
            if self._running:
                logger.info("BufferService already running")
                return True
                
            try:
                # Try to open the camera
                self._camera = self._open_camera()
                
                if self._camera is None or not self._camera.isOpened():
                    logger.error("Failed to open camera")
                    return False
                
                # Reset stop event
                self._stop_event.clear()
                
                # Start buffer thread
                self._running = True
                self._buffer_thread = threading.Thread(
                    target=self._buffer_frames_loop,
                    daemon=True,
                    name="BufferThread"
                )
                self._buffer_thread.start()
                
                logger.info(f"BufferService started successfully at {self._width}x{self._height}@{self._fps}fps")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start BufferService: {e}")
                self.stop()
                return False

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Try to open the camera with the best available settings.
        Returns the camera object if successful, None otherwise.
        """
        # Release any existing camera
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            time.sleep(self._reconnect_delay)
        
        # SIMPLE CAMERA DETECTION - NO MORE HARDCODED MESS
        import glob
        available_devices = sorted(glob.glob('/dev/video*'))
        logger.info(f"Found video devices: {available_devices}")
        
        # ONE SIMPLE PRIORITY: Try devices in order until one works
        configs_to_try = []
        for device_path in available_devices:
            device_index = int(device_path.replace('/dev/video', ''))
            configs_to_try.append((device_path, device_index, f"Device {device_path}", self._width, self._height, self._fps))
        
        # Fallback to indices if devices don't work
        for idx in [0, 1, 2]:
            configs_to_try.append((None, idx, f"Index {idx}", self._width, self._height, self._fps))
            
        # Try each configuration until one works
        for device_path, index, name, width, height, fps in configs_to_try:
            # Camera captures at landscape 1280x720, then we rotate to portrait 720x1280
            camera_width, camera_height = 1280, 720
            try:
                logger.info(f"Trying camera {name}")
                
                # Open by direct device path if specified
                if device_path:
                    camera = cv2.VideoCapture(device_path)
                else:
                    camera = cv2.VideoCapture(index)
                
                if not camera.isOpened():
                    logger.warning(f"Failed to open camera {name}")
                    camera.release()
                    continue
                
                # Force MJPEG format FIRST for better FPS performance on Logitech cameras
                if device_path == '/dev/video1':
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    logger.info(f"Set MJPEG format for Logitech camera {device_path}")
                
                # Set camera properties AFTER setting format  
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                camera.set(cv2.CAP_PROP_FPS, fps)
                
                # Set additional camera properties for better stability
                camera.set(cv2.CAP_PROP_BUFFERSIZE, self._buffersize)
                
                # Test capture with multiple retries and FPS validation
                success = False
                test_frames = 0
                start_time = time.time()
                
                for retry in range(self._reconnect_attempts):
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        test_frames += 1
                        # For IMX477 camera (/dev/video0), be more strict about performance
                        if device_path == '/dev/video0':
                            # Test for at least 2 frames in 1 second to ensure decent FPS
                            if test_frames >= 2:
                                elapsed = time.time() - start_time
                                if elapsed < 1.0:
                                    time.sleep(1.0 - elapsed)  # Wait for full second
                                actual_fps = test_frames / (time.time() - start_time)
                                if actual_fps < 2.0:  # Reject if less than 2 FPS
                                    logger.warning(f"Camera {name} has poor performance ({actual_fps:.1f} FPS), skipping")
                                    break
                        success = True
                        break
                    logger.warning(f"Camera {name} retry {retry+1}/{self._reconnect_attempts}")
                    time.sleep(self._reconnect_delay)
                
                if not success:
                    logger.warning(f"Camera {name} opened but failed to read frame or has poor performance")
                    camera.release()
                    continue
                
                # Successfully opened camera
                self._camera_index = index
                logger.info(f"Successfully opened camera {name} with resolution {frame.shape[1]}x{frame.shape[0]}")
                
                # REMOVED: No more preferred device saving logic
                # Camera detection is now automatic with no preferences
                
                return camera
                
            except Exception as e:
                logger.warning(f"Error opening camera {name}: {e}")
                continue
        
        # Failed to open any camera - fail fast for production DePIN device
        logger.error("No physical camera detected. DePIN camera device requires working camera hardware.")
        return None

    def _save_preferred_device(self, device_path):
        """Save the preferred device to the config file"""
        config_path = os.path.expanduser("~/mmoment/app/orin_nano/camera_service/config/camera_config.json")
        try:
            import json
            config = self._config.copy()
            
            # Update the top-level preferred device (not nested under 'camera')
            config['preferred_device'] = device_path
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Updated preferred device to {device_path} in config file")
        except Exception as e:
            logger.error(f"Error saving preferred device to config file: {e}")

    def _buffer_frames_loop(self) -> None:
        """
        Continuous loop that captures frames from the camera and adds them to the buffer.
        This is the core function that maintains the buffer as the source of truth.
        """
        last_frame_time = time.time()
        frames_captured = 0
        consecutive_failures = 0
        max_failures = self._reconnect_attempts
        camera_reset_count = 0
        max_camera_resets = 3
        
        logger.info("Buffer frames loop started")
        
        try:
            while not self._stop_event.is_set() and self._camera and self._camera.isOpened():
                # Capture frame
                try:
                    ret, frame = self._camera.read()
                except Exception as e:
                    logger.error(f"Exception during frame capture: {e}")
                    ret, frame = False, None
                
                # Rotate frame to vertical 16:9 for mobile viewing (720x1280)
                if ret and frame is not None and frame.size > 0:
                    # Rotate 90 degrees counterclockwise: 1280x720 -> 720x1280
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Check if frame capture was successful
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning(f"Failed to capture frame. Consecutive failures: {consecutive_failures}/{max_failures}")
                    
                    # Try to recover
                    if consecutive_failures >= max_failures:
                        camera_reset_count += 1
                        logger.error(f"Too many consecutive frame capture failures. Reopening camera (Reset {camera_reset_count}/{max_camera_resets})...")
                        
                        if camera_reset_count >= max_camera_resets:
                            logger.error("Maximum camera resets reached. Trying alternate video devices...")
                            # Force a full device scan by forgetting the preferred device
                            self._preferred_device = None
                            camera_reset_count = 0
                        
                        # Try to reopen the camera
                        self._camera = self._open_camera()
                        consecutive_failures = 0
                        
                        if self._camera is None:
                            logger.error("Failed to reopen camera. Stopping buffer service.")
                            break
                        
                        # Allow camera to stabilize
                        time.sleep(1.0)
                        continue
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counters on successful capture
                consecutive_failures = 0
                if frames_captured == 0:
                    camera_reset_count = 0  # Reset this counter once we get a successful frame
                
                # Get current timestamp for this frame
                current_time = time.time()
                
                # Update buffer with captured frame (thread-safe)
                with self._buffer_lock:
                    # Store frame in buffer (only keep reference, don't copy yet)
                    self._frame_buffer.append((frame, current_time))
                    # Update latest frame reference (copy only once)
                    self._latest_frame = frame  # Keep reference, copy only when requested
                    self._latest_timestamp = current_time
                    self._frame_counter += 1
                
                # Calculate actual FPS periodically
                frames_captured += 1
                self._frames_since_last_calc += 1
                
                if current_time - self._last_fps_calc_time >= 1.0:
                    with self._stats_lock:
                        elapsed = current_time - self._last_fps_calc_time
                        self._fps_actual = self._frames_since_last_calc / elapsed
                        self._frames_since_last_calc = 0
                        self._last_fps_calc_time = current_time
                        
                        # Log FPS every 10 seconds
                        if self._frame_counter % (10 * self._fps) < self._fps:
                            logger.info(f"Camera running at {self._fps_actual:.1f} FPS")
                
                # Health check
                if current_time - self._last_health_check >= 10.0:
                    self._health_check()
                    self._last_health_check = current_time
                
                # Let camera run at natural speed - don't artificially limit FPS
                # The camera will naturally provide frames at its maximum supported rate
                last_frame_time = time.time()
                
        except Exception as e:
            logger.error(f"Error in buffer frames loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Buffer frames loop stopped")
            with self._lock:
                self._running = False
                
            # Make sure to release the camera
            if self._camera:
                try:
                    self._camera.release()
                    self._camera = None
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")

    def get_frame(self, processed=False) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest raw frame from the buffer.
        If processed=True, returns a frame processed with visualizations from services.
        Returns a tuple of (frame, timestamp) or (None, 0) if no frame is available.
        """
        with self._buffer_lock:
            if self._latest_frame is not None:
                # Return a copy to prevent external modification
                return self._latest_frame.copy(), self._latest_timestamp
            return None, 0

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest raw frame from the buffer (for service compatibility).
        Returns a copy of the frame or None if no frame is available.
        """
        with self._buffer_lock:
            if self._latest_frame is not None:
                # Return a copy to prevent external modification
                return self._latest_frame.copy()
            return None

    def get_processed_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest frame with processing applied (face detection, gesture, etc).
        This requires the face and gesture services to be injected.
        Returns a tuple of (processed_frame, timestamp) or (None, 0) if no frame is available.
        """
        frame, timestamp = self.get_frame()
        
        if frame is None:
            return None, 0
            
        # Apply processing if the services are available (frame is already a copy from get_frame)
        processed_frame = frame  # Use the copy from get_frame, don't copy again
        
        # Services will be injected after initialization
        # GPU face service only (no CPU fallback)
        if hasattr(self, '_gpu_face_service'):
            processed_frame = self._gpu_face_service.get_processed_frame(processed_frame)
            
        if hasattr(self, '_gesture_service'):
            processed_frame = self._gesture_service.get_processed_frame(processed_frame)
        
        return processed_frame, timestamp

    def inject_services(self, gesture_service=None, gpu_face_service=None):
        """
        Inject services for frame processing.
        This is called after all services are initialized.
        """
        if gesture_service:
            self._gesture_service = gesture_service
            logger.info(f"✅ Injected gesture service: {gesture_service}")

        if gpu_face_service:
            self._gpu_face_service = gpu_face_service
            logger.info(f"✅ Injected GPU face service: {gpu_face_service}")

    def get_buffer_frames(self, count: int = 30) -> List[Tuple[np.ndarray, float]]:
        """
        Get a specified number of recent frames from the buffer.
        Returns a list of (frame, timestamp) tuples.
        Optimized to reduce memory usage.
        """
        with self._buffer_lock:
            # Limit count to prevent excessive memory usage
            count = min(count, 10)  # Maximum 10 frames to prevent memory issues
            frames_count = min(count, len(self._frame_buffer))
            # Return copies only when necessary
            recent_frames = list(self._frame_buffer)[-frames_count:]
            return [(frame.copy(), ts) for frame, ts in recent_frames]

    def get_jpeg_frame(self, quality: int = 90, processed: bool = False) -> Tuple[Optional[bytes], float]:
        """
        Get the latest frame encoded as JPEG.
        If processed=True, returns a processed frame with visualizations.
        Returns a tuple of (jpeg_bytes, timestamp).
        """
        if processed and hasattr(self, '_gpu_face_service'):
            frame, timestamp = self.get_processed_frame()
        else:
            frame, timestamp = self.get_frame()
            
        if frame is None:
            return None, 0
            
        # Encode the frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes(), timestamp

    def _health_check(self) -> None:
        """
        Perform a health check on the buffer service.
        Updates the health status.
        """
        if not self._running:
            self._health_status = "Not running"
            return
            
        # Check if frames are being captured
        current_time = time.time()
        with self._buffer_lock:
            if self._latest_timestamp == 0:
                self._health_status = "No frames captured yet"
                return
                
            frame_age = current_time - self._latest_timestamp
            if frame_age > 1.0:
                self._health_status = f"Frame capture delay: {frame_age:.2f}s"
                return
            
            buffer_size = len(self._frame_buffer)
            if buffer_size < self._frame_buffer.maxlen * 0.5:
                self._health_status = f"Buffer below 50% capacity: {buffer_size}/{self._frame_buffer.maxlen}"
                return
                
            # All checks passed
            self._health_status = "Healthy"

    def get_status(self) -> Dict:
        """
        Get the current status of the buffer service.
        """
        with self._stats_lock:
            fps = self._fps_actual
            
        with self._buffer_lock:
            buffer_size = len(self._frame_buffer)
            buffer_capacity = self._frame_buffer.maxlen
            frame_count = self._frame_counter
            
        return {
            "running": self._running,
            "health": self._health_status,
            "fps": round(fps, 2),
            "buffer_size": buffer_size,
            "buffer_capacity": buffer_capacity,
            "frame_count": frame_count,
            "resolution": f"{self._width}x{self._height}"
        }

    def stop(self) -> None:
        """
        Stop the buffer service and release the camera.
        """
        logger.info("Stopping BufferService")
        
        # Signal the buffer thread to stop
        self._stop_event.set()
        
        # Wait for the buffer thread to stop
        if self._buffer_thread and self._buffer_thread.is_alive():
            self._buffer_thread.join(timeout=2.0)
        
        # Release the camera
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        
        with self._lock:
            self._running = False
            
        logger.info("BufferService stopped")

# Global function to get the buffer service instance
def get_buffer_service() -> BufferService:
    """
    Get the singleton BufferService instance.
    """
    return BufferService() 