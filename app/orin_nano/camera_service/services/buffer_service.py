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
        self._frame_buffer = deque(maxlen=180)  # 6 seconds at 30fps
        self._latest_frame = None
        self._latest_timestamp = 0
        self._frame_counter = 0
        
        # Load camera settings from config
        self._config = self._load_camera_config()
        
        # Camera settings
        self._camera_index = 0
        # Check for environment variable override for camera device
        env_camera_device = os.environ.get('CAMERA_DEVICE')
        if env_camera_device:
            logger.info(f"Using camera device from environment variable: {env_camera_device}")
            self._preferred_device = env_camera_device
        else:
            self._preferred_device = self._config.get('camera', {}).get('preferred_device', '/dev/video1')
            logger.info(f"Using camera device from config: {self._preferred_device}")
            
        self._width = self._config.get('camera', {}).get('width', 1280)
        self._height = self._config.get('camera', {}).get('height', 720)
        self._fps = self._config.get('camera', {}).get('fps', 30)
        self._buffersize = self._config.get('camera', {}).get('buffersize', 3)
        self._reconnect_attempts = self._config.get('camera', {}).get('reconnect_attempts', 5)
        self._reconnect_delay = self._config.get('camera', {}).get('reconnect_delay', 0.5)
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._fps_actual = 0
        self._last_fps_calc_time = time.time()
        self._frames_since_last_calc = 0
        
        # Health check
        self._last_health_check = 0
        self._health_status = "Not started"
        
        logger.info(f"BufferService initialized with settings: {self._width}x{self._height}@{self._fps}fps, preferred device: {self._preferred_device}")

    def _load_camera_config(self):
        """Load camera configuration from file"""
        config_path = os.path.expanduser("~/mmoment/app/orin_nano/camera_service_new/config/camera_config.json")
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
                'preferred_device': '/dev/video1',
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
        
        # Get list of available video devices
        import glob
        available_devices = sorted(glob.glob('/dev/video*'))
        logger.info(f"Found video devices: {available_devices}")
        
        # Define a list of configurations to try
        # Format: (device_path, device_index, name, width, height, fps)
        configs_to_try = []
        
        # First, try the preferred device from config if it exists
        if self._preferred_device in available_devices:
            device_index = int(self._preferred_device.replace('/dev/video', ''))
            configs_to_try.append((self._preferred_device, device_index, f"Preferred device {self._preferred_device}", self._width, self._height, self._fps))
        
        # Then add the rest of the devices
        for device_path in available_devices:
            if device_path != self._preferred_device:  # Skip preferred device as it's already added
                device_index = int(device_path.replace('/dev/video', ''))
                configs_to_try.append((device_path, device_index, f"Device {device_path}", self._width, self._height, self._fps))
        
        # Also try common indices as backup
        for idx in [1, 0, 2]:
            configs_to_try.append((None, idx, f"Index {idx}", self._width, self._height, self._fps))
            
        # Try each configuration until one works
        for device_path, index, name, width, height, fps in configs_to_try:
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
                
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                camera.set(cv2.CAP_PROP_FPS, fps)
                
                # Set additional camera properties for better stability
                camera.set(cv2.CAP_PROP_BUFFERSIZE, self._buffersize)
                
                # Test capture with multiple retries
                success = False
                for retry in range(self._reconnect_attempts):
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        success = True
                        break
                    logger.warning(f"Camera {name} retry {retry+1}/{self._reconnect_attempts}")
                    time.sleep(self._reconnect_delay)
                
                if not success:
                    logger.warning(f"Camera {name} opened but failed to read frame")
                    camera.release()
                    continue
                
                # Successfully opened camera
                self._camera_index = index
                logger.info(f"Successfully opened camera {name} with resolution {frame.shape[1]}x{frame.shape[0]}")
                
                # Save this as the preferred device for next time
                if device_path and self._preferred_device != device_path:
                    self._preferred_device = device_path
                    self._save_preferred_device(device_path)
                
                return camera
                
            except Exception as e:
                logger.warning(f"Error opening camera {name}: {e}")
                continue
        
        # Failed to open any camera, create a virtual test camera
        logger.warning("No physical camera detected, creating virtual test camera")
        
        class VirtualCamera:
            def __init__(self, width, height, fps):
                self.width = width
                self.height = height
                self.fps = fps
                self.counter = 0
                self.opened = True
                
            def read(self):
                # Create a test pattern frame
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                # Add some dynamic element
                self.counter += 1
                
                # Draw a moving circle
                radius = 50
                x = int(self.width/2 + radius * np.sin(self.counter / 30))
                y = int(self.height/2 + radius * np.cos(self.counter / 30))
                
                # Draw circle and text
                cv2.circle(frame, (x, y), 50, (0, 165, 255), -1)
                cv2.putText(frame, "Virtual Camera - No Device Connected", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {self.counter}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                return True, frame
                
            def isOpened(self):
                return self.opened
                
            def release(self):
                self.opened = False
                
            def set(self, prop, value):
                return True
        
        virtual_camera = VirtualCamera(self._width, self._height, self._fps)
        self._camera_index = -1
        logger.info(f"Virtual test camera created with resolution {self._width}x{self._height}")
        return virtual_camera

    def _save_preferred_device(self, device_path):
        """Save the preferred device to the config file"""
        config_path = os.path.expanduser("~/mmoment/app/orin_nano/camera_service_new/config/camera_config.json")
        try:
            import json
            config = self._config.copy()
            
            # Update the preferred device
            if 'camera' not in config:
                config['camera'] = {}
            config['camera']['preferred_device'] = device_path
            
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
                    # Store frame in buffer
                    self._frame_buffer.append((frame.copy(), current_time))
                    # Update latest frame reference
                    self._latest_frame = frame.copy()
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
                
                # Precise timing to maintain requested FPS
                ideal_frame_time = 1.0 / self._fps
                elapsed = time.time() - last_frame_time
                sleep_time = max(0, ideal_frame_time - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
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

    def get_processed_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest frame with processing applied (face detection, gesture, etc).
        This requires the face and gesture services to be injected.
        Returns a tuple of (processed_frame, timestamp) or (None, 0) if no frame is available.
        """
        frame, timestamp = self.get_frame()
        
        if frame is None:
            return None, 0
            
        # Apply processing if the services are available
        processed_frame = frame.copy()
        
        # Services will be injected after initialization
        if hasattr(self, '_face_service'):
            processed_frame = self._face_service.get_processed_frame(processed_frame)
            
        if hasattr(self, '_gesture_service'):
            processed_frame = self._gesture_service.get_processed_frame(processed_frame)
        
        return processed_frame, timestamp

    def inject_services(self, face_service=None, gesture_service=None):
        """
        Inject services for frame processing.
        This is called after all services are initialized.
        """
        if face_service:
            self._face_service = face_service
            
        if gesture_service:
            self._gesture_service = gesture_service

    def get_buffer_frames(self, count: int = 30) -> List[Tuple[np.ndarray, float]]:
        """
        Get a specified number of recent frames from the buffer.
        Returns a list of (frame, timestamp) tuples.
        """
        with self._buffer_lock:
            # Get at most 'count' frames from the buffer
            frames_count = min(count, len(self._frame_buffer))
            # Return copies to prevent external modification
            return [(frame.copy(), ts) for frame, ts in list(self._frame_buffer)[-frames_count:]]

    def get_jpeg_frame(self, quality: int = 90, processed: bool = False) -> Tuple[Optional[bytes], float]:
        """
        Get the latest frame encoded as JPEG.
        If processed=True, returns a processed frame with visualizations.
        Returns a tuple of (jpeg_bytes, timestamp).
        """
        if processed and hasattr(self, '_face_service'):
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