import logging
import subprocess
import cv2
import os
import threading
from threading import Lock
import time
from .buffer_service import BufferService
from .stream_service import LivepeerStreamService
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class CameraService:
    _instance = None
    _lock = Lock()
    MAX_RECORDING_DURATION = 60  # Reduced from 300 to 60 seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.buffer_service = BufferService()
        self.stream_service = LivepeerStreamService()
        
        # Start buffer immediately - it's our highest priority
        self.buffer_service.start_buffering()
        
        # Resource monitoring
        self._resource_check_interval = 2  # Reduced from 5 to 2 seconds
        self._last_resource_check = 0
        self._system_healthy = True
        self._recording_active = False
        self._recording_start_time = None

    def _check_system_resources(self, streaming_active=False) -> bool:
        """Check if system has enough resources to perform operations"""
        current_time = time.time()
        if current_time - self._last_resource_check < self._resource_check_interval:
            return self._system_healthy

        try:
            import psutil
            
            # Check memory - more strict limits
            memory = psutil.virtual_memory()
            if memory.available < 750 * 1024 * 1024:  # Increased from 500MB to 750MB
                logger.warning("Low memory available")
                self._system_healthy = False
                return False
                
            # Check CPU temperature
            temp = self.buffer_service._check_temperature()
            if temp > 70:  # Reduced from 75°C to 70°C
                logger.warning(f"High CPU temperature: {temp}°C")
                self._system_healthy = False
                return False
                
            # Check CPU usage - more strict limit
            if psutil.cpu_percent(interval=0.1) > 80:  # Reduced from 90 to 80
                logger.warning("High CPU usage")
                self._system_healthy = False
                return False
                
            self._system_healthy = True
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False
        finally:
            self._last_resource_check = current_time

    def _get_safe_recording_duration(self) -> int:
        """Determine safe recording duration based on system conditions"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            temp = self.buffer_service._check_temperature()
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Base duration when streaming is active
            if self.stream_service.is_streaming:
                base_duration = 5  # Start with proven stable duration
                
                # Only increase if conditions are very good
                if (memory.available > 2500 * 1024 * 1024 and  # More than 2.5GB free
                    temp < 55 and                              # Very cool
                    cpu < 60):                                # Low CPU usage
                    base_duration = 10
                
                return base_duration
            
            # When not streaming, we can be more generous
            if memory.available < 1500 * 1024 * 1024:  # Less than 1.5GB
                return 10
            elif temp > 65:  # High temperature
                return 10
            elif cpu > 75:  # High CPU
                return 10
            elif memory.available > 3000 * 1024 * 1024:  # More than 3GB free
                return 15
            else:
                return 12
                
        except Exception as e:
            logger.error(f"Error determining recording duration: {e}")
            return 5  # Fall back to safe duration

    def start_recording(self, duration_seconds=5) -> dict:
        """
        Start video recording. 
        If recording is already active, returns its info.
        If resources are insufficient, returns error info.
        """
        logger.info(f"Starting recording for {duration_seconds} seconds")
        
        with self._lock:
            if self._recording_active:
                logger.info("Recording already in progress")
                return {"status": "recorded", "filename": self._recording_start_time, "duration": duration_seconds}
            
            # Check if we have enough system resources to record
            if not self._check_system_resources():
                logger.error("Insufficient system resources for recording")
                return {"status": "error", "message": "Insufficient resources for recording"}
            
            # Check if we're streaming, and if so, can we record simultaneously
            if self.stream_service.is_streaming:
                # More stringent resource check when streaming is active
                if not self._check_system_resources(streaming_active=True):
                    logger.error("Insufficient resources for recording while streaming")
                    return {"status": "error", "message": "Insufficient resources for recording while streaming"}
            
            # Get safe recording duration based on system resources
            safe_duration = self._get_safe_recording_duration()
            if duration_seconds > safe_duration:
                logger.warning(f"Requested duration {duration_seconds}s exceeds safe duration {safe_duration}s, limiting recording")
                duration_seconds = safe_duration
            
            try:
                # Set recording flag first to prevent race conditions
                self._recording_active = True
                self._recording_start_time = time.time()
                
                # Get video segment directly
                video_info = self.buffer_service.get_video_segment(duration_seconds)
                
                # Set recording status to completed and return info
                self._recording_active = False
                
                if not video_info:
                    logger.error("Failed to create video segment")
                    return {"status": "error", "message": "Failed to create video segment"}
                
                # Create a dict with filename and status (for API response)
                result = {
                    "filename": video_info["mov_filename"],
                    "status": "recorded",
                    "duration": video_info.get("duration", duration_seconds),
                    "frame_count": video_info.get("frame_count", 0),
                    "path": video_info.get("path", "")
                }
                
                logger.info(f"Recording completed: {result}")
                return result
                
            except Exception as e:
                logger.error(f"Recording error: {e}")
                self._recording_active = False
                return {"status": "error", "message": str(e)}

    def stop_recording(self):
        """Stop current recording"""
        with self._lock:
            if not self._recording_active:
                return {"status": "not_recording", "message": "No active recording"}
            
            self._recording_active = False
            logger.info("Recording stopped")
            return {"status": "stopped", "message": "Recording completed"}

    def _can_handle_simultaneous_operations(self) -> bool:
        """Check if system can handle combined operations"""
        if not self._check_system_resources():
            return False
            
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            # Even stricter limits for simultaneous operations
            if memory.available < 2000 * 1024 * 1024:  # Need 2GB+ free
                logger.warning("Insufficient memory for simultaneous operations")
                return False
                
            # Check CPU temperature
            temp = self.buffer_service._check_temperature()
            if temp > 60:  # Even stricter temperature limit
                logger.warning(f"Temperature too high for simultaneous operations: {temp}°C")
                return False
                
            # Check CPU usage
            if psutil.cpu_percent(interval=0.1) > 70:  # Stricter CPU limit
                logger.warning("CPU usage too high for simultaneous operations")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking simultaneous operations: {e}")
            return False

    def take_picture(self):
        """Take picture with resource check"""
        if not self._check_system_resources():
            raise RuntimeError("Insufficient system resources for picture capture")

        try:
            # Just get a frame from the buffer without resetting anything
            jpeg_data = self.buffer_service.get_jpeg_frame()
            if jpeg_data:
                return jpeg_data
            else:
                raise RuntimeError("Failed to capture photo - no frame available in buffer")
        except Exception as e:
            logger.error(f"Photo capture error: {e}")
            raise