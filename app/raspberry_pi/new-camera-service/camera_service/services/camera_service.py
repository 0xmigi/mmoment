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
    MAX_RECORDING_DURATION = 30  # Reduced maximum recording duration

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
        
        # Resource monitoring (minimal)
        self._resource_check_interval = 10  # Check less frequently
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
            
            # Only check for extremely low memory
            memory = psutil.virtual_memory()
            if memory.available < 150 * 1024 * 1024:  # Very minimal requirement (150MB)
                logger.warning("Critically low memory available")
                self._system_healthy = False
                return False
                
            # Only check for extreme temperature
            temp = self.buffer_service._check_temperature()
            if temp > 85:  # Only block at very high temperatures
                logger.warning(f"Very high CPU temperature: {temp}°C")
                self._system_healthy = False
                return False
                
            self._system_healthy = True
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Default to allowing operation if check fails
        finally:
            self._last_resource_check = current_time

    def _get_safe_recording_duration(self) -> int:
        """Determine safe recording duration based on system conditions"""
        # Simple fixed durations
        if self.stream_service.is_streaming:
            return 5  # Short recording while streaming
        else:
            return 10  # Longer when not streaming
                
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
            
            # Simple resource check
            if not self._check_system_resources():
                logger.error("Insufficient system resources for recording")
                return {"status": "error", "message": "Insufficient resources for recording"}
            
            # Simple duration limitation
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

    def take_picture(self):
        """Take picture with minimal resource check"""
        # Simplified check
        if not self._system_healthy:
            raise RuntimeError("System health check failed")

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