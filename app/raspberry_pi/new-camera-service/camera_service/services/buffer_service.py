# buffer_service.py
import subprocess
import threading
import time
import cv2
import numpy as np
import logging
import os
from picamera2 import Picamera2
import libcamera
from threading import Lock
from ..config.settings import Settings
from collections import deque
from queue import Queue
from typing import Optional, Tuple, List
import glob
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BufferService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BufferService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        # STRIPPED DOWN: Minimal framerate and resolution
        self.frame_rate = 15  # Increased framerate for better streaming quality
        self.buffer_size = 100  # Small buffer size
        self.width = 640  # Lower resolution
        self.height = 360  # Lower resolution
        
        # Core buffer components
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Camera components
        self.picam2 = None
        self.latest_frame = None
        
        # Health monitoring
        self.last_frame_time = 0
        self.health_lock = Lock()
        self.last_temp_check = 0
        self.temp_check_interval = 2  # Check less frequently
        
        # Cleanup settings
        self.max_video_age_days = 1
        self.max_videos_to_keep = 10  # Keep fewer videos
        self.cleanup_interval = 3600
        self.last_cleanup_time = 0
        
        logger.info("Buffer Service initialized with minimal settings")
        
        # Start buffering automatically
        self.start_buffering()

    def _check_temperature(self) -> float:
        """Check CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0

    def get_frame_count(self) -> int:
        """Get the current number of frames in the buffer"""
        with self.buffer_lock:
            return len(self.frame_buffer)

    def start_buffering(self):
        """Start the camera capture and buffering"""
        with self._lock:
            if self.is_running:
                return False

            try:
                if self.picam2:
                    self.picam2.stop()
                    self.picam2.close()
                    self.picam2 = None
                    time.sleep(0.5)
                    
                # Additional delay to ensure camera is released
                time.sleep(1.0)

                self.picam2 = Picamera2()
                # Add back autofocus settings but remove the vflip/hflip transform
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={
                        "FrameRate": self.frame_rate,
                        # Add back auto-focus controls
                        "AfMode": 2,                  # Continuous auto focus (2 is better than 1)
                        "AfMetering": 0,              # Auto focus metering mode
                        "AfRange": 0,                 # Normal range
                        "AfSpeed": 1,                 # Normal AF speed
                        "AfTrigger": 0                # Trigger autofocus now
                    }
                    # Remove transform to avoid double-flip with get_latest_frame
                )
                
                self.picam2.configure(config)
                
                # Try to apply controls after configuration as well to ensure focus works
                try:
                    logger.info("Setting autofocus controls after configuration")
                    self.picam2.set_controls({
                        "AfMode": 2,                # Continuous autofocus mode
                        "AfTrigger": 0              # Trigger autofocus now
                    })
                    logger.info("Successfully set autofocus controls")
                except Exception as e:
                    logger.warning(f"Could not set autofocus controls: {e}")
                
                # Start the camera and wait for it to initialize
                self.picam2.start()
                time.sleep(2.0)  # Give it more time to initialize
                
                logger.info(f"Camera started with autofocus settings and no transform")

                self.is_running = True
                self.shutdown_event.clear()
                
                self._buffer_thread = threading.Thread(
                    target=self._buffer_frames,
                    daemon=True
                )
                self._buffer_thread.start()

                return True

            except Exception as e:
                logger.error(f"Failed to start buffer: {e}")
                self.cleanup()
                return False

    def cleanup_old_videos(self, force=False):
        """Clean up old videos to save disk space"""
        current_time = time.time()
        
        # Only run cleanup at intervals unless forced
        if not force and (current_time - self.last_cleanup_time) < self.cleanup_interval:
            return
            
        self.last_cleanup_time = current_time
        
        try:
            logger.info("Running video cleanup...")
            
            # List all video files in the videos directory
            video_files = glob.glob(os.path.join(Settings.VIDEOS_DIR, "video_*.mov"))
            
            if not video_files:
                logger.info("No video files to clean up")
                return
                
            # Sort by modification time (oldest first)
            video_files.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove files older than max_video_age_days
            cutoff_time = time.time() - (self.max_video_age_days * 86400)
            deleted_count = 0
            
            # First pass: delete by age
            for video_file in video_files[:]:
                if os.path.getmtime(video_file) < cutoff_time:
                    try:
                        os.remove(video_file)
                        deleted_count += 1
                        video_files.remove(video_file)
                        logger.debug(f"Deleted old video: {video_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete {video_file}: {e}")
            
            # Second pass: delete excess files if there are still too many
            if len(video_files) > self.max_videos_to_keep:
                # Calculate how many to delete
                to_delete = len(video_files) - self.max_videos_to_keep
                
                for video_file in video_files[:to_delete]:
                    try:
                        os.remove(video_file)
                        deleted_count += 1
                        logger.debug(f"Deleted excess video: {video_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete {video_file}: {e}")
            
            logger.info(f"Video cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"Error during video cleanup: {e}")

    def get_video_segment(self, duration_seconds: int = 5) -> Optional[dict]:
        """Extract video segment directly to MOV format"""
        logger.info("Starting video extraction")
        
        try:
            # Run cleanup to ensure we have disk space
            self.cleanup_old_videos(force=True)
            
            num_frames = min(duration_seconds * self.frame_rate, len(self.frame_buffer))
            if num_frames < 15:  # Minimum frames needed
                raise RuntimeError("Insufficient frames in buffer")

            with self.buffer_lock:
                frames = list(self.frame_buffer)[-num_frames:]
            
            timestamp = int(time.time())
            mov_filename = f"video_{timestamp}.mov"
            
            # Ensure videos directory exists
            os.makedirs(Settings.VIDEOS_DIR, exist_ok=True)
            mov_path = os.path.join(Settings.VIDEOS_DIR, mov_filename)
            logger.info(f"Will save video to: {mov_path}")

            # Create a temporary file for frames - manually rotate each frame before writing
            temp_path = f"/tmp/frames_{timestamp}.raw"
            logger.info(f"Writing frames to temporary file: {temp_path}")
            with open(temp_path, "wb") as temp_file:
                for frame, _ in frames:
                    # Try a different method to correct orientation
                    # Rotate each frame 180 degrees (flip horizontally and vertically)
                    frame = cv2.flip(frame, -1)  # -1 means flip both horizontally and vertically
                    temp_file.write(frame.tobytes())

            # Use a direct approach for ffmpeg without additional transformations
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "rgb24",
                "-r", str(self.frame_rate),
                "-i", temp_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-b:v", "1M",  # Low bitrate
                # Don't use any additional transformations since we've already rotated the frames
                "-movflags", "+faststart",
                mov_path
            ]

            logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout, stderr = process.communicate(timeout=10)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

            # Clean up temp file
            try:
                os.remove(temp_path)
                logger.info("Cleaned up temporary file")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

            # Verify the file exists and has content
            if not os.path.exists(mov_path):
                raise RuntimeError("Video file was not created")
            
            file_size = os.path.getsize(mov_path)
            if file_size == 0:
                raise RuntimeError("Video file is empty")
                
            logger.info(f"Successfully created video with frame-by-frame rotation: {mov_filename} (size: {file_size} bytes)")
            return {
                "mov_filename": mov_filename,
                "frame_count": num_frames,
                "duration": duration_seconds,
                "path": mov_path
            }

        except Exception as e:
            logger.error(f"Failed to create video segment: {e}")
            raise

    def _buffer_frames(self):
        """Continuously buffer frames from the camera"""
        frame_interval = 1.0 / self.frame_rate
        last_temp_check = time.time()
        frames_processed = 0
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Temperature check less frequently
                if current_time - last_temp_check >= self.temp_check_interval:
                    temp = self._check_temperature()
                    if temp > 80:
                        logger.error(f"Critical temperature: {temp}°C")
                        self.cleanup()
                        break
                    last_temp_check = current_time

                # Basic frame capture
                frame = self.picam2.capture_array()
                frames_processed += 1
                
                # Log less frequently
                if frames_processed % 200 == 0:
                    logger.debug(f"Processed {frames_processed} frames, Buffer size: {len(self.frame_buffer)}")
                
                with self.buffer_lock:
                    self.frame_buffer.append((frame, current_time))
                    self.latest_frame = frame
                    self.last_frame_time = current_time

                # Basic timing
                sleep_time = frame_interval - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error buffering frame: {e}")
                time.sleep(0.1)

    def get_latest_frame(self):
        """Thread-safe access to the latest frame with orientation correction"""
        with self.buffer_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                # Apply orientation correction
                frame = cv2.flip(frame, -1)  # -1 means flip both horizontally and vertically
                return frame
            return None

    def get_jpeg_frame(self):
        """Get latest frame as JPEG with manual rotation for consistent orientation"""
        frame = self.get_latest_frame()  # Already includes orientation correction
        if frame is not None:
            try:
                # Basic JPEG encoding (frame is already rotated in get_latest_frame)
                _, jpeg_data = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 80,  # Lower quality
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                logger.debug("Created JPEG with orientation correction")
                return jpeg_data.tobytes()
            except Exception as e:
                logger.error(f"Error encoding JPEG: {e}")
        return None

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up buffer service")
        self.shutdown_event.set()
        self.is_running = False
        
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None

        with self.buffer_lock:
            self.frame_buffer.clear()
            self.latest_frame = None

        logger.info("Buffer service cleanup complete")