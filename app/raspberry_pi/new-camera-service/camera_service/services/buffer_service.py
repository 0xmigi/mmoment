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
        self.buffer_size = 450  # Increased buffer size for 30 second recordings
        self.width = 640  # Lower resolution
        self.height = 360  # Lower resolution
        self.jpeg_quality = 90  # Quality setting for JPEG compression
        
        # Core buffer components
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Camera components
        self.picam2 = None
        self.latest_frame = None
        self.latest_jpeg = None  # Store the latest frame as JPEG
        
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
        
        logger.info("Buffer Service initialized with MJPEG settings")
        
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
                # Configure for JPEG encoding directly
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={
                        "FrameRate": self.frame_rate,
                        # Auto-focus controls
                        "AfMode": 2,                  # Continuous auto focus
                        "AfMetering": 0,              # Auto focus metering mode
                        "AfRange": 0,                 # Normal range
                        "AfSpeed": 1,                 # Normal AF speed
                        "AfTrigger": 0                # Trigger autofocus now
                    },
                    transform=libcamera.Transform(hflip=1, vflip=1)  # Hardware flip for correct orientation
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
                
                logger.info(f"Camera started with MJPEG buffer and hardware orientation correction")

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
        """Extract video segment directly from MJPEG frames"""
        logger.info("Starting video extraction from MJPEG buffer")
        
        try:
            # Run cleanup to ensure we have disk space
            self.cleanup_old_videos(force=True)
            
            num_frames = min(duration_seconds * self.frame_rate, len(self.frame_buffer))
            if num_frames < 15:  # Minimum frames needed
                raise RuntimeError("Insufficient frames in buffer")

            with self.buffer_lock:
                # Get the jpeg frames with timestamps
                frames = list(self.frame_buffer)[-num_frames:]
            
            timestamp = int(time.time())
            mov_filename = f"video_{timestamp}.mov"
            
            # Get start and end timestamps for audio sync
            start_time = frames[0][1]  # First frame timestamp
            end_time = frames[-1][1]   # Last frame timestamp
            
            # Ensure videos directory exists
            os.makedirs(Settings.VIDEOS_DIR, exist_ok=True)
            mov_path = os.path.join(Settings.VIDEOS_DIR, mov_filename)
            logger.info(f"Will save video to: {mov_path}")

            # Create a temporary file for MJPEG frames
            temp_path = f"/tmp/frames_{timestamp}.mjpeg"
            logger.info(f"Writing MJPEG frames to: {temp_path}")
            
            with open(temp_path, "wb") as temp_file:
                for jpeg_data, _ in frames:
                    temp_file.write(jpeg_data)
            
            # Try to get audio from AudioBufferService if available
            has_audio = False
            audio_temp = f"/tmp/audio_{timestamp}.wav"
            
            try:
                from .audio_buffer_service import AudioBufferService
                audio_buffer = AudioBufferService()
                
                # Only try to use audio if the service is running
                if audio_buffer.is_running:
                    # Save audio segment to temp file
                    if audio_buffer.save_audio_segment(start_time, end_time, audio_temp):
                        has_audio = True
                        logger.info(f"Audio segment saved to {audio_temp}")
                    else:
                        logger.warning("No audio data available for the time range")
                else:
                    logger.info("Audio buffer service not running, creating video without audio")
            except Exception as e:
                logger.warning(f"Could not get audio segment: {e}")
            
            # Prepare FFmpeg command based on whether we have audio
            if has_audio:
                # Use ffmpeg to convert MJPEG and WAV to MP4/MOV
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "mjpeg",
                    "-i", temp_path,
                    "-i", audio_temp,  # Add audio input
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "1M",
                    "-c:a", "aac",     # Audio codec
                    "-b:a", "128k",    # Audio bitrate
                    "-movflags", "+faststart",
                    mov_path
                ]
            else:
                # Video only
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "mjpeg",
                    "-i", temp_path,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "1M",
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

            # Clean up temp files
            try:
                os.remove(temp_path)
                if has_audio and os.path.exists(audio_temp):
                    os.remove(audio_temp)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")

            # Verify the file exists and has content
            if not os.path.exists(mov_path):
                raise RuntimeError("Video file was not created")
            
            file_size = os.path.getsize(mov_path)
            if file_size == 0:
                raise RuntimeError("Video file is empty")
                
            logger.info(f"Successfully created {'audio+video' if has_audio else 'video-only'} from MJPEG frames: {mov_filename} (size: {file_size} bytes)")
            return {
                "mov_filename": mov_filename,
                "frame_count": num_frames,
                "duration": duration_seconds,
                "path": mov_path,
                "has_audio": has_audio
            }

        except Exception as e:
            logger.error(f"Failed to create video segment: {e}")
            raise

    def _buffer_frames(self):
        """Continuously buffer frames from the camera as JPEG"""
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

                # Capture frame
                frame = self.picam2.capture_array()
                frames_processed += 1
                
                # Convert to JPEG immediately to preserve color accuracy
                _, jpeg_data = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                
                # Store JPEG data in buffer
                with self.buffer_lock:
                    self.frame_buffer.append((jpeg_data.tobytes(), current_time))
                    self.latest_frame = frame
                    self.latest_jpeg = jpeg_data.tobytes()
                    self.last_frame_time = current_time
                
                # Log less frequently
                if frames_processed % 200 == 0:
                    logger.debug(f"Processed {frames_processed} frames, Buffer size: {len(self.frame_buffer)}")
                
                # Basic timing
                sleep_time = frame_interval - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error buffering frame: {e}")
                time.sleep(0.1)

    def get_latest_frame(self):
        """Thread-safe access to the latest frame"""
        with self.buffer_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_jpeg_frame(self):
        """Get latest frame as JPEG directly from buffer"""
        with self.buffer_lock:
            if self.latest_jpeg is not None:
                return self.latest_jpeg
            
            # Fallback if latest_jpeg not available
            if self.latest_frame is not None:
                try:
                    _, jpeg_data = cv2.imencode('.jpg', self.latest_frame, [
                        cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
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
            self.latest_jpeg = None

        logger.info("Buffer service cleanup complete")