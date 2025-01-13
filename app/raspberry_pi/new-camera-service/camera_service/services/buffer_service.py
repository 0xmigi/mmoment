# buffer_service.py
import subprocess
import threading
import time
import cv2
import numpy as np
import logging
import os
from picamera2 import Picamera2
from threading import Lock
from ..config.settings import Settings
from collections import deque
from queue import Queue
from typing import Optional, Tuple, List

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
        self.frame_rate = 10  # Reduced from 20 to 10 fps
        self.buffer_size = 100  # Reduced from 200 to 100 frames (10 seconds at 10fps)
        self.width = 960  # Reduced from 1280 to 960
        self.height = 540  # Reduced from 720 to 540
        
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
        self.temp_check_interval = 1  # Check temp every second
        
        logger.info("Buffer Service initialized with lower resolution and frame rate")

    def _check_temperature(self) -> float:
        """Check CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0

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

                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={
                        "FrameRate": self.frame_rate,
                        "ExposureTime": 25000,  # Further reduced from 50000
                        "AnalogueGain": 1.5,    # Further reduced from 2.0
                        "NoiseReductionMode": 2, # Increased noise reduction
                        "Brightness": 0.1,       # Reduced from 0.2
                        "Contrast": 1.1,         # Reduced from 1.2
                        "Saturation": 1.0,       # Reduced from 1.1
                        "Sharpness": 0.8,        # Reduced from 1.0
                        "AeEnable": True,
                        "AwbEnable": True,
                    },
                    buffer_count=2  # Reduced from 3
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(0.5)

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

    def get_video_segment(self, duration_seconds: int = 5) -> Optional[dict]:
        """Extract video segment directly to MOV format"""
        logger.info("Starting video extraction")
        
        try:
            num_frames = min(duration_seconds * self.frame_rate, len(self.frame_buffer))
            if num_frames < 15:  # Minimum frames needed
                raise RuntimeError("Insufficient frames in buffer")

            with self.buffer_lock:
                frames = list(self.frame_buffer)[-num_frames:]
            
            timestamp = int(time.time())
            mov_path = os.path.join(Settings.VIDEOS_DIR, f"video_{timestamp}.mov")

            # Create a temporary file for frames
            with open("/tmp/frames.raw", "wb") as temp_file:
                for frame, _ in frames:
                    temp_file.write(frame.tobytes())

            # Use the temporary file as input with optimized libx264 settings
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "rgb24",
                "-r", str(self.frame_rate),
                "-i", "/tmp/frames.raw",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",
                "-b:v", "1.5M",
                "-bufsize", "2M",
                "-maxrate", "2M",
                "-movflags", "+faststart",
                mov_path
            ]

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            _, stderr = process.communicate(timeout=10)

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

            # Clean up temp file
            try:
                os.remove("/tmp/frames.raw")
            except:
                pass

            return {
                "mov_path": mov_path,
                "mov_filename": os.path.basename(mov_path),
                "frame_count": len(frames)
            }

        except Exception as e:
            logger.error(f"Failed to create video: {e}", exc_info=True)
            if 'mov_path' in locals() and os.path.exists(mov_path):
                os.remove(mov_path)
            try:
                os.remove("/tmp/frames.raw")
            except:
                pass
            return None

    def _buffer_frames(self):
        """Continuously buffer frames from the camera"""
        frame_interval = 1.0 / self.frame_rate
        last_temp_check = time.time()
        frames_processed = 0
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Temperature check every second
                if current_time - last_temp_check >= self.temp_check_interval:
                    temp = self._check_temperature()
                    if temp > 70:  # Reduced from 75
                        logger.error(f"Critical temperature: {temp}°C")
                        self.cleanup()
                        break
                    last_temp_check = current_time

                frame = self.picam2.capture_array()
                frames_processed += 1
                
                # Log performance metrics every 100 frames
                if frames_processed % 100 == 0:
                    logger.debug(f"Processed {frames_processed} frames, Buffer size: {len(self.frame_buffer)}")
                
                with self.buffer_lock:
                    self.frame_buffer.append((frame, current_time))
                    self.latest_frame = frame
                    self.last_frame_time = current_time

                # Maintain frame rate with better timing
                elapsed = time.time() - current_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > frame_interval * 1.5:
                    logger.warning(f"Frame processing taking too long: {elapsed:.3f}s")

            except Exception as e:
                logger.error(f"Error buffering frame: {e}")
                time.sleep(0.1)

    def get_latest_frame(self):
        """Thread-safe access to the latest frame"""
        with self.buffer_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_jpeg_frame(self):
        """Get latest frame as JPEG with optimized settings"""
        frame = self.get_latest_frame()
        if frame is not None:
            try:
                _, jpeg_data = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 85,  # Reduced quality
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

        logger.info("Buffer service cleanup complete")