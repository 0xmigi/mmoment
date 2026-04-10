import logging
from .buffer_service import BufferService
import subprocess
import threading
from threading import Lock
import time
import os
import io
from ..config.settings import Settings
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class LivepeerStreamService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LivepeerStreamService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized") or not self._initialized:
            self._initialized = True
            self._is_streaming = False
            self._is_starting = False
            self._lock = Lock()
            self.streaming_process = None
            self.stream_thread = None
            self.buffer_service = BufferService()
            self.api_key = os.getenv("LIVEPEER_API_KEY")
            self.stream_key = os.getenv("LIVEPEER_STREAM_KEY")
            self.ingest_url = os.getenv("LIVEPEER_INGEST_URL")
            # STRIPPED DOWN: Use minimal resolution and framerate
            self.width = 640  # Match buffer service
            self.height = 360  # Match buffer service
            self.frame_rate = 15  # Match buffer service frame rate
            self.stop_streaming_event = threading.Event()
            self.ffmpeg_process = None
            
            # Check if hardware acceleration is available
            self.hw_accel_available = self._check_hw_acceleration()
            logger.debug(f"LivepeerStreamService initialized for MJPEG buffer. Hardware acceleration available: {self.hw_accel_available}")

    def _check_hw_acceleration(self):
        """Check if h264_v4l2m2m hardware acceleration is available"""
        try:
            # Try to query the hardware encoder
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, 
                text=True, 
                check=False
            )
            return "h264_v4l2m2m" in result.stdout
        except Exception as e:
            logger.warning(f"Failed to check hardware acceleration: {e}")
            return False

    @property
    def is_streaming(self):
        return self._is_streaming

    @is_streaming.setter
    def is_streaming(self, value):
        self._is_streaming = value

    def start_stream(self) -> dict:
        """Start streaming to Livepeer"""
        with self._lock:
            if self.is_streaming:
                return {"status": "already_streaming"}

            try:
                logger.info("=== Starting Livepeer Stream ===")
                
                # Check if Livepeer is configured
                if not self.ingest_url or not self.stream_key:
                    logger.error("Livepeer not configured - missing environment variables")
                    return {"status": "error", "message": "Livepeer not configured"}
                    
                # Set full ingest URL with stream key
                ingest_url = f"{self.ingest_url}/{self.stream_key}"
                logger.info(f"Using Livepeer ingest URL: {ingest_url}")
                
                # Start with basic command
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-f", "mjpeg",  # Input format is MJPEG
                    "-i", "-",  # Read from stdin
                ]
                
                # Simplify encoding to troubleshoot connectivity issues
                # Use software encoding with minimal settings
                logger.info("Using simplified encoding settings for troubleshooting")
                ffmpeg_cmd.extend([
                    "-c:v", "libx264",
                    "-preset", "ultrafast",  # Use ultrafast preset to minimize CPU usage
                    "-tune", "zerolatency",  # Optimize for streaming
                    "-pix_fmt", "yuv420p",   # Required for compatibility
                    "-b:v", "500k",          # Lower bitrate
                    "-maxrate", "500k",      # Lower max bitrate 
                    "-bufsize", "1000k",     # Smaller buffer
                    "-g", str(self.frame_rate),  # Set keyframe interval to 1 second
                    "-r", str(self.frame_rate),  # Framerate
                    "-f", "flv",             # Output format
                    ingest_url
                ])

                logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
                
                # Check system resources before starting
                if not self._check_system_resources():
                    logger.error("Insufficient system resources to start streaming")
                    return {"status": "error", "message": "Insufficient system resources"}
                
                # Start FFmpeg process
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0  # No buffering for less memory usage
                )

                # Start the streaming thread
                self.is_streaming = True
                self.stop_streaming_event.clear()
                self._stream_thread = threading.Thread(
                    target=self._stream_frames,
                    daemon=True
                )
                self._stream_thread.start()

                logger.info("Stream started successfully")
                return {
                    "status": "streaming",
                    "stream_key": self.stream_key,
                    "ingest_url": self.ingest_url
                }

            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                self.stop_stream()  # Cleanup on failure
                raise RuntimeError(f"Failed to start stream: {e}")
    
    def _check_system_resources(self):
        """Check if system has enough resources to stream"""
        try:
            # Check CPU temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read()) / 1000.0
                    logger.info(f"Current CPU temperature: {temp}°C")
                    if temp > 75:
                        logger.warning(f"CPU temperature too high: {temp}°C")
                        return False
            except Exception as e:
                logger.warning(f"Failed to check CPU temperature: {e}")
            
            # Check available memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                logger.info(f"Available memory: {mem.available / 1024 / 1024:.1f} MB")
                if mem.available < 300 * 1024 * 1024:  # Less than 300MB available
                    logger.warning(f"Not enough memory available: {mem.available / 1024 / 1024:.1f} MB")
                    return False
            except Exception as e:
                logger.warning(f"Failed to check memory: {e}")
            
            # Check CPU load
            try:
                load = os.getloadavg()[0]
                logger.info(f"Current CPU load: {load}")
                if load > 3.5:  # High load for a 4-core Pi
                    logger.warning(f"CPU load too high: {load}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to check CPU load: {e}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return True  # Default to allow if check fails

    def _stream_frames(self):
        """Stream MJPEG frames to FFmpeg process"""
        frames_sent = 0
        start_time = time.time()
        last_log = start_time
        last_resource_check = start_time
        error_count = 0
        max_errors = 5  # Maximum number of errors before giving up
        
        try:
            while not self.stop_streaming_event.is_set():
                if not self.buffer_service.is_running:
                    logger.error("Buffer service not running")
                    break

                current_time = time.time()
                
                # Check system resources every 10 seconds
                if current_time - last_resource_check >= 10:
                    if not self._check_system_resources():
                        logger.warning("Stopping stream due to insufficient system resources")
                        break
                    last_resource_check = current_time

                # Get JPEG frame directly from buffer
                jpeg_data = self.buffer_service.get_jpeg_frame()
                if jpeg_data is None:
                    time.sleep(0.01)  # Sleep if no frame
                    continue

                try:
                    # Write JPEG frame to FFmpeg's stdin
                    self.ffmpeg_process.stdin.write(jpeg_data)
                    self.ffmpeg_process.stdin.flush()
                    frames_sent += 1
                    error_count = 0  # Reset error count on successful frame

                    # Log streaming stats every 10 seconds (less frequently)
                    if current_time - last_log >= 10:
                        elapsed = current_time - start_time
                        fps = frames_sent / elapsed
                        logger.debug(f"Streaming stats - Frames sent: {frames_sent}, FPS: {fps:.1f}")
                        last_log = current_time
                        
                        # Check FFmpeg stderr for errors (non-blocking)
                        if hasattr(self.ffmpeg_process, 'stderr') and self.ffmpeg_process.stderr:
                            stderr_data = None
                            try:
                                # Try to read stderr without blocking
                                import select
                                if select.select([self.ffmpeg_process.stderr], [], [], 0)[0]:
                                    stderr_data = self.ffmpeg_process.stderr.read(1024).decode('utf-8', errors='ignore')
                                    if stderr_data and 'Error' in stderr_data:
                                        logger.warning(f"FFmpeg error: {stderr_data}")
                            except Exception as e:
                                logger.error(f"Error checking FFmpeg stderr: {e}")

                except BrokenPipeError:
                    error_count += 1
                    logger.error(f"FFmpeg process pipe broken (error {error_count}/{max_errors})")
                    if error_count >= max_errors:
                        logger.error("Too many pipe errors, stopping stream")
                        break
                    # Brief pause to avoid spinning on errors
                    time.sleep(0.5)
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error writing frame to FFmpeg: {e} (error {error_count}/{max_errors})")
                    if error_count >= max_errors:
                        logger.error("Too many errors, stopping stream")
                        break
                    time.sleep(0.5)

                # More adaptive rate control
                target_interval = 1.0 / self.frame_rate
                elapsed = time.time() - current_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        except Exception as e:
            logger.error(f"Streaming thread error: {e}")
        finally:
            self.stop_stream()

    def stop_stream(self) -> dict:
        """Stop the current stream"""
        with self._lock:
            if not self.is_streaming:
                return {"status": "not_streaming"}

            try:
                logger.info("Stopping stream...")
                self.stop_streaming_event.set()

                if self.ffmpeg_process:
                    # Gracefully stop FFmpeg
                    try:
                        self.ffmpeg_process.stdin.close()
                        self.ffmpeg_process.wait(timeout=5)
                    except:
                        self.ffmpeg_process.kill()
                    finally:
                        self.ffmpeg_process = None

                self.is_streaming = False
                logger.info("Stream stopped successfully")
                return {"status": "stopped"}

            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
                return {"status": "error", "message": str(e)}

    def cleanup_stream(self):
        """Clean up stream resources"""
        logger.debug("Cleaning up stream resources")
        self._is_streaming = False

        if self.streaming_process:
            try:
                if self.streaming_process.stdin:
                    self.streaming_process.stdin.close()
                self.streaming_process.terminate()
                try:
                    self.streaming_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.streaming_process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up stream process: {e}")
            self.streaming_process = None

        # Don't try to join the thread from within itself
        if self.stream_thread and self.stream_thread != threading.current_thread():
            try:
                self.stream_thread.join(timeout=1)
            except Exception as e:
                logger.error(f"Error joining stream thread: {e}")
            self.stream_thread = None

    def get_stream_info(self):
        """Get current stream status and information"""
        logger.debug(f"Getting stream info. Is streaming: {self.is_streaming}")
        return {
            "isActive": self.is_streaming,
            "playbackId": "a1831mk3ncwwk8cu"  # Your actual playback ID
        }