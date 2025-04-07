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
            logger.debug("LivepeerStreamService initialized for MJPEG buffer")

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
                
                # Modify FFmpeg command to accept MJPEG input
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-f", "mjpeg",  # Input format is MJPEG
                    "-i", "-",  # Read from stdin
                    "-c:v", "libx264",
                    "-preset", "veryfast",  # Faster encoding, less CPU
                    "-tune", "zerolatency",  # Optimize for streaming
                    "-pix_fmt", "yuv420p",  # Required for compatibility
                    "-b:v", "1M",  # Lower bitrate
                    "-maxrate", "1M",  # Lower max bitrate
                    "-bufsize", "2M",  # Smaller buffer
                    "-g", str(self.frame_rate * 2),  # Set keyframe interval to 2 seconds
                    "-force_key_frames", f"expr:gte(t,n_forced*{2})",  # Force keyframe every 2 seconds
                    "-x264opts", "no-scenecut",  # Disable scene detection to maintain keyframe interval
                    "-f", "flv",  # Output format
                    ingest_url
                ]

                logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
                
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

    def _stream_frames(self):
        """Stream MJPEG frames to FFmpeg process"""
        frames_sent = 0
        start_time = time.time()
        last_log = start_time
        
        try:
            while not self.stop_streaming_event.is_set():
                if not self.buffer_service.is_running:
                    logger.error("Buffer service not running")
                    break

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

                    # Log streaming stats every 10 seconds (less frequently)
                    current_time = time.time()
                    if current_time - last_log >= 10:
                        elapsed = current_time - start_time
                        fps = frames_sent / elapsed
                        logger.debug(f"Streaming stats - Frames sent: {frames_sent}, FPS: {fps:.1f}")
                        last_log = current_time

                except BrokenPipeError:
                    logger.error("FFmpeg process pipe broken")
                    break
                except Exception as e:
                    logger.error(f"Error writing frame to FFmpeg: {e}")
                    break

                # Simple rate control
                time.sleep(1.0 / self.frame_rate)

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