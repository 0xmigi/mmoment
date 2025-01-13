import logging
from .buffer_service import BufferService
import subprocess
import threading
from threading import Lock
import time
import os
from ..config.settings import Settings
import cv2

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
            logger.debug("LivepeerStreamService initialized")

    @property
    def is_streaming(self):
        return self._is_streaming

    @is_streaming.setter
    def is_streaming(self, value):
        self._is_streaming = value

    def start_streaming(self):
        """Start RTMP stream using buffer frames"""
        logger.debug("=== Stream Start Sequence ===")
        with self._lock:
            if self._is_streaming:
                logger.info("Stream already running")
                return True

            try:
                self._is_starting = True

                # Check buffer service is running
                if not self.buffer_service or not self.buffer_service.is_running:
                    logger.error("Buffer service not running")
                    return False

                if not self.stream_key or not self.ingest_url:
                    logger.error("Missing Livepeer credentials")
                    return False

                rtmp_url = f"{self.ingest_url}/{self.stream_key}"
                logger.info(f"Streaming to: {rtmp_url}")

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-s", "960x540",
                    "-r", "10",
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "superfast",
                    "-tune", "zerolatency",
                    "-profile:v", "baseline",
                    "-level", "3.0",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "500k",
                    "-maxrate", "600k",
                    "-bufsize", "1200k",
                    "-g", "20",
                    "-x264opts", "no-scenecut:nal-hrd=cbr",
                    "-f", "flv",
                    rtmp_url
                ]

                logger.debug(f"Starting ffmpeg with command: {' '.join(cmd)}")
                
                self.streaming_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=4096
                )

                def _stream_frames():
                    frames_sent = 0
                    frame_interval = 1.0 / 10  # 10fps
                    start_time = time.time()
                    next_frame_time = start_time
                    last_log_time = start_time
                    
                    try:
                        while self._is_streaming and self.streaming_process:
                            current_time = time.time()
                            
                            # Temperature check every 30 frames
                            if frames_sent % 30 == 0:
                                try:
                                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                                        temp = float(f.read()) / 1000.0
                                        if temp > 65:
                                            logger.error(f"High temperature: {temp}°C")
                                            self._is_streaming = False
                                            return
                                except Exception as e:
                                    logger.error(f"Failed to read temperature: {e}")

                            # Wait until next frame is due
                            sleep_time = next_frame_time - current_time
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                            # Get frame
                            frame = self.buffer_service.get_latest_frame()
                            if frame is not None:
                                try:
                                    self.streaming_process.stdin.write(frame.tobytes())
                                    frames_sent += 1
                                    
                                    # Log progress every 5 seconds
                                    if current_time - last_log_time >= 5:
                                        elapsed = current_time - start_time
                                        actual_fps = frames_sent / elapsed if elapsed > 0 else 0
                                        logger.debug(f"Streaming stats - Frames sent: {frames_sent}, FPS: {actual_fps:.1f}")
                                        last_log_time = current_time
                                        
                                except (IOError, BrokenPipeError) as e:
                                    logger.error(f"Stream write failed: {e}")
                                    self._is_streaming = False
                                    break

                            # Check FFmpeg process health
                            if self.streaming_process.poll() is not None:
                                stderr = self.streaming_process.stderr.read()
                                logger.error(f"FFmpeg process died: {stderr.decode()}")
                                self._is_streaming = False
                                break

                            # Update next frame time
                            next_frame_time += frame_interval
                            
                            # Reset timing if we're too far behind
                            if current_time - next_frame_time > frame_interval * 2:
                                logger.warning("Stream timing reset - too far behind")
                                next_frame_time = current_time + frame_interval
                            
                    except Exception as e:
                        logger.error(f"Error in stream_frames: {e}")
                        self._is_streaming = False
                    finally:
                        elapsed = time.time() - start_time
                        actual_fps = frames_sent / elapsed if elapsed > 0 else 0
                        logger.info(f"Stream frame loop ended. Frames sent: {frames_sent}, Average FPS: {actual_fps:.1f}")
                        if self.streaming_process and self.streaming_process.stdin:
                            try:
                                self.streaming_process.stdin.close()
                            except:
                                pass

                self._is_streaming = True
                self.stream_thread = threading.Thread(target=_stream_frames)
                self.stream_thread.daemon = True
                self.stream_thread.start()

                # Wait a moment to check if ffmpeg starts successfully
                time.sleep(1)
                if self.streaming_process.poll() is not None:
                    stderr = self.streaming_process.stderr.read()
                    logger.error(f"FFmpeg failed to start: {stderr.decode()}")
                    self._is_streaming = False
                    self.cleanup_stream()
                    return False

                logger.debug("Stream initialization completed successfully")
                return True

            except Exception as e:
                logger.error(f"Stream start failed: {str(e)}", exc_info=True)
                self._is_streaming = False
                self.cleanup_stream()
                return False
            finally:
                self._is_starting = False

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

    def stop_streaming(self):
        """Stop streaming safely"""
        logger.debug("Stopping stream")
        with self._lock:
            self._is_streaming = False
            self.cleanup_stream()
            logger.debug("Stream stopped")

    def get_stream_info(self):
        """Get current stream status and information"""
        logger.debug(f"Getting stream info. Is streaming: {self.is_streaming}")
        return {
            "isActive": self.is_streaming,
            "playbackId": "a1831mk3ncwwk8cu"  # Your actual playback ID
        }