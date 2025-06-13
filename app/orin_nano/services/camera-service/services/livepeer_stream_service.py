"""
Livepeer Stream Service for Jetson Orin Nano

Provides Livepeer streaming functionality that takes JPEG frames from the buffer service
and streams them to Livepeer for low-latency distribution. This maintains all existing
camera functionality while adding high-quality streaming capabilities.

Enhanced with better error handling and automatic restart capabilities.
"""

import logging
import subprocess
import threading
from threading import Lock
import time
import os
import select
import signal
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
            self.buffer_service = None  # Will be injected
            self._load_config()
            
            # Stream settings optimized for Jetson
            self.width = 1280
            self.height = 720
            self.frame_rate = 15
            self.stop_streaming_event = threading.Event()
            self.ffmpeg_process = None
            self._restart_count = 0
            self._max_restarts = 3
            self._last_restart_time = 0
            
            # Monitoring and health check
            self._last_frame_time = 0
            self._frames_sent = 0
            self._monitoring_thread = None
            self._health_check_interval = 10  # seconds
            
            # Check if hardware acceleration is available on Jetson
            self.hw_accel_available = self._check_hw_acceleration()
            logger.debug(f"LivepeerStreamService initialized. Hardware acceleration available: {self.hw_accel_available}")

    def _load_config(self):
        """Load configuration from environment variables"""
        # Get from environment variables or use defaults
        self.api_key = os.environ.get('LIVEPEER_API_KEY', "522d8091-867f-42b3-8f62-5eeeab60f000")
        self.stream_key = os.environ.get('LIVEPEER_STREAM_KEY', "2458-aycn-mgfp-2dze")
        self.ingest_url = os.environ.get('LIVEPEER_INGEST_URL', "rtmp://rtmp.livepeer.com/live")
        self.playback_id = os.environ.get('LIVEPEER_PLAYBACK_ID', "24583deg6syfcql")
        
        logger.info(f"[LIVEPEER-CONFIG] Configuration loaded successfully!")
        logger.info(f"[LIVEPEER-CONFIG] Stream Key: {self.stream_key}")
        logger.info(f"[LIVEPEER-CONFIG] Playback ID: {self.playback_id}")
        logger.info(f"[LIVEPEER-CONFIG] Using environment variables: API_KEY={bool(os.environ.get('LIVEPEER_API_KEY'))}, STREAM_KEY={bool(os.environ.get('LIVEPEER_STREAM_KEY'))}")

    def reload_config(self):
        """Reload configuration from environment variables"""
        self._load_config()

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)"""
        with cls._lock:
            cls._instance = None

    def set_buffer_service(self, buffer_service):
        """Inject the buffer service dependency"""
        self.buffer_service = buffer_service

    def _check_hw_acceleration(self):
        """Check if hardware acceleration is available on Jetson"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, 
                text=True, 
                check=False
            )
            return "h264_nvenc" in result.stdout or "h264_v4l2m2m" in result.stdout
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
        """Start streaming to Livepeer with improved error handling"""
        with self._lock:
            if self.is_streaming:
                return {"status": "already_streaming"}

            try:
                logger.info("=== Starting Livepeer Stream ===")
                
                # Check if buffer service is available
                if not self.buffer_service:
                    logger.error("Buffer service not available")
                    return {"status": "error", "message": "Buffer service not available"}
                
                # Check if Livepeer is configured
                if not self.ingest_url or not self.stream_key:
                    logger.error("Livepeer not configured")
                    return {"status": "error", "message": "Livepeer not configured"}
                    
                # Set full ingest URL with stream key
                ingest_url = f"{self.ingest_url}/{self.stream_key}"
                logger.info(f"Using Livepeer ingest URL: {ingest_url}")
                
                # Build ultra-stable FFmpeg command optimized for Livepeer (anti-jitter)
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "mjpeg",
                    "-r", "15",
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "faster",
                    "-pix_fmt", "yuv420p",
                    # CRITICAL: Consistent keyframe settings to eliminate jitter warnings
                    "-g", "30",                    # Keyframe every 2 seconds (15fps * 2)
                    "-keyint_min", "30",           # Force minimum keyframe interval = max
                    "-sc_threshold", "0",          # Disable scene cut detection
                    "-force_key_frames", "expr:gte(t,n_forced*2)",  # Force keyframes every 2 seconds
                    # Disable B-frames for consistent delivery (required for low latency)
                    "-bf", "0",                    # No B-frames
                    "-b_strategy", "0",            # Disable B-frame strategy
                    # Optimized rate control for stable streaming
                    "-b:v", "1200k",               # Slightly higher base bitrate
                    "-maxrate", "1500k",           # Maximum bitrate cap
                    "-bufsize", "2400k",           # 2x maxrate for stable buffer
                    "-crf", "23",                  # Constant quality baseline
                    # Additional stability settings
                    "-threads", "2",               # Limit threads for consistency
                    "-slices", "1",                # Single slice for lower latency
                    "-refs", "1",                  # Single reference frame
                    "-f", "flv",
                    ingest_url
                ]

                logger.info(f"Starting FFmpeg with optimized settings")
                
                # Start FFmpeg process
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )

                # Start the streaming thread
                self.is_streaming = True
                self.stop_streaming_event.clear()
                self._stream_thread = threading.Thread(
                    target=self._stream_frames_optimized,
                    daemon=True
                )
                self._stream_thread.start()
                
                # Start monitoring thread
                self._monitoring_thread = threading.Thread(
                    target=self._monitor_stream_health,
                    daemon=True
                )
                self._monitoring_thread.start()

                logger.info("Livepeer stream started successfully")
                return {
                    "status": "streaming",
                    "stream_key": self.stream_key,
                    "ingest_url": self.ingest_url,
                    "hardware_acceleration": self.hw_accel_available
                }

            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                self.stop_stream()
                return {"status": "error", "message": str(e)}

    def _stream_frames_optimized(self):
        """Ultra-stable streaming with aggressive anti-jitter timing"""
        logger.info("Starting ultra-stable frame streaming...")
        
        import time
        target_fps = 15.0
        frame_interval = 1.0 / target_fps  # Exactly 66.67ms between frames
        
        # Initialize timing with more aggressive control
        start_time = time.time()
        frame_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        # Frame buffer and timing control
        last_frame_data = None
        last_send_time = start_time
        frame_queue_size = 0
        max_queue_size = 3  # Limit frame queue to prevent buildup
        
        while not self.stop_streaming_event.is_set():
            try:
                current_time = time.time()
                
                # Check if FFmpeg process is still alive
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process died")
                    break
                
                # Calculate if we should send a frame now
                time_since_last_send = current_time - last_send_time
                
                # Only proceed if enough time has passed (strict timing)
                if time_since_last_send < frame_interval:
                    # Sleep for the remaining time, but not too long
                    sleep_time = min(frame_interval - time_since_last_send, 0.01)
                    time.sleep(sleep_time)
                    continue
                
                # Get fresh frame from buffer service
                frame_data = self.buffer_service.get_jpeg_frame(quality=90, processed=False)
                if frame_data is not None and frame_data[0] is not None:
                    jpeg_bytes = frame_data[0]
                    if jpeg_bytes and len(jpeg_bytes) > 1000:
                        last_frame_data = jpeg_bytes
                        frame_queue_size = 0  # Reset queue size on fresh frame
                
                # Use last good frame if current frame is invalid
                if last_frame_data is None:
                    time.sleep(0.001)
                    continue
                
                # Prevent frame queue buildup (anti-jitter measure)
                frame_queue_size += 1
                if frame_queue_size > max_queue_size:
                    # Skip this frame to prevent jitter
                    frame_queue_size = 0
                    last_send_time = current_time
                    continue
                
                # Send frame to FFmpeg with strict timing
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    try:
                        self.ffmpeg_process.stdin.write(last_frame_data)
                        self.ffmpeg_process.stdin.flush()
                        consecutive_errors = 0
                        frame_count += 1
                        last_send_time = current_time
                        
                        # Update monitoring stats
                        self._frames_sent = frame_count
                        self._last_frame_time = current_time
                        
                        # Log timing every 150 frames (10 seconds)
                        if frame_count % 150 == 0:
                            actual_fps = frame_count / (current_time - start_time)
                            logger.debug(f"Streaming at {actual_fps:.2f} FPS (target: {target_fps})")
                        
                    except (BrokenPipeError, OSError) as e:
                        consecutive_errors += 1
                        logger.warning(f"FFmpeg pipe error: {e} (error {consecutive_errors}/{max_consecutive_errors})")
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        time.sleep(0.01)
                        continue
                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(f"Error writing to FFmpeg: {e} (error {consecutive_errors}/{max_consecutive_errors})")
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        time.sleep(0.01)
                        continue
                else:
                    break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in streaming loop: {e} (error {consecutive_errors}/{max_consecutive_errors})")
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(0.01)
        
        logger.info(f"Ultra-stable frame streaming stopped after {frame_count} frames")
        self.is_streaming = False

    def _monitor_stream_health(self):
        """Monitor stream health and restart if needed"""
        logger.info("Starting stream health monitoring...")
        
        while not self.stop_streaming_event.is_set():
            try:
                time.sleep(self._health_check_interval)
                
                if not self.is_streaming:
                    break
                
                current_time = time.time()
                time_since_last_frame = current_time - self._last_frame_time
                
                # Check if streaming has stalled (no frames sent in 30 seconds)
                if time_since_last_frame > 30:
                    logger.warning(f"Stream stalled - no frames sent for {time_since_last_frame:.1f} seconds")
                    self._restart_stream()
                    break
                
                # Check if FFmpeg process died
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process died, restarting stream")
                    self._restart_stream()
                    break
                
                # Log health status
                if self._frames_sent > 0:
                    logger.debug(f"Stream health: {self._frames_sent} frames sent, last frame {time_since_last_frame:.1f}s ago")
                
            except Exception as e:
                logger.error(f"Error in stream monitoring: {e}")
                time.sleep(5)
        
        logger.info("Stream health monitoring stopped")

    def _restart_stream(self):
        """Restart the stream automatically"""
        current_time = time.time()
        
        # Prevent rapid restarts
        if current_time - self._last_restart_time < 30:
            logger.warning("Preventing rapid restart - waiting...")
            return
        
        self._restart_count += 1
        if self._restart_count > self._max_restarts:
            logger.error(f"Max restarts ({self._max_restarts}) exceeded, stopping stream")
            self._force_stop_stream()
            return
        
        logger.info(f"Attempting stream restart #{self._restart_count}")
        self._last_restart_time = current_time
        
        # Schedule restart in a separate thread to avoid deadlock
        restart_thread = threading.Thread(
            target=self._perform_restart,
            daemon=True
        )
        restart_thread.start()

    def _perform_restart(self):
        """Perform the actual restart in a separate thread"""
        try:
            # Force stop current stream
            self._force_stop_stream()
            
            # Wait a moment for cleanup
            time.sleep(3)
            
            # Restart stream
            self.start_stream()
            logger.info("Stream restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart stream: {e}")

    def _force_stop_stream(self):
        """Force stop the stream without waiting for threads"""
        logger.info("Force stopping Livepeer stream...")
        
        # Signal stop
        self.stop_streaming_event.set()
        self.is_streaming = False
        
        # Clean up FFmpeg process immediately
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.poll() is None:  # Still running
                    self.ffmpeg_process.terminate()
                    # Don't wait long for graceful shutdown
                    try:
                        self.ffmpeg_process.wait(timeout=1)
                    except:
                        self.ffmpeg_process.kill()
            except:
                pass
            finally:
                self.ffmpeg_process = None
        
        # Reset monitoring stats
        self._last_frame_time = 0
        self._frames_sent = 0
        
        logger.info("Force stop completed")

    def stop_stream(self) -> dict:
        """Stop streaming to Livepeer"""
        with self._lock:
            if not self.is_streaming:
                return {"status": "not_streaming"}

            logger.info("Stopping Livepeer stream...")
            
            # Signal stop
            self.stop_streaming_event.set()
            self.is_streaming = False
            
            # Clean up FFmpeg process
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=3)
                except:
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass
                finally:
                    self.ffmpeg_process = None
            
            # Wait for thread to finish
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=3)
            
            # Wait for monitoring thread to finish
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=3)
            
            # Reset monitoring stats
            self._last_frame_time = 0
            self._frames_sent = 0
            
            logger.info("Livepeer stream stopped")
            return {"status": "stopped"}

    def cleanup_stream(self):
        """Clean up streaming resources"""
        self.stop_stream()

    def get_stream_status(self):
        """Get current stream status and health information"""
        import time
        current_time = time.time()
        
        return {
            "status": "streaming" if self.is_streaming else "stopped",
            "stream_key": self.stream_key if hasattr(self, 'stream_key') else None,
            "playback_id": self.playback_id if hasattr(self, 'playback_id') else None,
            "ingest_url": self.ingest_url if hasattr(self, 'ingest_url') else None,
            "hardware_acceleration": self.hw_accel_available,
            "frames_sent": getattr(self, '_frames_sent', 0),
            "last_frame_time": getattr(self, '_last_frame_time', 0),
            "seconds_since_last_frame": current_time - getattr(self, '_last_frame_time', current_time),
            "ffmpeg_process_alive": self.ffmpeg_process is not None and self.ffmpeg_process.poll() is None if self.ffmpeg_process else False,
            "restart_count": self._restart_count
        }

    def get_stream_info(self):
        """Get current stream information"""
        return {
            "is_streaming": self.is_streaming,
            "stream_key": self.stream_key if hasattr(self, 'stream_key') else None,
            "playback_id": self.playback_id if hasattr(self, 'playback_id') else None,
            "hardware_acceleration": self.hw_accel_available,
            "restart_count": self._restart_count
        } 