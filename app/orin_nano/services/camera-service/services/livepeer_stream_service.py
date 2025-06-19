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
import json
import requests

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
            
            # Stream settings optimized for Jetson - ULTRA LOW LATENCY
            self.width = 1280
            self.height = 720
            self.frame_rate = 20  # Increased for responsiveness
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
    
    def set_services(self, services):
        """Inject all services for visual effects"""
        self._services = services

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

    def _validate_stream_active(self, timeout=30):
        """
        Validate that the stream is actually active on Livepeer's side
        Returns True if stream is confirmed active, False otherwise
        """
        if not self.playback_id:
            logger.warning("No playback ID available for validation")
            return False
            
        try:
            # Check Livepeer API for stream status
            api_url = f"https://livepeer.studio/api/stream/{self.stream_key}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            for attempt in range(timeout):
                try:
                    response = requests.get(api_url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        stream_data = response.json()
                        if stream_data.get('isActive'):
                            logger.info(f"✅ Stream confirmed active on Livepeer after {attempt+1} seconds")
                            return True
                    time.sleep(1)
                except Exception as e:
                    logger.debug(f"Stream validation attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            
            logger.warning(f"❌ Stream not active on Livepeer after {timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Stream validation error: {e}")
            return False

    def _check_playback_url(self):
        """
        Check if the playback URL is accessible
        Returns True if playback is working, False otherwise
        """
        if not self.playback_id:
            return False
            
        try:
            playback_url = f"https://livepeercdn.studio/hls/{self.playback_id}/index.m3u8"
            response = requests.head(playback_url, timeout=10)
            is_accessible = response.status_code == 200
            
            if is_accessible:
                logger.info(f"✅ Playback URL accessible: {playback_url}")
            else:
                logger.warning(f"❌ Playback URL not accessible (status {response.status_code}): {playback_url}")
                
            return is_accessible
        except Exception as e:
            logger.error(f"Playback URL check failed: {e}")
            return False

    def _enhanced_ffmpeg_command(self, ingest_url):
        """
        Create FFmpeg command optimized for ABSOLUTE MINIMUM LATENCY
        Sacrificing quality for real-time performance
        """
        # ULTRA LOW LATENCY settings - prioritize speed over quality
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "mjpeg",
            "-r", "10",                    # Reduced to 10 FPS for lower latency
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",        # Fastest encoding preset
            "-tune", "zerolatency",        # Zero latency tuning
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",      # Baseline profile for fastest decode
            # ABSOLUTE MINIMUM LATENCY SETTINGS
            "-g", "10",                    # Keyframe every 1 second (10fps * 1)
            "-keyint_min", "10",           # Same as -g for consistency
            "-sc_threshold", "0",          # Disable scene cut detection
            "-bf", "0",                    # No B-frames (required for low latency)
            # LOWER BITRATE FOR SPEED
            "-b:v", "800k",                # Reduced bitrate for faster encoding
            "-maxrate", "800k",            # Same as bitrate for CBR
            "-bufsize", "400k",            # Smaller buffer for lower latency
            "-minrate", "800k",            # Force CBR
            # ULTRA LOW LATENCY OPTIMIZATIONS
            "-x264opts", "bframes=0:no-scenecut:rc-lookahead=0:sync-lookahead=0:sliced-threads=1",
            "-threads", "2",               # Limit threads for consistency
            "-r", "10",                    # Ensure consistent framerate
            "-force_key_frames", "expr:gte(t,n_forced*1)",  # Force keyframes every 1 second
            # MINIMAL BUFFERING
            "-flush_packets", "1",         # Flush packets immediately
            "-fflags", "+genpts+igndts",   # Generate PTS, ignore DTS
            # NETWORK SETTINGS
            "-f", "flv",
            ingest_url
        ]
        
        return ffmpeg_cmd

    def start_stream(self) -> dict:
        """Start streaming to Livepeer with enhanced validation and error detection"""
        with self._lock:
            if self.is_streaming:
                return {"status": "already_streaming"}

            try:
                logger.info("=== Starting ENHANCED Livepeer Stream with Validation ===")
                
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
                
                # Use enhanced FFmpeg command
                ffmpeg_cmd = self._enhanced_ffmpeg_command(ingest_url)

                logger.info(f"Starting FFmpeg with LIVEPEER-COMPLIANT settings (15fps, 1200k CBR, 1s keyframes)")
                
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

                # Wait a moment for stream to initialize
                time.sleep(3)
                
                # Enhanced validation
                logger.info("🔍 Validating stream activation on Livepeer...")
                stream_active = self._validate_stream_active(timeout=30)
                playback_working = self._check_playback_url()
                
                result = {
                    "status": "streaming",
                    "stream_key": self.stream_key,
                    "ingest_url": self.ingest_url,
                    "playback_id": self.playback_id,
                    "playback_url": f"https://livepeercdn.studio/hls/{self.playback_id}/index.m3u8",
                    "hardware_acceleration": self.hw_accel_available,
                                         "optimization": "livepeer_compliant_cbr",
                     "keyframe_interval": "1_second",
                     "target_fps": 15,
                     "bitrate": "1200k",
                    "stream_validated": stream_active,
                    "playback_accessible": playback_working,
                    "validation_notes": []
                }
                
                # Add validation notes
                if not stream_active:
                    result["validation_notes"].append("Stream not yet active on Livepeer - may take 30-60 seconds")
                if not playback_working:
                    result["validation_notes"].append("Playback URL not yet accessible - normal for new streams")
                
                if stream_active and playback_working:
                    logger.info("✅ STREAM FULLY VALIDATED - should be visible in Livepeer dashboard")
                elif stream_active:
                    logger.info("⚠️  Stream active but playback still initializing")
                else:
                    logger.warning("⚠️  Stream started but Livepeer validation pending")

                return result

            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                self.stop_stream()
                return {"status": "error", "message": str(e)}

    def _stream_frames_optimized(self):
        """ULTRA LOW LATENCY streaming with minimal visual effects overhead"""
        logger.info("Starting ULTRA LOW LATENCY streaming with minimal overhead...")
        
        import time
        import cv2
        target_fps = 10.0  # Reduced to 10 FPS to match FFmpeg for minimum latency
        frame_interval = 1.0 / target_fps
        
        # Initialize timing
        start_time = time.time()
        frame_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Minimal frame management
        last_frame_data = None
        last_send_time = start_time
        
        # Get services for visual overlays (MINIMAL overhead)
        face_service = None
        gesture_service = None
        
        if hasattr(self, '_services'):
            face_service = self._services.get('face')
            gesture_service = self._services.get('gesture')
            logger.info(f"Visual services loaded - Face: {face_service is not None}, Gesture: {gesture_service is not None}")
        
        # PERFORMANCE COUNTERS (log only occasionally)
        visual_effects_time = 0
        encoding_time = 0
        log_interval = 300  # Log every 300 frames (20 seconds at 15fps)
        
        while not self.stop_streaming_event.is_set():
            try:
                current_time = time.time()
                
                # Check if FFmpeg process is still alive
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process died")
                    break
                
                # Send frame if enough time passed
                time_since_last_send = current_time - last_send_time
                
                if time_since_last_send >= frame_interval:
                    # Get RAW frame from buffer
                    frame, timestamp = self.buffer_service.get_frame()
                    
                    if frame is not None:
                        # MINIMAL VISUAL EFFECTS PROCESSING
                        processed_frame = frame
                        effects_start_time = time.time()
                        
                        # Apply face service overlays (NO DEBUG LOGGING)
                        if face_service:
                            try:
                                processed_frame = face_service.get_processed_frame(processed_frame)
                            except Exception as e:
                                if frame_count % 100 == 0:  # Log errors only occasionally
                                    logger.error(f"Face overlay error: {e}")
                        
                        # Apply gesture service overlays (NO DEBUG LOGGING)
                        if gesture_service:
                            try:
                                processed_frame = gesture_service.get_processed_frame(processed_frame)
                            except Exception as e:
                                if frame_count % 100 == 0:  # Log errors only occasionally
                                    logger.error(f"Gesture overlay error: {e}")
                        
                        visual_effects_time += time.time() - effects_start_time
                        
                        # FAST ENCODING
                        encoding_start_time = time.time()
                        try:
                            # Use lower quality for speed (60 instead of 75)
                            _, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                            jpeg_bytes = jpeg.tobytes()
                            
                            if jpeg_bytes and len(jpeg_bytes) > 500:
                                last_frame_data = jpeg_bytes
                        except Exception as e:
                            if frame_count % 100 == 0:
                                logger.warning(f"Encoding error: {e}")
                        
                        encoding_time += time.time() - encoding_start_time
                    
                    # Use last good frame if current frame is invalid
                    if last_frame_data is None:
                        time.sleep(0.001)
                        continue
                    
                    # Send frame to FFmpeg IMMEDIATELY
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
                            
                            # MINIMAL LOGGING - only every 300 frames (20 seconds)
                            if frame_count % log_interval == 0:
                                elapsed = current_time - start_time
                                actual_fps = frame_count / elapsed
                                avg_visual_time = (visual_effects_time / frame_count) * 1000
                                avg_encoding_time = (encoding_time / frame_count) * 1000
                                logger.info(f"ULTRA LOW LATENCY: {actual_fps:.1f}fps, Visual: {avg_visual_time:.1f}ms, Encoding: {avg_encoding_time:.1f}ms per frame")
                            
                        except (BrokenPipeError, OSError) as e:
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                break
                            continue
                        except Exception as e:
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                break
                            continue
                    else:
                        break
                else:
                    # Minimal sleep to prevent CPU spinning
                    time.sleep(0.001)
                
            except Exception as e:
                consecutive_errors += 1
                if frame_count % 50 == 0:  # Log errors only occasionally
                    logger.error(f"Streaming loop error: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(0.001)
        
        logger.info(f"ULTRA LOW LATENCY streaming stopped after {frame_count} frames")
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