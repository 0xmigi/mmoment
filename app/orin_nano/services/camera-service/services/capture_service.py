"""
Capture Service

Handles photo and video capture without modifying the source buffer.
Provides clean storage and retrieval of media files.
"""

import os
import cv2
import time
import uuid
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptureService")

class CaptureService:
    """
    Service for capturing photos and videos from the buffer.
    This service reads from the buffer without modifying it.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CaptureService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        
        # Recording state
        self._recording_lock = threading.Lock()
        self._recording_active = False
        self._recording_thread = None
        self._stop_recording = threading.Event()
        self._current_writer = None
        self._current_recording_path = None
        
        # Storage paths - use Docker volume mount paths
        self._base_dir = "/app"
        self._photos_dir = os.path.join(self._base_dir, "photos")
        self._videos_dir = os.path.join(self._base_dir, "videos")
        
        # Create directories if they don't exist
        Path(self._photos_dir).mkdir(parents=True, exist_ok=True)
        Path(self._videos_dir).mkdir(parents=True, exist_ok=True)
        
        # Storage limits
        self._max_photos = 100
        self._max_videos = 20
        self._cleanup_threshold = 0.9  # Cleanup when 90% full
        
        # Photo capture settings
        self._jpeg_quality = 95
        
        # Video capture settings - OPTIMIZED FOR SMOOTH PLAYBACK
        self._video_format = 'mp4'  # MP4 format for universal compatibility
        self._video_codec = 'mp4v'  # mp4v codec for better compression
        self._video_fps = 30  # Match camera FPS for smooth playback
        
        logger.info("CaptureService initialized with mp4v codec and MOV format for IPFS compatibility")
    
    def capture_photo(self, buffer_service, user_id: str = None) -> Dict:
        """
        Capture a photo from the buffer.
        
        Args:
            buffer_service: The buffer service to get frames from
            user_id: Optional user identifier for the filename
            
        Returns:
            Dict with photo information (success, path, etc.)
        """
        try:
            # Get the latest frame from the buffer
            frame, timestamp = buffer_service.get_frame()
            
            if frame is None:
                logger.error("Failed to capture photo: No frame available")
                return {
                    "success": False,
                    "error": "No frame available"
                }
            
            # Generate a filename
            filename = self._generate_photo_filename(user_id)
            filepath = os.path.join(self._photos_dir, filename)
            
            # Save the frame as a JPEG
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            
            if not success:
                logger.error(f"Failed to save photo to {filepath}")
                return {
                    "success": False,
                    "error": "Failed to save photo"
                }
            
            # Get file info
            filesize = os.path.getsize(filepath)
            
            logger.info(f"Photo captured successfully: {filepath} ({filesize} bytes)")
            
            # Run cleanup if needed
            self._cleanup_photos()
            
            # Return photo info
            return {
                "success": True,
                "path": filepath,
                "filename": filename,
                "timestamp": int(timestamp * 1000),
                "size": filesize,
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error capturing photo: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def start_recording(self, buffer_service, user_id: str = None, duration_seconds: int = 0) -> Dict:
        """
        Start recording a video from the buffer - COMPLETELY REWRITTEN FOR RELIABILITY.
        
        Args:
            buffer_service: The buffer service to get frames from
            user_id: Optional user identifier for the filename
            duration_seconds: Optional duration in seconds (0 means record until stopped)
            
        Returns:
            Dict with recording information (success, path, etc.)
        """
        with self._recording_lock:
            try:
                # Generate a filename
                filename = self._generate_video_filename(user_id)
                filepath = os.path.join(self._videos_dir, filename)
                
                # Get a frame for dimensions
                frame, _ = buffer_service.get_frame()
                
                if frame is None:
                    logger.error("Failed to start recording: No frame available")
                    return {
                        "success": False,
                        "error": "No frame available"
                    }
                
                # Get frame dimensions
                height, width = frame.shape[:2]
                logger.info(f"Starting video recording: {width}x{height} at {self._video_fps} FPS")
                
                # Try multiple codec options for maximum compatibility (mp4v first for quality)
                codecs_to_try = [
                    ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Original quality codec
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # High bitrate fallback
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Legacy fallback
                ]
                
                writer = None
                for codec_name, fourcc in codecs_to_try:
                    logger.info(f"Trying codec: {codec_name}")
                    writer = cv2.VideoWriter(filepath, fourcc, self._video_fps, (width, height))
                    
                    if writer.isOpened():
                        logger.info(f"Successfully initialized video writer with {codec_name} codec")
                        break
                    else:
                        logger.warning(f"Failed to initialize with {codec_name} codec")
                        writer.release()
                        writer = None
                
                if writer is None or not writer.isOpened():
                    logger.error("Failed to create video writer with any codec")
                    return {
                        "success": False,
                        "error": "Failed to create video writer - no supported codec found"
                    }
                
                # Set recording state
                self._recording_active = True
                self._stop_recording.clear()
                self._current_writer = writer
                self._current_recording_path = filepath
                
                # Start recording thread with simpler logic
                self._recording_thread = threading.Thread(
                    target=self._simple_recording_loop,
                    args=(buffer_service, duration_seconds),
                    daemon=True,
                    name="VideoRecordingThread"
                )
                self._recording_thread.start()
                
                logger.info(f"Started recording to {filepath}")
                
                return {
                    "success": True,
                    "path": filepath,
                    "filename": filename,
                    "recording": True,
                    "duration_limit": duration_seconds
                }
                
            except Exception as e:
                logger.error(f"Error starting recording: {e}")
                # Clean up on error
                with self._recording_lock:
                    self._recording_active = False
                    if self._current_writer:
                        self._current_writer.release()
                        self._current_writer = None
                    self._current_recording_path = None
                
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def stop_recording(self) -> Dict:
        """
        Stop the current recording.
        
        Returns:
            Dict with recording information (success, path, etc.)
        """
        with self._recording_lock:
            if not self._recording_active:
                logger.warning("No recording in progress")
                return {
                    "success": False,
                    "error": "No recording in progress"
                }
            
            # Signal the recording thread to stop
            self._stop_recording.set()
            
            # Wait for the recording thread to finish (with timeout)
            if self._recording_thread and self._recording_thread.is_alive():
                self._recording_thread.join(timeout=5.0)
                if self._recording_thread.is_alive():
                    logger.warning("Recording thread did not stop within timeout")
            
            # Get the final file info
            filepath = self._current_recording_path
            filename = os.path.basename(filepath) if filepath else None
            filesize = 0
            
            if filepath and os.path.exists(filepath):
                filesize = os.path.getsize(filepath)
                logger.info(f"Recording stopped: {filepath} ({filesize} bytes)")
            else:
                logger.warning("Recording file not found after stopping")
            
            # Clean up state
            self._recording_active = False
            self._current_writer = None
            self._current_recording_path = None
            
            return {
                "success": True,
                "path": filepath,
                "filename": filename,
                "size": filesize,
                "recording": False
            }
    
    def _simple_recording_loop(self, buffer_service, duration_seconds: int) -> None:
        """
        HIGH-PERFORMANCE recording loop optimized for smooth video capture.
        Captures frames at the target FPS rate for consistent playback.
        """
        start_time = time.time()
        frame_count = 0
        target_frame_interval = 1.0 / self._video_fps  # Time between frames for target FPS
        next_frame_time = start_time
        
        logger.info(f"Recording loop started (duration: {duration_seconds}s, target FPS: {self._video_fps})")
        
        try:
            while not self._stop_recording.is_set():
                current_time = time.time()
                
                # Check duration limit
                if duration_seconds > 0 and (current_time - start_time) >= duration_seconds:
                    logger.info(f"Recording reached {duration_seconds}s duration limit")
                    break
                
                # SAFETY: Force stop any recording that runs longer than 10 minutes
                if (current_time - start_time) >= 600:  # 10 minutes
                    logger.warning(f"Recording force-stopped after 10 minutes for safety")
                    break
                
                # Check if it's time for the next frame
                if current_time >= next_frame_time:
                    # Get frame from buffer
                    frame, timestamp = buffer_service.get_frame()
                    
                    if frame is not None:
                        # Write frame to video
                        if self._current_writer and self._current_writer.isOpened():
                            self._current_writer.write(frame)
                            frame_count += 1
                            
                            # Schedule next frame capture
                            next_frame_time += target_frame_interval
                            
                            # If we're falling behind, catch up
                            if next_frame_time < current_time:
                                next_frame_time = current_time + target_frame_interval
                            
                            # Log progress every second (at 30 FPS = every 30 frames)
                            if frame_count % self._video_fps == 0:
                                elapsed = current_time - start_time
                                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                                logger.debug(f"Recording: {frame_count} frames, {elapsed:.1f}s, {actual_fps:.1f} FPS")
                        else:
                            logger.error("Video writer is not available")
                            break
                else:
                    # Sleep until next frame time
                    sleep_time = next_frame_time - current_time
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.001))  # Small sleep to prevent CPU spinning
                
        except Exception as e:
            logger.error(f"Error in recording loop: {e}")
        finally:
            # Clean up
            if self._current_writer:
                self._current_writer.release()
                logger.info("Video writer released")
            
            # Final stats
            duration = time.time() - start_time
            avg_fps = frame_count / duration if duration > 0 else 0
            logger.info(f"Recording completed: {frame_count} frames in {duration:.2f}s (avg {avg_fps:.1f} FPS)")
            
            # Save the recording path before resetting it
            recording_path_for_conversion = self._current_recording_path
            
            # Reset recording state - CRITICAL FIX
            with self._recording_lock:
                self._recording_active = False
                self._current_writer = None
                self._current_recording_path = None
            
            # Convert MOV to MP4 for better browser compatibility
            if recording_path_for_conversion and os.path.exists(recording_path_for_conversion):
                logger.info(f"Starting MP4 conversion for: {recording_path_for_conversion}")
                self._convert_to_mp4(recording_path_for_conversion)
            
            # Run cleanup
            self._cleanup_videos()
    
    def _generate_photo_filename(self, user_id: str = None) -> str:
        """Generate a unique filename for a photo"""
        timestamp = int(time.time())
        prefix = f"{user_id}_" if user_id else ""
        return f"{prefix}photo_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    
    def _generate_video_filename(self, user_id: str = None) -> str:
        """Generate a unique filename for a video"""
        timestamp = int(time.time())
        prefix = f"{user_id}_" if user_id else ""
        # Use .mov extension for initial recording, will be converted to .mp4
        return f"{prefix}video_{timestamp}_{uuid.uuid4().hex[:6]}.mov"
    
    def _convert_to_mp4(self, mov_path: str) -> None:
        """
        Convert MOV file to MP4 with H.264 codec for better browser compatibility.
        This runs in the background and doesn't block the recording response.
        """
        try:
            # Generate MP4 filename
            mp4_path = mov_path.replace('.mov', '.mp4')
            
            # Skip if MP4 already exists
            if os.path.exists(mp4_path):
                logger.info(f"MP4 version already exists: {mp4_path}")
                return
            
            logger.info(f"Converting {mov_path} to MP4 for browser compatibility...")
            
            # Use ffmpeg to convert with H.264 codec
            import subprocess
            
            # FFmpeg command for efficient conversion with original quality settings
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', mov_path,  # Input file
                '-c:v', 'libx264',  # H.264 video codec
                '-preset', 'fast',  # Fast encoding preset (original setting)
                '-crf', '23',  # Good quality/size balance (original setting)
                '-r', str(self._video_fps),  # Ensure correct frame rate
                '-c:a', 'aac',  # AAC audio codec (if audio exists)
                '-movflags', '+faststart',  # Optimize for web streaming
                mp4_path  # Output file
            ]
            
            # Run conversion in background
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode == 0:
                # Check if the MP4 file was created and has reasonable size
                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000:
                    logger.info(f"Successfully converted to MP4: {mp4_path}")
                else:
                    logger.warning(f"MP4 conversion produced invalid file: {mp4_path}")
                    # Clean up invalid file
                    if os.path.exists(mp4_path):
                        os.remove(mp4_path)
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out for {mov_path}")
        except Exception as e:
            logger.error(f"Error converting {mov_path} to MP4: {e}")
    
    def _cleanup_photos(self) -> None:
        """Clean up old photos if the limit is reached"""
        try:
            # Get all photos
            photos = list(Path(self._photos_dir).glob("*.jpg"))
            
            # Check if cleanup is needed
            if len(photos) < self._max_photos * self._cleanup_threshold:
                return
                
            # Sort by modification time (oldest first)
            photos.sort(key=lambda x: x.stat().st_mtime)
            
            # Calculate how many to delete
            to_delete = len(photos) - self._max_photos
            
            if to_delete <= 0:
                return
                
            # Delete oldest photos
            for photo in photos[:to_delete]:
                try:
                    photo.unlink()
                    logger.debug(f"Deleted old photo: {photo}")
                except Exception as e:
                    logger.error(f"Failed to delete photo {photo}: {e}")
                    
            logger.info(f"Cleaned up {to_delete} old photos")
            
        except Exception as e:
            logger.error(f"Error cleaning up photos: {e}")
    
    def _cleanup_videos(self) -> None:
        """Clean up old videos if the limit is reached"""
        try:
            # Get all videos (both .mov and .mp4)
            mov_videos = list(Path(self._videos_dir).glob("*.mov"))
            mp4_videos = list(Path(self._videos_dir).glob("*.mp4"))
            
            # Group videos by base name (without extension)
            video_groups = {}
            for video in mov_videos + mp4_videos:
                # Extract base name without extension and suffix
                base_name = video.stem
                if base_name not in video_groups:
                    video_groups[base_name] = []
                video_groups[base_name].append(video)
            
            # Check if cleanup is needed (count unique videos, not files)
            if len(video_groups) <= self._max_videos:
                return
                
            # Sort groups by oldest file modification time
            sorted_groups = sorted(
                video_groups.items(),
                key=lambda x: min(f.stat().st_mtime for f in x[1])
            )
            
            # Calculate how many video groups to delete
            to_delete = len(video_groups) - self._max_videos
            
            if to_delete <= 0:
                return
                
            # Delete oldest video groups (both .mov and .mp4)
            deleted_count = 0
            for base_name, files in sorted_groups[:to_delete]:
                for video_file in files:
                    try:
                        video_file.unlink()
                        logger.debug(f"Deleted old video: {video_file}")
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete video {video_file}: {e}")
                    
            logger.info(f"Cleaned up {to_delete} old videos")
            
        except Exception as e:
            logger.error(f"Error cleaning up videos: {e}")
    
    def get_photos(self, limit: int = 10) -> List[Dict]:
        """
        Get a list of recent photos.
        
        Args:
            limit: Maximum number of photos to return
            
        Returns:
            List of photo information dictionaries
        """
        try:
            # Get all photos
            photos = list(Path(self._photos_dir).glob("*.jpg"))
            
            # Sort by modification time (newest first)
            photos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Limit the number of results
            photos = photos[:limit]
            
            # Build result list
            result = []
            for photo in photos:
                try:
                    stat = photo.stat()
                    result.append({
                        "filename": photo.name,
                        "path": str(photo),
                        "size": stat.st_size,
                        "timestamp": int(stat.st_mtime * 1000),
                        "url": f"/photos/{photo.name}"
                    })
                except Exception as e:
                    logger.error(f"Error getting info for photo {photo}: {e}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting photos: {e}")
            return []
    
    def get_videos(self, limit: int = 10) -> List[Dict]:
        """
        Get a list of recent videos.
        
        Args:
            limit: Maximum number of videos to return
            
        Returns:
            List of video information dictionaries
        """
        try:
            # Get all videos
            videos = list(Path(self._videos_dir).glob(f"*.{self._video_format}"))
            
            # Sort by modification time (newest first)
            videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Limit the number of results
            videos = videos[:limit]
            
            # Build result list
            result = []
            for video in videos:
                try:
                    stat = video.stat()
                    result.append({
                        "filename": video.name,
                        "path": str(video),
                        "size": stat.st_size,
                        "timestamp": int(stat.st_mtime * 1000),
                        "url": f"/videos/{video.name}"
                    })
                except Exception as e:
                    logger.error(f"Error getting info for video {video}: {e}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting videos: {e}")
            return []
    
    def is_recording(self) -> bool:
        """Check if recording is currently active"""
        with self._recording_lock:
            return self._recording_active
    
    def get_photo_path(self, filename: str) -> Optional[str]:
        """Get the full path for a photo filename"""
        path = os.path.join(self._photos_dir, filename)
        return path if os.path.exists(path) else None
    
    def get_video_path(self, filename: str) -> Optional[str]:
        """Get the full path for a video filename"""
        path = os.path.join(self._videos_dir, filename)
        return path if os.path.exists(path) else None

def get_capture_service() -> CaptureService:
    """Get the singleton CaptureService instance."""
    return CaptureService() 