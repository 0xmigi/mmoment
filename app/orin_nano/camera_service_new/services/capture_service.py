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
        
        # Storage paths
        self._base_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service_new")
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
        
        # Video capture settings
        self._video_format = 'mp4'
        self._video_codec = 'avc1'  # H.264
        
        logger.info("CaptureService initialized")
    
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
        Start recording a video from the buffer.
        
        Args:
            buffer_service: The buffer service to get frames from
            user_id: Optional user identifier for the filename
            duration_seconds: Optional duration in seconds (0 means record until stopped)
            
        Returns:
            Dict with recording information (success, path, etc.)
        """
        with self._recording_lock:
            if self._recording_active:
                logger.warning("Recording already in progress")
                return {
                    "success": False,
                    "error": "Recording already in progress"
                }
            
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
                fps = 30  # Target FPS
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*self._video_codec)
                writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                
                if not writer.isOpened():
                    logger.error(f"Failed to create video writer for {filepath}")
                    return {
                        "success": False,
                        "error": "Failed to create video writer"
                    }
                
                # Set recording state
                self._recording_active = True
                self._stop_recording.clear()
                
                # Start recording thread
                self._recording_thread = threading.Thread(
                    target=self._recording_loop,
                    args=(buffer_service, writer, duration_seconds, filepath),
                    daemon=True,
                    name="RecordingThread"
                )
                self._recording_thread.start()
                
                logger.info(f"Started recording to {filepath}")
                
                return {
                    "success": True,
                    "path": filepath,
                    "filename": filename,
                    "started_at": int(time.time() * 1000),
                    "duration": duration_seconds,
                    "width": width,
                    "height": height
                }
                
            except Exception as e:
                logger.error(f"Error starting recording: {e}")
                self._recording_active = False
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
            
            try:
                # Signal the recording thread to stop
                self._stop_recording.set()
                
                # Wait for the recording thread to finish
                if self._recording_thread and self._recording_thread.is_alive():
                    self._recording_thread.join(timeout=5.0)
                
                # Update recording state
                self._recording_active = False
                
                logger.info("Recording stopped")
                
                return {
                    "success": True,
                    "stopped_at": int(time.time() * 1000)
                }
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
                self._recording_active = False
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def _recording_loop(self, buffer_service, writer, duration_seconds: int, filepath: str) -> None:
        """
        Loop that writes frames to the video file.
        
        Args:
            buffer_service: The buffer service to get frames from
            writer: The OpenCV VideoWriter object
            duration_seconds: The duration to record (0 means until stopped)
            filepath: The path to the video file
        """
        try:
            start_time = time.time()
            frame_count = 0
            last_frame_time = start_time
            
            logger.info(f"Recording loop started with duration: {duration_seconds} seconds")
            
            # Record until stopped or duration reached
            while not self._stop_recording.is_set():
                # Check if duration limit reached
                if duration_seconds > 0 and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Recording reached duration limit of {duration_seconds} seconds")
                    break
                
                # Get the latest frame from the buffer
                frame, timestamp = buffer_service.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Write the frame to the video
                writer.write(frame)
                frame_count += 1
                
                # Calculate and maintain FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Aim for 30 FPS
                target_frame_time = 1.0 / 30.0
                sleep_time = max(0, target_frame_time - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
                last_frame_time = time.time()
                
        except Exception as e:
            logger.error(f"Error in recording loop: {e}")
        finally:
            # Release the video writer
            if writer:
                writer.release()
            
            # Update recording state
            with self._recording_lock:
                self._recording_active = False
            
            # Run cleanup if needed
            self._cleanup_videos()
            
            # Log completion
            duration = time.time() - start_time
            filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            logger.info(f"Recording completed: {filepath} ({frame_count} frames, {duration:.2f} seconds, {filesize} bytes)")
    
    def _generate_photo_filename(self, user_id: str = None) -> str:
        """Generate a unique filename for a photo"""
        timestamp = int(time.time())
        prefix = f"{user_id}_" if user_id else ""
        return f"{prefix}photo_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    
    def _generate_video_filename(self, user_id: str = None) -> str:
        """Generate a unique filename for a video"""
        timestamp = int(time.time())
        prefix = f"{user_id}_" if user_id else ""
        return f"{prefix}video_{timestamp}_{uuid.uuid4().hex[:6]}.{self._video_format}"
    
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
            # Get all videos
            videos = list(Path(self._videos_dir).glob(f"*.{self._video_format}"))
            
            # Check if cleanup is needed
            if len(videos) < self._max_videos * self._cleanup_threshold:
                return
                
            # Sort by modification time (oldest first)
            videos.sort(key=lambda x: x.stat().st_mtime)
            
            # Calculate how many to delete
            to_delete = len(videos) - self._max_videos
            
            if to_delete <= 0:
                return
                
            # Delete oldest videos
            for video in videos[:to_delete]:
                try:
                    video.unlink()
                    logger.debug(f"Deleted old video: {video}")
                except Exception as e:
                    logger.error(f"Failed to delete video {video}: {e}")
                    
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
                    
                    # Get video duration and dimensions
                    duration = 0
                    width = 0
                    height = 0
                    
                    try:
                        cap = cv2.VideoCapture(str(video))
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = frame_count / fps if fps > 0 else 0
                        cap.release()
                    except Exception as ve:
                        logger.error(f"Error getting video info for {video}: {ve}")
                    
                    result.append({
                        "filename": video.name,
                        "path": str(video),
                        "size": stat.st_size,
                        "timestamp": int(stat.st_mtime * 1000),
                        "duration": duration,
                        "width": width,
                        "height": height,
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
        if os.path.isfile(path):
            return path
        return None
    
    def get_video_path(self, filename: str) -> Optional[str]:
        """Get the full path for a video filename"""
        path = os.path.join(self._videos_dir, filename)
        if os.path.isfile(path):
            return path
        return None

# Global function to get the capture service instance
def get_capture_service() -> CaptureService:
    """
    Get the singleton CaptureService instance.
    """
    return CaptureService() 