"""
Recording Module

Handles video recording functionality.
"""

import os
import time
import threading
import json
import cv2
import uuid
import shutil
import config
import camera
import session

# Global recording state
recording_active = False
recording_start_time = 0
recording_writer = None
recording_thread = None
recording_filename = None
recording_lock = threading.Lock()

def start_from_gesture():
    """Start recording triggered by a gesture"""
    global recording_active, recording_frames, recording_start_time, recording_thread
    
    # Create a temporary session if needed
    temp_session_id, temp_wallet_address = session.create_temp_session()
    
    with recording_lock:
        if recording_active:
            return temp_session_id
        
        recording_active = True
        recording_frames = []
        recording_start_time = time.time()
        
        # Create directories if needed
        config.create_directories()
        
        # Set up recording for reliable AVI format
        timestamp = int(time.time())
        filename = f"{config.CAMERA_VIDEOS_DIR}/{temp_wallet_address}_{timestamp}.avi"
        
        print(f"Starting recording to {filename}")
        
        # Start recording thread
        recording_thread = threading.Thread(
            target=record_video_frames,
            args=(filename, 30, temp_wallet_address, temp_session_id),
            daemon=True
        )
        recording_thread.start()
        print("Recording thread started from thumbs up gesture")
        
    return temp_session_id

def start_recording(max_duration=30):
    """Start recording video"""
    global recording_active, recording_start_time, recording_writer, recording_thread, recording_filename
    
    with recording_lock:
        if recording_active:
            return False, "Recording already in progress"
        
        # Create videos directory if it doesn't exist
        os.makedirs(config.CAMERA_VIDEOS_DIR, exist_ok=True)
        
        # Generate timestamp-based filename
        timestamp = int(time.time())
        base_filename = f"video_{timestamp}"
        
        # Try to select the most reliable codec and container for the system
        codecs_to_try = [
            # Codec, extension
            ('mp4v', '.mp4'),  # MPEG-4, widely compatible
            ('avc1', '.mp4'),  # H.264, sometimes requires specific builds
            ('XVID', '.avi'),  # XVID in AVI - reliable but larger files
        ]
        
        # Get frame dimensions
        frame = camera.get_raw_frame()  # Use raw frame without overlays
        if frame is None:
            return False, "No camera frame available"
            
        height, width = frame.shape[:2]
        fps = 30
        
        # Try different codecs until one works
        for codec, extension in codecs_to_try:
            recording_filename = f"{base_filename}{extension}"
            output_path = os.path.join(config.CAMERA_VIDEOS_DIR, recording_filename)
            
            try:
                print(f"Trying codec {codec} with container {extension}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if writer.isOpened():
                    recording_writer = writer
                    print(f"Successfully opened video writer with codec {codec}")
                    break
                else:
                    print(f"Failed to open writer with codec {codec}")
                    writer.release()
            except Exception as e:
                print(f"Error with codec {codec}: {e}")
        
        # If all normal codecs failed, try hardware-accelerated encoding as last resort
        if recording_writer is None or not recording_writer.isOpened():
            try:
                recording_filename = f"{base_filename}.mp4"
                output_path = os.path.join(config.CAMERA_VIDEOS_DIR, recording_filename)
                gst_out = f"appsrc ! video/x-raw, format=BGR ! nvvidconv ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location={output_path}"
                recording_writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height))
                if recording_writer.isOpened():
                    print("Using hardware-accelerated H.264 encoding")
                else:
                    recording_writer = None
            except Exception as e:
                print(f"Hardware encoding failed: {e}")
        
        # Final fallback to most reliable format
        if recording_writer is None or not recording_writer.isOpened():
            recording_filename = f"{base_filename}.avi"
            output_path = os.path.join(config.CAMERA_VIDEOS_DIR, recording_filename)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG - very reliable
            recording_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print("Falling back to Motion JPEG encoding")
            
        if not recording_writer or not recording_writer.isOpened():
            return False, f"Failed to open video writer with any codec"
            
        # Set recording state
        recording_active = True
        recording_start_time = time.time()
        
        # Start recording thread
        recording_thread = threading.Thread(target=record_frames, args=(max_duration,), daemon=True)
        recording_thread.start()
        
        return True, {"filename": recording_filename, "path": output_path}

def stop_recording():
    """Stop recording video"""
    global recording_active, recording_writer, recording_filename
    
    with recording_lock:
        if not recording_active:
            return False, "No recording in progress"
        
        # Set flag to stop recording
        recording_active = False
        
        # Get full path to the recorded video
        output_path = os.path.join(config.CAMERA_VIDEOS_DIR, recording_filename)
        
        # Wait briefly for recording thread to finish
        for _ in range(5):  # Wait up to 0.5 seconds
            if recording_thread and recording_thread.is_alive():
                time.sleep(0.1)
            else:
                break
    
    # Second lock section after waiting for thread
    with recording_lock:
        # Close writer if it exists
        if recording_writer is not None:
            try:
                recording_writer.release()
                print(f"Video writer released for {output_path}")
                
                # Wait a moment for file to finalize
                time.sleep(0.5)
                
                # Check if file exists and has content
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    
                    if file_size > 0:
                        print(f"Recorded video: {output_path}, size: {file_size} bytes")
                        recording_writer = None
                        return True, {"filename": recording_filename, "path": output_path, "size": file_size}
                    else:
                        print(f"Warning: Video file exists but is empty: {output_path}")
                else:
                    print(f"Warning: Video file does not exist: {output_path}")
            except Exception as e:
                print(f"Error releasing video writer: {e}")
                import traceback
                traceback.print_exc()
            finally:
                recording_writer = None
        
        # If we get here, something went wrong
        return False, "Recording failed or produced empty file"

def get_recording_status():
    """Get current recording status"""
    global recording_active, recording_start_time, recording_filename
    
    with recording_lock:
        if not recording_active:
            return {
                "recording": False,
                "duration": 0,
                "filename": None
            }
            
        # Calculate elapsed time
        elapsed = int(time.time() - recording_start_time)
        
        return {
            "recording": True,
            "duration": elapsed,
            "elapsed_formatted": format_duration(elapsed),
            "filename": recording_filename
        }

def record_frames(max_duration):
    """Record frames from camera in a separate thread"""
    global recording_active, recording_writer, recording_start_time
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while recording_active:
            # Check if max duration exceeded
            if max_duration > 0 and (time.time() - recording_start_time) >= max_duration:
                # Auto-stop recording
                with recording_lock:
                    recording_active = False
                print("RECORDING: Max duration reached, stopping recording")
                break
            
            # Get frame directly from the raw frame buffer, same as our capture functions
            with camera.frame_lock:
                # Access raw buffer directly
                if camera.raw_frame is None:
                    print("RECORDING: No raw frame available, skipping frame")
                    time.sleep(0.033)  # Wait a bit and try again
                    continue
                    
                # Make a direct copy of the raw frame
                frame = camera.raw_frame.copy()
                
            # Check if we have a writer ready
            if frame is not None and recording_writer is not None:
                # Write frame to video
                try:
                    recording_writer.write(frame)
                    frame_count += 1
                    
                    # Log progress occasionally
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"RECORDING: {frame_count} frames, {fps:.1f} FPS")
                except Exception as e:
                    print(f"RECORDING ERROR: Error writing frame: {e}")
                
            # Control frame rate to avoid excessive CPU usage
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"Error in recording thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure writer is released
        with recording_lock:
            if recording_writer is not None:
                try:
                    # Add a small delay before releasing to ensure all frames are written
                    time.sleep(0.1)
                    recording_writer.release()
                    
                    # Log completion stats
                    elapsed = time.time() - start_time
                    print(f"Recording complete: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
                except Exception as e:
                    print(f"Error releasing writer: {e}")
                recording_writer = None
            recording_active = False
        print("Recording thread ended")

def format_duration(seconds):
    """Format seconds as MM:SS"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def record_video_frames(filename, duration, wallet_address, session_id):
    """Thread function to record video frames"""
    global recording_active, video_writer
    
    # Reset state
    video_writer = None
    recording_frames = []
    recording_ok = False
    
    start_time = time.time()
    
    try:
        print(f"Starting video recording to {filename}")
        
        # Create file path
        abs_path = os.path.abspath(filename)
        
        # Define file properties for video
        frame_size = None
        if camera.raw_frame is not None:
            h, w = camera.raw_frame.shape[:2]
            frame_size = (w, h)
        else:
            # Default size if no frame available
            frame_size = (640, 480)
        
        w, h = frame_size
        fps = 20.0
        
        # Try different reliable codec options in order of preference
        codec_options = [
            ('avc1', '.mp4'),  # H.264 in MP4 (may require specific builds)
            ('mp4v', '.mp4'),  # MPEG-4 in MP4 - most compatible
            ('XVID', '.avi'),  # XVID in AVI (reliable but larger)
            ('MJPG', '.avi')   # Motion JPEG (fallback)
        ]
        
        success = False
        working_codec = None
        
        for codec, container in codec_options:
            try:
                # Update filename with correct extension
                filename_with_ext = os.path.splitext(abs_path)[0] + container
                
                print(f"Trying codec {codec} with container {container}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(filename_with_ext, fourcc, fps, (w, h))
                
                if not video_writer.isOpened():
                    print(f"Failed to open video writer with codec {codec}, trying next codec")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    continue
                
                print(f"Successfully opened video writer with codec {codec}, container {container}")
                abs_path = filename_with_ext
                working_codec = codec
                success = True
                break
                
            except Exception as e:
                print(f"Error with codec {codec}: {e}")
                if video_writer:
                    video_writer.release()
                    video_writer = None
        
        if not success:
            # Try hardware-accelerated encoding with GStreamer as last resort
            try:
                filename_with_ext = os.path.splitext(abs_path)[0] + ".mp4"
                gst_out = f"appsrc ! video/x-raw, format=BGR ! nvvidconv ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location={filename_with_ext}"
                video_writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w, h))
                if video_writer.isOpened():
                    abs_path = filename_with_ext
                    working_codec = "h264 (hardware)"
                    success = True
                    print("Using hardware-accelerated H.264 encoding")
                else:
                    raise Exception("Failed to open GStreamer pipeline")
            except Exception as e:
                print(f"Hardware encoding failed: {e}")
                
        if not success:
            raise Exception("Failed to initialize video writer with any codec")
        
        # Record for the specified duration
        frame_count = 0
        recording_active = True
        
        # Wait for the raw_frame to be available
        max_wait = 5.0  # seconds
        wait_start = time.time()
        frame_available = False
        
        while time.time() - wait_start < max_wait and not frame_available:
            with camera.frame_lock:
                if camera.raw_frame is not None:
                    frame_available = True
                    test_frame = camera.raw_frame.copy()
            
            if not frame_available:
                print("Waiting for frame...")
                time.sleep(0.1)
        
        if not frame_available:
            print("Error: No frame available after waiting")
            recording_active = False
            return
        
        h, w = test_frame.shape[:2]
        print(f"Got frame with dimensions {w}x{h}")
        
        # Record frames until manually stopped or max duration reached
        last_frame_time = 0
        frame_interval = 1.0 / fps  # Time between frames to maintain target FPS
        
        max_duration = min(30, duration)  # Cap to 30 seconds
        print(f"Recording for up to {max_duration} seconds at {fps} FPS")
        
        # Record at least 3 seconds of frames, even if recording_active is set to False
        min_recording_time = start_time + 3.0
        
        while (time.time() - start_time < max_duration and recording_active) or time.time() < min_recording_time:
            current_time = time.time()
            
            # Only capture a new frame if enough time has passed
            if current_time - last_frame_time >= frame_interval:
                with camera.frame_lock:
                    if camera.raw_frame is not None:
                        frame_to_write = camera.raw_frame.copy()
                        
                        try:
                            video_writer.write(frame_to_write)
                            frame_count += 1
                            last_frame_time = current_time
                            
                            # Log progress
                            if frame_count % 30 == 0:  # Report every 30 frames
                                elapsed = current_time - start_time
                                print(f"Recorded {frame_count} frames ({frame_count/elapsed:.1f} FPS)")
                        except Exception as e:
                            print(f"Error writing frame: {e}")
            
            # Brief sleep to prevent 100% CPU usage
            time.sleep(0.005)  # Reduced sleep time for better responsiveness
        
        # Make sure we've recorded at least some frames
        if frame_count > 0:
            recording_ok = True
            elapsed_time = time.time() - start_time
            print(f"Recording complete: {frame_count} frames in {elapsed_time:.1f}s")
        else:
            print("No frames were recorded")
    
    except Exception as e:
        print(f"Recording error: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up resources
        try:
            if video_writer:
                video_writer.release()
                print("Video writer released")
        except Exception as e:
            print(f"Error releasing video writer: {e}")
        
        # Mark recording as complete
        with recording_lock:
            recording_active = False
        
        # Generate completion data if recording was successful
        if recording_ok and abs_path:
            try:
                # Verify the file exists and has content
                if not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0:
                    print(f"Error: Video file not found or empty: {abs_path}")
                    return
                     
                print(f"Video file exists: {abs_path}, size: {os.path.getsize(abs_path)} bytes")
                
                # Generate URL for the video
                video_url = f"/videos/{os.path.basename(abs_path)}"
                recorded_duration = time.time() - start_time
                
                # Store completion data
                completion_data = {
                    "success": True,
                    "message": "Video recording completed",
                    "duration": recorded_duration,
                    "filename": abs_path,
                    "video_url": video_url,
                    "frame_count": frame_count,
                    "codec": working_codec,
                    "player_url": f"/video-player/{os.path.basename(abs_path)}",
                    "timestamp": int(time.time())
                }
                
                # Write completion data to file
                completion_file = f"{os.path.splitext(abs_path)[0]}_completion.json"
                with open(completion_file, 'w') as f:
                    json.dump(completion_data, f)
                
                print(f"Saved completion data to {completion_file}")
                
                # Make sure the file is also available in the alternate location
                alt_dir = config.ALT_VIDEOS_DIR
                os.makedirs(alt_dir, exist_ok=True)
                
                alt_path = os.path.join(alt_dir, os.path.basename(abs_path))
                if abs_path != alt_path:
                    try:
                        print(f"Copying {abs_path} to {alt_path}")
                        shutil.copy2(abs_path, alt_path)
                        print(f"Copied video to {alt_path}")
                        print(f"File exists? {os.path.exists(alt_path)}, Size: {os.path.getsize(alt_path)} bytes")
                        
                        # Also copy the completion file
                        alt_completion = os.path.join(alt_dir, os.path.basename(completion_file))
                        shutil.copy2(completion_file, alt_completion)
                        print(f"Copied completion data to {alt_completion}")
                    except Exception as e:
                        print(f"Error copying to alternate location: {e}")
                
                # Create a thumbnail
                try:
                    cap = cv2.VideoCapture(abs_path)
                    if cap.isOpened():
                        ret, thumb_frame = cap.read()
                        if ret:
                            thumb_file = f"{os.path.splitext(abs_path)[0]}_thumb.jpg"
                            cv2.imwrite(thumb_file, thumb_frame)
                            print(f"Created thumbnail: {thumb_file}")
                            
                            # Copy thumbnail to alternate location
                            alt_thumb = os.path.join(alt_dir, os.path.basename(thumb_file))
                            shutil.copy2(thumb_file, alt_thumb)
                            print(f"Copied thumbnail to {alt_thumb}")
                        else:
                            print("Failed to read frame for thumbnail")
                        cap.release()
                    else:
                        print(f"Could not open video file for thumbnail: {abs_path}")
                except Exception as e:
                    print(f"Error creating thumbnail: {e}")
            
            except Exception as e:
                print(f"Error generating completion data: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Recording failed, no completion data generated")

def get_video_list():
    """Get a list of all recorded videos with metadata"""
    videos = []
    
    # Create directories if needed
    config.create_directories()
    
    # Check main directory
    try:
        for file in os.listdir(config.CAMERA_VIDEOS_DIR):
            if file.endswith((".mov", ".mp4", ".avi")) and not file.startswith("."):
                try:
                    filepath = os.path.join(config.CAMERA_VIDEOS_DIR, file)
                    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                        # Check if thumbnail exists
                        thumbnail_path = os.path.splitext(filepath)[0] + "_thumb.jpg"
                        has_thumbnail = os.path.exists(thumbnail_path)
                        
                        # Check if completion data exists
                        completion_path = os.path.splitext(filepath)[0] + "_completion.json"
                        completion_data = None
                        if os.path.exists(completion_path):
                            try:
                                with open(completion_path, 'r') as f:
                                    completion_data = json.load(f)
                            except:
                                pass
                        
                        # Get file information
                        file_stats = os.stat(filepath)
                        
                        # Add to list
                        videos.append({
                            "filename": file,
                            "path": filepath,
                            "size": file_stats.st_size,
                            "created": int(file_stats.st_ctime),
                            "thumbnail_url": f"/videos/{file}?thumbnail=true" if has_thumbnail else None,
                            "video_url": f"/videos/{file}",
                            "completion_data": completion_data
                        })
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    except Exception as e:
        print(f"Error listing videos in main directory: {e}")
    
    # Check alternate directory for any files not in main directory
    try:
        for file in os.listdir(config.ALT_VIDEOS_DIR):
            if file.endswith((".mov", ".mp4", ".avi")) and not file.startswith("."):
                # Skip if already in list
                if any(v["filename"] == file for v in videos):
                    continue
                
                try:
                    filepath = os.path.join(config.ALT_VIDEOS_DIR, file)
                    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                        # Check if thumbnail exists
                        thumbnail_path = os.path.splitext(filepath)[0] + "_thumb.jpg"
                        has_thumbnail = os.path.exists(thumbnail_path)
                        
                        # Check if completion data exists
                        completion_path = os.path.splitext(filepath)[0] + "_completion.json"
                        completion_data = None
                        if os.path.exists(completion_path):
                            try:
                                with open(completion_path, 'r') as f:
                                    completion_data = json.load(f)
                            except:
                                pass
                        
                        # Get file information
                        file_stats = os.stat(filepath)
                        
                        # Add to list
                        videos.append({
                            "filename": file,
                            "path": filepath,
                            "size": file_stats.st_size,
                            "created": int(file_stats.st_ctime),
                            "thumbnail_url": f"/videos/{file}?thumbnail=true" if has_thumbnail else None,
                            "video_url": f"/videos/{file}",
                            "completion_data": completion_data
                        })
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    except Exception as e:
        print(f"Error listing videos in alternate directory: {e}")
    
    # Sort by creation time, newest first
    videos.sort(key=lambda x: x["created"], reverse=True)
    
    return videos

def clear_all_videos():
    """Delete all recorded videos"""
    files_cleared = 0
    errors = []
    
    # Ensure directories exist
    config.create_directories()
    
    # Clear main directory
    for file in os.listdir(config.CAMERA_VIDEOS_DIR):
        if file.endswith((".avi", ".mp4", ".mov", ".jpg", ".json")) and not file.startswith("."):
            try:
                filepath = os.path.join(config.CAMERA_VIDEOS_DIR, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    files_cleared += 1
            except Exception as e:
                errors.append(f"Error removing {file}: {e}")
    
    # Clear alternate directory
    for file in os.listdir(config.ALT_VIDEOS_DIR):
        if file.endswith((".avi", ".mp4", ".mov", ".jpg", ".json")) and not file.startswith("."):
            try:
                filepath = os.path.join(config.ALT_VIDEOS_DIR, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    files_cleared += 1
            except Exception as e:
                errors.append(f"Error removing {file} from alternate location: {e}")
    
    return files_cleared, errors 