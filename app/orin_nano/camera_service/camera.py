"""
Camera Module

Handles camera operations, frame capturing, and processing.
"""

import os
import cv2
import sys
import time
import numpy as np
import threading

# Direct import instead of relative import
import config

# Global variables
camera = None
last_frame = None
raw_frame = None
frame_lock = threading.Lock()
hands_lock = threading.Lock()  # Lock for mediapipe hands processing
detected_faces = []
identified_user = None  # Initialize as None to fix the undefined variable error
recognized_faces = {}  # Dictionary to store recognized faces with their positions

# Add site-packages paths to find libraries
potential_paths = [
    os.path.expanduser(f'~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'),
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    os.path.expanduser('~/.local/lib/python3.9/site-packages'),
    os.path.expanduser('~/.local/lib/python3.8/site-packages'),
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/lib/python3/dist-packages'
]

# Add potential paths to sys.path
for path in potential_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Add MediaPipe for hand tracking (ensure it's installed)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Configuration for hand detection
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5
    
    # Initialize hand detection
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    
    MEDIAPIPE_AVAILABLE = True
    print(f"MediaPipe hands module loaded successfully with detection confidence {min_detection_confidence} and tracking confidence {min_tracking_confidence}")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")

# Add face recognition library (ensure it's installed)
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("Face recognition library loaded successfully")
except ImportError:
    # Try alternative paths
    for path in potential_paths:
        if FACE_RECOGNITION_AVAILABLE:
            break
        try:
            if path not in sys.path:
                sys.path.append(path)
            import face_recognition
            FACE_RECOGNITION_AVAILABLE = True
            print(f"Face recognition library loaded from {path}")
            break
        except ImportError:
            continue
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("Warning: face_recognition library not available. Install with: pip install face_recognition")

# Initialize face detection
try:
    # Load pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Face detection initialized successfully")
except Exception as e:
    print(f"Error initializing face detection: {e}")
    face_cascade = None

# Initialize camera
def init_camera():
    """Initialize the camera device"""
    global camera, identified_user
    
    # Ensure identified_user is defined
    identified_user = None
    
    # Reset any stuck recording state
    import recording
    recording.recording_active = False
    print("Initialized with recording_active set to False")
    
    # Change camera index from 0 to 1 to use the Logitech StreamCam
    camera = cv2.VideoCapture(1)  # Using /dev/video1 for Logitech StreamCam
    if not camera.isOpened():
        print("Error: Could not open camera at index 1. Trying index 0...")
        # Fallback to camera index 0
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open any camera.")
            return False
        else:
            print("Successfully opened camera at index 0.")
    else:
        print("Successfully opened camera at index 1 (Logitech StreamCam).")
    return True

# Detect faces in frame
def detect_faces(frame, apply_visualization=None, identified_user=None, for_recording=False):
    """Detect faces in a frame and optionally visualize them
    
    Args:
        frame: The frame to detect faces in
        apply_visualization: Whether to draw face boxes (None uses global setting)
        identified_user: Name of identified user if known
        for_recording: If True, never draw face boxes even if visualization is enabled
    """
    global face_cascade, detected_faces, recognized_faces
    
    # CRITICAL SAFETY CHECK
    if for_recording:
        print("DETECT_FACES: for_recording=True, returning original frame with NO face detection")
        # Return the exact frame we were given, with no modifications whatsoever
        return frame, []
    
    # If we're not recording, continue with normal face detection
    if not config.enable_face_detection:
        # Face detection is disabled, return frame as-is
        return frame, []
    
    # Make a copy before processing to avoid modifying the input frame
    output_frame = frame.copy()
    
    # Determine if we should draw face boxes
    should_visualize = False
    if apply_visualization is None:
        # Use global setting if no explicit parameter given
        should_visualize = config.enable_face_visualization
    else:
        # Otherwise use the provided parameter
        should_visualize = apply_visualization
    
    # Skip visualization if for_recording is True (extra safety check)
    if for_recording:
        should_visualize = False
    
    # Perform face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,  # Higher value (7) is more selective, reducing false positives
        minSize=(50, 50)  # Larger min size (50x50) ignores small face-like patterns
    )
    
    # Update detected faces list
    detected_faces = faces
    
    # Only perform visualization if requested
    if should_visualize:
        print(f"DETECT_FACES: Drawing face boxes on frame. Box count: {len(faces)}")
        
        # Visualize faces on the output frame only
        for (x, y, w, h) in faces:
            # Get the face coordinates
            face_coords = (x, y, w, h)
            
            # Determine box color based on recognition
            box_color = (255, 0, 0)  # Blue for unrecognized faces (default)
            text = "Unknown"
            font_color = (0, 0, 0)  # Black text
            bg_color = (255, 255, 255)  # White background
            
            # Look up this face in our recognition cache
            if identified_user:
                # User is logged in and identified - show green box
                box_color = (0, 255, 0)  # Green for recognized faces
                text = identified_user
                bg_color = (230, 255, 230)  # Light green background
            
            # Draw the face box
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Add user label with white background for better readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            text_width, text_height = text_size
            
            # Create background rectangle for text
            cv2.rectangle(
                output_frame,
                (x, y - text_height - 8),
                (x + text_width + 8, y),
                bg_color,
                -1  # Filled rectangle
            )
            
            # Draw border around text background with same color as face box
            cv2.rectangle(
                output_frame,
                (x, y - text_height - 8),
                (x + text_width + 8, y),
                box_color,
                1  # 1-pixel border
            )
            
            # Draw the text
            cv2.putText(
                output_frame,
                text,
                (x + 4, y - 5),
                font,
                font_scale,
                font_color,
                1
            )
    else:
        print(f"DETECT_FACES: Skipping face box visualization. Face count: {len(faces)}")
    
    # Return the processed frame and detected faces
    return output_frame, faces

# Detect and display hand gestures
def detect_gestures(frame, gesture_name, confidence, apply_visualization=None):
    """Detect and visualize gestures in the frame"""
    # Use the visualization parameter if provided, otherwise use the global setting
    if apply_visualization is None:
        apply_visualization = config.enable_gesture_visualization
    
    # Skip visualization if disabled or gesture is "none" with low confidence
    if not apply_visualization or (gesture_name == "none" and confidence < 0.3):
        return frame
    
    # IMPORTANT: Don't add any text or overlays to the frame
    # Return the original frame without modifications
    return frame

# Frame capture thread
def capture_frames():
    """Main thread that captures frames from the camera
    
    This critical function maintains two separate frame buffers:
    1. raw_frame - Unprocessed frame direct from camera (no face boxes or overlays)
    2. last_frame - Processed frame with face detection and visualization applied
    
    Using this dual-buffer architecture ensures that services can access either:
    - The raw frame (for recording, photos, or clean processing)
    - The processed frame (for visualization with face boxes)
    
    The frame flow is:
    Camera → raw_frame buffer → processing → last_frame buffer
    """
    global camera, last_frame, raw_frame, identified_user
    
    # Ensure identified_user is defined
    if identified_user is None:
        identified_user = None
    
    # Hand gesture processing interval (every 5 frames)
    gesture_frame_count = 0
    gesture_interval = 5
    
    # Face recognition interval (every 10 frames)
    recognition_frame_count = 0
    recognition_interval = 10
    
    # Debug output interval (every 100 frames)
    debug_frame_count = 0
    debug_interval = 100
    
    # Import here to avoid circular imports
    import gestures
    
    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
            
        success, frame = camera.read()
        if success:
            # STEP 1: Store the raw frame immediately in the buffer BEFORE any processing
            # This ensures we always have access to untouched frames from the camera
            with frame_lock:
                # IMPORTANT: This is where the raw frame is stored - NEVER modify this frame
                # and store the ORIGINAL frame without ANY modifications
                raw_frame = frame.copy()
                print(f"DEBUG: CAPTURE_FRAMES - Raw frame captured with shape {raw_frame.shape}")
            
            # Process hand gestures periodically to avoid CPU overload
            gesture_frame_count += 1
            if gesture_frame_count >= gesture_interval:
                gesture_frame_count = 0
                # Fixed: Process hands WITHOUT passing a frame parameter - it will get its own copy
                threading.Thread(target=gestures.process_hands_frame, daemon=True).start()
            
            # Process face recognition periodically
            recognition_frame_count += 1
            if recognition_frame_count >= recognition_interval:
                recognition_frame_count = 0
                # Process face recognition with a COPY of the frame
                if FACE_RECOGNITION_AVAILABLE:
                    threading.Thread(target=lambda: recognize_detected_faces(frame.copy()), daemon=True).start()
            
            # Debug output periodically
            debug_frame_count += 1
            if debug_frame_count >= debug_interval:
                debug_frame_count = 0
                print(f"DEBUG: CAPTURE_FRAMES - Face detection: enabled={config.enable_face_detection}, " +
                     f"visualization={config.enable_face_visualization}")
            
            # STEP 2: Create a completely separate copy for processing that won't affect the raw frame
            processed_frame = frame.copy()
            
            # STEP 3: Apply face detection and visualization only to the processed frame
            if config.enable_face_detection:
                try:
                    # Get current visualization setting from config
                    face_vis = config.enable_face_visualization
                    
                    # Apply face detection with or without visualization ONLY to processed_frame
                    if not config.enable_face_visualization:
                        # Just detect faces without drawing anything
                        _, detected = detect_faces(processed_frame.copy(), False, identified_user)
                        # Do not modify the processed_frame at all
                    else:
                        # Normal case - apply face detection with visualization
                        processed_frame, detected = detect_faces(
                            processed_frame, 
                            True,  # Explicitly enable visualization
                            identified_user
                        )
                except Exception as e:
                    print(f"Error in face detection: {e}")
            
            # STEP 4: Apply gesture visualization if enabled (only to processed frame)
            if config.enable_gesture_visualization:
                try:
                    # Apply gesture visualization ONLY to processed_frame
                    processed_frame = detect_gestures(
                        processed_frame, 
                        gestures.active_gesture.get("gesture", "none"),
                        gestures.active_gesture.get("confidence", 0),
                        config.enable_gesture_visualization
                    )
                except Exception as e:
                    print(f"Error in gesture visualization: {e}")
            
            # STEP 5: Store the processed frame in its own buffer
            # This way, get_current_frame() returns the visualization version
            # while get_raw_frame() always returns clean frames
            with frame_lock:
                # IMPORTANT: Only ever store the processed frame in last_frame, never in raw_frame
                last_frame = processed_frame
                
                # Verify that raw_frame and last_frame are different objects (not the same memory reference)
                if id(raw_frame) == id(processed_frame) or id(raw_frame) == id(last_frame):
                    print("ERROR: CAPTURE_FRAMES - raw_frame has same memory address as processed frames!")
        
        time.sleep(0.033)  # ~30 FPS

# Function to get a copy of the current frame
def get_current_frame():
    """Get a copy of the current camera frame"""
    with frame_lock:
        if last_frame is None:
            return None
        return last_frame.copy()

# Function to get a copy of the raw frame
def get_raw_frame():
    """Get a copy of the raw camera frame without any processing
    
    This function returns a frame directly from the raw buffer. The raw_frame is captured
    and stored in the buffer BEFORE any face detection, recognition, or visualization
    processing is applied. This makes it ideal for:
    
    1. Recording raw video without face boxes or overlays
    2. Capturing photos without any visual elements 
    3. Passing clean frames to external processing systems
    
    The architecture guarantees that these frames are pristine copies of what the
    camera captures, completely untouched by any drawing or processing functions.
    """
    with frame_lock:
        if raw_frame is None:
            print("DEBUG: RAW_FRAME_ACCESS - raw_frame is None!")
            return None
        
        # Debug output to track frame access
        debug_frame = raw_frame.copy()
        print(f"DEBUG: RAW_FRAME_ACCESS - Returning raw frame with shape {debug_frame.shape}")
        
        # DO NOT MODIFY THIS FRAME - Return direct copy
        return raw_frame.copy()

# Function to attempt recognition on detected faces
def recognize_detected_faces(frame):
    """Try to recognize any faces detected in the frame"""
    global recognized_faces, identified_user
    
    # Skip if face recognition is not available
    if not FACE_RECOGNITION_AVAILABLE:
        return
    
    try:
        # Create a copy to avoid multi-threading issues
        faces_to_recognize = detected_faces.copy() if detected_faces else []
        
        if not faces_to_recognize:
            return
            
        # Convert to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If we have a global identified user, use that directly (reliable)
        if identified_user:
            current_time = time.time()
            # For all faces, set them as identified with the current user
            for face_info in faces_to_recognize:
                x, y, w, h = face_info["x"], face_info["y"], face_info["width"], face_info["height"]
                face_id = f"{x}_{y}_{w}_{h}"
                
                # Add or update the face recognition with a long expiration
                recognized_faces[face_id] = {
                    "name": identified_user,
                    "timestamp": current_time,
                    "is_global_user": True,  # Flag to indicate this comes from the global identified user
                    "expires_at": current_time + 60  # Make this valid for a long time (60 seconds)
                }
            return
        
        # Without a global identified user, we maintain existing facial recognition but don't try to recognize new ones
        # Clear expired face recognitions (older than 5 seconds) if no global user
        current_time = time.time()
        for face_id in list(recognized_faces.keys()):
            face_info = recognized_faces[face_id]
            # If this isn't from the global identified user (which has longer expiration)
            # or it's past the explicit expiration time
            if (not face_info.get("is_global_user") and current_time - face_info.get("timestamp", 0) > 3) or \
               (current_time > face_info.get("expires_at", 0)):
                del recognized_faces[face_id]
                
    except Exception as e:
        print(f"Error in face recognition process: {e}") 