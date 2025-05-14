"""
Gestures Module

Handles gesture detection and processing.
"""

import threading
import time
import cv2
import numpy as np
import config
import camera
import recording
import session

# Global variable to track current active gesture
active_gesture = {"gesture": "none", "confidence": 0}
identified_user = None  # Currently identified user, if any

def detect_hand_gesture(hand_landmarks):
    """Detect hand gesture from landmarks"""
    # Skip if no landmarks
    if hand_landmarks is None:
        return "none", 0.0
    
    # Extract landmark positions (simplified for testing)
    points = []
    for landmark in hand_landmarks.landmark:
        points.append((landmark.x, landmark.y, landmark.z))
    
    # Skip if not enough points
    if len(points) < 21:
        return "none", 0.0
    
    # Check that hand is visible - wrist must be below fingertips
    wrist_y = points[0][1]
    fingertips_y = [points[4][1], points[8][1], points[12][1], points[16][1], points[20][1]]
    avg_fingertip_y = sum(fingertips_y) / len(fingertips_y)
    
    # Ensure hand is clearly in frame with wrist below fingertips
    if not (wrist_y > avg_fingertip_y):
        return "none", 0.0
    
    # Check for thumbs up gesture - moderately strict
    if "thumbs_up" in config.allowed_gestures:
        # For thumbs up: thumb is pointing up, other fingers are curled
        thumb_tip = points[4]
        # Ensure thumb is pointing up (moderate thresholds)
        thumb_up = (thumb_tip[1] < points[3][1] - 0.02) and (thumb_tip[1] < points[2][1] - 0.03)
        
        # Check if other fingers are curled - with moderate conditions
        fingers_curled = True
        for finger_base, finger_mid, finger_tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
            # Finger is curled if fingertip is lower than middle joint
            if points[finger_tip][1] < points[finger_mid][1]:
                fingers_curled = False
                break
        
        if thumb_up and fingers_curled:
            return "thumbs_up", 0.9
    
    # Check for peace sign - medium strictness
    if "peace" in config.allowed_gestures:
        # For peace sign: index and middle fingers extended upward, others curled
        # Moderate extension of index and middle fingers
        index_extended = points[8][1] < points[5][1] - 0.05
        middle_extended = points[12][1] < points[9][1] - 0.05
        
        # Require curling of ring and pinky fingers
        ring_curled = points[16][1] > points[14][1]
        pinky_curled = points[20][1] > points[18][1]
        
        if index_extended and middle_extended and ring_curled and pinky_curled:
            return "peace", 0.85
    
    # Check for open palm - medium strictness
    if "open_palm" in config.allowed_gestures:
        # For open palm: all fingers extended 
        extended_fingers_count = 0
        
        # Count extended fingers
        for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            if points[tip][1] < points[base][1] - 0.03:
                extended_fingers_count += 1
        
        # Check thumb separately
        thumb_extended = False
        
        # Right hand check (thumb points left)
        if points[4][0] < points[2][0] - 0.02:
            thumb_extended = True
        # Left hand check (thumb points right)
        elif points[4][0] > points[2][0] + 0.02:
            thumb_extended = True
            
        # Need at least 3 fingers extended plus thumb
        if extended_fingers_count >= 3 and thumb_extended:
            return "open_palm", 0.8
    
    # Default: no gesture detected
    return "none", 0.0

def process_hands_frame():
    """Detect and process hands in the current frame"""
    global active_gesture
    
    # Check if MediaPipe is available and we have a frame to process
    if not camera.MEDIAPIPE_AVAILABLE or camera.raw_frame is None:
        print("MediaPipe not available or no frame to process")
        return
    
    try:
        # Get a copy of the current frame
        with camera.hands_lock:
            # Only copy the frame if we can get the lock
            if camera.raw_frame is not None:
                frame_copy = camera.raw_frame.copy()
            else:
                return
        
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = camera.hands.process(rgb_frame)
        
        # Default to no gesture
        gesture_name = "none"
        confidence = 0.0
        
        # Check if we detected any hand landmarks
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Detect the gesture
            gesture_name, confidence = detect_hand_gesture(hand_landmarks)
            
            # If confidence is high enough, update the active gesture
            if confidence >= 0.65:  # Moderate threshold
                previous_gesture = active_gesture.get("gesture", "none")
                
                # Only update if the gesture is stable (stays the same for consecutive detections)
                # This helps prevent flickering between gestures
                if gesture_name == previous_gesture:
                    # Increase confidence for stable gestures
                    confidence = min(confidence + 0.05, 1.0)
                
                # Update the active gesture state
                active_gesture = {
                    "gesture": gesture_name,
                    "confidence": confidence
                }
                
                print(f"Detected gesture: {gesture_name} with confidence {confidence:.2f}")
                
                # Only respond to gesture change events (not continuous detection of same gesture)
                if gesture_name != previous_gesture:
                    print(f"GESTURE CHANGED: {previous_gesture} -> {gesture_name}")
                    
                    # Handle gesture-triggered recording
                    if gesture_name == "thumbs_up":
                        # Start recording when thumbs up is detected
                        print("THUMBS UP GESTURE DETECTED - Starting recording")
                        
                        # Only start if not already recording
                        with recording.recording_lock:
                            if not recording.recording_active:
                                try:
                                    # Create a temporary session if needed
                                    temp_session_id = recording.start_from_gesture()
                                    print(f"Recording started with session ID: {temp_session_id}")
                                except Exception as e:
                                    print(f"Error starting recording from thumbs up gesture: {e}")
                    
                    elif gesture_name == "open_palm":
                        # Stop recording when open palm is detected
                        print("OPEN PALM GESTURE DETECTED - Stopping recording")
                        
                        with recording.recording_lock:
                            if recording.recording_active:
                                try:
                                    # Set flag to stop the recording
                                    recording.recording_active = False
                                    print("Recording stopped by open palm gesture")
                                except Exception as e:
                                    print(f"Error stopping recording from open palm gesture: {e}")
                    
                    elif gesture_name == "peace":
                        # Take a photo when peace sign is detected
                        print("PEACE SIGN DETECTED - Taking photo")
                        try:
                            # Import only when needed to avoid circular imports
                            from routes import capture_photo_gesture
                            capture_photo_gesture()
                        except Exception as e:
                            print(f"Error taking photo from peace gesture: {e}")
            else:
                # Gradually decay confidence for low-confidence gestures to avoid flickering
                if active_gesture["gesture"] != "none" and active_gesture["confidence"] > 0.1:
                    active_gesture["confidence"] -= 0.05  # Reduced decay rate
                    if active_gesture["confidence"] < 0.1:
                        active_gesture = {"gesture": "none", "confidence": 0.0}
    
    except Exception as e:
        print(f"Error processing hands: {e}")
        import traceback
        traceback.print_exc() 