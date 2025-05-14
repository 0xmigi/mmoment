import os
import cv2
import time
import threading
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
camera = None
frame_buffer = []
buffer_lock = threading.Lock()
face_detection = None
hands = None
is_running = False
shutdown_event = threading.Event()
last_face_save_time = 0  # Timestamp of last face save
detected_gestures = {}  # Track detected gestures by hand ID

# Create necessary directories
os.makedirs('faces', exist_ok=True)

def init_camera(camera_id=1, width=1280, height=720, fps=30):
    """Initialize the camera"""
    global camera
    try:
        camera = cv2.VideoCapture(camera_id)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Get actual camera properties to verify settings
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized with resolution {actual_width}x{actual_height} @ {actual_fps} FPS")
        return True
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False

def init_face_detection():
    """Initialize face detection"""
    global face_detection, hands
    try:
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # 0 for short range, 1 for full range
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        logger.info("Face detection initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing face detection: {e}")
        return False

def detect_gesture(hand_landmarks):
    """
    Detect gesture from hand landmarks
    Returns: gesture name or None
    """
    # Get all landmark positions
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append((landmark.x, landmark.y, landmark.z))
    
    # Check for thumbs up gesture
    # For thumbs up: thumb tip (4) should be above thumb IP (3), and other fingers should be curled
    if (landmarks[4][1] < landmarks[3][1] and  # Thumb pointing up
        landmarks[8][1] > landmarks[5][1] and  # Index finger curled
        landmarks[12][1] > landmarks[9][1] and  # Middle finger curled
        landmarks[16][1] > landmarks[13][1] and  # Ring finger curled
        landmarks[20][1] > landmarks[17][1]):   # Pinky finger curled
        return "thumbs_up"
    
    # Check for peace/victory sign
    # For peace sign: index (8) and middle (12) fingers extended, other fingers curled
    if (landmarks[8][1] < landmarks[5][1] and  # Index finger extended
        landmarks[12][1] < landmarks[9][1] and  # Middle finger extended
        landmarks[16][1] > landmarks[13][1] and  # Ring finger curled
        landmarks[20][1] > landmarks[17][1]):   # Pinky finger curled
        return "peace"
    
    return None

def buffer_frames():
    """Thread to continuously buffer frames"""
    global frame_buffer, is_running, last_face_save_time, detected_gestures
    
    logger.info("Starting frame buffering")
    frame_counter = 0
    start_time = time.time()
    
    # Calculate buffer size for 30 seconds at current FPS
    fps = camera.get(cv2.CAP_PROP_FPS) if camera and camera.isOpened() else 30
    buffer_size = int(fps * 30)  # 30 seconds of frames
    logger.info(f"Setting buffer size to {buffer_size} frames for 30 seconds at {fps} FPS")
    
    # Face saving interval (in seconds)
    face_save_interval = 2.0
    
    # Gesture duration requirement (in seconds)
    gesture_duration_required = 1.0
    
    while not shutdown_event.is_set():
        try:
            if not camera or not camera.isOpened():
                logger.error("Camera not available")
                time.sleep(1)
                continue
            
            # Capture frame
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_counter += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Log FPS every 30 frames
            if frame_counter >= 30:
                actual_fps = frame_counter / elapsed_time
                logger.info(f"Current capture rate: {actual_fps:.2f} FPS")
                frame_counter = 0
                start_time = current_time
            
            # Process frame (face detection every 6 frames to save resources at 30fps)
            processed_frame = frame.copy()
            if frame_counter % 6 == 0 and face_detection:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                face_results = face_detection.process(rgb_frame)
                faces_detected = []
                
                if face_results.detections:
                    for i, detection in enumerate(face_results.detections):
                        # Draw face detection box
                        mp_drawing.draw_detection(processed_frame, detection)
                        
                        # Extract face bounding box
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = processed_frame.shape
                        x = max(0, int(bboxC.xmin * iw))
                        y = max(0, int(bboxC.ymin * ih))
                        w = min(int(bboxC.width * iw), iw - x)
                        h = min(int(bboxC.height * ih), ih - y)
                        
                        # Add confidence score
                        confidence = detection.score[0]
                        cv2.putText(processed_frame, f"{confidence:.2f}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Save face image periodically
                        if current_time - last_face_save_time > face_save_interval:
                            # Add some margin (20%)
                            margin_x = int(w * 0.2)
                            margin_y = int(h * 0.2)
                            face_x = max(0, x - margin_x)
                            face_y = max(0, y - margin_y)
                            face_w = min(iw - face_x, w + 2*margin_x)
                            face_h = min(ih - face_y, h + 2*margin_y)
                            
                            face_img = frame[face_y:face_y+face_h, face_x:face_x+face_w]
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            face_filename = f"faces/face_{timestamp}_{i}.jpg"
                            cv2.imwrite(face_filename, face_img)
                            logger.info(f"Saved face image to {face_filename}")
                            
                            # Remember detected face location
                            faces_detected.append((face_x, face_y, face_w, face_h))
                            
                            last_face_save_time = current_time
                
                # Hand gesture detection
                hand_results = hands.process(rgb_frame)
                
                # Reset current detections
                current_gestures = {}
                
                if hand_results.multi_hand_landmarks:
                    for hand_idx, (hand_landmarks, handedness) in enumerate(
                        zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                        
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            processed_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Get hand ID (left/right)
                        hand_id = f"{handedness.classification[0].label}_{hand_idx}"
                        
                        # Detect gesture
                        gesture = detect_gesture(hand_landmarks)
                        
                        if gesture:
                            # Add label to the frame
                            # Get wrist position to place the label
                            wrist = hand_landmarks.landmark[0]
                            x_wrist = int(wrist.x * iw)
                            y_wrist = int(wrist.y * ih)
                            cv2.putText(processed_frame, gesture, (x_wrist, y_wrist), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            
                            # Update gesture detection time
                            if hand_id not in detected_gestures:
                                detected_gestures[hand_id] = {"gesture": gesture, "start_time": current_time}
                            elif detected_gestures[hand_id]["gesture"] != gesture:
                                # Gesture changed, reset timer
                                detected_gestures[hand_id] = {"gesture": gesture, "start_time": current_time}
                            
                            # Check if gesture held long enough and near a face
                            if (current_time - detected_gestures[hand_id]["start_time"] > gesture_duration_required):
                                # Check if gesture is near a detected face
                                for face_x, face_y, face_w, face_h in faces_detected:
                                    # Calculate distance between hand wrist and face center
                                    face_center_x = face_x + face_w // 2
                                    face_center_y = face_y + face_h // 2
                                    distance = ((x_wrist - face_center_x)**2 + (y_wrist - face_center_y)**2) ** 0.5
                                    
                                    # Check if hand is close to face (within ~1.5x face width)
                                    if distance < face_w * 1.5:
                                        # Gesture detected near face, perform action
                                        logger.info(f"Gesture '{gesture}' detected near face for {current_time - detected_gestures[hand_id]['start_time']:.1f} seconds")
                                        
                                        # Save special labeled gesture image
                                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                                        gesture_filename = f"faces/gesture_{gesture}_{timestamp}.jpg"
                                        cv2.imwrite(gesture_filename, frame)
                                        logger.info(f"Saved gesture image to {gesture_filename}")
                                        
                                        # Reset timer
                                        detected_gestures[hand_id]["start_time"] = current_time
                            
                            # Record current gesture
                            current_gestures[hand_id] = gesture
                
                # Clean up gestures that are not currently detected
                for hand_id in list(detected_gestures.keys()):
                    if hand_id not in current_gestures:
                        del detected_gestures[hand_id]
            
            # Store frame in buffer (keep only last 30 seconds of frames)
            with buffer_lock:
                frame_buffer.append((processed_frame, time.time()))
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)
            
            # Sleep to maintain frame rate (less aggressive for higher FPS)
            time.sleep(0.001)
            
        except Exception as e:
            logger.error(f"Error in buffer thread: {e}", exc_info=True)
            time.sleep(0.1)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "camera_running": camera is not None and camera.isOpened(),
        "buffer_size": len(frame_buffer),
        "face_detection_enabled": face_detection is not None,
        "usb_webcam": True,
        "resolution": f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}" if camera and camera.isOpened() else "unavailable",
        "fps": camera.get(cv2.CAP_PROP_FPS) if camera and camera.isOpened() else 0
    })

@app.route('/capture', methods=['POST'])
def capture():
    """Capture an image"""
    try:
        with buffer_lock:
            if not frame_buffer:
                return jsonify({"error": "No frames in buffer"}), 500
            
            # Get latest frame
            frame, _ = frame_buffer[-1]
        
        # Convert to JPEG
        _, jpeg_data = cv2.imencode('.jpg', frame)
        
        return send_file(
            io.BytesIO(jpeg_data.tobytes()),
            mimetype='image/jpeg'
        )
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/faces', methods=['GET'])
def list_faces():
    """List saved faces"""
    try:
        # Get list of face files
        face_files = [f for f in os.listdir('faces') if f.startswith('face_')]
        face_files.sort(reverse=True)  # Newest first
        
        # Get basic info
        faces = []
        for face_file in face_files[:20]:  # Limit to 20 most recent
            timestamp_str = face_file.split('_')[1].split('.')[0]
            faces.append({
                "filename": face_file,
                "path": f"faces/{face_file}",
                "captured_at": timestamp_str,
                "url": f"/faces/{face_file}"
            })
        
        return jsonify({"faces": faces})
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/gestures', methods=['GET'])
def list_gestures():
    """List saved gesture images"""
    try:
        # Get list of gesture files
        gesture_files = [f for f in os.listdir('faces') if f.startswith('gesture_')]
        gesture_files.sort(reverse=True)  # Newest first
        
        # Get basic info
        gestures = []
        for gesture_file in gesture_files[:20]:  # Limit to 20 most recent
            parts = gesture_file.split('_')
            gesture_type = parts[1]
            timestamp_str = parts[2].split('.')[0]
            gestures.append({
                "filename": gesture_file,
                "path": f"faces/{gesture_file}",
                "gesture_type": gesture_type,
                "captured_at": timestamp_str,
                "url": f"/faces/{gesture_file}"
            })
        
        return jsonify({"gestures": gestures})
    except Exception as e:
        logger.error(f"Error listing gestures: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/faces/<filename>', methods=['GET'])
def get_face(filename):
    """Get a specific face image"""
    try:
        # Safety check - prevent directory traversal
        if '..' in filename or '/' in filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        file_path = os.path.join('faces', filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error retrieving face: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    """Main function"""
    global is_running
    
    # Initialize camera
    if not init_camera():
        logger.error("Failed to initialize camera")
        return
    
    # Initialize face detection
    init_face_detection()
    
    # Start buffer thread
    is_running = True
    buffer_thread = threading.Thread(target=buffer_frames, daemon=True)
    buffer_thread.start()
    
    # Make sure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Run Flask app
    logger.info("Starting web server on port 5000")
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        # Clean up
        shutdown_event.set()
        if camera:
            camera.release()
        logger.info("Cleaned up resources")

if __name__ == "__main__":
    main() 