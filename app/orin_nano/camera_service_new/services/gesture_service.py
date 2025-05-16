"""
Gesture Recognition Service

Provides gesture detection and recognition functionality.
Works with the buffer service to process frames without modifying the source.
"""

import cv2
import logging
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GestureService")

# Check for mediapipe availability
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe hands module loaded successfully")
except ImportError:
    logger.warning("MediaPipe not available. Install with: pip install mediapipe")

class GestureType(Enum):
    """Enum for supported gesture types"""
    UNKNOWN = "unknown"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    PALM = "palm"
    NONE = "none"

class GestureService:
    """
    Service for gesture detection and recognition.
    This is a processor that consumes frames from the buffer service.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GestureService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        self._processing_enabled = False
        self._visualization_enabled = False
        self._processing_thread = None
        self._stop_event = threading.Event()
        
        # Gesture detection settings
        self._detection_interval = 0.1  # Seconds between detection runs
        self._last_detection_time = 0
        self._confidence_threshold = 0.7
        
        # MediaPipe hands module
        self._hands = None
        if MEDIAPIPE_AVAILABLE:
            self._hands = mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Gesture detection results
        self._results_lock = threading.Lock()
        self._current_gesture = GestureType.NONE
        self._gesture_confidence = 0.0
        self._hand_landmarks = None
        
        logger.info("GestureService initialized")
        
    def start(self, buffer_service) -> bool:
        """
        Start the gesture processing thread.
        """
        with self._lock:
            if self._processing_enabled:
                logger.info("GestureService already running")
                return True
                
            if not MEDIAPIPE_AVAILABLE:
                logger.warning("MediaPipe not available, limited functionality")
                return False
            
            # Store reference to buffer service
            self._buffer_service = buffer_service
            
            # Reset stop event
            self._stop_event.clear()
            
            # Start processing thread
            self._processing_enabled = True
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="GestureProcessingThread"
            )
            self._processing_thread.start()
            
            logger.info("GestureService started")
            return True
    
    def stop(self) -> None:
        """
        Stop the gesture processing thread.
        """
        logger.info("Stopping GestureService")
        
        # Signal the processing thread to stop
        self._stop_event.set()
        
        # Wait for the processing thread to stop
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        # Clean up MediaPipe resources
        if self._hands:
            self._hands.close()
        
        with self._lock:
            self._processing_enabled = False
            
        logger.info("GestureService stopped")
    
    def _processing_loop(self) -> None:
        """
        Main processing loop that runs gesture detection.
        """
        logger.info("Gesture processing loop started")
        
        try:
            while not self._stop_event.is_set():
                # Get the latest frame from the buffer
                frame, timestamp = self._buffer_service.get_frame()
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # Run gesture detection at regular intervals
                if current_time - self._last_detection_time >= self._detection_interval:
                    self._detect_gestures(frame)
                    self._last_detection_time = current_time
                
                # Sleep to avoid consuming too much CPU
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in gesture processing loop: {e}")
        finally:
            logger.info("Gesture processing loop stopped")
            with self._lock:
                self._processing_enabled = False
    
    def _detect_gestures(self, frame: np.ndarray) -> None:
        """
        Detect gestures in the given frame using MediaPipe.
        Updates self._current_gesture with the results.
        """
        if not MEDIAPIPE_AVAILABLE or self._hands is None:
            return
            
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self._hands.process(rgb_frame)
            
            # Default to no gesture
            gesture = GestureType.NONE
            confidence = 0.0
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if this is left or right hand
                    handedness = results.multi_handedness[idx].classification[0].label if results.multi_handedness else "Unknown"
                    
                    # Analyze gesture from landmarks
                    detected_gesture, detected_confidence = self._analyze_gesture(hand_landmarks, handedness)
                    
                    # Only update if confidence is higher than previous gestures
                    if detected_confidence > confidence:
                        gesture = detected_gesture
                        confidence = detected_confidence
            
            # Update gesture info
            with self._results_lock:
                self._current_gesture = gesture
                self._gesture_confidence = confidence
                self._hand_landmarks = results.multi_hand_landmarks
                
        except Exception as e:
            logger.error(f"Error detecting gestures: {e}")
            with self._results_lock:
                self._current_gesture = GestureType.NONE
                self._gesture_confidence = 0.0
                self._hand_landmarks = None
    
    def _analyze_gesture(self, landmarks, handedness: str) -> Tuple[GestureType, float]:
        """
        Analyze hand landmarks to detect specific gestures.
        Returns tuple of (gesture_type, confidence).
        """
        try:
            # Extract landmark points to numpy array for easier calculations
            points = []
            for landmark in landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])
            points = np.array(points)
            
            # Check for thumbs up gesture
            if self._is_thumbs_up(points, handedness):
                return GestureType.THUMBS_UP, 0.85
                
            # Check for thumbs down gesture
            if self._is_thumbs_down(points, handedness):
                return GestureType.THUMBS_DOWN, 0.85
                
            # Check for peace sign
            if self._is_peace_sign(points):
                return GestureType.PEACE, 0.9
                
            # Check for palm (open hand)
            if self._is_palm(points):
                return GestureType.PALM, 0.8
                
            # No recognized gesture
            return GestureType.UNKNOWN, 0.3
            
        except Exception as e:
            logger.error(f"Error analyzing gesture: {e}")
            return GestureType.UNKNOWN, 0.0
    
    def _is_thumbs_up(self, points, handedness: str) -> bool:
        """
        Check if the hand position represents a thumbs up gesture.
        
        Requires:
        - Thumb is pointing up (y-coord of tip is less than base)
        - Other fingers are curled (tips are close to palm)
        """
        try:
            # MediaPipe hand landmark indices:
            # - Thumb: 1 (base), 2, 3, 4 (tip)
            # - Index: 5 (base), 6, 7, 8 (tip)
            # - Middle: 9 (base), 10, 11, 12 (tip)
            # - Ring: 13 (base), 14, 15, 16 (tip)
            # - Pinky: 17 (base), 18, 19, 20 (tip)
            
            # Check thumb orientation (vertical)
            thumb_base = points[1]
            thumb_tip = points[4]
            
            # Thumb should be extended upward (lower y value is higher up)
            thumb_is_up = thumb_tip[1] < thumb_base[1]
            
            # Check if other fingers are curled (tips close to palm)
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]
            
            palm = points[0]  # Wrist point as reference
            
            # Calculate average distance of finger tips to palm
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            avg_distance = np.mean([np.linalg.norm(tip - palm) for tip in finger_tips])
            
            # Calculate distance of thumb tip to palm for comparison
            thumb_distance = np.linalg.norm(thumb_tip - palm)
            
            # For thumbs up, other fingers should be curled (closer to palm)
            fingers_curled = avg_distance < thumb_distance * 0.7
            
            return thumb_is_up and fingers_curled
            
        except Exception as e:
            logger.error(f"Error checking thumbs up: {e}")
            return False
    
    def _is_thumbs_down(self, points, handedness: str) -> bool:
        """
        Check if the hand position represents a thumbs down gesture.
        
        Requires:
        - Thumb is pointing down (y-coord of tip is greater than base)
        - Other fingers are curled (tips are close to palm)
        """
        try:
            # Similar to thumbs up but with inverted vertical check
            thumb_base = points[1]
            thumb_tip = points[4]
            
            # Thumb should be extended downward (higher y value is lower down)
            thumb_is_down = thumb_tip[1] > thumb_base[1]
            
            # Check if other fingers are curled (tips close to palm)
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]
            
            palm = points[0]  # Wrist point as reference
            
            # Calculate average distance of finger tips to palm
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            avg_distance = np.mean([np.linalg.norm(tip - palm) for tip in finger_tips])
            
            # Calculate distance of thumb tip to palm for comparison
            thumb_distance = np.linalg.norm(thumb_tip - palm)
            
            # For thumbs down, other fingers should be curled (closer to palm)
            fingers_curled = avg_distance < thumb_distance * 0.7
            
            return thumb_is_down and fingers_curled
            
        except Exception as e:
            logger.error(f"Error checking thumbs down: {e}")
            return False
    
    def _is_peace_sign(self, points) -> bool:
        """
        Check if the hand position represents a peace sign.
        
        Requires:
        - Index and middle fingers extended (tips far from palm)
        - Other fingers curled (tips close to palm)
        """
        try:
            # Get finger tips
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]
            
            # Get finger bases for reference
            index_base = points[5]
            middle_base = points[9]
            
            palm = points[0]  # Wrist point as reference
            
            # Check if index and middle are extended
            index_extended = np.linalg.norm(index_tip - palm) > np.linalg.norm(index_base - palm) * 1.5
            middle_extended = np.linalg.norm(middle_tip - palm) > np.linalg.norm(middle_base - palm) * 1.5
            
            # Check if ring and pinky are curled
            ring_base = points[13]
            pinky_base = points[17]
            
            ring_curled = np.linalg.norm(ring_tip - palm) < np.linalg.norm(ring_base - palm) * 1.25
            pinky_curled = np.linalg.norm(pinky_tip - palm) < np.linalg.norm(pinky_base - palm) * 1.25
            
            # Index and middle should be apart (V shape)
            fingers_apart = np.linalg.norm(index_tip - middle_tip) > np.linalg.norm(index_base - middle_base) * 1.2
            
            return index_extended and middle_extended and ring_curled and pinky_curled and fingers_apart
            
        except Exception as e:
            logger.error(f"Error checking peace sign: {e}")
            return False
    
    def _is_palm(self, points) -> bool:
        """
        Check if the hand position represents an open palm.
        
        Requires:
        - All fingers extended (tips far from palm)
        - Fingers spread apart
        """
        try:
            # Get finger tips
            thumb_tip = points[4]
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]
            
            # Get finger bases for reference
            thumb_base = points[1]
            index_base = points[5]
            middle_base = points[9]
            ring_base = points[13]
            pinky_base = points[17]
            
            palm = points[0]  # Wrist point as reference
            
            # Check if all fingers are extended
            thumb_extended = np.linalg.norm(thumb_tip - palm) > np.linalg.norm(thumb_base - palm) * 1.2
            index_extended = np.linalg.norm(index_tip - palm) > np.linalg.norm(index_base - palm) * 1.5
            middle_extended = np.linalg.norm(middle_tip - palm) > np.linalg.norm(middle_base - palm) * 1.5
            ring_extended = np.linalg.norm(ring_tip - palm) > np.linalg.norm(ring_base - palm) * 1.5
            pinky_extended = np.linalg.norm(pinky_tip - palm) > np.linalg.norm(pinky_base - palm) * 1.3
            
            all_extended = thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended
            
            # Check if fingers are spread
            spread_distances = [
                np.linalg.norm(thumb_tip - index_tip),
                np.linalg.norm(index_tip - middle_tip),
                np.linalg.norm(middle_tip - ring_tip),
                np.linalg.norm(ring_tip - pinky_tip)
            ]
            
            avg_spread = np.mean(spread_distances)
            base_width = np.linalg.norm(index_base - pinky_base)
            
            fingers_spread = avg_spread > base_width * 0.4
            
            return all_extended and fingers_spread
            
        except Exception as e:
            logger.error(f"Error checking palm: {e}")
            return False
    
    def get_processed_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gesture visualization to a copy of the frame.
        This does NOT modify the original frame from the buffer.
        """
        if frame is None or not self._visualization_enabled:
            return frame
            
        # Make a copy to avoid modifying the original
        output = frame.copy()
        
        with self._results_lock:
            if self._hand_landmarks:
                for idx, hand_landmarks in enumerate(self._hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        output,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                # Draw gesture label if detected
                if self._current_gesture != GestureType.NONE and self._gesture_confidence > self._confidence_threshold:
                    gesture_name = self._current_gesture.value
                    confidence = self._gesture_confidence
                    
                    label = f"{gesture_name.upper()} ({confidence:.2f})"
                    
                    # Draw label background
                    cv2.rectangle(output, (10, 10), (300, 60), (0, 0, 0), -1)
                    
                    # Draw label text
                    cv2.putText(
                        output,
                        label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
        
        return output
    
    def get_current_gesture(self) -> Dict:
        """
        Get the current detected gesture.
        """
        with self._results_lock:
            return {
                "gesture": self._current_gesture.value,
                "confidence": round(self._gesture_confidence, 2),
                "timestamp": int(time.time() * 1000)
            }
    
    def enable_visualization(self, enabled: bool) -> None:
        """
        Enable or disable gesture visualization.
        """
        self._visualization_enabled = enabled
        logger.info(f"Gesture visualization {'enabled' if enabled else 'disabled'}")

# Global function to get the gesture service instance
def get_gesture_service() -> GestureService:
    """
    Get the singleton GestureService instance.
    """
    return GestureService() 