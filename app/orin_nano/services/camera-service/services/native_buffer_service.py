"""
Native Buffer Service

Gets frames from the native C++ inference server (runs in same container, started by entrypoint.sh).
Python does NOT access the camera directly - native C++ server handles camera capture and inference.

Architecture (all inside container):
  Camera → Native C++ Server → inference → frames+results → Python (this service) → WebRTC
"""

import cv2
import time
import threading
import numpy as np
import logging
import sys
import math
from collections import deque
from typing import Optional, Dict, List, Tuple

# Add native client path
sys.path.insert(0, '/app/native')

from native_client import NativeInferenceClient
from .native_identity_service import get_native_identity_service

logger = logging.getLogger("NativeBufferService")


class NativeBufferService:
    """
    Buffer service that receives frames from the native C++ inference server.
    Provides the same interface as BufferService for compatibility with existing code.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NativeBufferService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._client = None
        self._running = False
        self._buffer_thread = None
        self._stop_event = threading.Event()

        # Frame buffer
        self._buffer_lock = threading.Lock()
        self._buffer_size = 15
        self._frame_buffer = deque(maxlen=self._buffer_size)
        self._latest_frame = None
        self._latest_timestamp = 0
        self._frame_counter = 0

        # Detection results from native server
        self._latest_persons = []
        self._latest_faces = []
        self._latest_timing = {}

        # Settings (will be updated from actual frames)
        self._width = 1280
        self._height = 720
        self._fps = 30
        self._camera_index = -1  # Native mode - no direct camera access

        # Statistics
        self._stats_lock = threading.Lock()
        self._fps_actual = 0
        self._last_fps_calc_time = time.time()
        self._frames_since_last_calc = 0

        # Health
        self._health_status = "Not started"

        # Injected services (for annotation drawing)
        self._gesture_service = None
        self._gpu_face_service = None
        self._pose_service = None
        self._app_manager = None

        # Native identity service (lightweight, no PyTorch needed)
        self._identity_service = get_native_identity_service()

        logger.info("NativeBufferService initialized - will connect to native C++ server")

    def start(self) -> bool:
        """Start the native buffer service by connecting to the C++ server."""
        with self._lock:
            if self._running:
                logger.info("NativeBufferService already running")
                return True

            try:
                # Connect to native server
                logger.info("Connecting to native inference server...")
                self._client = NativeInferenceClient()

                if not self._client.connect():
                    logger.error("Failed to connect to native inference server")
                    logger.error("Native C++ server should be started by entrypoint.sh!")
                    return False

                logger.info("Connected to native inference server")

                # Start buffer thread
                self._stop_event.clear()
                self._running = True
                self._buffer_thread = threading.Thread(
                    target=self._buffer_frames_loop,
                    daemon=True,
                    name="NativeBufferThread"
                )
                self._buffer_thread.start()

                logger.info("NativeBufferService started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start NativeBufferService: {e}")
                self.stop()
                return False

    def _buffer_frames_loop(self) -> None:
        """Continuously fetch frames from native server."""
        logger.info("Native buffer loop started")

        # Rate limiting - cap at 30fps to avoid wasting CPU on duplicate frames
        target_fps = 30
        min_frame_time = 1.0 / target_fps  # ~33ms between frames
        last_frame_time = 0

        while not self._stop_event.is_set() and self._running:
            try:
                # Rate limit to target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

                # Get frame from native server
                result = self._client.get_frame()
                last_frame_time = time.time()

                if result is None:
                    # Connection lost, try to reconnect
                    logger.warning("Lost connection to native server, reconnecting...")
                    if self._client.connect():
                        continue
                    else:
                        logger.error("Failed to reconnect to native server")
                        time.sleep(1)
                        continue

                # Extract frame and results
                frame = result.get('frame')
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Use last_frame_time for timestamp (already computed above)
                current_time = last_frame_time

                # Frame is already rotated on GPU by native server (720x1280 portrait)
                # No CPU rotation needed!

                # Extract detection results from native server
                persons = result.get('persons', [])
                faces = result.get('faces', [])
                timing = result.get('timing', {})

                # Run identity matching OUTSIDE the lock (can be slow due to embedding comparisons)
                # This prevents health check timeouts by not holding _buffer_lock during slow operations
                id_elapsed = 0.0  # Initialize for diagnostic logging
                if self._identity_service:
                    id_start = time.time()
                    persons = self._identity_service.process_native_results(persons, faces, frame)
                    id_elapsed = (time.time() - id_start) * 1000
                    if id_elapsed > 100:  # Log if identity processing takes >100ms
                        logger.warning(f"⚠️ SLOW identity processing: {id_elapsed:.1f}ms")

                # Update buffer - quick assignments
                with self._buffer_lock:
                    self._frame_buffer.append((frame, current_time))
                    self._latest_frame = frame
                    self._latest_timestamp = current_time
                    self._frame_counter += 1
                    self._latest_timing = timing
                    self._latest_persons = persons
                    self._latest_faces = faces
                    # Update dimensions
                    self._height, self._width = frame.shape[:2]
                # Process CV apps if one is active
                if self._app_manager and self._app_manager.active_app:
                    self._process_cv_app(persons, current_time)

                # Calculate FPS
                self._frames_since_last_calc += 1
                if current_time - self._last_fps_calc_time >= 1.0:
                    with self._stats_lock:
                        elapsed = current_time - self._last_fps_calc_time
                        self._fps_actual = self._frames_since_last_calc / elapsed
                        self._frames_since_last_calc = 0
                        self._last_fps_calc_time = current_time

                        if self._frame_counter % 100 == 0:
                            timing = self._latest_timing
                            logger.info(f"Native buffer: {self._fps_actual:.1f} FPS, "
                                       f"inference: {timing.get('total_ms', 0):.1f}ms, "
                                       f"persons: {len(self._latest_persons)}, "
                                       f"faces: {len(self._latest_faces)}")

                self._health_status = "Healthy"

            except Exception as e:
                logger.error(f"Error in native buffer loop: {e}")
                self._health_status = f"Error: {e}"
                time.sleep(0.1)

        logger.info("Native buffer loop stopped")
        self._running = False

    def get_frame(self, processed=False) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame from the buffer."""
        with self._buffer_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._latest_timestamp
            return None, 0

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (no copy for performance)."""
        with self._buffer_lock:
            return self._latest_frame

    def get_processed_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame with annotations drawn."""
        with self._buffer_lock:
            if self._latest_frame is None:
                return None, 0
            frame = self._latest_frame.copy()
            timestamp = self._latest_timestamp
            persons = self._latest_persons.copy()
            faces = self._latest_faces.copy()

        # Draw native detection results
        frame = self._draw_native_annotations(frame, persons, faces)

        return frame, timestamp

    def get_clean_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get raw frame without annotations."""
        return self.get_frame()

    def get_annotated_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame with all annotations."""
        return self.get_processed_frame()

    def _transform_point_90ccw(self, x: float, y: float, orig_width: int = 1280) -> Tuple[int, int]:
        """
        Transform a point from landscape (1280x720) to portrait (720x1280) coordinates.
        90° counterclockwise rotation: (x, y) -> (y, orig_width - 1 - x)
        """
        new_x = int(y)
        new_y = int(orig_width - 1 - x)
        return new_x, new_y

    def _transform_bbox_90ccw(self, x1: float, y1: float, x2: float, y2: float,
                               orig_width: int = 1280) -> Tuple[int, int, int, int]:
        """
        Transform a bounding box from landscape to portrait coordinates.
        For 90° CCW rotation, the corners transform and we need to recalculate min/max.
        """
        # Transform all 4 corners
        corners = [
            self._transform_point_90ccw(x1, y1, orig_width),
            self._transform_point_90ccw(x2, y1, orig_width),
            self._transform_point_90ccw(x1, y2, orig_width),
            self._transform_point_90ccw(x2, y2, orig_width),
        ]
        # Find new bounding box from transformed corners
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _process_cv_app(self, persons: list, timestamp: float) -> None:
        """
        Process frame data through active CV app.
        Transforms keypoints from landscape to portrait coordinates.
        """
        ORIG_WIDTH = 1280

        # Build keypoints list for apps (transformed to portrait coordinates)
        all_keypoints = []
        for person in persons:
            track_id = person.get('track_id')
            raw_kps = person.get('keypoints', [])

            if len(raw_kps) < 17:
                continue

            # Transform keypoints to portrait coordinates
            transformed_kps = []
            for kp in raw_kps:
                if len(kp) >= 3:
                    px, py = self._transform_point_90ccw(kp[0], kp[1], ORIG_WIDTH)
                    transformed_kps.append([float(px), float(py), float(kp[2])])
                else:
                    transformed_kps.append([0.0, 0.0, 0.0])

            all_keypoints.append({
                'track_id': track_id,
                'keypoints': np.array(transformed_kps, dtype=np.float32)
            })

        # Build frame_data for app
        frame_data = {
            'detections': persons,  # Contains wallet_address, track_id, bbox, etc.
            'keypoints': all_keypoints,  # Transformed to portrait coords
            'timestamp': timestamp
        }

        try:
            self._app_manager.process_frame(frame_data)
        except Exception as e:
            logger.error(f"CV app processing error: {e}")

    # COCO skeleton connections (pairs of keypoint indices)
    # 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
    # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
    # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
    # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
    SKELETON = [
        # Head connections disabled - no face annotations
        # (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    # Face keypoint indices to skip (nose, eyes, ears)
    FACE_KEYPOINTS = {0, 1, 2, 3, 4}
    KPS_THRESHOLD = 0.5  # Keypoint confidence threshold

    def _draw_native_annotations(self, frame: np.ndarray, persons: list, faces: list) -> np.ndarray:
        """Draw detection results from native server on frame.

        Note: Detection runs on 1280x720 landscape frame, but we display on 720x1280 portrait.
        All coordinates must be transformed for the 90° CCW rotation.
        Uses skeleton drawing approach from cyrusbehr/YOLOv8-TensorRT-CPP.
        """
        ORIG_WIDTH = 1280
        h, w = frame.shape[:2]

        for person in persons:
            conf = person.get('confidence', 0)
            wallet = person.get('wallet_address')
            track = person.get('track_id')
            # Removed excessive per-frame logging - was causing WebRTC freeze
            if conf < 0.5:
                continue

            # Draw bounding box
            bbox = person.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = self._transform_bbox_90ccw(
                    bbox[0], bbox[1], bbox[2], bbox[3], ORIG_WIDTH
                )

                # Check if person is recognized
                wallet_address = person.get('wallet_address')
                unified_confidence = person.get('unified_confidence')
                confidence_source = person.get('confidence_source')

                if wallet_address:
                    # Use cached display_name from person dict (populated by _enhance_persons_with_identity)
                    # This avoids lock contention - we don't acquire identity_store lock on every frame
                    display_name = person.get('display_name')

                    # Fallback if display_name not cached (shouldn't happen normally)
                    if not display_name:
                        display_name = wallet_address[:8]

                    # Build label with unified confidence score
                    # Format: "Name (0.85)" for combined, "Name (0.85 F)" for face-only, "Name (0.78 B)" for body-only
                    if unified_confidence is not None:
                        if confidence_source:
                            label_text = f"{display_name} ({unified_confidence:.2f} {confidence_source})"
                        else:
                            label_text = f"{display_name} ({unified_confidence:.2f})"
                    else:
                        label_text = display_name

                    # Recognized - draw green box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Draw compact label with black background and green border
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                    # Position label above the box
                    label_x = x1
                    label_y = y1 - 8
                    padding = 3

                    # Label box coordinates
                    box_x1 = label_x - padding
                    box_y1 = label_y - text_h - padding
                    box_x2 = label_x + text_w + padding
                    box_y2 = label_y + baseline + padding

                    # Draw black filled background
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

                    # Draw green border to match person bounding box
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 1)

                    # Draw white text
                    cv2.putText(frame, label_text, (label_x, label_y),
                               font, font_scale, (255, 255, 255), thickness)
                else:
                    # Unknown - draw yellow box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Show label for unrecognized people
                    # Use face_similarity from person dict if available
                    face_sim = person.get('face_similarity')
                    label_text = "Unrecognized"
                    if face_sim is not None:
                        label_text = f"Unrecognized ({face_sim:.2f})"

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                    label_x = x1
                    label_y = y1 - 8
                    padding = 3

                    # Label box coordinates
                    box_x1 = label_x - padding
                    box_y1 = label_y - text_h - padding
                    box_x2 = label_x + text_w + padding
                    box_y2 = label_y + baseline + padding

                    # Draw dark background
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

                    # Draw yellow border to match person bounding box
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 1)

                    cv2.putText(frame, label_text, (label_x, label_y),
                               font, font_scale, (255, 255, 255), thickness)

            # Get and transform keypoints
            keypoints = person.get('keypoints', [])
            if len(keypoints) < 17:
                continue

            # Transform all keypoints first
            kps_transformed = []
            for kp in keypoints:
                if len(kp) >= 3:
                    px, py = self._transform_point_90ccw(kp[0], kp[1], ORIG_WIDTH)
                    kps_transformed.append((px, py, kp[2]))
                else:
                    kps_transformed.append((0, 0, 0))

            # Draw skeleton lines (only if BOTH endpoints are confident)
            for (i, j) in self.SKELETON:
                if i < len(kps_transformed) and j < len(kps_transformed):
                    p1 = kps_transformed[i]
                    p2 = kps_transformed[j]
                    # Only draw if both keypoints are confident
                    if p1[2] > self.KPS_THRESHOLD and p2[2] > self.KPS_THRESHOLD:
                        x1, y1 = p1[0], p1[1]
                        x2, y2 = p2[0], p2[1]
                        # Bounds check
                        if (0 <= x1 < w and 0 <= y1 < h and
                            0 <= x2 < w and 0 <= y2 < h):
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw keypoint circles (only confident ones, skip face keypoints)
            # Hand indices: 9=left_wrist, 10=right_wrist
            # Foot indices: 15=left_ankle, 16=right_ankle
            HAND_KEYPOINTS = {9, 10}
            FOOT_KEYPOINTS = {15, 16}

            for idx, (px, py, pc) in enumerate(kps_transformed):
                if idx in self.FACE_KEYPOINTS:
                    continue  # Skip face keypoints
                if pc > self.KPS_THRESHOLD and 0 <= px < w and 0 <= py < h:
                    if idx in HAND_KEYPOINTS:
                        # Draw hand marker - larger cyan circle with palm indicator
                        cv2.circle(frame, (px, py), 10, (255, 255, 0), 2)  # Cyan outline
                        cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)  # Cyan filled center
                        # Draw palm direction based on elbow position
                        elbow_idx = 7 if idx == 9 else 8  # left_elbow for left_wrist, right_elbow for right_wrist
                        if elbow_idx < len(kps_transformed) and kps_transformed[elbow_idx][2] > self.KPS_THRESHOLD:
                            ex, ey = kps_transformed[elbow_idx][0], kps_transformed[elbow_idx][1]
                            # Direction away from elbow (towards fingertips)
                            dx, dy = px - ex, py - ey
                            length = max(1, (dx*dx + dy*dy) ** 0.5)
                            dx, dy = dx / length, dy / length
                            # Draw small lines for "fingers"
                            finger_len = 12
                            for angle_offset in [-0.4, -0.15, 0.1, 0.35]:
                                fx = int(px + finger_len * (dx * math.cos(angle_offset) - dy * math.sin(angle_offset)))
                                fy = int(py + finger_len * (dx * math.sin(angle_offset) + dy * math.cos(angle_offset)))
                                if 0 <= fx < w and 0 <= fy < h:
                                    cv2.line(frame, (px, py), (fx, fy), (255, 255, 0), 2)
                    elif idx in FOOT_KEYPOINTS:
                        # Draw foot marker - elongated magenta shape
                        cv2.circle(frame, (px, py), 8, (255, 0, 255), 2)  # Magenta outline
                        cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)  # Magenta filled center
                        # Draw foot direction based on knee position
                        knee_idx = 13 if idx == 15 else 14  # left_knee for left_ankle, right_knee for right_ankle
                        if knee_idx < len(kps_transformed) and kps_transformed[knee_idx][2] > self.KPS_THRESHOLD:
                            kx, ky = kps_transformed[knee_idx][0], kps_transformed[knee_idx][1]
                            # Direction away from knee (towards toes)
                            dx, dy = px - kx, py - ky
                            length = max(1, (dx*dx + dy*dy) ** 0.5)
                            dx, dy = dx / length, dy / length
                            # Draw elongated foot shape
                            foot_len = 18
                            toe_x = int(px + foot_len * dx)
                            toe_y = int(py + foot_len * dy)
                            if 0 <= toe_x < w and 0 <= toe_y < h:
                                cv2.line(frame, (px, py), (toe_x, toe_y), (255, 0, 255), 4)
                                cv2.circle(frame, (toe_x, toe_y), 5, (255, 0, 255), -1)
                    else:
                        # Regular joint - small blue circle
                        cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

        # Face detection disabled - was causing purple line artifacts
        # TODO: Fix face coordinate handling if needed later

        return frame

    def get_detection_results(self) -> Dict:
        """Get the latest detection results from native server."""
        with self._buffer_lock:
            return {
                'persons': self._latest_persons.copy(),
                'faces': self._latest_faces.copy(),
                'timing': self._latest_timing.copy(),
            }

    def get_buffer_frames(self, count: int = 30) -> List[Tuple[np.ndarray, float]]:
        """Get recent frames from buffer."""
        with self._buffer_lock:
            count = min(count, 10)
            frames_count = min(count, len(self._frame_buffer))
            recent_frames = list(self._frame_buffer)[-frames_count:]
            return [(frame.copy(), ts) for frame, ts in recent_frames]

    def inject_services(self, gesture_service=None, gpu_face_service=None,
                       pose_service=None, app_manager=None):
        """Inject services (for compatibility, but native server handles detection)."""
        self._gesture_service = gesture_service
        self._gpu_face_service = gpu_face_service
        self._pose_service = pose_service
        self._app_manager = app_manager
        logger.info("Services injected (note: native server handles detection)")

    def get_status(self) -> Dict:
        """Get service status."""
        with self._stats_lock:
            fps = self._fps_actual

        with self._buffer_lock:
            buffer_size = len(self._frame_buffer)
            buffer_capacity = self._frame_buffer.maxlen
            frame_count = self._frame_counter
            timing = self._latest_timing.copy()

        return {
            "running": self._running,
            "health": self._health_status,
            "fps": round(fps, 2),
            "buffer_size": buffer_size,
            "buffer_capacity": buffer_capacity,
            "frame_count": frame_count,
            "resolution": f"{self._width}x{self._height}",
            "mode": "native",
            "inference_ms": timing.get('total_ms', 0),
            "persons_detected": len(self._latest_persons),
            "faces_detected": len(self._latest_faces),
        }

    def stop(self) -> None:
        """Stop the service."""
        logger.info("Stopping NativeBufferService")
        self._stop_event.set()

        if self._buffer_thread and self._buffer_thread.is_alive():
            self._buffer_thread.join(timeout=2.0)

        if self._client:
            self._client.disconnect()
            self._client = None

        self._running = False
        logger.info("NativeBufferService stopped")


# Global accessor
_native_buffer_service = None


def get_native_buffer_service() -> NativeBufferService:
    """Get the singleton NativeBufferService instance."""
    global _native_buffer_service
    if _native_buffer_service is None:
        _native_buffer_service = NativeBufferService()
    return _native_buffer_service
