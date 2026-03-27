"""
Buffer Service — Pi Zero 2W

Captures frames from Pi Camera Module 3 via Picamera2 into a rolling JPEG buffer.
Provides instant photo access and retroactive video clip extraction.

Portrait 9:16 orientation is set at the sensor/ISP level — no CPU rotation.
"""

import subprocess
import threading
import time
import cv2
import logging
import os
from picamera2 import Picamera2
import libcamera
from threading import Lock
from collections import deque
from typing import Optional, Tuple, List

from ..config.settings import Settings

logger = logging.getLogger(__name__)

# Pi Zero 2W has 512MB RAM — keep buffer lean
# 720x1280 JPEG @ quality 80 ≈ 80-100KB per frame
# 300 frames @ 15fps = 20s buffer ≈ 25-30MB
FRAME_RATE = 15
BUFFER_SECONDS = 20
BUFFER_SIZE = FRAME_RATE * BUFFER_SECONDS  # 300 frames
WIDTH = 720
HEIGHT = 1280
JPEG_QUALITY = 80
TEMP_CRITICAL = 80.0  # °C — shut down capture above this


class BufferService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
        self._buffer_lock = Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame = None  # numpy, for WHIP publisher
        self._latest_timestamp: float = 0

        self._picam2: Optional[Picamera2] = None
        self._running = False
        self._stop_event = threading.Event()
        self._buffer_thread: Optional[threading.Thread] = None

        self._fps_actual = 0.0
        self._frame_counter = 0
        self._fps_window_start = time.time()
        self._fps_window_frames = 0

        logger.info(f"BufferService init — {WIDTH}x{HEIGHT} @ {FRAME_RATE}fps, {BUFFER_SIZE} frame buffer")
        self.start()

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            try:
                if self._picam2:
                    self._picam2.stop()
                    self._picam2.close()
                    self._picam2 = None
                    time.sleep(0.5)

                self._picam2 = Picamera2()

                # Portrait 9:16 via 90° rotation at the ISP level — no CPU overhead
                config = self._picam2.create_video_configuration(
                    main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
                    controls={
                        "FrameRate": FRAME_RATE,
                        "AfMode": 2,       # Continuous autofocus
                        "AfRange": 0,      # Normal range
                        "AfSpeed": 1,      # Normal speed
                    },
                    transform=libcamera.Transform(rotation=90),
                )
                self._picam2.configure(config)
                self._picam2.start()
                time.sleep(1.5)  # Allow AWB and AE to settle

                self._running = True
                self._stop_event.clear()
                self._buffer_thread = threading.Thread(
                    target=self._capture_loop,
                    daemon=True,
                    name="BufferCapture",
                )
                self._buffer_thread.start()

                logger.info("BufferService started")
                return True

            except Exception as e:
                logger.error(f"Failed to start BufferService: {e}")
                self._cleanup_camera()
                return False

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / FRAME_RATE
        last_temp_check = time.time()
        temp_check_interval = 5.0

        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                # Periodic temperature check
                if loop_start - last_temp_check >= temp_check_interval:
                    temp = self._read_temp()
                    last_temp_check = loop_start
                    if temp > TEMP_CRITICAL:
                        logger.error(f"Critical temperature {temp:.1f}°C — stopping capture")
                        self._running = False
                        break

                frame = self._picam2.capture_array()
                timestamp = time.time()

                _, jpeg = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                )
                jpeg_bytes = jpeg.tobytes()

                with self._buffer_lock:
                    self._frame_buffer.append((jpeg_bytes, timestamp))
                    self._latest_jpeg = jpeg_bytes
                    self._latest_frame = frame
                    self._latest_timestamp = timestamp
                    self._frame_counter += 1
                    self._fps_window_frames += 1

                # Update FPS counter every second
                elapsed = timestamp - self._fps_window_start
                if elapsed >= 1.0:
                    self._fps_actual = self._fps_window_frames / elapsed
                    self._fps_window_frames = 0
                    self._fps_window_start = timestamp

                sleep = frame_interval - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

        logger.info("Capture loop stopped")

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Return the latest JPEG frame — instant, no blocking."""
        with self._buffer_lock:
            return self._latest_jpeg

    def get_latest_frame(self):
        """Return the latest raw numpy frame for the WHIP publisher."""
        with self._buffer_lock:
            return self._latest_frame

    def get_video_segment(self, duration_seconds: int = 10) -> dict:
        """
        Extract last N seconds from the rolling buffer and encode to H.264 MOV.
        Tries hardware encoder (h264_v4l2m2m) first, falls back to libx264.
        Returns dict with path, duration, frame_count.
        """
        num_frames = min(duration_seconds * FRAME_RATE, len(self._frame_buffer))
        if num_frames < FRAME_RATE:  # Need at least 1 second
            raise RuntimeError(f"Only {num_frames} frames in buffer, need at least {FRAME_RATE}")

        with self._buffer_lock:
            frames = list(self._frame_buffer)[-num_frames:]

        timestamp = int(time.time())
        os.makedirs(Settings.VIDEOS_DIR, exist_ok=True)
        mjpeg_path = f"/tmp/capture_{timestamp}.mjpeg"
        out_path = os.path.join(Settings.VIDEOS_DIR, f"video_{timestamp}.mov")

        with open(mjpeg_path, "wb") as f:
            for jpeg_bytes, _ in frames:
                f.write(jpeg_bytes)

        encoder = self._detect_hw_encoder()
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "mjpeg",
            "-framerate", str(FRAME_RATE),
            "-i", mjpeg_path,
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            *([] if encoder == "h264_v4l2m2m" else ["-preset", "veryfast", "-b:v", "1M"]),
            "-movflags", "+faststart",
            out_path,
        ]

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode())
        finally:
            try:
                os.remove(mjpeg_path)
            except OSError:
                pass

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("FFmpeg produced empty output")

        actual_duration = round(num_frames / FRAME_RATE, 1)
        logger.info(f"Video segment: {out_path} ({actual_duration}s, encoder={encoder})")
        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "duration": actual_duration,
            "frame_count": num_frames,
        }

    def get_status(self) -> dict:
        with self._buffer_lock:
            buf_len = len(self._frame_buffer)
        return {
            "running": self._running,
            "fps": round(self._fps_actual, 1),
            "buffer_frames": buf_len,
            "buffer_seconds": round(buf_len / FRAME_RATE, 1),
            "resolution": f"{WIDTH}x{HEIGHT}",
            "temperature": self._read_temp(),
        }

    def stop(self) -> None:
        logger.info("Stopping BufferService")
        self._stop_event.set()
        if self._buffer_thread and self._buffer_thread.is_alive():
            self._buffer_thread.join(timeout=3.0)
        self._cleanup_camera()
        self._running = False

    def _cleanup_camera(self) -> None:
        if self._picam2:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

    @staticmethod
    def _detect_hw_encoder() -> str:
        """Return h264_v4l2m2m if VideoCore hardware encoder is available, else libx264."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, timeout=5,
            )
            if b"h264_v4l2m2m" in result.stdout:
                return "h264_v4l2m2m"
        except Exception:
            pass
        return "libx264"

    @staticmethod
    def _read_temp() -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return 0.0


def get_buffer_service() -> BufferService:
    return BufferService()
