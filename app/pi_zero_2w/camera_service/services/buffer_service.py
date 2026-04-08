"""
Buffer Service — Pi Zero 2W

Provides photo and video capture from the H.264 segment buffer written
by StreamManager. No camera access — rpicam-vid owns the camera.

Photos: decode 1 frame on-demand from the latest segment via ffmpeg.
Videos: concatenate segments with -c copy (zero CPU re-encode).
"""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Optional, List

from ..config.settings import Settings

logger = logging.getLogger(__name__)

SEGMENT_DIR = Path("/tmp/mmoment_segments")
WIDTH = 720
HEIGHT = 1280
FRAMERATE = 15
SEGMENT_DURATION = 5


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
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"BufferService init — reading segments from {SEGMENT_DIR}")

    def _get_segments(self) -> List[Path]:
        """Return segment files sorted by modification time (oldest first)."""
        if not SEGMENT_DIR.exists():
            return []
        segments = list(SEGMENT_DIR.glob("seg*.ts"))
        segments.sort(key=lambda p: p.stat().st_mtime)
        return segments

    def _get_latest_segment(self) -> Optional[Path]:
        """Return the most recently written segment."""
        segments = self._get_segments()
        return segments[-1] if segments else None

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Decode 1 frame from the latest segment. Returns JPEG bytes or None."""
        seg = self._get_latest_segment()
        if not seg:
            return None

        # Use a temp file — Pi's ffmpeg mjpeg encoder fails with image2pipe.
        # Skip -sseof which also breaks the mjpeg encoder on this build.
        tmp_path = "/tmp/mmoment_frame.jpg"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(seg),
                    "-frames:v", "1",
                    "-update", "1",
                    "-q:v", "5",
                    tmp_path,
                ],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Frame decode failed: {e}")

        return None

    def get_latest_frame(self):
        """Return latest frame as numpy array (for QR scanner)."""
        import cv2
        import numpy as np

        jpeg = self.get_jpeg_frame()
        if jpeg is None:
            return None

        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def get_video_segment(self, duration_seconds: int = 10) -> dict:
        """
        Extract last N seconds from the rolling segment buffer.
        Concatenates H.264 segments with -c copy (zero re-encode).
        Returns dict with path, filename, duration, frame_count.
        """
        segments = self._get_segments()
        if not segments:
            raise RuntimeError("No segments available")

        # Calculate how many segments we need
        segments_needed = max(1, (duration_seconds + SEGMENT_DURATION - 1) // SEGMENT_DURATION)
        selected = segments[-segments_needed:]

        timestamp = int(time.time())
        os.makedirs(Settings.VIDEOS_DIR, exist_ok=True)
        out_path = os.path.join(Settings.VIDEOS_DIR, f"video_{timestamp}.mov")
        concat_file = f"/tmp/concat_{timestamp}.txt"

        # Write ffmpeg concat file
        with open(concat_file, "w") as f:
            for seg in selected:
                f.write(f"file '{seg}'\n")

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    out_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()}")
        finally:
            try:
                os.remove(concat_file)
            except OSError:
                pass

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("ffmpeg produced empty output")

        actual_duration = len(selected) * SEGMENT_DURATION
        frame_count = actual_duration * FRAMERATE

        logger.info(f"Video segment: {out_path} (~{actual_duration}s from {len(selected)} segments)")
        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "duration": actual_duration,
            "frame_count": frame_count,
        }

    def get_status(self) -> dict:
        segments = self._get_segments()
        return {
            "running": len(segments) > 0,
            "fps": float(FRAMERATE),
            "buffer_frames": len(segments) * SEGMENT_DURATION * FRAMERATE,
            "buffer_seconds": float(len(segments) * SEGMENT_DURATION),
            "resolution": f"{WIDTH}x{HEIGHT}",
            "temperature": self._read_temp(),
        }

    @staticmethod
    def _read_temp() -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return 0.0


def get_buffer_service() -> BufferService:
    return BufferService()
