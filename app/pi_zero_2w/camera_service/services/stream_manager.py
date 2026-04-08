"""
Stream Manager — Pi Zero 2W

Manages hardware-accelerated H.264 streaming via rpicam-vid + ffmpeg.
rpicam-vid uses the VideoCore GPU for H.264 encoding at near-zero CPU cost.
ffmpeg pushes the stream via RTSP to MediaMTX (zero transcode, just remux).
A second ffmpeg writes rolling H.264 segments for retroactive video clips.

Architecture:
  rpicam-vid (HW H.264) → tee
      ├── ffmpeg → RTSP push to MediaMTX
      └── ffmpeg → rolling 5s segment files
"""

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from ..config.settings import Settings

logger = logging.getLogger(__name__)

SEGMENT_DIR = Path("/tmp/mmoment_segments")
FIFO_STREAM = Path("/tmp/mmoment_fifo_stream")
FIFO_SEGMENTS = Path("/tmp/mmoment_fifo_segments")
SEGMENT_DURATION = 5
SEGMENT_WRAP = 6  # Keep last 30s (6 × 5s)

WIDTH = 720
HEIGHT = 1280
FRAMERATE = 15
BITRATE = 1500000


class StreamManager:
    _instance = None
    _lock = threading.Lock()

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

        self.stream_name: Optional[str] = None
        self.mediamtx_url = Settings.MEDIAMTX_URL
        self._running = False
        self._streaming = False  # True when RTSP push is active

        self._rpicam_proc: Optional[subprocess.Popen] = None
        self._tee_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_rtsp_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_segment_proc: Optional[subprocess.Popen] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_count = 0
        self._max_restarts = 20

        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

    def set_stream_name(self, name: str):
        self.stream_name = name

    @property
    def rtsp_url(self) -> Optional[str]:
        if not self.stream_name:
            return None
        # MediaMTX RTSP port is typically 8554
        base = self.mediamtx_url.replace(":8889", ":8554")
        return f"{base}/{self.stream_name}"

    @property
    def whep_url(self) -> Optional[str]:
        if not self.stream_name:
            return None
        return f"{self.mediamtx_url}/{self.stream_name}/"

    def start(self, stream_name: str = None) -> bool:
        if stream_name:
            self.stream_name = stream_name

        if self._running:
            logger.info("StreamManager already running")
            return True

        try:
            self._setup_fifos()
            self._start_pipeline()
            self._running = True
            self._restart_count = 0
            self._stop_event.clear()

            self._watchdog_thread = threading.Thread(
                target=self._watchdog, daemon=True, name="StreamWatchdog"
            )
            self._watchdog_thread.start()

            logger.info(f"StreamManager started (stream: {self.stream_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to start StreamManager: {e}")
            self.stop()
            return False

    def stop(self):
        self._running = False
        self._streaming = False
        self._stop_event.set()

        for proc_name in ("_ffmpeg_rtsp_proc", "_ffmpeg_segment_proc", "_tee_proc", "_rpicam_proc"):
            proc = getattr(self, proc_name, None)
            if proc and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            setattr(self, proc_name, None)

        self._cleanup_fifos()
        logger.info("StreamManager stopped")

    def _setup_fifos(self):
        self._cleanup_fifos()
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        # Only create FIFOs when we need tee (i.e., streaming + segments)
        if self.stream_name:
            for fifo in (FIFO_STREAM, FIFO_SEGMENTS):
                os.mkfifo(str(fifo))

    def _cleanup_fifos(self):
        for fifo in (FIFO_STREAM, FIFO_SEGMENTS):
            try:
                fifo.unlink(missing_ok=True)
            except Exception:
                pass

    def _start_pipeline(self):
        # 1. rpicam-vid: HW H.264 encode → stdout
        # rpicam-vid at 720x1280 portrait — the ISP handles the crop.
        # No --rotation needed; the sensor outputs the requested dimensions.
        rpicam_cmd = [
            "rpicam-vid", "-t", "0",
            "--width", str(WIDTH),
            "--height", str(HEIGHT),
            "--framerate", str(FRAMERATE),
            "--codec", "h264",
            "--profile", "baseline",
            "--bitrate", str(BITRATE),
            "--inline",
            "-n",  # No preview
            "-o", "-",
        ]
        self._rpicam_proc = subprocess.Popen(
            rpicam_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"rpicam-vid started (PID {self._rpicam_proc.pid})")

        segment_pattern = str(SEGMENT_DIR / "seg%03d.ts")
        segment_cmd = [
            "ffmpeg", "-y",
            "-fflags", "nobuffer",
            "-f", "h264",
        ]

        if self.stream_name:
            # Full pipeline: tee to RTSP + segments via FIFOs
            # Open FIFOs in background threads to avoid blocking
            self._tee_proc = subprocess.Popen(
                ["tee", str(FIFO_STREAM)],
                stdin=self._rpicam_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._rpicam_proc.stdout.close()
            logger.info(f"tee started (PID {self._tee_proc.pid})")

            # Segment writer reads from tee's stdout
            self._ffmpeg_segment_proc = subprocess.Popen(
                segment_cmd + [
                    "-i", "pipe:0",
                    "-c:v", "copy",
                    "-f", "segment",
                    "-segment_time", str(SEGMENT_DURATION),
                    "-segment_wrap", str(SEGMENT_WRAP),
                    "-reset_timestamps", "1",
                    segment_pattern,
                ],
                stdin=self._tee_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._tee_proc.stdout.close()
            logger.info(f"ffmpeg segment writer started (PID {self._ffmpeg_segment_proc.pid})")

            # RTSP push reads from FIFO
            self._start_rtsp_push()
        else:
            # Segments only: pipe rpicam-vid directly to ffmpeg segment writer
            self._ffmpeg_segment_proc = subprocess.Popen(
                segment_cmd + [
                    "-i", "pipe:0",
                    "-c:v", "copy",
                    "-f", "segment",
                    "-segment_time", str(SEGMENT_DURATION),
                    "-segment_wrap", str(SEGMENT_WRAP),
                    "-reset_timestamps", "1",
                    segment_pattern,
                ],
                stdin=self._rpicam_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._rpicam_proc.stdout.close()
            logger.info(f"ffmpeg segment writer started — segments only (PID {self._ffmpeg_segment_proc.pid})")

    def _start_rtsp_push(self):
        if not self.stream_name:
            return

        rtsp_cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-f", "h264",
            "-i", str(FIFO_STREAM),
            "-c:v", "copy",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]
        self._ffmpeg_rtsp_proc = subprocess.Popen(
            rtsp_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._streaming = True
        logger.info(f"ffmpeg RTSP push started → {self.rtsp_url} (PID {self._ffmpeg_rtsp_proc.pid})")

    def _watchdog(self):
        # Give pipeline time to initialize before first check
        time.sleep(10)

        while not self._stop_event.is_set():
            time.sleep(5)

            if not self._running:
                break

            # Check if core processes are alive
            rpicam_alive = self._rpicam_proc and self._rpicam_proc.poll() is None
            # tee only runs when streaming (with stream_name)
            tee_alive = self._tee_proc is None or self._tee_proc.poll() is None
            segment_alive = self._ffmpeg_segment_proc and self._ffmpeg_segment_proc.poll() is None

            if not (rpicam_alive and tee_alive and segment_alive):
                self._restart_count += 1
                if self._restart_count > self._max_restarts:
                    logger.error("Max restarts reached, giving up")
                    self._running = False
                    break

                logger.warning(f"Pipeline died, restarting (attempt {self._restart_count})")
                self.stop()
                time.sleep(2)
                self._running = True
                self._stop_event.clear()
                try:
                    self._setup_fifos()
                    self._start_pipeline()
                except Exception as e:
                    logger.error(f"Restart failed: {e}")
                    time.sleep(5)

            # Check RTSP push separately (non-critical)
            if self._streaming and self._ffmpeg_rtsp_proc and self._ffmpeg_rtsp_proc.poll() is not None:
                logger.warning("RTSP push died, restarting")
                try:
                    self._start_rtsp_push()
                except Exception as e:
                    logger.error(f"RTSP restart failed: {e}")

    def is_streaming(self) -> bool:
        return self._streaming and self._ffmpeg_rtsp_proc is not None and self._ffmpeg_rtsp_proc.poll() is None

    def get_segment_dir(self) -> Path:
        return SEGMENT_DIR

    def get_status(self) -> dict:
        segments = sorted(SEGMENT_DIR.glob("seg*.ts")) if SEGMENT_DIR.exists() else []
        return {
            "running": self._running,
            "streaming": self.is_streaming(),
            "stream_name": self.stream_name,
            "rtsp_url": self.rtsp_url,
            "whep_url": self.whep_url,
            "segment_count": len(segments),
            "buffer_seconds": len(segments) * SEGMENT_DURATION,
            "restart_count": self._restart_count,
        }


_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    global _manager
    if _manager is None:
        _manager = StreamManager()
    return _manager
