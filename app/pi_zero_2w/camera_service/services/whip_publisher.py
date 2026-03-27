"""
WHIP Publisher — Pi Zero 2W

Pushes a single clean stream to MediaMTX via WHIP.
MediaMTX fans out to all viewers via WHEP — device only encodes once.

Architecture:
  Pi → WHIP → MediaMTX (Oracle VPS) → WHEP → up to N viewers
"""

import asyncio
import logging
import threading
import time
import cv2
import numpy as np
from fractions import Fraction
from typing import Optional

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from ..config.settings import Settings

logger = logging.getLogger(__name__)

WIDTH = 720
HEIGHT = 1280


class CameraVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, buffer_service, fps: int = 15):
        super().__init__()
        self.buffer_service = buffer_service
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0.0
        self.frame_count = 0
        self._placeholder: Optional[np.ndarray] = None

    async def recv(self) -> VideoFrame:
        now = time.time()
        wait = self.frame_interval - (now - self.last_frame_time)
        if wait > 0:
            await asyncio.sleep(wait)

        self.frame_count += 1
        self.last_frame_time = time.time()

        try:
            frame = self.buffer_service.get_latest_frame()
            if frame is None:
                frame = self._get_placeholder()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Frame error: {e}")
            frame_rgb = self._get_placeholder()

        av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        pts, time_base = await self.next_timestamp()
        av_frame.pts = pts
        av_frame.time_base = time_base

        if self.frame_count % (self.fps * 30) == 0:
            logger.info(f"WHIP: {self.frame_count} frames published")

        return av_frame

    def _get_placeholder(self) -> np.ndarray:
        if self._placeholder is None:
            self._placeholder = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
            cv2.putText(
                self._placeholder, "Starting...",
                (WIDTH // 2 - 80, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2,
            )
        return self._placeholder


class WHIPPublisher:
    def __init__(self, stream_name: str = None, fps: int = 15):
        self.mediamtx_url = Settings.MEDIAMTX_URL
        self.stream_name = stream_name  # set after registration
        self.fps = fps
        self.buffer_service = None

        self.running = False
        self.connected = False
        self.pc: Optional[RTCPeerConnection] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.reconnect_delay = 5
        self.reconnect_attempts = 0
        self.max_reconnects = 20

    @property
    def whip_url(self) -> str:
        return f"{self.mediamtx_url}/{self.stream_name}/whip"

    @property
    def whep_url(self) -> str:
        return f"{self.mediamtx_url}/{self.stream_name}/"

    def set_buffer_service(self, buf):
        self.buffer_service = buf

    def set_stream_name(self, name: str):
        self.stream_name = name

    def start(self) -> bool:
        if self.running:
            return True
        if not self.buffer_service:
            logger.error("Buffer service not set")
            return False
        if not self.stream_name:
            logger.error("Stream name (camera PDA) not set")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="WHIPPublisher")
        self.thread.start()
        logger.info(f"WHIP publisher started → {self.whip_url}")
        return True

    def stop(self):
        self.running = False
        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._cleanup(), self.event_loop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _run_loop(self):
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        try:
            self.event_loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"WHIP loop error: {e}")
        finally:
            self.event_loop.close()

    async def _main_loop(self):
        while self.running:
            try:
                await self._connect_and_publish()
            except Exception as e:
                self.connected = False
                logger.error(f"WHIP connection lost: {e}")
                if not self.running:
                    break
                self.reconnect_attempts += 1
                if self.reconnect_attempts > self.max_reconnects:
                    logger.error("Max WHIP reconnects reached")
                    self.running = False
                    break
                logger.info(f"Reconnecting in {self.reconnect_delay}s (attempt {self.reconnect_attempts})")
                await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_publish(self):
        self.pc = RTCPeerConnection()
        track = CameraVideoTrack(self.buffer_service, fps=self.fps)
        self.pc.addTrack(track)

        @self.pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"WHIP state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                self.connected = True
                self.reconnect_attempts = 0
            elif self.pc.connectionState in ("failed", "closed", "disconnected"):
                self.connected = False

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering
        start = time.time()
        while self.pc.iceGatheringState != "complete":
            if time.time() - start > 5:
                break
            await asyncio.sleep(0.1)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.whip_url,
                data=self.pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 201:
                    body = await resp.text()
                    raise Exception(f"WHIP handshake failed {resp.status}: {body}")
                answer_sdp = await resp.text()

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
        logger.info(f"WHIP live → {self.whep_url}")

        while self.running and self.pc.connectionState == "connected":
            await asyncio.sleep(1)

    async def _cleanup(self):
        if self.pc:
            try:
                await self.pc.close()
            except Exception:
                pass
            self.pc = None
        self.connected = False

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "connected": self.connected,
            "stream_name": self.stream_name,
            "whip_url": self.whip_url if self.stream_name else None,
            "whep_url": self.whep_url if self.stream_name else None,
            "reconnect_attempts": self.reconnect_attempts,
        }


_publisher: Optional[WHIPPublisher] = None


def get_whip_publisher() -> WHIPPublisher:
    global _publisher
    if _publisher is None:
        _publisher = WHIPPublisher()
    return _publisher
