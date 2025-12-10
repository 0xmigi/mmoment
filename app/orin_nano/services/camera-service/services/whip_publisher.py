"""
WHIP Publisher Service for Remote WebRTC Streaming

Publishes camera frames to MediaMTX server via WHIP protocol.
This enables remote viewers to watch via WHEP without P2P connection issues.

Architecture:
- Jetson camera → WHIP → MediaMTX (Oracle VPS) → WHEP → Remote viewers
- Local viewers can still use direct P2P WebRTC for lower latency
"""

import asyncio
import logging
import threading
import time
import os
import cv2
import numpy as np
from typing import Optional, Callable
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fractions import Fraction

logger = logging.getLogger("WHIPPublisher")


class BaseVideoTrack(VideoStreamTrack):
    """
    Base video track class with shared functionality for WHIP publishing.
    """
    kind = "video"

    def __init__(self, buffer_service, fps: int = 30, track_name: str = "base"):
        super().__init__()
        self.buffer_service = buffer_service
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self.frame_count = 0
        self.time_base = Fraction(1, 90000)  # Standard 90kHz RTP clock
        self.track_name = track_name

    def _get_frame_data(self):
        """Override in subclass to get the appropriate frame type"""
        raise NotImplementedError

    async def recv(self) -> VideoFrame:
        """Get the next video frame from buffer service"""
        # Maintain frame rate
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_interval:
            await asyncio.sleep(self.frame_interval - elapsed)

        self.frame_count += 1

        try:
            # Get frame from buffer service (method determined by subclass)
            frame_data = self._get_frame_data()

            # Handle tuple return format
            if isinstance(frame_data, tuple) and len(frame_data) > 0:
                frame = frame_data[0]
            else:
                frame = frame_data

            # Validate frame
            if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
                frame = self._create_placeholder_frame()

            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = self._create_placeholder_frame()

            # Create VideoFrame
            av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            pts, time_base = await self.next_timestamp()
            av_frame.pts = pts
            av_frame.time_base = time_base

            self.last_frame_time = time.time()

            # Log periodically
            if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
                logger.info(f"📡 WHIP [{self.track_name}]: Published {self.frame_count} frames")

            return av_frame

        except Exception as e:
            logger.error(f"Error getting frame [{self.track_name}]: {e}")
            return self._create_error_frame()

    def _create_placeholder_frame(self) -> np.ndarray:
        """Create a placeholder frame when camera isn't ready"""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 40  # Dark gray
        cv2.putText(frame, "Camera Starting...", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return frame

    def _create_error_frame(self) -> VideoFrame:
        """Create an error frame"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 2] = 100  # Red tint
        cv2.putText(frame, "Stream Error", (450, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        av_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame.pts = int(time.time() * 90000)
        av_frame.time_base = self.time_base
        return av_frame


class CleanVideoTrack(BaseVideoTrack):
    """
    Video track that streams CLEAN frames (no CV annotations).
    Used for the default stream without overlays.
    """
    def __init__(self, buffer_service, fps: int = 30):
        super().__init__(buffer_service, fps, track_name="clean")

    def _get_frame_data(self):
        """Get clean frame without annotations"""
        return self.buffer_service.get_clean_frame()


class AnnotatedVideoTrack(BaseVideoTrack):
    """
    Video track that streams ANNOTATED frames (with CV overlays always on).
    Used for the annotated stream with face boxes, skeletons, etc.
    """
    def __init__(self, buffer_service, fps: int = 30):
        super().__init__(buffer_service, fps, track_name="annotated")

    def _get_frame_data(self):
        """Get annotated frame with CV overlays"""
        return self.buffer_service.get_annotated_frame()


# Legacy alias for backwards compatibility
class CameraVideoTrack(BaseVideoTrack):
    """
    Legacy video track - now defaults to annotated frames for backwards compatibility.
    Use CleanVideoTrack or AnnotatedVideoTrack directly instead.
    """
    def __init__(self, buffer_service, fps: int = 30):
        super().__init__(buffer_service, fps, track_name="legacy")

    def _get_frame_data(self):
        """Get processed frame (toggle-based, for backwards compatibility)"""
        return self.buffer_service.get_processed_frame()


class WHIPPublisher:
    """
    Publishes camera stream to MediaMTX via WHIP protocol.

    Usage:
        publisher = WHIPPublisher(
            mediamtx_url="http://129.80.99.75:8889",
            stream_name="camera-abc123"
        )
        publisher.set_buffer_service(buffer_service)
        publisher.start()
    """

    def __init__(
        self,
        mediamtx_url: str = None,
        stream_name: str = None,
        fps: int = 30,
        video_track_class: type = None
    ):
        # MediaMTX configuration
        self.mediamtx_url = mediamtx_url or os.environ.get(
            'MEDIAMTX_URL', 'http://129.80.99.75:8889'
        )

        # Stream name - use camera PDA for unique identification
        self.stream_name = stream_name or os.environ.get(
            'CAMERA_PDA', 'jetson-camera'
        )

        self.fps = fps
        self.buffer_service = None

        # Video track class - default to CleanVideoTrack for clean stream
        self.video_track_class = video_track_class or CleanVideoTrack

        # Connection state
        self.running = False
        self.connected = False
        self.pc: Optional[RTCPeerConnection] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        # Reconnection settings
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        self.reconnect_attempts = 0

        track_type = "annotated" if video_track_class == AnnotatedVideoTrack else "clean"
        logger.info(f"WHIPPublisher initialized [{track_type}]: {self.whip_url}")

    @property
    def whip_url(self) -> str:
        """Get the WHIP endpoint URL"""
        return f"{self.mediamtx_url}/{self.stream_name}/whip"

    @property
    def whep_url(self) -> str:
        """Get the WHEP playback URL for viewers"""
        return f"{self.mediamtx_url}/{self.stream_name}/"

    def set_buffer_service(self, buffer_service):
        """Set the buffer service instance"""
        self.buffer_service = buffer_service
        logger.info("Buffer service set for WHIP publisher")

    def start(self) -> bool:
        """Start the WHIP publisher"""
        if self.running:
            logger.warning("WHIP publisher already running")
            return True

        if not self.buffer_service:
            logger.error("Buffer service not set - cannot start WHIP publisher")
            return False

        logger.info(f"🚀 Starting WHIP publisher to {self.whip_url}")

        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

        return True

    def stop(self):
        """Stop the WHIP publisher"""
        logger.info("Stopping WHIP publisher...")
        self.running = False

        if self.event_loop and not self.event_loop.is_closed():
            # Schedule cleanup in the event loop
            asyncio.run_coroutine_threadsafe(self._cleanup(), self.event_loop)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        logger.info("WHIP publisher stopped")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        try:
            self.event_loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"WHIP publisher error: {e}")
        finally:
            self.event_loop.close()

    async def _main_loop(self):
        """Main publishing loop with auto-reconnect"""
        while self.running:
            try:
                await self._connect_and_publish()
            except Exception as e:
                logger.error(f"WHIP connection error: {e}")
                self.connected = False

                if self.running:
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts <= self.max_reconnect_attempts:
                        logger.info(
                            f"🔄 Reconnecting in {self.reconnect_delay}s "
                            f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
                        )
                        await asyncio.sleep(self.reconnect_delay)
                    else:
                        logger.error("Max reconnect attempts reached, stopping")
                        self.running = False

    async def _connect_and_publish(self):
        """Establish WHIP connection and publish stream"""
        logger.info(f"📡 Connecting to MediaMTX: {self.whip_url}")

        # Create peer connection
        self.pc = RTCPeerConnection()

        # Add video track (using configured track class - CleanVideoTrack or AnnotatedVideoTrack)
        video_track = self.video_track_class(self.buffer_service, fps=self.fps)
        self.pc.addTrack(video_track)

        # Set up connection state monitoring
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"🔗 WHIP connection state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                self.connected = True
                self.reconnect_attempts = 0  # Reset on successful connection
                logger.info(f"✅ WHIP stream live at: {self.whep_url}")
            elif self.pc.connectionState in ["failed", "closed", "disconnected"]:
                self.connected = False

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            logger.debug(f"🧊 ICE state: {self.pc.iceConnectionState}")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering
        logger.info("🧊 Gathering ICE candidates...")
        timeout = 5.0
        start = time.time()
        while self.pc.iceGatheringState != "complete":
            if time.time() - start > timeout:
                logger.warning("ICE gathering timeout, proceeding anyway")
                break
            await asyncio.sleep(0.1)

        logger.info(f"🧊 ICE gathering complete ({time.time() - start:.1f}s)")

        # Send offer to MediaMTX via WHIP
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.whip_url,
                data=self.pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"}
            ) as resp:
                if resp.status == 201:
                    answer_sdp = await resp.text()

                    # Set remote description
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type="answer")
                    )

                    logger.info("✅ WHIP handshake complete!")
                    logger.info(f"📺 View stream at: {self.whep_url}")

                    # Wait for connection to be established
                    logger.info("⏳ Waiting for WebRTC connection to establish...")
                    wait_start = time.time()
                    while self.running and self.pc.connectionState in ["new", "connecting"]:
                        if time.time() - wait_start > 10:  # 10 second timeout
                            logger.warning("Connection establishment timeout")
                            break
                        await asyncio.sleep(0.1)

                    if self.pc.connectionState == "connected":
                        logger.info("🟢 WHIP stream connected and publishing!")

                    # Keep connection alive
                    while self.running and self.pc.connectionState == "connected":
                        await asyncio.sleep(1)

                else:
                    error_text = await resp.text()
                    raise Exception(f"WHIP handshake failed: {resp.status} - {error_text}")

    async def _cleanup(self):
        """Clean up resources"""
        if self.pc:
            try:
                await self.pc.close()
            except Exception as e:
                logger.warning(f"Error closing peer connection: {e}")
            self.pc = None
        self.connected = False

    def get_status(self) -> dict:
        """Get publisher status"""
        return {
            "running": self.running,
            "connected": self.connected,
            "mediamtx_url": self.mediamtx_url,
            "stream_name": self.stream_name,
            "whip_url": self.whip_url,
            "whep_url": self.whep_url,
            "reconnect_attempts": self.reconnect_attempts
        }


# Singleton instance
_whip_publisher_instance: Optional[WHIPPublisher] = None


def get_whip_publisher() -> WHIPPublisher:
    """Get the singleton WHIP publisher instance"""
    global _whip_publisher_instance
    if _whip_publisher_instance is None:
        _whip_publisher_instance = WHIPPublisher()
    return _whip_publisher_instance


def reset_whip_publisher():
    """Reset the WHIP publisher instance"""
    global _whip_publisher_instance
    if _whip_publisher_instance:
        _whip_publisher_instance.stop()
    _whip_publisher_instance = None
