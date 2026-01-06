"""
native_client.py - Client for native C++ inference server

Python client that connects to the native C++ TensorRT server via Unix socket.
Both run inside the same container - C++ server started by entrypoint.sh.

Usage:
    from native_client import NativeInferenceClient

    client = NativeInferenceClient()
    if client.connect():
        result = client.get_frame()
        print(f"Detected {len(result['persons'])} persons")
"""

import socket
import struct
import json
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default socket path (must match server)
DEFAULT_SOCKET_PATH = "/tmp/native_inference.sock"


class NativeInferenceClient:
    """Client for the native C++ inference server (in-container, via Unix socket)"""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self.socket: Optional[socket.socket] = None
        self.connected = False

        # Stats
        self.total_frames = 0
        self.total_time_ms = 0.0

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the native inference server"""
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect(self.socket_path)
            self.connected = True

            # Ping to verify connection
            response = self._send_request({'cmd': 'ping'})
            if response and response.get('status') == 'ok':
                logger.info(f"Connected to native server - version {response.get('version')}")
                return True
            else:
                logger.error("Server ping failed")
                self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to native server: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False

    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self.connected and self.socket is not None

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest frame and detection results from the native server.

        The native server captures camera and runs inference continuously.
        This method fetches the latest processed frame and results.

        Returns:
            Dict with 'frame' (numpy array), 'persons', 'faces', 'timing' or None on error
        """
        if not self.connected:
            logger.error("Not connected to native server")
            return None

        try:
            # Send get_frame request
            request = {'cmd': 'get_frame'}
            req_data = json.dumps(request).encode()
            self.socket.sendall(struct.pack('>I', len(req_data)) + req_data)

            # Receive response header
            header = self._recv_exact(4)
            if not header:
                raise ConnectionError("Failed to receive response header")

            resp_len = struct.unpack('>I', header)[0]
            resp_data = self._recv_exact(resp_len)
            if not resp_data:
                raise ConnectionError("Failed to receive response")

            result = json.loads(resp_data.decode())

            # Check for error
            if 'error' in result:
                logger.error(f"Server error: {result['error']}")
                return None

            # Receive frame data
            frame_header = self._recv_exact(4)
            if not frame_header:
                raise ConnectionError("Failed to receive frame header")

            frame_size = struct.unpack('>I', frame_header)[0]
            frame_data = self._recv_exact(frame_size)
            if not frame_data:
                raise ConnectionError("Failed to receive frame data")

            # Reconstruct frame
            width = result['width']
            height = result['height']
            channels = result['channels']
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, channels))

            result['frame'] = frame

            # Update stats
            self.total_frames += 1
            if 'timing' in result:
                self.total_time_ms += result['timing'].get('total_ms', 0)

            return result

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            self.disconnect()
            return None

    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _send_request(self, request: dict) -> Optional[dict]:
        """Send a request and receive response"""
        try:
            req_data = json.dumps(request).encode()
            self.socket.sendall(struct.pack('>I', len(req_data)) + req_data)

            header = self._recv_exact(4)
            if not header:
                return None

            resp_len = struct.unpack('>I', header)[0]
            resp_data = self._recv_exact(resp_len)
            if not resp_data:
                return None

            return json.loads(resp_data.decode())

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def get_stats(self) -> Optional[dict]:
        """Get server statistics"""
        if not self.connected:
            return None
        return self._send_request({'cmd': 'stats'})

    def process_image(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process an arbitrary image through the native TensorRT pipeline.

        Used for phone selfie face registration - ensures the same ArcFace model
        is used for both registration and runtime matching.

        Args:
            frame: BGR image as numpy array (height, width, 3)

        Returns:
            Dict with 'faces' array containing embeddings, or None on error
        """
        if not self.connected:
            logger.error("Not connected to native server")
            return None

        try:
            height, width, channels = frame.shape

            # Send process_image request with dimensions
            request = {
                'cmd': 'process_image',
                'width': width,
                'height': height,
                'channels': channels
            }
            req_data = json.dumps(request).encode()
            self.socket.sendall(struct.pack('>I', len(req_data)) + req_data)

            # Send raw image bytes
            frame_bytes = frame.astype(np.uint8).tobytes()
            self.socket.sendall(frame_bytes)

            # Receive response
            header = self._recv_exact(4)
            if not header:
                raise ConnectionError("Failed to receive response header")

            resp_len = struct.unpack('>I', header)[0]
            resp_data = self._recv_exact(resp_len)
            if not resp_data:
                raise ConnectionError("Failed to receive response")

            result = json.loads(resp_data.decode())

            if 'error' in result:
                logger.error(f"Server error: {result['error']}")
                return None

            return result

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.disconnect()
            return None

    def __del__(self):
        self.disconnect()


# ============================================================================
# Drop-in replacement service for camera-service integration
# ============================================================================

class NativeInferenceService:
    """
    Drop-in replacement for camera service's existing inference.

    Uses the native C++ server for inference (in-container).
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self._client: Optional[NativeInferenceClient] = None
        self._initialized = False

        # Configuration
        self.person_conf_threshold = 0.5
        self.face_conf_threshold = 0.5
        self.face_recognition_enabled = True

        # Stats
        self.total_frames = 0
        self.total_persons = 0
        self.total_faces = 0

    def initialize(self) -> bool:
        """Initialize connection to native server"""
        if self._initialized:
            logger.warning("Already initialized")
            return True

        logger.info(f"Connecting to native inference server at {self.socket_path}...")

        self._client = NativeInferenceClient(self.socket_path)
        if self._client.connect():
            self._initialized = True
            logger.info("Connected to native inference server")
            return True
        else:
            logger.error("Failed to connect to native inference server")
            return False

    def shutdown(self):
        """Shutdown the client connection"""
        if self._client:
            self._client.disconnect()
            self._client = None
        self._initialized = False
        logger.info("Native client disconnected")

    def is_ready(self) -> bool:
        """Check if connected and ready"""
        return self._initialized and self._client and self._client.is_connected()

    def process_image(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process an arbitrary image through the native TensorRT pipeline.

        Used for phone selfie face registration - ensures the same ArcFace model
        is used for both registration and runtime matching.

        Args:
            frame: BGR image as numpy array (height, width, 3)

        Returns:
            Dict with 'faces' array containing embeddings, or None on error
        """
        if not self._initialized:
            raise RuntimeError("Native service not initialized")

        result = self._client.process_image(frame)

        if result is None:
            # Connection lost, try to reconnect
            logger.warning("Lost connection during process_image, attempting reconnect...")
            if self._client.connect():
                result = self._client.process_image(frame)

        if result is None:
            raise RuntimeError("Failed to process image - server connection lost")

        return result

    def get_frame(self) -> Dict[str, Any]:
        """
        Get the latest frame and detection results from the native server.

        Returns:
            Dict with 'frame', 'persons', 'faces', 'timing'
        """
        if not self._initialized:
            raise RuntimeError("Native service not initialized")

        result = self._client.get_frame()

        if result is None:
            # Connection lost, try to reconnect
            logger.warning("Lost connection, attempting reconnect...")
            if self._client.connect():
                result = self._client.get_frame()

        if result is None:
            raise RuntimeError("Failed to get frame - server connection lost")

        # Update stats
        self.total_frames += 1
        self.total_persons += len(result.get('persons', []))
        self.total_faces += len(result.get('faces', []))

        # Add local stats to result
        result['stats'] = {
            'total_frames': self.total_frames,
            'total_persons': self.total_persons,
            'total_faces': self.total_faces,
        }

        return result


# Global instance for easy integration
_native_service: Optional[NativeInferenceService] = None


def get_native_service() -> NativeInferenceService:
    """Get or create the global native service instance"""
    global _native_service
    if _native_service is None:
        _native_service = NativeInferenceService()
    return _native_service


def init_native_service() -> bool:
    """Initialize the global native service"""
    service = get_native_service()
    return service.initialize()
