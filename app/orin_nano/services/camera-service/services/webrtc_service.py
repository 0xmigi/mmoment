"""
WebRTC Service for Jetson Camera

Provides WebRTC streaming capability that connects to the backend signaling server
and streams video frames directly from buffer_service to web browsers with sub-second latency.
"""

import cv2
import time
import threading
import logging
import json
import os
import numpy as np
import socketio
from typing import Optional, Dict, Any, Callable
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
import asyncio
import queue

# FORCE aiortc to recognize correct network interface for ICE candidates
import aioice
import socket
import subprocess
_original_get_host_addresses = aioice.ice.get_host_addresses

def get_jetson_ip():
    """Dynamically get Jetson's actual IP address"""
    try:
        # Try to get the primary network interface IP (excluding docker interfaces)
        result = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Parse output like: "1.1.1.1 via 192.168.1.1 dev wlP1p1s0 src 192.168.1.232"
            parts = result.stdout.split()
            if 'src' in parts:
                src_index = parts.index('src')
                if src_index + 1 < len(parts):
                    ip = parts[src_index + 1]
                    logger.info(f"🌐 Detected Jetson IP dynamically: {ip}")
                    return ip
    except Exception as e:
        logger.error(f"Failed to get IP dynamically: {e}")
    
    # Fallback to current IP if dynamic detection fails
    fallback_ip = '192.168.1.232'
    logger.warning(f"Using fallback IP: {fallback_ip}")
    return fallback_ip

def _force_jetson_interface(use_ipv4=True, use_ipv6=False):
    """Force aiortc to use Jetson's actual IP instead of 0.0.0.0"""
    if use_ipv4:
        jetson_ip = get_jetson_ip()
        return [jetson_ip]
    return []

# Patch aiortc's interface discovery
aioice.ice.get_host_addresses = _force_jetson_interface
from av import VideoFrame

logger = logging.getLogger("WebRTCService")

class BufferVideoTrack(VideoStreamTrack):
    """
    A video track that streams frames from the buffer service
    """
    kind = "video"
    
    def __init__(self, buffer_service, fps=30):
        super().__init__()
        self.buffer_service = buffer_service
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        
        # Initialize time_base properly for aiortc
        from fractions import Fraction
        self.time_base = Fraction(1, 90000)  # Standard 90kHz clock
        
    async def recv(self):
        """
        Receive the next video frame
        """
        # Only log every 30th frame to avoid spam
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
        else:
            self._frame_count = 1
            
        if self._frame_count % 30 == 0:
            logger.info(f"🎥 WebRTC video track generating frame #{self._frame_count}")
        current_time = time.time()
        
        # Maintain frame rate
        if current_time - self.last_frame_time < self.frame_interval:
            await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
        
        try:
            # Get processed frame from buffer service (includes face detection boxes)
            frame_data = self.buffer_service.get_processed_frame()
            
            # Handle different return formats from buffer service
            if isinstance(frame_data, tuple) and len(frame_data) > 0:
                frame = frame_data[0]
            else:
                frame = frame_data
            
            # Check if frame is valid
            if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
                # Create a test pattern frame if no camera frame available
                frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128  # Gray frame
                # Add some pattern to verify it's working
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
                cv2.putText(frame, f"WebRTC Test {int(current_time) % 100}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Ensure frame is the right format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB for WebRTC
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Fallback for unexpected format
                frame_rgb = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            
            # Create VideoFrame
            av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            
            # WORKING TIMESTAMP SOLUTION
            try:
                # Use aiortc's built-in timestamp method if available
                av_frame.pts = self.next_timestamp()
                av_frame.time_base = self.time_base
            except:
                # Fallback to manual timestamp 
                from fractions import Fraction
                av_frame.pts = int(time.time() * 90000)  # 90kHz clock
                av_frame.time_base = Fraction(1, 90000)
            
            self.last_frame_time = time.time()
            return av_frame
            
        except Exception as e:
            logger.error(f"Error in BufferVideoTrack.recv: {e}")
            # Return a red error frame
            error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            error_frame[:, :, 0] = 255  # Red channel
            cv2.putText(error_frame, "WebRTC Error", (400, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            av_frame = VideoFrame.from_ndarray(error_frame, format="rgb24")
            
            # WORKING TIMESTAMP SOLUTION  
            try:
                # Use aiortc's built-in timestamp method if available
                av_frame.pts = self.next_timestamp()
                av_frame.time_base = self.time_base
            except:
                # Fallback to manual timestamp 
                from fractions import Fraction
                av_frame.pts = int(time.time() * 90000)  # 90kHz clock
                av_frame.time_base = Fraction(1, 90000)
            
            self.last_frame_time = time.time()
            return av_frame

class WebRTCService:
    """
    WebRTC streaming service that integrates with the camera service architecture
    """
    
    def __init__(self):
        self.buffer_service = None
        self.sio = None
        self.running = False
        self.connections = {}  # viewer_id -> RTCPeerConnection
        self.event_loop = None
        self.thread = None
        
        # Get backend URL - prefer BACKEND_URL env var, fallback to constructing from host/port
        self.backend_url = os.getenv('BACKEND_URL')
        if not self.backend_url:
            # Fallback to constructing from host/port for backward compatibility
            backend_host = os.getenv('BACKEND_HOST', '192.168.1.80')
            backend_port = os.getenv('BACKEND_PORT', '3001')
            self.backend_url = f"http://{backend_host}:{backend_port}"
        logger.info(f"WebRTC service will connect to backend at: {self.backend_url}")
        
        # Use Camera PDA as the identifier (matches your existing architecture)
        self.camera_pda = os.environ.get('CAMERA_PDA')
        if not self.camera_pda:
            logger.warning("CAMERA_PDA not set - WebRTC will use fallback device ID")
            self.camera_pda = self._get_device_id()
        logger.info(f"Camera PDA: {self.camera_pda}")
        
        # Get external IP for WebRTC dynamically
        self.external_ip = os.environ.get('WEBRTC_EXTERNAL_IP', get_jetson_ip())
        
        # ENHANCED WEBRTC WITH MULTIPLE STUN/TURN SERVERS
        from aiortc import RTCIceServer
        
        # Extract hostname from backend URL for TURN server
        turn_hostname = self.backend_url.replace("https://", "").replace("http://", "")
        if ":" in turn_hostname:
            turn_hostname = turn_hostname.split(":")[0]
        
        # Oracle Cloud CoTURN server with time-based auth
        import time
        import hmac
        import hashlib
        import base64
        
        # Generate time-based TURN credentials
        timestamp = int(time.time()) + 86400  # Valid for 24 hours
        username = str(timestamp)
        secret = 'mmoment-webrtc-secret-2025'
        
        # Generate HMAC-SHA1 credential
        mac = hmac.new(secret.encode(), username.encode(), hashlib.sha1)
        credential = base64.b64encode(mac.digest()).decode()
        
        self.rtc_config = RTCConfiguration(
            iceServers=[
                # Google STUN servers first (working for same-network)
                RTCIceServer(urls=['stun:stun.l.google.com:19302']),
                RTCIceServer(urls=['stun:stun1.l.google.com:19302']),
                RTCIceServer(urls=['stun:stun.cloudflare.com:3478']),
                # Oracle Cloud CoTURN server for cross-network - UDP and TCP
                RTCIceServer(
                    urls=[
                        'turn:129.80.99.75:3478',           # UDP TURN
                        'turn:129.80.99.75:3478?transport=tcp'  # TCP TURN for mobile networks
                    ],
                    username=username,
                    credential=credential
                ),
                RTCIceServer(urls=['stun:129.80.99.75:3478']),
            ]
        )
        
        # Configure port range commonly auto-forwarded by UPnP
        self.media_port_range = (8000, 8100)  # Range commonly used by media applications
        
        logger.info(f"WebRTC service configured with external IP: {self.external_ip}")
    
    def _get_device_id(self) -> str:
        """Get device ID from registration system (PDA-based)"""
        try:
            # Try to read device config for camera PDA (the actual identifier used by frontend)
            config_path = "/app/config/device_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'camera_pda' in config:
                        logger.info(f"Using camera PDA from device config: {config['camera_pda']}")
                        return config['camera_pda']
            
            # Fallback to hostname or generate if no registration has occurred yet
            import socket
            fallback_id = f"jetson-{socket.gethostname()}"
            logger.warning(f"No camera PDA found in device config, using fallback: {fallback_id}")
            return fallback_id
        except Exception as e:
            logger.warning(f"Could not determine device ID: {e}")
            return f"jetson-{int(time.time())}"
    
    def set_buffer_service(self, buffer_service):
        """Set the buffer service instance"""
        self.buffer_service = buffer_service
        logger.info("Buffer service injected into WebRTC service")
    
    def start(self):
        """Start the WebRTC service"""
        logger.info(f"🚀 WebRTC service start() called - buffer_service: {self.buffer_service}, running: {self.running}")
        
        if self.running:
            logger.warning("WebRTC service already running")
            return
        
        if not self.buffer_service:
            logger.error("❌ Buffer service not set - cannot start WebRTC service")
            return False
        
        logger.info("✅ Starting WebRTC service with buffer service")
        
        self.running = True
        
        # Start the async event loop in a separate thread
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
        logger.info("WebRTC service started")
        return True
    
    def _run_async_loop(self):
        """Run the async event loop"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        try:
            self.event_loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"Error in WebRTC async loop: {e}")
        finally:
            self.event_loop.close()
    
    async def _async_main(self):
        """Main async function"""
        # Initialize Socket.IO client
        self.sio = socketio.AsyncClient()
        
        # Set up Socket.IO event handlers
        self._setup_socketio_handlers()
        
        try:
            # Connect to backend WebRTC namespace
            logger.info(f"🔌 Attempting to connect to {self.backend_url}")
            logger.info(f"🔌 Socket.IO client state: {self.sio}")
            await self.sio.connect(self.backend_url, wait_timeout=10)
            logger.info(f"✅ Connected to signaling server at {self.backend_url}")
            logger.info(f"✅ Socket.IO connected: {self.sio.connected}")
            logger.info(f"✅ Socket.IO sid: {self.sio.sid}")
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(1)
            
            # Register as camera
            await self._register_camera()
            
            # Keep the connection alive
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ WebRTC service error: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        finally:
            if self.sio and self.sio.connected:
                await self.sio.disconnect()
    
    def _setup_socketio_handlers(self):
        """Set up Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            logger.info("🔗 Connected to signaling server - Socket.IO connected successfully")
            # MINIMAL WebRTC: No port redirection needed
            logger.info("🚀 MINIMAL WebRTC: Starting with natural port bindings (no socat)")
        
        @self.sio.event
        async def disconnect():
            logger.warning("🔌 Disconnected from signaling server")
        
        @self.sio.event
        async def connect_error(data):
            logger.error(f"❌ Socket.IO connection error: {data}")
        
        # Add handlers for acknowledgments
        @self.sio.event  
        async def register_ack(data):
            logger.info(f"📨 Registration acknowledgment received: {data}")
            
        @self.sio.event
        async def error(data):
            logger.error(f"❌ Socket.IO error event: {data}")
        
        # Test handler first
        @self.sio.on('viewer-wants-connection')
        async def handle_viewer_wants_connection(data):
            print(f"HANDLER CALLED: viewer-wants-connection with data: {data}")
            logger.info(f"Handler called: viewer-wants-connection with data: {data}")
            
            viewer_id = data.get('viewerId') if isinstance(data, dict) else str(data)
            logger.info(f"Processing viewer {viewer_id}")
            
            # Create WebRTC peer connection and offer
            await self._create_peer_connection_and_offer(viewer_id)
        
        @self.sio.on('webrtc-answer')
        async def handle_webrtc_answer(data):
            """Handle WebRTC answer from viewer"""
            logger.info(f"WEBRTC ANSWER RECEIVED from {data.get('senderId')}")
            sender_id = data['senderId']
            answer = data['answer']
            
            if sender_id in self.connections:
                pc = self.connections[sender_id]
                
                try:
                    # Set remote description (the viewer's answer)
                    await pc.setRemoteDescription(RTCSessionDescription(
                        sdp=answer['sdp'], 
                        type=answer['type']
                    ))
                    
                    logger.info(f"Successfully set remote description for viewer {sender_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to set remote description for viewer {sender_id}: {e}")
            else:
                logger.warning(f"Received answer from unknown viewer {sender_id}")
        
        @self.sio.on('webrtc-ice-candidate')  
        async def handle_webrtc_ice_candidate(data):
            """Handle ICE candidate from viewer"""
            logger.info(f"🔥 ICE CANDIDATE EVENT RECEIVED from frontend: {data}")
            sender_id = data['senderId']
            candidate = data['candidate']
            logger.info(f"🔥 Processing ICE candidate from sender {sender_id}: {candidate}")
            
            if sender_id in self.connections:
                pc = self.connections[sender_id]
                
                if candidate:
                    try:
                        from aiortc import RTCIceCandidate
                        # Parse the candidate string to extract components
                        candidate_parts = candidate['candidate'].split()
                        
                        ice_candidate = RTCIceCandidate(
                            component=candidate.get('component', 1),
                            foundation=candidate.get('foundation', candidate_parts[0] if len(candidate_parts) > 0 else ''),
                            ip=candidate_parts[4] if len(candidate_parts) > 4 else '',
                            port=int(candidate_parts[5]) if len(candidate_parts) > 5 else 0,
                            priority=candidate.get('priority', 0),
                            protocol=candidate_parts[2].lower() if len(candidate_parts) > 2 else 'udp',
                            type=candidate.get('type', candidate_parts[7] if len(candidate_parts) > 7 else 'host'),
                            sdpMid=candidate.get('sdpMid'),
                            sdpMLineIndex=candidate.get('sdpMLineIndex')
                        )
                        
                        await pc.addIceCandidate(ice_candidate)
                        logger.debug(f"Added ICE candidate from {sender_id}")
                    except Exception as e:
                        logger.warning(f"Failed to add ICE candidate from {sender_id}: {e}")
        
        # Catch-all event handler for debugging
        @self.sio.event
        async def catch_all(event, *args):
            """Catch all unhandled events for debugging"""
            if event not in ['connect', 'disconnect', 'connect_error']:
                logger.info(f"🔥 SOCKET.IO EVENT: {event} with data: {args}")
    
    async def _register_camera(self):
        """Register this camera with the signaling server"""
        logger.info(f"REGISTERING CAMERA WITH BACKEND: {self.camera_pda}")
        
        # Test basic Socket.IO functionality first
        logger.info("Testing Socket.IO connection with ping")
        try:
            await self.sio.emit('ping', {'test': 'data'})
            logger.info("Sent ping event")
        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
        
        # Simple registration data
        registration_data = {
            'cameraId': self.camera_pda
        }
        
        # Try the most likely correct event name
        try:
            await self.sio.emit('register-camera', registration_data)
            logger.info(f"Sent register-camera event: {registration_data}")
        except Exception as e:
            logger.error(f"Failed to send register-camera: {e}")
        
        # Try joining the room
        room_name = f"webrtc-{self.camera_pda}"
        try:
            await self.sio.emit('join', room_name)
            logger.info(f"Sent join event for room: {room_name}")
        except Exception as e:
            logger.error(f"Failed to join room: {e}")
        
        logger.info(f"Registration complete for camera PDA: {self.camera_pda}")
        
        # Test sending a webrtc-offer event to verify Socket.IO works
        try:
            test_offer = {
                'offer': {'type': 'offer', 'sdp': 'test'},
                'targetId': 'test-viewer',
                'cameraId': self.camera_pda
            }
            await self.sio.emit('webrtc-offer', test_offer)
            logger.info(f"Test webrtc-offer sent successfully: {test_offer}")
        except Exception as e:
            logger.error(f"Failed to send test webrtc-offer: {e}")
    
    async def _create_peer_connection_and_offer(self, viewer_id: str):
        """Create a new WebRTC peer connection and send offer"""
        logger.info(f"CRITICAL: Starting _create_peer_connection_and_offer for viewer {viewer_id}")
        
        try:
            # Create peer connection with standard configuration
            pc = RTCPeerConnection(configuration=self.rtc_config)
            logger.info(f"🔍 Created peer connection with standard configuration")
            
            self.connections[viewer_id] = pc
            logger.info(f"CRITICAL: Created RTCPeerConnection for viewer {viewer_id} with external IP {self.external_ip}")
            
            # Add video track from buffer service
            if not self.buffer_service:
                logger.error(f"CRITICAL: Buffer service is None!")
                return
                
            video_track = BufferVideoTrack(self.buffer_service)
            pc.addTrack(video_track)
            logger.info(f"CRITICAL: Added video track to peer connection for viewer {viewer_id}")
            
            # MINIMAL APPROACH: No port redirection - let aiortc handle ports naturally
            logger.info(f"🧊 MINIMAL WebRTC: Using aiortc's natural port binding for viewer {viewer_id}")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create peer connection: {e}")
            import traceback
            logger.error(f"CRITICAL: Traceback: {traceback.format_exc()}")
            return
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"🔗 WebRTC connection to viewer {viewer_id} state: {pc.connectionState}")
            
            if pc.connectionState == "connected":
                logger.info(f"🎉 WEBRTC CONNECTION ESTABLISHED with viewer {viewer_id}")
            elif pc.connectionState == "failed":
                # Clean up failed connection
                logger.info(f"🧹 Cleaning up failed connection for viewer {viewer_id}")
                if viewer_id in self.connections:
                    del self.connections[viewer_id]
                    logger.warning(f"❌ Removed failed connection for viewer {viewer_id}")
            elif pc.connectionState == "closed":
                # Clean up closed connection
                logger.info(f"🧹 Cleaning up closed connection for viewer {viewer_id}")
        
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"🧊 ICE connection to viewer {viewer_id} state: {pc.iceConnectionState}")
            if pc.iceConnectionState == "connected":
                logger.info(f"🎉 ICE CONNECTION ESTABLISHED! Media flow should start now for viewer {viewer_id}")
            elif pc.iceConnectionState == "failed":
                logger.error(f"❌ ICE connection failed for viewer {viewer_id} - cleaning up")
                if viewer_id in self.connections:
                    del self.connections[viewer_id]
            elif pc.iceConnectionState == "disconnected":
                logger.warning(f"⚠️ ICE connection disconnected for viewer {viewer_id}")
            
        @pc.on("icegatheringstatechange") 
        async def on_icegatheringstatechange():
            logger.info(f"🔍 ICE gathering to viewer {viewer_id} state: {pc.iceGatheringState}")
            if pc.iceGatheringState == "complete":
                logger.info(f"🧊 ICE gathering complete for viewer {viewer_id} - should have sent candidates")
                
                # SIMPLE APPROACH: Let aiortc handle ICE candidates naturally
                logger.info(f"🧊 SIMPLE WebRTC: Relying on aiortc's natural ICE candidate generation")
        
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                # Track candidates being sent
                pc._candidates_sent = getattr(pc, '_candidates_sent', 0) + 1
                
                # Log full candidate details for debugging
                candidate_ip = candidate.ip
                candidate_type = candidate.type
                
                # Detailed logging for cross-network debugging
                if candidate_type == 'host':
                    logger.info(f"🏠 HOST candidate: {candidate_ip}:{candidate.port} ({candidate.protocol})")
                elif candidate_type == 'srflx':
                    logger.info(f"🌐 STUN candidate (server reflexive): {candidate_ip}:{candidate.port} ({candidate.protocol})")
                elif candidate_type == 'relay':
                    logger.info(f"🔄 TURN relay candidate: {candidate_ip}:{candidate.port} ({candidate.protocol}) - CRITICAL for cross-network!")
                else:
                    logger.info(f"🧊 ICE candidate ({candidate_type}): {candidate_ip}:{candidate.port} ({candidate.protocol})")
            else:
                logger.info(f"✅ ICE gathering complete signal for viewer {viewer_id}")
                return
            
            # CRITICAL: Check for Docker-internal IPs only on host candidates
            if candidate_type == 'host' and (candidate_ip.startswith('172.17.') or candidate_ip.startswith('172.18.')):
                logger.warning(f"🚨 Docker-internal IP detected in host candidate: {candidate_ip}")
                # Force external IP for Docker-internal candidates
                candidate_ip = self.external_ip
                logger.info(f"🧊 Replaced Docker IP with external IP: {candidate_ip}")
            
            # Use aiortc's actual port without redirection
            actual_port = candidate.port
            
            # Build proper ICE candidate string
            ice_candidate_str = f"candidate:{candidate.foundation} {candidate.component} {candidate.protocol.upper()} {candidate.priority} {candidate_ip} {actual_port} typ {candidate.type}"
            
            # Add generation for relay candidates (TURN)
            if candidate.type == 'relay' and hasattr(candidate, 'relatedAddress') and candidate.relatedAddress:
                ice_candidate_str += f" raddr {candidate.relatedAddress} rport {candidate.relatedPort}"
            
            candidate_data = {
                'candidate': {
                        'candidate': ice_candidate_str,
                        'sdpMid': candidate.sdpMid,
                        'sdpMLineIndex': candidate.sdpMLineIndex,
                        # FRONTEND COMPATIBILITY: Add parsed fields that frontend might expect
                        'protocol': candidate.protocol.lower(),
                        'address': candidate_ip,
                        'port': actual_port,
                        'type': candidate.type,
                        'priority': candidate.priority,
                        'foundation': candidate.foundation,
                        'component': candidate.component
                    },
                    'targetId': viewer_id,
                    'cameraId': self.camera_pda
                }
            await self.sio.emit('webrtc-ice-candidate', candidate_data)
            logger.info(f"🧊 Sent ICE candidate #{pc._candidates_sent} to viewer {viewer_id}: {candidate_data}")
        
        logger.info(f"Created peer connection for viewer {viewer_id}")
        
        # Create and send WebRTC offer
        try:
            logger.info(f"CRITICAL: Creating WebRTC offer for viewer {viewer_id}")
            offer = await pc.createOffer()
            logger.info(f"CRITICAL: Created offer, setting local description for viewer {viewer_id}")
            
            await pc.setLocalDescription(offer)
            logger.info(f"CRITICAL: Set local description for viewer {viewer_id}")
            
            # CRITICAL FIX: Wait for ICE gathering to complete before sending offer
            # This ensures all ICE candidates (including TURN relay) are included in the SDP
            logger.info(f"🧊 Waiting for ICE gathering to complete...")
            ice_gathering_timeout = 5.0  # 5 second timeout for ICE gathering
            start_time = time.time()
            
            while pc.iceGatheringState != 'complete' and (time.time() - start_time) < ice_gathering_timeout:
                await asyncio.sleep(0.1)
                logger.info(f"🧊 ICE gathering state: {pc.iceGatheringState}")
            
            if pc.iceGatheringState == 'complete':
                logger.info(f"✅ ICE gathering completed after {time.time() - start_time:.2f}s")
            else:
                logger.warning(f"⚠️ ICE gathering timed out after {ice_gathering_timeout}s, state: {pc.iceGatheringState}")
            
            # Now use pc.localDescription which includes ALL gathered ICE candidates
            final_offer = RTCSessionDescription(
                type=pc.localDescription.type,
                sdp=pc.localDescription.sdp
            )
            
            logger.info(f"🎯 Using complete SDP with all ICE candidates")
            logger.info(f"🔍 Final SDP preview: {final_offer.sdp[:500]}...")
            
            # Use the complete SDP with all ICE candidates
            complete_sdp = final_offer.sdp
            
            # Log ICE candidates found in SDP for debugging
            ice_candidate_lines = [line for line in complete_sdp.split('\n') if line.startswith('a=candidate:')]
            logger.info(f"📊 Found {len(ice_candidate_lines)} ICE candidates in SDP:")
            for candidate in ice_candidate_lines[:5]:  # Log first 5 candidates
                logger.info(f"  - {candidate}")
            
            # DON'T modify IP addresses - let TURN servers handle NAT traversal properly
            
            # Let the TURN servers and STUN servers handle all the connectivity
            # Don't add fake relay candidates - rely on real ICE negotiation
            
            # CRITICAL FIX: Find what port aiortc actually bound to and use that
            import re
            actual_rtp_port = None
            actual_rtcp_port = None
            
            # Try to extract actual bound ports from aiortc
            try:
                # Check if we can get the actual bound port from aiortc internals
                if hasattr(pc, '_sctp') and hasattr(pc._sctp, '_transport'):
                    dtls_transport = pc._sctp._transport
                    if hasattr(dtls_transport, '_transport'):
                        ice_transport = dtls_transport._transport
                        if hasattr(ice_transport, '_connection') and hasattr(ice_transport._connection, '_local_address'):
                            local_addr = ice_transport._connection._local_address
                            if local_addr and len(local_addr) > 1:
                                actual_rtp_port = local_addr[1]
                                actual_rtcp_port = actual_rtp_port + 1
                                logger.info(f"🔍 Found aiortc actual bound port: {actual_rtp_port}")
            except Exception as e:
                logger.warning(f"Could not extract actual aiortc port: {e}")
            
            # Use the actual port if detected, otherwise use our target range
            if actual_rtp_port:
                target_rtp_port = actual_rtp_port
                target_rtcp_port = actual_rtcp_port
                logger.info(f"🔍 Using detected aiortc port: {target_rtp_port}")
            else:
                # Default to 10000 range for SDP but let aiortc choose actual port
                target_rtp_port = 10000
                target_rtcp_port = 10001
                logger.info(f"🔍 Using default ports in SDP: {target_rtp_port}/{target_rtcp_port}")
            
            # SIMPLE APPROACH: Don't modify ports - use aiortc's natural ports
            logger.info(f"🔍 SIMPLE WebRTC: Using aiortc's natural ports without redirection")
            logger.info(f"🔍 Complete SDP ready with all ICE candidates")
            
            # Send offer to viewer via signaling server with complete SDP
            offer_data = {
                'offer': {
                    'type': offer.type,
                    'sdp': complete_sdp
                },
                'targetId': viewer_id,
                'cameraId': self.camera_pda
            }
            
            logger.info(f"CRITICAL: Sending webrtc-offer event with data: {offer_data}")
            await self.sio.emit('webrtc-offer', offer_data)
            logger.info(f"CRITICAL: Successfully sent WebRTC offer to viewer {viewer_id}")
            
            # MINIMAL APPROACH: No manual ICE candidates - trust aiortc
            logger.info(f"🧊 MINIMAL WebRTC: Letting aiortc generate all ICE candidates naturally")
            
            # Wait for ICE connection with timeout
            logger.info(f"🧊 Waiting for ICE connection establishment...")
            
            async def monitor_ice_connection():
                ice_timeout = 10.0  # 10 second timeout for ICE
                start_time = time.time()
                
                while time.time() - start_time < ice_timeout:
                    if viewer_id not in self.connections:
                        logger.warning(f"Viewer {viewer_id} connection removed during ICE wait")
                        return
                        
                    if pc.iceConnectionState == "connected":
                        logger.info(f"🎉 ICE connection established for viewer {viewer_id} after {time.time() - start_time:.2f}s")
                        return
                    elif pc.iceConnectionState == "failed":
                        logger.error(f"❌ ICE connection failed for viewer {viewer_id} after {time.time() - start_time:.2f}s")
                        return
                        
                    await asyncio.sleep(0.5)
                
                # ICE timeout - log detailed state
                logger.warning(f"⏰ ICE connection timeout after {ice_timeout}s for viewer {viewer_id}")
                logger.warning(f"⏰ Final states - ICE: {pc.iceConnectionState}, Connection: {pc.connectionState}")
                logger.warning(f"⏰ ICE Gathering: {pc.iceGatheringState}, Signaling: {pc.signalingState}")
            
            # Start ICE monitoring
            asyncio.create_task(monitor_ice_connection())
            
            # MINIMAL WebRTC: No manual ICE candidates - rely entirely on aiortc
            logger.info("🚀 MINIMAL WebRTC: Relying on aiortc's automatic ICE candidate generation only")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to create WebRTC offer for viewer {viewer_id}: {e}")
            import traceback
            logger.error(f"CRITICAL: Offer creation traceback: {traceback.format_exc()}")
            # Clean up on failure
            if viewer_id in self.connections:
                del self.connections[viewer_id]
    
    def _setup_port_redirection_for_viewer(self, pc, viewer_id):
        """Set up dynamic port redirection after ICE gathering is complete"""
        logger.info(f"🔄 Setting up port redirection monitoring for viewer {viewer_id}")
        
        @pc.on("icegatheringstatechange")
        async def on_ice_gathering_for_redirection():
            logger.info(f"🔍 Port redirection handler - ICE gathering state for viewer {viewer_id}: {pc.iceGatheringState}")
            if pc.iceGatheringState == "complete":
                logger.info(f"🔍 ICE gathering complete for viewer {viewer_id}, discovering actual ports...")
                await self._discover_and_redirect_ports(viewer_id)
                
        # NOTE: Disabled aggressive health monitoring as it was too strict
        # asyncio.create_task(self._monitor_connection_health(viewer_id))
    
    async def _discover_and_redirect_ports(self, viewer_id):
        """Discover what ports aiortc actually bound to and set up redirection"""
        try:
            # Read current UDP bindings to find aiortc ports
            with open('/proc/net/udp', 'r') as f:
                udp_data = f.readlines()
            
            # Find recently bound high ports (likely aiortc)
            current_ports = []
            for line in udp_data[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    local_address = parts[1]
                    if ':' in local_address:
                        port_hex = local_address.split(':')[1]
                        port_dec = int(port_hex, 16)
                        # Look for ports in aiortc's typical range
                        if port_dec >= 30000:
                            current_ports.append(port_dec)
            
            if not current_ports:
                logger.warning(f"No high ports found for viewer {viewer_id}")
                return
            
            # Select the two most recently bound ports (likely RTP/RTCP pair)
            current_ports.sort(reverse=True)  # Most recent first
            aiortc_rtp_port = current_ports[0] if len(current_ports) > 0 else None
            aiortc_rtcp_port = current_ports[1] if len(current_ports) > 1 else None
            
            if aiortc_rtp_port:
                logger.info(f"🔍 Detected aiortc RTP port: {aiortc_rtp_port}")
                # Set up socat redirection from 10000 to actual aiortc port
                await self._setup_port_redirect(10000, aiortc_rtp_port, viewer_id)
            
            if aiortc_rtcp_port:
                logger.info(f"🔍 Detected aiortc RTCP port: {aiortc_rtcp_port}")
                # Set up socat redirection from 10001 to actual aiortc port
                await self._setup_port_redirect(10001, aiortc_rtcp_port, viewer_id)
                
        except Exception as e:
            logger.error(f"Failed to discover aiortc ports for viewer {viewer_id}: {e}")
    
    async def _setup_port_redirect(self, advertised_port, actual_port, viewer_id):
        """Set up socat to redirect traffic from advertised_port to actual_port"""
        import subprocess
        try:
            # Kill any existing socat process for this advertised port
            subprocess.run(['pkill', '-f', f'socat.*UDP-LISTEN:{advertised_port}'], 
                         capture_output=True, check=False)
            
            # Find what IP the aiortc port is actually bound to
            target_ip = "127.0.0.1"  # Default
            try:
                result = subprocess.run(['netstat', '-uln'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if f':{actual_port}' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            bind_address = parts[3]
                            if ':' in bind_address:
                                ip_part = bind_address.split(':')[0]
                                if ip_part == '0.0.0.0':
                                    target_ip = "127.0.0.1"  # Use localhost for 0.0.0.0
                                elif ip_part.startswith('172.17.'):
                                    target_ip = ip_part  # Use Docker IP directly
                                else:
                                    target_ip = ip_part
                                logger.info(f"🔍 Found aiortc port {actual_port} bound to {ip_part}, using {target_ip}")
                                break
            except Exception as e:
                logger.warning(f"Could not detect aiortc binding IP for port {actual_port}: {e}")
            
            # Start socat to redirect UDP traffic
            cmd = [
                'socat', 
                f'UDP-LISTEN:{advertised_port},bind=0.0.0.0,reuseaddr,fork',
                f'UDP-CONNECT:{target_ip}:{actual_port}'
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"🔀 Started port redirection: {advertised_port} → {target_ip}:{actual_port} for viewer {viewer_id}")
            logger.info(f"🔀 Socat command: {' '.join(cmd)}")
            
            # Store the process so we can clean it up later
            if not hasattr(self, '_port_redirects'):
                self._port_redirects = {}
            self._port_redirects[f"{viewer_id}_{advertised_port}"] = process
            
        except Exception as e:
            logger.error(f"Failed to set up port redirect {advertised_port}→{actual_port}: {e}")

    async def _cleanup_viewer_redirections(self, viewer_id):
        """Clean up port redirections for a specific viewer"""
        if hasattr(self, '_port_redirects'):
            keys_to_remove = []
            for key, process in self._port_redirects.items():
                if key.startswith(f"{viewer_id}_"):
                    try:
                        process.terminate()
                        try:
                            process.wait(timeout=2)  # Wait for clean termination
                        except:
                            process.kill()  # Force kill if needed
                        logger.info(f"🧹 Cleaned up port redirection: {key}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up redirection {key}: {e}")
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._port_redirects[key]
                
    async def _monitor_connection_health(self, viewer_id):
        """Monitor WebRTC connection health and handle timeouts"""
        try:
            # Wait for ICE to complete (with timeout)
            timeout = 15.0  # Increased from 10s to 15s
            start_time = asyncio.get_event_loop().time()
            
            while True:
                if viewer_id not in self.connections:
                    logger.info(f"❌ Connection {viewer_id} no longer exists, stopping health monitor")
                    return
                    
                pc = self.connections[viewer_id]
                current_time = asyncio.get_event_loop().time()
                
                # Check ICE connection state
                ice_state = pc.iceConnectionState
                connection_state = pc.connectionState
                
                if ice_state in ['connected', 'completed']:
                    logger.info(f"✅ ICE connection successful for viewer {viewer_id} after {current_time - start_time:.1f}s")
                    return
                    
                if ice_state in ['failed', 'disconnected']:
                    logger.warning(f"❌ ICE connection failed for viewer {viewer_id}: {ice_state}")
                    await self._handle_connection_failure(viewer_id)
                    return
                    
                if connection_state in ['failed', 'closed']:
                    logger.warning(f"❌ Peer connection failed for viewer {viewer_id}: {connection_state}")
                    await self._handle_connection_failure(viewer_id)
                    return
                    
                if current_time - start_time > timeout:
                    logger.warning(f"⏰ ICE connection timeout after {timeout}s for viewer {viewer_id}")
                    logger.warning(f"⏰ Final states - ICE: {ice_state}, Connection: {connection_state}")
                    logger.warning(f"⏰ ICE Gathering: {pc.iceGatheringState}, Signaling: {pc.signalingState}")
                    await self._handle_connection_timeout(viewer_id)
                    return
                
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"❌ Error monitoring connection health for {viewer_id}: {e}")
            
    async def _handle_connection_failure(self, viewer_id):
        """Handle WebRTC connection failure"""
        try:
            logger.warning(f"🔄 Handling connection failure for viewer {viewer_id}")
            
            # Clean up the failed connection
            await self._cleanup_viewer_connection(viewer_id)
            
            # Notify frontend about the failure
            try:
                await self.sio.emit('webrtc-connection-failed', {
                    'viewerId': viewer_id,
                    'reason': 'Connection failed - please try refreshing'
                })
            except Exception as e:
                logger.error(f"Failed to notify frontend of connection failure: {e}")
                
        except Exception as e:
            logger.error(f"Error handling connection failure for {viewer_id}: {e}")
            
    async def _handle_connection_timeout(self, viewer_id):
        """Handle ICE connection timeout"""
        try:
            logger.warning(f"🔄 Handling connection timeout for viewer {viewer_id}")
            
            # Try restarting the connection once
            if not hasattr(self, '_restart_attempts'):
                self._restart_attempts = {}
                
            attempts = self._restart_attempts.get(viewer_id, 0)
            if attempts < 1:  # Only try once
                self._restart_attempts[viewer_id] = attempts + 1
                logger.info(f"🔄 Attempting connection restart for viewer {viewer_id} (attempt {attempts + 1})")
                
                # Clean up current connection
                await self._cleanup_viewer_connection(viewer_id)
                
                # Wait a moment
                await asyncio.sleep(2)
                
                # Restart the connection
                await self._create_peer_connection_and_offer(viewer_id)
            else:
                logger.warning(f"❌ Giving up on viewer {viewer_id} after {attempts} restart attempts")
                await self._cleanup_viewer_connection(viewer_id)
                
                # Notify frontend about the timeout
                try:
                    await self.sio.emit('webrtc-connection-timeout', {
                        'viewerId': viewer_id,
                        'reason': 'Connection timed out - please refresh and try again'
                    })
                except Exception as e:
                    logger.error(f"Failed to notify frontend of timeout: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling connection timeout for {viewer_id}: {e}")
            
    async def _cleanup_viewer_connection(self, viewer_id):
        """Clean up all resources for a viewer connection"""
        try:
            # Close peer connection
            if viewer_id in self.connections:
                try:
                    await self.connections[viewer_id].close()
                except Exception as e:
                    logger.warning(f"Error closing peer connection for {viewer_id}: {e}")
                del self.connections[viewer_id]
                
            # Clean up port redirections
            logger.info(f"🧹 Minimal cleanup for viewer {viewer_id} - no redirections to clean")
            
            # Clean up restart attempts
            if hasattr(self, '_restart_attempts') and viewer_id in self._restart_attempts:
                del self._restart_attempts[viewer_id]
                
            logger.info(f"🧹 Cleaned up all resources for viewer {viewer_id}")
            
        except Exception as e:
            logger.error(f"Error during viewer cleanup for {viewer_id}: {e}")
            
    async def _cleanup_all_socat_processes(self):
        """Clean up all existing socat processes on startup"""
        import subprocess
        try:
            # Kill all socat processes that might be left from previous runs
            result = subprocess.run(['pkill', '-f', 'UDP-LISTEN:10000'], capture_output=True, check=False)
            result = subprocess.run(['pkill', '-f', 'UDP-LISTEN:10001'], capture_output=True, check=False)
            
            # Wait for processes to terminate
            await asyncio.sleep(1)
            
            logger.info("🧹 Cleaned up all existing socat processes on startup")
            
        except Exception as e:
            logger.warning(f"Error cleaning up socat processes on startup: {e}")

    async def handle_webrtc_answer(self, data):
        """Handle WebRTC answer from client"""
        try:
            viewer_id = data.get('viewerId') or data.get('targetId')
            answer_data = data.get('answer')
            
            if not viewer_id or not answer_data:
                logger.error(f"Invalid answer data: {data}")
                return
            
            if viewer_id not in self.connections:
                logger.error(f"No peer connection found for viewer {viewer_id}")
                return
                
            pc = self.connections[viewer_id]
            answer = RTCSessionDescription(sdp=answer_data['sdp'], type=answer_data['type'])
            
            await pc.setRemoteDescription(answer)
            logger.info(f"✅ Set remote description (answer) for viewer {viewer_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle WebRTC answer: {e}")
            import traceback
            logger.error(f"Answer handling traceback: {traceback.format_exc()}")
    
    def stop(self):
        """Stop the WebRTC service"""
        logger.info("Stopping WebRTC service")
        self.running = False
        
        # Close all peer connections
        if self.event_loop and not self.event_loop.is_closed():
            for viewer_id, pc in self.connections.items():
                self.event_loop.create_task(pc.close())
            self.connections.clear()
        
        # Disconnect Socket.IO
        if self.sio and hasattr(self.sio, 'disconnect'):
            if self.event_loop and not self.event_loop.is_closed():
                self.event_loop.create_task(self.sio.disconnect())
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("WebRTC service stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebRTC service status"""
        return {
            'running': self.running,
            'connected': self.sio.connected if self.sio else False,
            'camera_pda': self.camera_pda,
            'backend_url': self.backend_url,
            'active_connections': len(self.connections),
            'connections': list(self.connections.keys())
        }

# Global instance
_webrtc_service_instance = None

def get_webrtc_service() -> WebRTCService:
    """Get the singleton WebRTC service instance"""
    global _webrtc_service_instance
    if _webrtc_service_instance is None:
        _webrtc_service_instance = WebRTCService()
    return _webrtc_service_instance

def reset_webrtc_service():
    """Reset the WebRTC service instance (for testing/restart)"""
    global _webrtc_service_instance
    if _webrtc_service_instance:
        _webrtc_service_instance.stop()
    _webrtc_service_instance = None