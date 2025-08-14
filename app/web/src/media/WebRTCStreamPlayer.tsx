import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useCamera } from '../camera/CameraProvider';
import { useParams } from 'react-router-dom';

interface WebRTCStreamPlayerProps {
  fallback?: React.ReactNode;
  onError?: (error: string) => void;
}

const WebRTCStreamPlayer: React.FC<WebRTCStreamPlayerProps> = ({ 
  fallback, 
  onError 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'failed' | 'disconnected'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [pendingStream, setPendingStream] = useState<MediaStream | null>(null);
  const { selectedCamera } = useCamera();
  const { cameraId } = useParams<{ cameraId: string }>();

  // Get current camera ID
  const currentCameraId = cameraId || selectedCamera?.publicKey || localStorage.getItem('directCameraId');

  const cleanup = useCallback(() => {
    console.log('[WebRTC] Cleaning up connection');
    
    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    
    // Disconnect socket
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    // Clear pending stream
    setPendingStream(null);
    setConnectionState('disconnected');
  }, []);

  const handleError = useCallback((errorMessage: string) => {
    console.error('[WebRTC] Error:', errorMessage);
    setError(errorMessage);
    setConnectionState('failed');
    onError?.(errorMessage);
    cleanup();
  }, [cleanup, onError]);

  const createPeerConnection = useCallback(() => {
    const config: RTCConfiguration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    };

    const peerConnection = new RTCPeerConnection(config);

    peerConnection.onicecandidate = (event) => {
      if (event.candidate && socketRef.current && currentCameraId) {
        console.log('[WebRTC] Sending ICE candidate');
        socketRef.current.emit('webrtc-ice-candidate', {
          candidate: event.candidate,
          targetId: 'camera', // Will be replaced by server with actual camera socket ID
          cameraId: currentCameraId
        });
      }
    };

    peerConnection.ontrack = (event) => {
      console.log('[WebRTC] Received remote stream:', event);
      console.log('[WebRTC] Number of streams:', event.streams.length);
      console.log('[WebRTC] Stream details:', event.streams[0]);
      
      const stream = event.streams[0];
      if (stream) {
        console.log('[WebRTC] Stream received, checking video element...');
        if (videoRef.current) {
          console.log('[WebRTC] Video element available, setting srcObject immediately');
          videoRef.current.srcObject = stream;
          setConnectionState('connected');
          setPendingStream(null);
        } else {
          console.log('[WebRTC] Video element not ready, storing stream for later');
          setPendingStream(stream);
          setConnectionState('connected');
        }
      } else {
        console.error('[WebRTC] No stream in track event');
      }
    };

    peerConnection.onconnectionstatechange = () => {
      const state = peerConnection.connectionState;
      console.log('[WebRTC] Connection state changed:', state);
      
      if (state === 'connected') {
        setConnectionState('connected');
        setError(null);
      } else if (state === 'failed' || state === 'disconnected') {
        handleError(`Connection ${state}`);
      }
    };

    peerConnection.oniceconnectionstatechange = () => {
      const iceState = peerConnection.iceConnectionState;
      console.log('[WebRTC] ICE connection state changed:', iceState);
    };

    peerConnection.onicegatheringstatechange = () => {
      const gatheringState = peerConnection.iceGatheringState;
      console.log('[WebRTC] ICE gathering state changed:', gatheringState);
    };

    peerConnection.onsignalingstatechange = () => {
      const signalingState = peerConnection.signalingState;
      console.log('[WebRTC] Signaling state changed:', signalingState);
    };

    return peerConnection;
  }, [currentCameraId, handleError]);

  const initializeWebRTC = useCallback(async () => {
    if (!currentCameraId) {
      handleError('No camera ID available');
      return;
    }

    try {
      console.log('[WebRTC] Initializing connection to camera:', currentCameraId);
      setConnectionState('connecting');
      setError(null);

      // Connect to signaling server (your backend)
      // Use the same backend URL that the Jetson is using
      const backendUrl = 'http://192.168.1.247:3001'; // Backend IP from Jetson logs
      console.log('[WebRTC] Connecting to backend:', backendUrl);
      
      const socket = io(backendUrl, {
        transports: ['websocket', 'polling'],
        timeout: 5000,
        forceNew: true
      });

      socketRef.current = socket;

      socket.on('connect', () => {
        console.log('[WebRTC] Connected to signaling server successfully');
        // Register as viewer
        console.log('[WebRTC] Registering as viewer for camera:', currentCameraId);
        socket.emit('register-viewer', { cameraId: currentCameraId });
      });

      socket.on('connect_error', (error) => {
        console.error('[WebRTC] Signaling server connection error:', error);
        handleError(`Signaling server connection failed: ${error.message}`);
      });

      socket.on('disconnect', (reason) => {
        console.log('[WebRTC] Signaling server disconnected:', reason);
        if (reason === 'io server disconnect') {
          // Server disconnected us, don't retry
          handleError('Signaling server disconnected');
        }
      });

      // Handle WebRTC offer from camera
      socket.on('webrtc-offer', async (data: { offer: RTCSessionDescriptionInit, senderId: string }) => {
        console.log('[WebRTC] Received offer from camera:', data.offer);
        console.log('[WebRTC] Offer SDP:', data.offer.sdp);
        
        try {
          const peerConnection = createPeerConnection();
          peerConnectionRef.current = peerConnection;

          await peerConnection.setRemoteDescription(data.offer);
          console.log('[WebRTC] Set remote description, creating answer...');
          console.log('[WebRTC] Remote streams after setRemoteDescription:', peerConnection.getRemoteStreams?.() || 'getRemoteStreams not available');
          console.log('[WebRTC] Transceivers:', peerConnection.getTransceivers());
          
          const answer = await peerConnection.createAnswer();
          await peerConnection.setLocalDescription(answer);
          console.log('[WebRTC] Created and set local description (answer)');

          socket.emit('webrtc-answer', {
            answer,
            targetId: data.senderId,
            cameraId: currentCameraId
          });

          console.log('[WebRTC] Sent answer to camera, connection state:', peerConnection.connectionState);
          console.log('[WebRTC] ICE connection state:', peerConnection.iceConnectionState);
          console.log('[WebRTC] ICE gathering state:', peerConnection.iceGatheringState);
        } catch (error) {
          handleError(`Failed to handle offer: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      });

      // Handle ICE candidates from camera
      socket.on('webrtc-ice-candidate', async (data: { candidate: RTCIceCandidateInit }) => {
        if (peerConnectionRef.current) {
          try {
            await peerConnectionRef.current.addIceCandidate(data.candidate);
            console.log('[WebRTC] Added ICE candidate');
          } catch (error) {
            console.warn('[WebRTC] Failed to add ICE candidate:', error);
          }
        }
      });

      socket.on('disconnect', () => {
        console.log('[WebRTC] Signaling server disconnected');
        handleError('Signaling server disconnected');
      });

    } catch (error) {
      handleError(`WebRTC initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [currentCameraId, createPeerConnection, handleError]);

  // Handle pending stream when video element becomes available
  useEffect(() => {
    if (pendingStream && videoRef.current) {
      console.log('[WebRTC] Video element now available, applying pending stream');
      console.log('[WebRTC] Stream active:', pendingStream.active);
      console.log('[WebRTC] Stream tracks:', pendingStream.getTracks());
      videoRef.current.srcObject = pendingStream;
      setPendingStream(null);
    }
  }, [pendingStream]);

  // Initialize WebRTC when camera changes
  useEffect(() => {
    if (currentCameraId) {
      initializeWebRTC();
    } else {
      cleanup();
    }

    return cleanup;
  }, [currentCameraId, initializeWebRTC, cleanup]);

  // Auto-retry on failure
  useEffect(() => {
    if (connectionState === 'failed' && currentCameraId) {
      const retryTimeout = setTimeout(() => {
        console.log('[WebRTC] Retrying connection...');
        initializeWebRTC();
      }, 5000);

      return () => clearTimeout(retryTimeout);
    }
  }, [connectionState, currentCameraId, initializeWebRTC]);


  return (
    <div className="aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
      />
    </div>
  );
};

export { WebRTCStreamPlayer };