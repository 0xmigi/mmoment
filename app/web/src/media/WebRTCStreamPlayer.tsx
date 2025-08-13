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
      console.log('[WebRTC] Received remote stream');
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
        setConnectionState('connected');
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
      const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';
      const socket = io(backendUrl, {
        transports: ['websocket', 'polling']
      });

      socketRef.current = socket;

      socket.on('connect', () => {
        console.log('[WebRTC] Connected to signaling server');
        // Register as viewer
        socket.emit('register-viewer', { cameraId: currentCameraId });
      });

      socket.on('connect_error', (error) => {
        handleError(`Signaling server connection failed: ${error.message}`);
      });

      // Handle WebRTC offer from camera
      socket.on('webrtc-offer', async (data: { offer: RTCSessionDescriptionInit, senderId: string }) => {
        console.log('[WebRTC] Received offer from camera');
        
        try {
          const peerConnection = createPeerConnection();
          peerConnectionRef.current = peerConnection;

          await peerConnection.setRemoteDescription(data.offer);
          const answer = await peerConnection.createAnswer();
          await peerConnection.setLocalDescription(answer);

          socket.emit('webrtc-answer', {
            answer,
            targetId: data.senderId,
            cameraId: currentCameraId
          });

          console.log('[WebRTC] Sent answer to camera');
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

  if (error) {
    return (
      <div className="aspect-[9/16] md:aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-400">
          <p>WebRTC connection failed</p>
          <p className="text-sm mt-1">{error}</p>
          {fallback && (
            <div className="mt-2 text-xs text-blue-400">
              Falling back to HLS stream...
            </div>
          )}
        </div>
      </div>
    );
  }

  if (connectionState === 'connecting') {
    return (
      <div className="aspect-[9/16] md:aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-400">
          <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2" />
          <p>Connecting to camera...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden relative">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
      />
      
      {connectionState === 'connected' && (
        <div className="absolute top-2 left-2 bg-green-500 bg-opacity-80 text-white text-xs px-2 py-1 rounded">
          WebRTC Live
        </div>
      )}
      
      {connectionState === 'failed' && fallback && (
        <div className="absolute inset-0">
          {fallback}
        </div>
      )}
    </div>
  );
};

export { WebRTCStreamPlayer };