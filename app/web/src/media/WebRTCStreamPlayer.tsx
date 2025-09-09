import { useCamera } from "../camera/CameraProvider";
import { CONFIG } from "../core/config";
import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { io, Socket } from "socket.io-client";

interface WebRTCStreamPlayerProps {
  fallback?: React.ReactNode;
  onError?: (error: string) => void;
}

const WebRTCStreamPlayer: React.FC<WebRTCStreamPlayerProps> = ({ onError }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const [connectionState, setConnectionState] = useState<
    "connecting" | "connected" | "failed" | "disconnected"
  >("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [pendingStream, setPendingStream] = useState<MediaStream | null>(null);
  const { selectedCamera } = useCamera();
  const { cameraId } = useParams<{ cameraId: string }>();

  // Get current camera ID
  const currentCameraId =
    cameraId ||
    selectedCamera?.publicKey ||
    localStorage.getItem("directCameraId");

  const cleanup = useCallback(() => {
    console.log("[WebRTC] Cleaning up connection");

    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // NOTE: Keep socket connection alive for recovery
    // Socket.IO signaling should persist through WebRTC failures

    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Clear pending stream
    setPendingStream(null);
    setConnectionState("disconnected");
  }, []);

  // Full cleanup for component unmount
  const fullCleanup = useCallback(() => {
    console.log("[WebRTC] Full cleanup including socket disconnect");

    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    // Disconnect socket on component unmount
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
    setConnectionState("disconnected");
  }, []);

  const handleError = useCallback(
    (errorMessage: string) => {
      console.error("[WebRTC] Error:", errorMessage);
      setError(errorMessage);
      setConnectionState("failed");
      onError?.(errorMessage);
      cleanup();
    },
    [cleanup, onError]
  );


  const createPeerConnection = useCallback(async () => {
    // Generate time-based TURN credentials
    const timestamp = Math.floor(Date.now() / 1000) + 86400; // Valid for 24 hours
    const username = timestamp.toString();
    const secret = 'mmoment-webrtc-secret-2025';
    
    let credential = '';
    try {
      // Generate HMAC-SHA1 credential
      const encoder = new TextEncoder();
      const secretBytes = encoder.encode(secret).buffer as ArrayBuffer;
      const usernameBytes = encoder.encode(username).buffer as ArrayBuffer;
      
      const key = await crypto.subtle.importKey(
        'raw',
        secretBytes,
        { name: 'HMAC', hash: 'SHA-1' } as HmacImportParams,
        false,
        ['sign']
      );
      const signature = await crypto.subtle.sign('HMAC', key, usernameBytes) as ArrayBuffer;
      credential = btoa(String.fromCharCode(...new Uint8Array(signature)));
      console.log("[WebRTC] ✅ Generated TURN credentials successfully:", { username, credential: credential.substring(0, 10) + '...' });
    } catch (error) {
      console.error("[WebRTC] ❌ TURN credential generation failed:", error);
      console.log("[WebRTC] 🔄 Falling back to simplified TURN config");
      // Fallback: still try connecting without TURN
    }

    const iceServers: RTCIceServer[] = [
      // STUN servers for NAT traversal
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" },
    ];

    // Add TURN server only if credential generation succeeded
    if (credential) {
      iceServers.push({
        urls: [
          "turn:129.80.99.75:3478",
          "turn:129.80.99.75:3478?transport=tcp"
        ],
        username: username,
        credential: credential
      });
      console.log("[WebRTC] 🔄 TURN server added to ICE configuration");
    } else {
      console.log("[WebRTC] ⚠️ TURN server skipped due to credential failure");
    }

    const config: RTCConfiguration = {
      iceServers,
      iceCandidatePoolSize: 10,
      iceTransportPolicy: "all", // Try both direct and relay
    };

    const peerConnection = new RTCPeerConnection(config);

    peerConnection.onicecandidate = (event) => {
      if (event.candidate && socketRef.current && currentCameraId) {
        console.log("[WebRTC] 🧊 BROWSER SENDING ICE candidate:", {
          type: event.candidate.type,
          protocol: event.candidate.protocol,
          address: event.candidate.address,
          port: event.candidate.port,
          priority: event.candidate.priority,
          foundation: event.candidate.foundation,
          component: event.candidate.component,
          candidate_string: event.candidate.candidate
        });
        socketRef.current.emit("webrtc-ice-candidate", {
          candidate: event.candidate,
          targetId: "camera",
          cameraId: currentCameraId,
        });
      } else if (event.candidate === null) {
        console.log("[WebRTC] 🧊 BROWSER ICE gathering completed");
      }
    };

    peerConnection.ontrack = (event) => {
      console.log("[WebRTC] Received remote stream:", event);
      console.log("[WebRTC] Number of streams:", event.streams.length);
      console.log("[WebRTC] Stream details:", event.streams[0]);

      const stream = event.streams[0];
      if (stream) {
        console.log("[WebRTC] Stream received, checking video element...");
        console.log("[WebRTC] Stream active:", stream.active);
        console.log("[WebRTC] Stream video tracks:", stream.getVideoTracks());

        if (videoRef.current) {
          console.log(
            "[WebRTC] Video element available, setting srcObject immediately"
          );
          videoRef.current.srcObject = stream;

          // Force video to load and play
          videoRef.current.load();
          videoRef.current.play().catch((error) => {
            console.warn("[WebRTC] Video play failed:", error);
          });

          setConnectionState("connected");
          setPendingStream(null);
        } else {
          console.log(
            "[WebRTC] Video element not ready, storing stream for later"
          );
          setPendingStream(stream);
          setConnectionState("connected");
        }
      } else {
        console.error("[WebRTC] No stream in track event");
      }
    };

    peerConnection.onconnectionstatechange = () => {
      const state = peerConnection.connectionState;
      console.log("[WebRTC] Connection state changed:", state);

      if (state === "connected") {
        console.log("[WebRTC] 🎉 Connection established successfully!");
        setConnectionState("connected");
        setError(null);
      } else if (state === "connecting") {
        console.log("[WebRTC] Connection in progress...");
        setConnectionState("connecting");
      } else if (state === "failed") {
        console.error(
          "[WebRTC] ❌ Connection failed - this might be a network/firewall issue"
        );
        handleError("Connection failed - check network connectivity");
      } else if (state === "disconnected") {
        console.warn("[WebRTC] ⚠️ Connection disconnected");
        handleError("Connection disconnected");
      }
    };

    peerConnection.oniceconnectionstatechange = () => {
      const iceState = peerConnection.iceConnectionState;
      console.log("[WebRTC] 🧊 ICE connection state changed:", iceState);

      if (iceState === "connected" || iceState === "completed") {
        console.log("[WebRTC] 🎉 ICE connection established successfully!");
        setConnectionState("connected");
        setError(null);
      } else if (iceState === "checking") {
        console.log("[WebRTC] 🔍 ICE candidates are being checked...");
        setConnectionState("connecting");
      } else if (iceState === "failed") {
        console.error(
          "[WebRTC] ❌ ICE connection failed - network connectivity issue"
        );
        console.error(
          "[WebRTC] This usually means the camera and viewer cannot reach each other"
        );
        handleError("Network connectivity failed - check if devices are on same network");
      } else if (iceState === "disconnected") {
        console.warn("[WebRTC] ⚠️ ICE connection disconnected");
        // Don't immediately fail on disconnect, might reconnect
        setTimeout(() => {
          if (peerConnection.iceConnectionState === "disconnected") {
            handleError("ICE connection lost");
          }
        }, 5000);
      }
    };

    peerConnection.onicegatheringstatechange = () => {
      const gatheringState = peerConnection.iceGatheringState;
      console.log("[WebRTC] ICE gathering state changed:", gatheringState);
    };

    peerConnection.onsignalingstatechange = () => {
      const signalingState = peerConnection.signalingState;
      console.log("[WebRTC] Signaling state changed:", signalingState);
    };

    return peerConnection;
  }, [currentCameraId, handleError]);

  const initializeWebRTC = useCallback(async () => {
    if (!currentCameraId) {
      handleError("No camera ID available");
      return;
    }

    try {
      console.log(
        "[WebRTC] Initializing connection to camera:",
        currentCameraId
      );
      setConnectionState("connecting");
      setError(null);

      // Connect to signaling server (running locally on Jetson now)
      const backendUrl = CONFIG.BACKEND_URL;
      console.log("[WebRTC] Connecting to backend:", backendUrl);
      console.log(
        "[WebRTC] Environment mode:",
        CONFIG.isProduction ? "Production" : "Development"
      );

      // Network connectivity test with environment-appropriate messaging
      if (CONFIG.isProduction) {
        console.log(
          "[WebRTC] 🌍 Production Mode: Using Railway backend for WebRTC signaling"
        );
      } else {
        console.log(
          "[WebRTC] 🌍 Development Mode: Using local backend. Are you on the same network?"
        );
      }

      const socket = io(backendUrl, {
        transports: ["websocket", "polling"],
        timeout: 20000,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      socketRef.current = socket;

      socket.on("connect", () => {
        console.log("[WebRTC] Connected to signaling server successfully");
        // Register as viewer
        console.log(
          "[WebRTC] Registering as viewer for camera:",
          currentCameraId
        );
        socket.emit("register-viewer", { cameraId: currentCameraId });
      });

      socket.on("connect_error", (error) => {
        console.error("[WebRTC] Signaling server connection error:", error);
        handleError(`Signaling server connection failed: ${error.message}`);
      });

      socket.on("disconnect", (reason) => {
        console.log("[WebRTC] Signaling server disconnected:", reason);
        if (reason === "io server disconnect") {
          // Server disconnected us, don't retry
          handleError("Signaling server disconnected");
        }
      });

      // Handle WebRTC offer from camera
      socket.on(
        "webrtc-offer",
        async (data: {
          offer: RTCSessionDescriptionInit;
          senderId: string;
        }) => {
          console.log("[WebRTC] Received offer from camera:", data.offer);
          console.log("[WebRTC] Offer SDP:", data.offer.sdp);

          try {
            console.log('[WebRTC] About to create peer connection...');
            const peerConnection = await createPeerConnection();
            console.log('[WebRTC] Peer connection created successfully');
            peerConnectionRef.current = peerConnection;

            await peerConnection.setRemoteDescription(data.offer);
            console.log("[WebRTC] Set remote description, creating answer...");
            console.log(
              "[WebRTC] Remote streams after setRemoteDescription:",
              (peerConnection as any).getRemoteStreams?.() ||
                "getRemoteStreams not available"
            );
            console.log(
              "[WebRTC] Transceivers:",
              peerConnection.getTransceivers()
            );

            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);
            console.log("[WebRTC] Created and set local description (answer)");

            socket.emit("webrtc-answer", {
              answer,
              targetId: data.senderId,
              cameraId: currentCameraId,
            });

            console.log(
              "[WebRTC] Sent answer to camera, connection state:",
              peerConnection.connectionState
            );
            console.log(
              "[WebRTC] ICE connection state:",
              peerConnection.iceConnectionState
            );
            console.log(
              "[WebRTC] ICE gathering state:",
              peerConnection.iceGatheringState
            );

            // Log all local candidates after answer
            setTimeout(() => {
              console.log("[WebRTC] 📊 ICE Connection Summary:");
              console.log(
                "  - Connection State:",
                peerConnection.connectionState
              );
              console.log(
                "  - ICE Connection State:",
                peerConnection.iceConnectionState
              );
              console.log(
                "  - ICE Gathering State:",
                peerConnection.iceGatheringState
              );
              console.log(
                "  - Signaling State:",
                peerConnection.signalingState
              );
            }, 2000);
          } catch (error) {
            handleError(
              `Failed to handle offer: ${
                error instanceof Error ? error.message : "Unknown error"
              }`
            );
          }
        }
      );

      // Handle ICE candidates from camera
      socket.on(
        "webrtc-ice-candidate",
        async (data: { candidate: RTCIceCandidateInit }) => {
          if (peerConnectionRef.current) {
            try {
              const candidateType = (data.candidate as any).type;
              const candidateInfo = {
                protocol: (data.candidate as any).protocol,
                address: (data.candidate as any).address,
                port: (data.candidate as any).port,
                type: candidateType,
                priority: (data.candidate as any).priority,
              };
              
              if (candidateType === 'host') {
                console.log('[WebRTC] 🏠 Received HOST candidate from camera:', candidateInfo);
              } else if (candidateType === 'srflx') {
                console.log('[WebRTC] 🌐 Received STUN candidate from camera:', candidateInfo);
              } else if (candidateType === 'relay') {
                console.log('[WebRTC] 🔄 Received TURN relay candidate from camera:', candidateInfo);
              } else {
                console.log('[WebRTC] Received ICE candidate from camera:', candidateInfo);
              }

              await peerConnectionRef.current.addIceCandidate(data.candidate);
              console.log("[WebRTC] Successfully added ICE candidate");
            } catch (error) {
              console.warn("[WebRTC] Failed to add ICE candidate:", error);
            }
          }
        }
      );
    } catch (error) {
      handleError(
        `WebRTC initialization failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }, [currentCameraId, createPeerConnection, handleError]);

  // Handle pending stream when video element becomes available
  useEffect(() => {
    if (pendingStream && videoRef.current) {
      console.log(
        "[WebRTC] Video element now available, applying pending stream"
      );
      console.log("[WebRTC] Stream active:", pendingStream.active);
      console.log("[WebRTC] Stream tracks:", pendingStream.getTracks());

      videoRef.current.srcObject = pendingStream;

      // Force video to load and play
      videoRef.current.load();
      videoRef.current.play().catch((error) => {
        console.warn("[WebRTC] Video play failed:", error);
      });

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

    return fullCleanup; // Use fullCleanup on unmount
  }, [currentCameraId]);

  // Auto-retry on failure with backoff
  useEffect(() => {
    if (connectionState === "failed" && currentCameraId) {
      const retryTimeout = setTimeout(() => {
        console.log("[WebRTC] 🔄 Retrying connection in 3 seconds...");
        cleanup(); // Clean up before retry
        setTimeout(() => initializeWebRTC(), 1000); // Brief pause before retry
      }, 3000);

      return () => clearTimeout(retryTimeout);
    }
  }, [connectionState, currentCameraId, initializeWebRTC, cleanup]);

  // Add video element event handlers for debugging
  const handleVideoLoadStart = () => {
    console.log("[WebRTC] Video load start");
  };

  const handleVideoLoadedMetadata = () => {
    console.log("[WebRTC] Video metadata loaded");
    if (videoRef.current) {
      console.log(
        "[WebRTC] Video dimensions:",
        videoRef.current.videoWidth,
        "x",
        videoRef.current.videoHeight
      );
    }
  };

  const handleVideoCanPlay = () => {
    console.log("[WebRTC] Video can play");
  };

  const handleVideoPlay = () => {
    console.log("[WebRTC] Video started playing");
  };

  const handleVideoError = (e: any) => {
    console.error("[WebRTC] Video error:", e.target.error);
  };

  return (
    <div className="px-2">
      <div className="aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden">
        {connectionState === "connecting" && (
          <div className="flex items-center justify-center h-full text-white">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
              <p>Connecting to camera...</p>
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-full text-red-400">
            <div className="text-center">
              <p>Connection failed</p>
              <p className="text-sm mt-1">{error}</p>
              {error?.includes("Network connectivity") && (
                <div className="text-xs mt-2 text-gray-400">
                  <p>Troubleshooting:</p>
                  <p>• Are you on the same WiFi as the camera?</p>
                  <p>• Can you ping 192.168.1.232?</p>
                  <p>• Check if ports 9, 5001-5003 are open</p>
                </div>
              )}
            </div>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          controls={false}
          className="w-full h-full object-contain"
          onLoadStart={handleVideoLoadStart}
          onLoadedMetadata={handleVideoLoadedMetadata}
          onCanPlay={handleVideoCanPlay}
          onPlay={handleVideoPlay}
          onError={handleVideoError}
        />
      </div>
    </div>
  );
};

export { WebRTCStreamPlayer };
