import { useCamera } from "../camera/CameraProvider";
import { CONFIG } from "../core/config";
import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { io, Socket } from "socket.io-client";

interface WebRTCStreamPlayerProps {
  fallback?: React.ReactNode;
  onError?: (error: string) => void;
  streamType?: 'clean' | 'annotated';  // Stream type: clean (default) or annotated (with CV overlays)
}

// WHEP fallback configuration
const WHEP_CONFIG = {
  // MediaMTX server URL - the camera publishes here via WHIP
  MEDIAMTX_URL: "http://129.80.99.75:8889",
  // Get WHEP URL for a specific camera and stream type
  getWhepUrl(cameraId: string, streamType: 'clean' | 'annotated' = 'clean') {
    const streamName = streamType === 'annotated' ? `${cameraId}-annotated` : cameraId;
    return `${this.MEDIAMTX_URL}/${streamName}/whep`;
  }
};

// Cellular connection detection helper
const detectCellularConnection = (): boolean => {
  // Method 1: Check Network Information API (if available)
  const connection = (navigator as any).connection || 
                    (navigator as any).mozConnection || 
                    (navigator as any).webkitConnection;
  
  if (connection) {
    const effectiveType = connection.effectiveType;
    const type = connection.type;
    
    console.log("[WebRTC] Network connection info:", {
      type,
      effectiveType,
      downlink: connection.downlink,
      rtt: connection.rtt
    });
    
    // Check if connection type indicates cellular
    if (type === 'cellular' || type === 'wimax') {
      console.log("[WebRTC] 📱 Cellular connection detected via Network API");
      return true;
    }
    
    // Check effective type for cellular patterns
    if (effectiveType === '2g' || effectiveType === '3g') {
      console.log("[WebRTC] 📱 Likely cellular based on effective type:", effectiveType);
      return true;
    }
  }
  
  // Method 2: Check if we're on a mobile device (heuristic)
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  
  // Method 3: Check screen size as additional hint
  const isSmallScreen = window.innerWidth <= 768;
  
  // If mobile device and not on known WiFi (no local network detection)
  if (isMobile && isSmallScreen) {
    console.log("[WebRTC] 📱 Possible cellular connection (mobile device detected)");
    // For mobile devices, default to cellular mode for safety
    return true;
  }
  
  console.log("[WebRTC] 📶 WiFi/Ethernet connection assumed");
  return false;
};

const WebRTCStreamPlayer: React.FC<WebRTCStreamPlayerProps> = ({ onError, streamType = 'clean' }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const cameraSocketIdRef = useRef<string | null>(null); // Store camera's socket ID for ICE candidates
  const [connectionState, setConnectionState] = useState<
    "connecting" | "connected" | "failed" | "disconnected"
  >("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [pendingStream, setPendingStream] = useState<MediaStream | null>(null);
  const connectionAttemptsRef = useRef<number>(0);
  const whepAttemptedRef = useRef<boolean>(false);
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

    // Clear camera socket ID (will be set again on next offer)
    cameraSocketIdRef.current = null;

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

    // Clear camera socket ID
    cameraSocketIdRef.current = null;

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

  // WHEP fallback connection - connects directly to MediaMTX server
  const connectViaWhep = useCallback(async () => {
    const whepUrl = currentCameraId
      ? WHEP_CONFIG.getWhepUrl(currentCameraId, streamType)
      : WHEP_CONFIG.getWhepUrl('jetson-camera', streamType);  // Fallback stream name

    console.log("[WHEP] 🔄 Falling back to WHEP via MediaMTX...");
    console.log("[WHEP] URL:", whepUrl);
    console.log("[WHEP] Stream type:", streamType);

    setConnectionState("connecting");
    setError(null);
    whepAttemptedRef.current = true;

    try {
      // Create peer connection with minimal config (no TURN needed - MediaMTX has public IP)
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });

      peerConnectionRef.current = pc;

      // Handle incoming video track
      pc.ontrack = (event) => {
        console.log("[WHEP] 📺 Received video track from MediaMTX");
        const stream = event.streams[0];
        if (stream && videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch((e) => console.warn("[WHEP] Play failed:", e));
          setConnectionState("connected");
          setError(null);
        }
      };

      pc.onconnectionstatechange = () => {
        console.log("[WHEP] Connection state:", pc.connectionState);
        if (pc.connectionState === "connected") {
          console.log("[WHEP] ✅ Connected to MediaMTX stream!");
          setConnectionState("connected");
        } else if (pc.connectionState === "failed") {
          console.error("[WHEP] ❌ WHEP connection failed");
          setError("WHEP connection to MediaMTX failed");
          setConnectionState("failed");
        }
      };

      // Add transceiver for receiving video
      pc.addTransceiver("video", { direction: "recvonly" });

      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          pc.onicegatheringstatechange = () => {
            if (pc.iceGatheringState === "complete") {
              resolve();
            }
          };
          // Timeout after 3 seconds
          setTimeout(resolve, 3000);
        }
      });

      console.log("[WHEP] 📤 Sending offer to MediaMTX...");

      // Send offer to MediaMTX WHEP endpoint
      const response = await fetch(whepUrl, {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: pc.localDescription?.sdp,
      });

      if (response.status !== 201) {
        throw new Error(`WHEP handshake failed: ${response.status}`);
      }

      const answerSdp = await response.text();
      console.log("[WHEP] 📥 Received answer from MediaMTX");

      await pc.setRemoteDescription({
        type: "answer",
        sdp: answerSdp,
      });

      console.log("[WHEP] ✅ WHEP handshake complete!");

    } catch (error) {
      console.error("[WHEP] ❌ WHEP connection failed:", error);
      setError(`WHEP fallback failed: ${error instanceof Error ? error.message : "Unknown error"}`);
      setConnectionState("failed");
    }
  }, [currentCameraId, streamType]);

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
          "turn:129.80.99.75:3478",                 // UDP first for best local WiFi performance
          "turn:129.80.99.75:3478?transport=tcp"   // TCP fallback for cellular
        ],
        username: username,
        credential: credential
      });
      console.log("[WebRTC] 🔄 Oracle TURN server added with UDP priority for local WiFi");
    } else {
      console.log("[WebRTC] ⚠️ TURN server skipped due to credential failure");
    }

    // Use relay-only mode on retry attempts
    const useRelayOnly = connectionAttemptsRef.current > 0;
    
    const config: RTCConfiguration = {
      iceServers,
      iceCandidatePoolSize: useRelayOnly ? 40 : 30, // More candidates for relay-only
      iceTransportPolicy: useRelayOnly ? "relay" : "all", // Force relay on retry
      bundlePolicy: "max-bundle", // Bundle everything for better relay compatibility
      rtcpMuxPolicy: "require",
    };

    if (useRelayOnly) {
      console.log("[WebRTC] 🔒 Using RELAY-ONLY mode for connection attempt", connectionAttemptsRef.current + 1);
    } else {
      console.log("[WebRTC] 🌐 Using ALL transport policies for first attempt");
    }

    const peerConnection = new RTCPeerConnection(config);

    peerConnection.onicecandidate = (event) => {
      if (event.candidate && socketRef.current && currentCameraId && cameraSocketIdRef.current) {
        console.log("[WebRTC] 🧊 BROWSER SENDING ICE candidate:", {
          type: event.candidate.type,
          protocol: event.candidate.protocol,
          address: event.candidate.address,
          port: event.candidate.port,
          priority: event.candidate.priority,
          foundation: event.candidate.foundation,
          component: event.candidate.component,
          candidate_string: event.candidate.candidate,
          targetSocketId: cameraSocketIdRef.current
        });
        socketRef.current.emit("webrtc-ice-candidate", {
          candidate: event.candidate,
          targetId: cameraSocketIdRef.current, // Use actual camera socket ID, not "camera"
          cameraId: currentCameraId,
        });
      } else if (event.candidate === null) {
        console.log("[WebRTC] 🧊 BROWSER ICE gathering completed");
      } else if (event.candidate && !cameraSocketIdRef.current) {
        console.warn("[WebRTC] ⚠️ ICE candidate generated but camera socket ID not yet known - candidate will be lost");
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
        connectionAttemptsRef.current = 0; // Reset counter on success
      } else if (state === "connecting") {
        console.log("[WebRTC] Connection in progress...");
        setConnectionState("connecting");
      } else if (state === "failed") {
        console.error(
          "[WebRTC] ❌ Connection failed - this might be a network/firewall issue"
        );
        // Increment connection attempts for relay-only retry
        connectionAttemptsRef.current += 1;
        
        if (connectionAttemptsRef.current === 1) {
          console.log("[WebRTC] 🔄 Connection failed, retrying with relay-only mode in 2 seconds...");
          handleError("Connection failed - retrying with relay-only mode");
          
          // Auto-retry with relay-only mode
          setTimeout(() => {
            if (currentCameraId) {
              console.log("[WebRTC] 🔒 Auto-retrying with relay-only mode...");
              initializeWebRTC();
            }
          }, 2000);
        } else if (!whepAttemptedRef.current) {
          // P2P and relay both failed - try WHEP fallback
          console.log("[WebRTC] ❌ Relay-only connection also failed - trying WHEP fallback...");
          cleanup();
          connectViaWhep();
        } else {
          console.log("[WebRTC] ❌ All connection methods failed (P2P, relay, WHEP)");
          handleError("Connection failed - all methods exhausted");
        }
      } else if (state === "disconnected") {
        console.warn("[WebRTC] ⚠️ Connection disconnected");
        handleError("Connection disconnected");
      }
    };

    peerConnection.oniceconnectionstatechange = () => {
      const iceState = peerConnection.iceConnectionState;
      const useRelayOnly = connectionAttemptsRef.current >= 1;
      console.log("[WebRTC] 🧊 ICE connection state changed:", iceState, useRelayOnly ? "(relay-only mode)" : "");

      if (iceState === "connected" || iceState === "completed") {
        console.log("[WebRTC] 🎉 ICE connection established successfully!");
        setConnectionState("connected");
        setError(null);
        connectionAttemptsRef.current = 0; // Reset counter on success
      } else if (iceState === "checking") {
        console.log("[WebRTC] 🔍 ICE candidates are being checked...");
        setConnectionState("connecting");
        
        // In relay-only mode, set up data flow detection instead of relying on ICE checks
        if (useRelayOnly) {
          console.log("[WebRTC] 🔒 Relay-only mode: Setting up data flow detection...");
          
          // Check for data flow after 8 seconds (longer than normal ICE timeout)
          setTimeout(() => {
            if (peerConnection.iceConnectionState === "checking" || peerConnection.iceConnectionState === "failed") {
              console.log("[WebRTC] 🔍 Relay-only mode: Checking for data flow...");
              
              peerConnection.getStats().then((stats) => {
                let hasDataFlow = false;
                let relayUsed = false;
                
                let relayPairsActive = 0;
                let relayAllocationsFound = false;
                
                stats.forEach((report) => {
                  // Check for successful or nominated candidate pairs using relay
                  if (report.type === 'candidate-pair') {
                    if (report.state === 'succeeded' || report.nominated) {
                      relayUsed = true;
                      console.log("[WebRTC] 📊 Found successful/nominated candidate pair:", {
                        state: report.state,
                        nominated: report.nominated,
                        localId: report.localCandidateId,
                        remoteId: report.remoteCandidateId
                      });
                    }
                    // Also check if relay candidates are being actively used
                    if (report.state === 'in-progress' || report.state === 'succeeded') {
                      // Check if this pair involves relay candidates
                      stats.forEach((candidate) => {
                        if ((candidate.id === report.localCandidateId || candidate.id === report.remoteCandidateId) && 
                            candidate.candidateType === 'relay' && 
                            candidate.address === '129.80.99.75') {
                          relayPairsActive++;
                          relayAllocationsFound = true;
                          console.log("[WebRTC] 📊 Active relay pair found using Oracle TURN:", {
                            candidateType: candidate.candidateType,
                            address: candidate.address,
                            port: candidate.port,
                            pairState: report.state
                          });
                        }
                      });
                    }
                  }
                  
                  // Check for any RTP data flow
                  if (report.type === 'inbound-rtp' && (report.bytesReceived > 0 || report.packetsReceived > 0)) {
                    hasDataFlow = true;
                    console.log("[WebRTC] 📊 Data flow detected:", {
                      bytes: report.bytesReceived,
                      packets: report.packetsReceived
                    });
                  }
                  
                  // Also check outbound RTP as camera might be sending
                  if (report.type === 'outbound-rtp' && (report.bytesSent > 0 || report.packetsSent > 0)) {
                    hasDataFlow = true;
                    console.log("[WebRTC] 📊 Outbound data flow detected:", {
                      bytes: report.bytesSent,
                      packets: report.packetsSent
                    });
                  }
                });
                
                console.log("[WebRTC] 📊 Relay connection analysis:", {
                  hasDataFlow,
                  relayUsed,
                  relayPairsActive,
                  relayAllocationsFound
                });
                
                if (hasDataFlow || relayUsed || relayAllocationsFound) {
                  console.log("[WebRTC] 🎉 Relay connection working despite ICE checks - bypassing ICE failure!");
                  setConnectionState("connected");
                  setError(null);
                  connectionAttemptsRef.current = 0;
                } else {
                  console.log("[WebRTC] ❌ No data flow detected in relay mode");
                  handleError("Relay connection failed - no data flow detected");
                }
              }).catch((e) => {
                console.error("[WebRTC] Failed to check data flow stats:", e);
                handleError("Could not verify relay connection");
              });
            }
          }, 8000);
        }
      } else if (iceState === "failed") {
        console.error(
          "[WebRTC] ❌ ICE connection failed - network connectivity issue"
        );
        
        if (useRelayOnly) {
          console.log("[WebRTC] 🔒 Relay-only mode failure - checking if data is actually flowing...");
          
          // Give relay more time to establish in case it's just slow
          setTimeout(() => {
            peerConnection.getStats().then((stats) => {
              let hasDataFlow = false;
              let relayAttempted = false;
              
              let relayPairsActive = 0;
              
              stats.forEach((report) => {
                // Check for any candidate pairs involving relay
                if (report.type === 'candidate-pair') {
                  if (report.localCandidateId && report.remoteCandidateId) {
                    // Check if any relay candidates were used
                    stats.forEach((candidate) => {
                      if ((candidate.id === report.localCandidateId || candidate.id === report.remoteCandidateId) && 
                          candidate.candidateType === 'relay' && 
                          candidate.address === '129.80.99.75') {
                        relayAttempted = true;
                        relayPairsActive++;
                        console.log("[WebRTC] 📊 Oracle TURN relay pair found in failure recovery:", {
                          pairState: report.state,
                          candidateType: candidate.candidateType,
                          address: candidate.address,
                          port: candidate.port
                        });
                      }
                    });
                  }
                }
                
                // Check for RTP data flow
                if (report.type === 'inbound-rtp' && (report.bytesReceived > 0 || report.packetsReceived > 0)) {
                  hasDataFlow = true;
                  console.log("[WebRTC] 📊 Inbound data flow detected despite ICE failure:", {
                    bytes: report.bytesReceived,
                    packets: report.packetsReceived
                  });
                }
                
                // Check for outbound RTP data flow  
                if (report.type === 'outbound-rtp' && (report.bytesSent > 0 || report.packetsSent > 0)) {
                  hasDataFlow = true;
                  console.log("[WebRTC] 📊 Outbound data flow detected despite ICE failure:", {
                    bytes: report.bytesSent,
                    packets: report.packetsSent
                  });
                }
              });
              
              console.log("[WebRTC] 📊 ICE failure recovery analysis:", {
                hasDataFlow,
                relayAttempted,
                relayPairsActive
              });
              
              if (hasDataFlow) {
                console.log("[WebRTC] 🎉 TURN relay working despite ICE failure - connection recovered!");
                setConnectionState("connected");
                setError(null);
                connectionAttemptsRef.current = 0;
                return;
              }
              
              // Give more time for relay connections since CoTURN logs show allocations working
              if (relayAttempted && relayPairsActive > 0) {
                console.log("[WebRTC] 🔒 Oracle TURN relay pairs found - treating as working connection despite ICE failure");
                setConnectionState("connected");
                setError(null);
                connectionAttemptsRef.current = 0;
                return;
              }
              
              if (relayAttempted) {
                console.log("[WebRTC] 🔒 Relay candidates found but no active pairs - connection truly failed");
              }
              
              // Log failed candidate pairs for debugging
              console.log("[WebRTC] 📊 ICE Connection Failure Analysis:");
              stats.forEach((report) => {
                if (report.type === 'candidate-pair') {
                  console.log("[WebRTC] 📋 Candidate pair:", {
                    state: report.state,
                    priority: report.priority,
                    nominated: report.nominated,
                    local: report.localCandidateId,
                    remote: report.remoteCandidateId
                  });
                }
                if (report.type === 'local-candidate' || report.type === 'remote-candidate') {
                  console.log(`[WebRTC] 🗳️ ${report.type}:`, {
                    id: report.id,
                    candidateType: report.candidateType,
                    protocol: report.protocol,
                    address: report.address,
                    port: report.port,
                    priority: report.priority
                  });
                }
              });
              
              handleError("Relay connection failed - TURN server not accessible");
            });
          }, 3000);
        } else {
          console.error(
            "[WebRTC] This usually means the camera and viewer cannot reach each other"
          );
          handleError("Network connectivity failed - check if devices are on same network");
        }
      } else if (iceState === "disconnected") {
        console.warn("[WebRTC] ⚠️ ICE connection disconnected");
        
        // Log candidate analysis on disconnect as well
        try {
          peerConnection.getStats().then((stats) => {
            console.log("[WebRTC] 📊 ICE Disconnect Analysis:");
            stats.forEach((report) => {
              if (report.type === 'candidate-pair') {
                console.log("[WebRTC] 📋 Candidate pair:", {
                  state: report.state,
                  priority: report.priority,
                  nominated: report.nominated,
                  local: report.localCandidateId,
                  remote: report.remoteCandidateId
                });
              }
              if (report.type === 'local-candidate' || report.type === 'remote-candidate') {
                console.log(`[WebRTC] 🗳️ ${report.type}:`, {
                  id: report.id,
                  candidateType: report.candidateType,
                  protocol: report.protocol,
                  address: report.address,
                  port: report.port,
                  priority: report.priority
                });
              }
            });
          });
        } catch (e) {
          console.log("[WebRTC] Could not get disconnect stats:", e);
        }
        
        // Give relay candidates much more time to establish connection
        console.log("[WebRTC] 🔄 Giving relay candidates extended time to connect...");
        
        setTimeout(() => {
          if (peerConnection.iceConnectionState === "disconnected") {
            console.log("[WebRTC] ⏰ ICE still disconnected after 20s timeout");
            handleError("ICE connection failed - relay candidates could not establish connection");
          }
        }, 20000); // Extended 20 second timeout for relay connections
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
        // Detect if we're on cellular by checking connection type and network info
        const isCellular = detectCellularConnection();
        
        // Register as viewer with cellular mode flag and stream type
        console.log(
          "[WebRTC] Registering as viewer for camera:",
          currentCameraId,
          "Cellular mode:", isCellular,
          "Stream type:", streamType
        );
        socket.emit("register-viewer", {
          cameraId: currentCameraId,
          cellularMode: isCellular,
          streamType: streamType  // 'clean' (default) or 'annotated'
        });
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
          console.log("[WebRTC] Camera socket ID:", data.senderId);

          // Store camera's socket ID for ICE candidate routing
          cameraSocketIdRef.current = data.senderId;

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
  }, [currentCameraId, createPeerConnection, handleError, streamType]);

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
