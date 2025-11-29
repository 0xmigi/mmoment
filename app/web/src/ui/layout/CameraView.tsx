/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */

import { IDL } from "../../anchor/idl";
import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from "../../anchor/setup";
import { TransactionModal } from "../../auth/components/TransactionModal";
import { CameraModal } from "../../camera/CameraModal";
import { useCamera, CameraData } from "../../camera/CameraProvider";
import { IRLAppsButton } from "../../camera/IRLAppsButton";
import { CompetitionScoreboard } from "../../camera/CompetitionScoreboard";
import { CompetitionControls } from "../../camera/CompetitionControls";
import { cameraStatus } from "../../camera/camera-status";
import { unifiedCameraService } from "../../camera/unified-camera-service";
import { useCameraStatus } from "../../camera/useCameraStatus";
import { unifiedCameraPolling, CameraStatusData } from "../../camera/unified-camera-polling";
import { CONFIG } from "../../core/config";
import { ToastMessage } from "../../core/types/toast";
import MediaGallery from "../../media/Gallery";
import { StreamPlayer } from "../../media/StreamPlayer";
import { unifiedIpfsService } from "../../storage/ipfs/unified-ipfs-service";
import { Timeline } from "../../timeline/Timeline";
import { timelineService } from "../../timeline/timeline-service";
import {
  TimelineEvent,
  TimelineEventType,
} from "../../timeline/timeline-types";
import { ToastContainer } from "../feedback/ToastContainer";
import { CameraControls } from "./MobileControls";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import {
  useDynamicContext,
  useEmbeddedWallet,
} from "@dynamic-labs/sdk-react-core";
import { useConnection } from "@solana/wallet-adapter-react";
import {
  PublicKey,
  Connection,
  Transaction,
  TransactionInstruction,
} from "@solana/web3.js";
import {
  StopCircle,
  Play,
  Camera,
  Video,
  Loader,
  Link2,
  CheckCircle,
} from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { useParams } from "react-router-dom";

// Update the CameraIdDisplay component to add a forced refresh when the modal is closed
const CameraIdDisplay = ({
  cameraId,
  selectedCamera,
  cameraAccount,
  timelineRef,
}: {
  cameraId: string | undefined;
  selectedCamera: CameraData | null;
  cameraAccount: string | null;
  timelineRef?: React.RefObject<{
    refreshTimeline?: () => void;
    refreshEvents?: () => void;
  }>;
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const cameraStatus = useCameraStatus(
    selectedCamera?.publicKey || cameraAccount || cameraId || ""
  );
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [isCheckedIn, setIsCheckedIn] = useState(false);

  // Get the direct ID from localStorage if available (most reliable source)
  const directId = localStorage.getItem("directCameraId");

  // Determine which ID to display (in order of preference)
  const displayId =
    selectedCamera?.publicKey || cameraAccount || directId || cameraId;

  // Simple blockchain check
  const checkBlockchainStatus = async () => {
    if (!displayId || !primaryWallet?.address || !connection) return;

    try {
      console.log(
        `[BLOCKCHAIN CHECK] Checking status for ${displayId.slice(0, 8)}...`
      );

      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: "confirmed" }
      );

      const program = new Program(
        IDL as any,
        CAMERA_ACTIVATION_PROGRAM_ID,
        provider
      );

      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("session"),
          new PublicKey(primaryWallet.address).toBuffer(),
          new PublicKey(displayId).toBuffer(),
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      try {
        const session = await (program.account as any).userSession.fetch(sessionPda);
        console.log(`[BLOCKCHAIN CHECK] ✅ SESSION FOUND:`, session);
        setIsCheckedIn(true);
      } catch (err) {
        console.log(`[BLOCKCHAIN CHECK] ❌ NO SESSION FOUND`);
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error("[BLOCKCHAIN CHECK] ERROR:", err);
    }
  };

  // Check blockchain status immediately when component loads or camera changes
  useEffect(() => {
    if (displayId && primaryWallet?.address && connection) {
      console.log(
        `[CAMERA ID DISPLAY] Component loaded/changed - checking blockchain status`
      );
      checkBlockchainStatus();
    }
  }, [displayId, primaryWallet?.address, connection]);

  // Add a function to handle status change from modal
  const handleCheckStatusChange = (newStatus: boolean) => {
    console.log("Status change from modal:", newStatus);
    setIsCheckedIn(newStatus);
    // If timelineRef exists, refresh it
    if (timelineRef?.current?.refreshTimeline) {
      timelineRef.current?.refreshTimeline();
    }
  };

  // Add a function to handle modal close with a forced refresh
  const handleModalClose = () => {
    setIsModalOpen(false);
    // Check blockchain status when modal closes
    checkBlockchainStatus();
  };

  // Handle case where ID might not be valid
  const formatId = (id: string | undefined | null) => {
    if (!id) return "None";
    try {
      return `${id.slice(0, 6)}...${id.slice(-6)}`;
    } catch (_) {
      return id;
    }
  };

  // The default camera PDA for development
  const defaultDevCameraPda = "EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA";

  return (
    <div>
      <h2 className="text-xl font-semibold">Camera</h2>
      {!displayId || displayId === "None" ? (
        <div
          onClick={() => setIsModalOpen(true)}
          className="text-sm text-red-500 font-medium hover:text-red-600 cursor-pointer"
        >
          No camera connected
        </div>
      ) : (
        <div
          onClick={() => setIsModalOpen(true)}
          className="text-sm text-gray-600 hover:text-blue-600 transition-colors cursor-pointer flex items-center"
        >
          <span>id: {formatId(displayId)}</span>
          <span id="check-in-status-icon">
            {isCheckedIn ? (
              <CheckCircle className="w-3.5 h-3.5 ml-1.5 text-green-500" />
            ) : (
              <Link2 className="w-3.5 h-3.5 ml-1.5 text-blue-500" />
            )}
          </span>
        </div>
      )}

      <CameraModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        onCheckStatusChange={handleCheckStatusChange}
        camera={{
          id: displayId && displayId !== "None" ? displayId : "",
          owner: selectedCamera?.owner || cameraStatus.owner || "",
          ownerDisplayName:
            selectedCamera?.metadata?.name || "newProgramCamera",
          model: selectedCamera?.metadata?.model || "pi5",
          isLive: cameraStatus.isLive || false,
          isStreaming: cameraStatus.isStreaming || false,
          status: "ok",

          // Add development info when no camera is connected
          showDevInfo: !displayId || displayId === "None",
          defaultDevCamera: defaultDevCameraPda,
        }}
      />
    </div>
  );
};

export function CameraView() {
  const { primaryWallet, user } = useDynamicContext();
  const { cameraId } = useParams<{ cameraId: string }>();
  useEmbeddedWallet();
  const { selectedCamera, setSelectedCamera, fetchCameraById } = useCamera();
  const { program } = useProgram();
  const { connection } = useConnection();
  const timelineRef = useRef<{
    addEvent?: (event: Omit<TimelineEvent, "id">) => void;
    refreshTimeline?: () => void;
    refreshEvents?: () => void;
  }>(null);
  const mobileTimelineRef = useRef<{
    addEvent?: (event: Omit<TimelineEvent, "id">) => void;
    refreshTimeline?: () => void;
    refreshEvents?: () => void;
  }>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [currentToast, setCurrentToast] = useState<ToastMessage | null>(null);
  const [loading] = useState(false);
  const [, setIsMobileView] = useState(window.innerWidth <= 768);
  const [isCheckedIn, setIsCheckedIn] = useState(false);

  // Add state to store video recording transaction signature
  const [_recordingTransactionSignature] = useState<string | null>(null);

  // Add state for gesture monitoring
  const [gestureMonitoring, setGestureMonitoring] = useState(false);
  const gestureCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Add state to track gesture controls status changes
  const [gestureControlsEnabled, setGestureControlsEnabled] = useState(false);

  // Add state for mobile camera modal
  const [isMobileCameraModalOpen, setIsMobileCameraModalOpen] = useState(false);

  // Helper function to detect if we're using the Jetson camera
  // const isJetsonCamera = (cameraId: string | null): boolean => {
  //   return cameraId === CONFIG.JETSON_CAMERA_PDA;
  // };

  // Update the states for TransactionModal
  const [showTransactionModal, setShowTransactionModal] = useState(false);
  const [transactionData, setTransactionData] = useState<{
    type: "photo" | "video" | "stream" | "initialize";
    cameraAccount: string;
  } | null>(null);

  // Remove local isStreaming state - query hardware instead
  const [hardwareState, setHardwareState] = useState<{
    isStreaming: boolean;
    isRecording: boolean;
    lastUpdated: number;
  }>({
    isStreaming: false,
    isRecording: false,
    lastUpdated: 0,
  });

  // Removed: Hardware state is now polled via unified service

  // Add camera-specific status hook to the main CameraView function
  const currentCameraId =
    cameraAccount || selectedCamera?.publicKey || cameraId || "";
  const currentCameraStatus = useCameraStatus(currentCameraId);

  // Add a function to create timeline events with Farcaster profile info
  const addTimelineEvent = (
    eventType: TimelineEventType,
    transactionId?: string,
    mediaUrl?: string
  ) => {
    if (primaryWallet && user) {
      const recentEventsKey = "recentTimelineEvents";

      try {
        // Get recent events from localStorage to check for duplicates
        const recentEventsStr = localStorage.getItem(recentEventsKey) || "[]";
        const recentEvents = JSON.parse(recentEventsStr);

        // Check if we've added this event recently (within last 30 seconds)
        const now = Date.now();
        const duplicateEvent = recentEvents.find(
          (event: any) =>
            event.type === eventType &&
            event.transactionId === transactionId &&
            now - event.timestamp < 30000 // 30 seconds
        );

        if (duplicateEvent) {
          console.log(
            "Skipping duplicate timeline event:",
            eventType,
            transactionId
          );
          return;
        }

        // Clean up old events (older than 5 minutes)
        const cleanedEvents = recentEvents.filter(
          (event: any) => now - event.timestamp < 300000 // 5 minutes
        );

        // Add this event to recent events
        cleanedEvents.push({
          type: eventType,
          transactionId,
          timestamp: now,
        });

        localStorage.setItem(recentEventsKey, JSON.stringify(cleanedEvents));
      } catch (e) {
        console.warn("Error checking for duplicate events:", e);
      }

      // Get the user's social credentials - prioritize Farcaster over Twitter
      const farcasterCred = user?.verifiedCredentials?.find(
        (cred) => cred.oauthProvider === "farcaster"
      );
      const twitterCred = user?.verifiedCredentials?.find(
        (cred) => cred.oauthProvider === "twitter"
      );

      // Use Farcaster if available, otherwise Twitter
      const socialCred = farcasterCred || twitterCred;

      // Create the timeline event with enriched user info
      // Try social accounts first, then fallback to Dynamic user profile
      const event: Omit<TimelineEvent, "id"> = {
        type: eventType,
        user: {
          address: primaryWallet.address,
          // Include profile info - prioritize social accounts, fallback to Dynamic user (NEVER use email)
          displayName: socialCred?.oauthDisplayName || user?.alias || undefined,
          username: socialCred?.oauthUsername || user?.username || undefined,
          pfpUrl: socialCred?.oauthAccountPhotos?.[0] || undefined,
          provider: socialCred?.oauthProvider,
        },
        timestamp: Date.now(),
        transactionId,
        mediaUrl,
        cameraId: cameraAccount || undefined,
      };

      console.log("Adding timeline event:", {
        type: event.type,
        transactionId: event.transactionId
          ? `${event.transactionId.slice(0, 8)}...`
          : "none",
        mediaUrl: event.mediaUrl ? "present" : "none",
        cameraId: event.cameraId,
        userInfo: {
          address: event.user.address,
          displayName: event.user.displayName || "(none)",
          username: event.user.username || "(none)",
          hasPfp: !!event.user.pfpUrl,
        },
      });

      // Emit the event to the timeline service
      timelineService.emitEvent(event);
    }
  };

  // Load camera from URL params if available - simplify to just set the ID
  useEffect(() => {
    if (!cameraId) return;

    try {
      // Log the camera ID (decoding it first)
      const decodedCameraId = decodeURIComponent(cameraId);
      console.log(`[CameraView] Using camera ID: ${decodedCameraId}`);

      // Store in localStorage for persistence
      localStorage.setItem("directCameraId", decodedCameraId);

      // Set camera account state
      setCameraAccount(decodedCameraId);

      // Attempt to fetch camera data if possible (but don't show errors if it fails)
      if (fetchCameraById) {
        fetchCameraById(decodedCameraId)
          .then((camera) => {
            if (camera) {
              console.log(`[CameraView] Camera data loaded:`, camera);
              setSelectedCamera(camera);
            }
          })
          .catch((err) => {
            // Just log the error but don't show to the user
            console.warn(
              `[CameraView] Non-critical error loading camera data:`,
              err
            );
          });
      }
    } catch (error) {
      console.error("[CameraView] Error processing camera ID:", error);
    }
  }, [cameraId, fetchCameraById, setSelectedCamera]);

  // Clear camera account when navigating to the default /app route
  useEffect(() => {
    // If we're on the default /app route (no cameraId in URL), clear the camera account
    if (!cameraId && window.location.pathname === "/app") {
      setCameraAccount(null);
      // Also clear the selected camera in the provider if needed
      if (selectedCamera) {
        setSelectedCamera(null);
      }
    }
  }, [cameraId, setSelectedCamera, selectedCamera]);

  // Also update cameraAccount whenever selectedCamera changes
  useEffect(() => {
    if (selectedCamera) {
      setCameraAccount(selectedCamera.publicKey);
    }
  }, [selectedCamera]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setIsMobileView(window.innerWidth <= 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Check if the user is using an embedded wallet
  useEffect(() => {
    if (primaryWallet) {
      const isEmbedded =
        primaryWallet.connector?.name.toLowerCase() !== "phantom";
      if (isEmbedded) {
        console.log("Using embedded wallet:", primaryWallet.connector?.name);
      }
    }
  }, [primaryWallet]);

  // Set wallet on CameraRegistry for request signing (ed25519 authentication)
  useEffect(() => {
    if (primaryWallet) {
      const { CameraRegistry } = require("../../camera/camera-registry");
      CameraRegistry.getInstance().setWallet(primaryWallet);
      console.log("[CameraView] Wallet set for request signing");
    }
  }, [primaryWallet]);

  // Sync UI state with actual camera state when component loads or camera changes
  // Uses unified polling service to avoid duplicate API calls
  useEffect(() => {
    if (!cameraAccount) return;

    console.log(`[CameraView] Syncing state for camera: ${cameraAccount}`);

    // Check if camera exists in registry
    if (!unifiedCameraService.hasCamera(cameraAccount)) {
      console.log(
        `[CameraView] Camera not in registry, skipping state sync`
      );
      return;
    }

    // Force a fresh check from unified polling (will reuse pending request if one exists)
    unifiedCameraPolling.forceCheck(cameraAccount).then((status) => {
      console.log(`[CameraView] Camera status from unified polling:`, status);
      // Recording state is managed by hardware state subscription
    }).catch((error) => {
      console.error(`[CameraView] Error syncing camera state:`, error);
    });
  }, [cameraAccount]);

  // Handle camera update from ActivateCamera component

  const updateToast = (type: "success" | "error" | "info", message: string) => {
    // Skip showing "Failed to fetch" network errors as toasts
    if (
      type === "error" &&
      (message.includes("Failed to fetch") ||
        message.includes("Network error") ||
        message.includes("Camera error") ||
        message.includes("CORS"))
    ) {
      console.warn("Suppressing network error toast:", message);
      return;
    }

    const id = Date.now().toString();
    setCurrentToast({ id, type, message });
  };

  const dismissToast = () => {
    setCurrentToast(null);
  };

  // Helper function to convert action type to event type
  const getEventType = (actionType: string): TimelineEventType => {
    switch (actionType) {
      case "photo":
        return "photo_captured";
      case "video":
        return "video_recorded";
      case "stream":
        return currentCameraStatus.isStreaming
          ? "stream_ended"
          : "stream_started";
      case "stream_start":
        return "stream_started";
      case "stream_stop":
        return "stream_ended";
      case "face_enrollment":
        return "face_enrolled";
      default:
        return "initialization"; // Changed from 'photo_captured' to avoid fake photo entries
    }
  };

  // Subscribe to unified polling service for hardware state (no separate polling needed)
  useEffect(() => {
    if (!cameraAccount) return;

    console.log('[CameraView] Subscribing to unified polling for hardware state');

    const unsubscribe = unifiedCameraPolling.subscribe(cameraAccount, (status: CameraStatusData) => {
      setHardwareState({
        isStreaming: status.isStreaming,
        isRecording: false, // TODO: Add isRecording to unified polling status
        lastUpdated: Date.now(),
      });
    });

    // Cleanup
    return () => {
      console.log('[CameraView] Unsubscribing from unified polling');
      unsubscribe();
    };
  }, [cameraAccount]);

  // Button handlers

  // Debug logs for program and connection
  useEffect(() => {
    console.log("Program initialized:", !!program);
    console.log(
      "Program details:",
      program
        ? {
            programId: program.programId.toString(),
            provider: !!program.provider,
          }
        : "No program"
    );

    console.log("Connection initialized:", !!connection);
    console.log(
      "Connection details:",
      connection
        ? {
            rpcEndpoint: connection.rpcEndpoint,
          }
        : "No connection"
    );
  }, [program, connection]);

  // More detailed error logging
  const logDetailedError = (error: any, context: string) => {
    console.error(`Error in ${context}:`, error);
    if (error instanceof Error) {
      console.error(`Name: ${error.name}, Message: ${error.message}`);
      console.error(`Stack: ${error.stack}`);
    } else {
      console.error(`Unknown error type: ${typeof error}`);
    }
  };

  // Update the Simple direct transaction function to return the signature and work with the existing Solana program
  const sendSimpleTransaction = async (
    actionType: string
  ): Promise<string | undefined> => {
    console.log("DIRECT TRANSACTION FUNCTION - Type:", actionType);
    let retryCount = 0;
    const MAX_RETRIES = 3;
    let currentConnection = connection;

    while (retryCount < MAX_RETRIES) {
      try {
        // Check for wallet, program, and connection
        if (!primaryWallet || !program || !currentConnection) {
          console.error("Missing required components for transaction");
          const missing = [];
          if (!primaryWallet) missing.push("wallet");
          if (!program) missing.push("program");
          if (!currentConnection) missing.push("connection");
          updateToast(
            "error",
            `Cannot send transaction: missing ${missing.join(", ")}`
          );
          return undefined;
        }

        // Get camera ID
        const cameraId =
          cameraAccount || localStorage.getItem("directCameraId");
        if (!cameraId) {
          console.error("No camera ID found");
          updateToast("error", "No camera ID found");
          return undefined;
        }

        // First, verify user is checked in
        const isCheckedInNow = await checkUserSession();
        if (!isCheckedInNow) {
          updateToast(
            "error",
            "You need to check in before performing camera actions"
          );
          return undefined;
        }

        // Create a simplified transaction that will update the session's lastActivity timestamp
        // This is the minimally invasive approach that works with the current program
        try {
          // Get recent blockhash with retry logic
          let blockhash;
          try {
            const { blockhash: newBlockhash } =
              await currentConnection.getLatestBlockhash("finalized");
            blockhash = newBlockhash;
          } catch (bhError) {
            console.error("Error getting blockhash:", bhError);
            const nextEndpoint = CONFIG.getNextEndpoint();
            console.log(`Switching to RPC endpoint: ${nextEndpoint}`);
            currentConnection = new Connection(nextEndpoint, "confirmed");
            throw new Error(
              "Failed to get blockhash, retrying with new endpoint"
            );
          }

          // Create a basic transaction that sends a minimal amount of SOL to yourself
          // This serves as a placeholder transaction that will be recorded on-chain
          const userPublicKey = new PublicKey(primaryWallet.address);
          const cameraPublicKey = new PublicKey(cameraId);

          // Find the session PDA

          const [_sessionPda] = PublicKey.findProgramAddressSync(
            [
              Buffer.from("session"),
              userPublicKey.toBuffer(),
              cameraPublicKey.toBuffer(),
            ],
            CAMERA_ACTIVATION_PROGRAM_ID
          );

          // Create a transaction with a memo instruction that identifies the camera action
          const memoProgram = new PublicKey(
            "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
          );

          // Create the memo data with the action type for on-chain recording
          const memoData = Buffer.from(
            `camera:${actionType}:${new Date().toISOString()}`
          );

          // Create the memo instruction
          const memoInstruction = new TransactionInstruction({
            keys: [{ pubkey: userPublicKey, isSigner: true, isWritable: true }],
            programId: memoProgram,
            data: memoData,
          });

          // Create the transaction
          const tx = new Transaction();
          tx.add(memoInstruction);

          // Add blockhash and payer
          tx.recentBlockhash = blockhash;
          tx.feePayer = userPublicKey;

          // Sign and send the transaction
          const signer = await (primaryWallet as any).getSigner();
          const signedTx = await signer.signTransaction(tx);
          const signature = await currentConnection.sendRawTransaction(
            signedTx.serialize()
          );

          // Wait for confirmation
          await currentConnection.confirmTransaction(signature, "confirmed");

          console.log(`Transaction confirmed: ${signature}`);
          updateToast(
            "success",
            `${actionType} transaction sent: ${signature.slice(0, 8)}...`
          );

          // Add the event to the timeline
          addTimelineEvent(getEventType(actionType), signature);

          return signature;
        } catch (error) {
          console.error("Error in transaction:", error);
          logDetailedError(error, "Transaction error");

          // Check if it's a blockhash error
          const errorMessage =
            error instanceof Error ? error.message : String(error);

          if (
            errorMessage.includes("Blockhash not found") ||
            errorMessage.includes("block height exceeded") ||
            errorMessage.includes("timeout")
          ) {
            retryCount++;
            if (retryCount < MAX_RETRIES) {
              console.log(
                `Retrying transaction (attempt ${
                  retryCount + 1
                }/${MAX_RETRIES})`
              );
              // Switch to next RPC endpoint
              const nextEndpoint = CONFIG.getNextEndpoint();
              console.log(`Switching to RPC endpoint: ${nextEndpoint}`);
              currentConnection = new Connection(nextEndpoint, "confirmed");
              continue;
            }
          }

          updateToast("error", `Transaction failed: ${errorMessage}`);
          return undefined;
        }
      } catch (error) {
        console.error("Outer error in transaction:", error);
        logDetailedError(error, "Outer transaction error");

        retryCount++;
        if (retryCount < MAX_RETRIES) {
          console.log(
            `Retrying entire transaction process (attempt ${
              retryCount + 1
            }/${MAX_RETRIES})`
          );
          continue;
        }

        updateToast(
          "error",
          `Error: ${error instanceof Error ? error.message : "Unknown error"}`
        );
        return undefined;
      }
    }

    updateToast("error", "Transaction failed after maximum retries");
    return undefined;
  };

  // Expose sendSimpleTransaction to window for Pi5Camera to use
  useEffect(() => {
    (window as any).sendSimpleTransaction = sendSimpleTransaction;

    // Cleanup on unmount
    return () => {
      delete (window as any).sendSimpleTransaction;
    };
  }, [sendSimpleTransaction]);

  // Replace the promptForCheckIn function with one that shows the modal
  const promptForCheckIn = (actionType: "photo" | "video" | "stream") => {
    if (!cameraAccount) {
      updateToast("error", "No camera connected");
      return false;
    }

    // Set transaction data and show modal
    setTransactionData({
      type: actionType,
      cameraAccount: cameraAccount,
    });
    setShowTransactionModal(true);

    // Track this as an "attempt"
    console.log(
      `User attempted to use camera for ${actionType} without checking in first`
    );

    return false;
  };

  // Update the checkUserSession function with similar throttling
  const checkUserSession = async (): Promise<boolean> => {
    if (!primaryWallet?.address || !cameraAccount || !program || !connection) {
      console.error("Missing required components for check-in status");
      return false;
    }

    try {
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(cameraAccount);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("session"),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer(),
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      try {
        // Try to fetch the session account
        const sessionAccount = await (program.account as any).userSession.fetch(
          sessionPda
        );
        if (sessionAccount) {
          if (!isCheckedIn) {
            console.log("[CameraView] Setting checked-in status to TRUE");
            setIsCheckedIn(true);
            // Refresh timeline if status changed
            if (timelineRef.current?.refreshTimeline) {
              timelineRef.current?.refreshTimeline();
            }
          }
          return true;
        }
      } catch (_) {
        console.log(
          "[CameraView] Session account not found, user is not checked in"
        );
        if (isCheckedIn) {
          console.log("[CameraView] Setting checked-in status to FALSE");
          setIsCheckedIn(false);
          // Refresh timeline if status changed
          if (timelineRef.current?.refreshTimeline) {
            timelineRef.current?.refreshTimeline();
          }
        }
        return false;
      }
    } catch (err) {
      console.error("[CameraView] Error checking session status:", err);
      // Don't update state on error to prevent UI flashing
      return isCheckedIn; // Return current state on error
    }

    return isCheckedIn;
  };

  // Update the periodic check-in status check to use a longer interval
  useEffect(() => {
    if (!primaryWallet?.address || !cameraAccount) return;

    console.log("[CameraView] Setting up check-in status monitoring");

    // Check status immediately
    checkUserSession();

    // Set up periodic check every 10 seconds (instead of 3)
    const intervalId = setInterval(() => {
      checkUserSession().then((isCheckedIn) => {
        console.log(
          "[CameraView] Periodic check result:",
          isCheckedIn ? "CHECKED IN" : "NOT CHECKED IN"
        );
      });
    }, 10000);

    // Clean up on unmount
    return () => {
      console.log("[CameraView] Cleaning up check-in status monitor");
      clearInterval(intervalId);
    };
  }, [primaryWallet, cameraAccount, program, connection]);

  // Sync gesture controls state with localStorage
  useEffect(() => {
    const updateGestureControlsState = async () => {
      const enabled = await unifiedCameraService.getGestureControlsStatus();
      console.log(
        "[CameraView] Gesture controls status from localStorage:",
        enabled
      );
      setGestureControlsEnabled(enabled);
    };

    // Initial sync
    updateGestureControlsState();

    // Listen for storage changes (when other tabs/components update localStorage)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === "jetson_gesture_controls_enabled") {
        updateGestureControlsState();
      }
    };

    window.addEventListener("storage", handleStorageChange);

    // Also check periodically in case localStorage is updated by the same tab
    const intervalId = setInterval(updateGestureControlsState, 1000);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(intervalId);
    };
  }, []);

  // Gesture monitoring effect for Jetson cameras
  useEffect(() => {
    // Only monitor gestures if:
    // 1. User is checked in
    // 2. Using Jetson camera
    // 3. Gesture controls are enabled
    const isJetson =
      cameraAccount && unifiedCameraService.hasCamera(cameraAccount);

    console.log("[CameraView] Gesture monitoring conditions:", {
      isCheckedIn,
      cameraAccount,
      isJetson,
      gestureControlsEnabled,
      gestureMonitoring,
    });

    const shouldMonitorGestures =
      isCheckedIn && cameraAccount && isJetson && gestureControlsEnabled;

    console.log("[CameraView] Should monitor gestures:", shouldMonitorGestures);

    // Clear any existing interval first
    if (gestureCheckIntervalRef.current) {
      console.log(
        "[CameraView] 🧹 Clearing existing gesture monitoring interval"
      );
      clearInterval(gestureCheckIntervalRef.current);
      gestureCheckIntervalRef.current = null;
    }

    if (shouldMonitorGestures) {
      console.log(
        "[CameraView] ✅ STARTING GESTURE MONITORING - All conditions met!"
      );

      // Don't use the gestureMonitoring state to control the interval
      // Just start it directly when conditions are met
      let lastGestureAction: string | null = null;
      let gestureActionCooldown = false;

      console.log("[CameraView] 🔄 Setting up gesture check interval...");
      gestureCheckIntervalRef.current = setInterval(async () => {
        try {
          console.log("[CameraView] 👀 Checking for gesture trigger...");
          const gestureCheck =
            await unifiedCameraService.checkForGestureTrigger(cameraAccount);
          console.log("[CameraView] 📊 Gesture check result:", gestureCheck);

          if (gestureCheck.shouldCapture && !gestureActionCooldown) {
            const gesture = gestureCheck.gesture;
            console.log(`[CameraView] 🎯 GESTURE TRIGGER DETECTED: ${gesture}`);

            // Prevent the same gesture from triggering multiple times
            if (lastGestureAction !== gesture) {
              lastGestureAction = gesture || null;
              gestureActionCooldown = true;

              // Trigger the appropriate action based on gesture
              if (gestureCheck.gestureType === "photo") {
                updateToast(
                  "info",
                  `📸 Gesture detected: ${gesture} - Taking photo...`
                );
                await handleDirectPhoto();
              } else if (gestureCheck.gestureType === "video") {
                updateToast(
                  "info",
                  `🎥 Gesture detected: ${gesture} - Recording video...`
                );
                await handleDirectVideo();
              }

              // Reset cooldown after 2 seconds
              setTimeout(() => {
                gestureActionCooldown = false;
                lastGestureAction = null;
              }, 2000);
            }
          }
        } catch (error) {
          console.error("[CameraView] Error checking gesture trigger:", error);
        }
      }, 500);

      console.log("[CameraView] ✅ Gesture monitoring interval started!");

      // Update the state to reflect that monitoring is active
      if (!gestureMonitoring) {
        setGestureMonitoring(true);
      }
    } else {
      console.log("[CameraView] ❌ Gesture monitoring conditions not met");
      console.log("[CameraView] Conditions:", {
        isCheckedIn,
        cameraAccount,
        isJetson,
        gestureControlsEnabled,
      });

      // Update the state to reflect that monitoring is inactive
      if (gestureMonitoring) {
        setGestureMonitoring(false);
      }
    }

    // Cleanup on unmount
    return () => {
      if (gestureCheckIntervalRef.current) {
        console.log(
          "[CameraView] 🧹 Cleaning up gesture monitoring on unmount"
        );
        clearInterval(gestureCheckIntervalRef.current);
        gestureCheckIntervalRef.current = null;
      }
    };
  }, [isCheckedIn, cameraAccount, gestureControlsEnabled]);

  // Update the button handlers to check for check-in status
  const handleDirectPhoto = async () => {
    if (!cameraAccount && !selectedCamera) {
      updateToast(
        "error",
        "No camera connected. Please connect to a camera first."
      );
      return;
    }

    const currentCameraId = cameraAccount || selectedCamera?.publicKey;
    if (!currentCameraId) {
      updateToast("error", "No camera ID available.");
      return;
    }

    if (!unifiedCameraService.hasCamera(currentCameraId)) {
      updateToast("error", "Camera not found in registry. Please reconnect.");
      return;
    }

    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn("photo");
    }

    try {
      updateToast("info", "Taking photo...");

      // Note: Pre-capture transactions removed - activities are now buffered on Jetson
      // and committed at checkout via the encrypted activity buffer system

      const isConnected = await unifiedCameraService.isConnected(
        currentCameraId
      );
      if (!isConnected && primaryWallet?.address) {
        await unifiedCameraService.connect(
          currentCameraId,
          primaryWallet.address
        );
      }

      const response = await unifiedCameraService.takePhoto(currentCameraId);

      if (response.success && response.data?.blob) {
        updateToast("info", "Photo captured, uploading to IPFS...");

        try {
          const results = await unifiedIpfsService.uploadFile(
            response.data.blob,
            primaryWallet?.address || "",
            "image",
            {
              cameraId: currentCameraId,
            }
          );

          if (results.length > 0) {
            updateToast("success", "Photo captured and uploaded to IPFS");
            // Note: Timeline event is handled by Jetson's timeline_activity_service
            // via buffer_photo_activity() - no frontend emission needed
            cameraStatus.setOnline(false);
          } else {
            updateToast("success", "Photo captured (upload to IPFS failed)");
          }
        } catch (uploadError) {
          updateToast("success", "Photo captured (upload to IPFS failed)");
        }

        // Refresh timeline to show event buffered by Jetson
        if (timelineRef.current?.refreshEvents) {
          timelineRef.current?.refreshEvents();
        }
      } else {
        updateToast(
          "error",
          `Failed to capture photo: ${response.error || "Unknown error"}`
        );
        // Note: Don't emit "photo_captured" when capture FAILED - the photo wasn't captured!
        // Jetson only buffers activities for successful captures
      }
    } catch (error) {
      updateToast(
        "error",
        `Error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  };

  const handleDirectVideo = async () => {
    if (isRecording) {
      return;
    }

    if (!cameraAccount && !selectedCamera) {
      updateToast(
        "error",
        "No camera connected. Please connect to a camera first."
      );
      return;
    }

    const currentCameraId = cameraAccount || selectedCamera?.publicKey;
    if (!currentCameraId) {
      updateToast("error", "No camera ID available.");
      return;
    }

    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn("video");
    }

    if (!primaryWallet?.address) {
      updateToast("error", "Wallet not connected");
      return;
    }

    // Note: Pre-capture transactions removed - activities are now buffered on Jetson
    // and committed at checkout via the encrypted activity buffer system

    try {
      setIsRecording(true);
      updateToast("info", "Starting video recording...");

      const isConnected = await unifiedCameraService.isConnected(
        currentCameraId
      );
      if (!isConnected && primaryWallet?.address) {
        await unifiedCameraService.connect(
          currentCameraId,
          primaryWallet.address
        );
      }

      const recordResponse = await unifiedCameraService.startVideoRecording(
        currentCameraId
      );

      if (!recordResponse.success) {
        throw new Error(`Failed to start recording: ${recordResponse.error}`);
      }

      updateToast("success", "Video recording started");

      // Wait for recording to complete - give it much more time
      let attempts = 0;
      const maxAttempts = 40; // 2 minutes max with 3 second intervals

      const checkRecordingStatus = async (): Promise<void> => {
        attempts++;

        if (attempts >= maxAttempts) {
          updateToast("error", "Recording timeout - stopping manually");
          setIsRecording(false);
          return;
        }

        const statusResponse = await unifiedCameraService.getStatus(
          currentCameraId
        );
        if (
          statusResponse.success &&
          statusResponse.data &&
          !statusResponse.data.isRecording
        ) {
          updateToast(
            "info",
            "Recording completed, waiting for video processing..."
          );

          // Wait additional time for video processing before trying to fetch
          await new Promise((resolve) => setTimeout(resolve, 8000));

          // Recording has naturally stopped, so try to get the most recent video instead
          const stopResponse = await unifiedCameraService.getMostRecentVideo(
            currentCameraId
          );

          if (stopResponse.success && stopResponse.data?.blob) {
            const videoBlob = stopResponse.data.blob;

            if (videoBlob.size < 100000) {
              updateToast(
                "error",
                "Video file appears to be corrupted. Try recording again."
              );
              setIsRecording(false);
              return;
            }

            updateToast("info", "Video processed, uploading to IPFS...");

            try {
              const ipfsResult = await unifiedIpfsService.uploadFile(
                videoBlob,
                primaryWallet.address,
                "video",
                {
                  cameraId: currentCameraId,
                }
              );

              if (ipfsResult && ipfsResult.length > 0) {
                updateToast("success", "Video uploaded to IPFS successfully!");
              } else {
                updateToast("error", "Video recorded but IPFS upload failed");
              }
            } catch (ipfsError) {
              updateToast("error", "Video recorded but IPFS upload failed");
            }
          } else {
            updateToast(
              "error",
              "Recording completed but no video file was created"
            );
          }
          setIsRecording(false);
        } else {
          setTimeout(checkRecordingStatus, 3000);
        }
      };

      // Wait longer before first check to give camera time to actually start recording
      setTimeout(checkRecordingStatus, 10000);
    } catch (error) {
      updateToast(
        "error",
        `Recording failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
      setIsRecording(false);
    }
  };

  const handleDirectStream = async () => {
    const currentCameraId =
      cameraAccount || selectedCamera?.publicKey || CONFIG.JETSON_CAMERA_PDA;
    if (!currentCameraId) {
      updateToast("error", "No camera selected");
      return;
    }

    // Check if user is checked in first
    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn("stream");
    }

    try {
      // Force fresh status check from unified polling
      await unifiedCameraPolling.forceCheck(currentCameraId);

      // Use the camera status hook for more reliable state
      const isCurrentlyStreaming = currentCameraStatus.isStreaming;
      console.log(
        `🔄 [STREAM DEBUG] Current streaming state:`,
        isCurrentlyStreaming
      );

      if (isCurrentlyStreaming) {
        // Stop streaming
        updateToast("info", "Stopping stream...");
        console.log(`🛑 [STREAM DEBUG] Attempting to stop stream...`);

        // Note: Pre-capture transactions removed - activities are now buffered on Jetson
        // and committed at checkout via the encrypted activity buffer system

        const response = await unifiedCameraService.stopStream(currentCameraId);
        console.log(`🛑 [STREAM DEBUG] Stop stream response:`, response);

        if (response.success) {
          // Clear the "stopping" toast immediately
          dismissToast();

          // Wait a bit longer for hardware to update, then check status
          setTimeout(async () => {
            await unifiedCameraPolling.forceCheck(currentCameraId);
            // Only show success if we can confirm the stream actually stopped
            const updatedState =
              await unifiedCameraService.getComprehensiveState(currentCameraId);
            if (
              updatedState.streamInfo.success &&
              !updatedState.streamInfo.data?.isActive
            ) {
              updateToast("success", "Stream stopped");
            }
          }, 2000);
        } else {
          updateToast(
            "error",
            `Failed to stop stream: ${response.error || "Unknown error"}`
          );
        }
      } else {
        // Start streaming
        updateToast("info", "Starting stream...");
        console.log(`▶️ [STREAM DEBUG] Attempting to start stream...`);

        // Note: Pre-capture transactions removed - activities are now buffered on Jetson
        // and committed at checkout via the encrypted activity buffer system

        // Connect to camera if not already connected
        const isConnected = await unifiedCameraService.isConnected(
          currentCameraId
        );
        if (!isConnected && primaryWallet?.address) {
          await unifiedCameraService.connect(
            currentCameraId,
            primaryWallet.address
          );
        }

        const response = await unifiedCameraService.startStream(
          currentCameraId
        );
        console.log(`▶️ [STREAM DEBUG] Start stream response:`, response);

        if (response.success) {
          // Clear the "starting" toast immediately
          dismissToast();

          // Wait a bit longer for hardware to update, then check status
          setTimeout(async () => {
            await unifiedCameraPolling.forceCheck(currentCameraId);
            // Only show success if we can confirm the stream actually started
            const updatedState =
              await unifiedCameraService.getComprehensiveState(currentCameraId);
            if (
              updatedState.streamInfo.success &&
              updatedState.streamInfo.data?.isActive
            ) {
              updateToast("success", "Stream started");
            }
          }, 2000);
        } else {
          updateToast(
            "error",
            `Failed to start stream: ${response.error || "Unknown error"}`
          );
        }
      }

      // Force additional checks at 2s and 5s intervals to catch delayed state changes
      setTimeout(() => unifiedCameraPolling.forceCheck(currentCameraId), 2000);
      setTimeout(() => unifiedCameraPolling.forceCheck(currentCameraId), 5000);

      // Refresh the timeline
      if (timelineRef.current?.refreshEvents) {
        timelineRef.current?.refreshEvents();
      }
    } catch (error) {
      console.error("🚨 [STREAM DEBUG] Error handling stream:", error);
      updateToast(
        "error",
        `Error handling stream: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  };

  // Add a useEffect to handle stream state changes
  useEffect(() => {
    // When streaming starts, clear any lingering "Starting stream..." toast
    if (
      hardwareState.isStreaming &&
      currentToast?.message?.includes("Starting stream")
    ) {
      dismissToast();
    }
  }, [hardwareState.isStreaming, currentToast]);

  // Helper function to handle stream errors more gracefully

  // Add a simple test function for gesture detection that we can call from console
  (window as any).testGestureAPI = async () => {
    console.log("[GESTURE TEST] Testing gesture API directly...");
    try {
      const currentCameraId =
        cameraAccount || selectedCamera?.publicKey || CONFIG.JETSON_CAMERA_PDA;
      const result = await unifiedCameraService.getCurrentGesture(
        currentCameraId
      );
      console.log("[GESTURE TEST] getCurrentGesture result:", result);

      const triggerCheck = await unifiedCameraService.checkForGestureTrigger();
      console.log(
        "[GESTURE TEST] checkForGestureTrigger result:",
        triggerCheck
      );

      return {
        getCurrentGesture: result,
        checkForGestureTrigger: triggerCheck,
      };
    } catch (error) {
      console.error("[GESTURE TEST] Error:", error);
      return { error };
    }
  };

  // Cleanup gesture monitoring on unmount
  useEffect(() => {
    return () => {
      if (gestureCheckIntervalRef.current) {
        clearInterval(gestureCheckIntervalRef.current);
        gestureCheckIntervalRef.current = null;
      }
    };
  }, []);

  return (
    <>
      <div className="pb-40">
        <div className="relative max-w-3xl mx-auto pt-0">
          <ToastContainer message={currentToast} onDismiss={dismissToast} />

          {/* Transaction Modal for embedded wallets */}
          <TransactionModal
            isOpen={showTransactionModal}
            onClose={() => setShowTransactionModal(false)}
            transactionData={transactionData || undefined}
            onSuccess={({ transactionId }) => {
              setShowTransactionModal(false);

              // After successful check-in and action, refresh check-in status
              checkUserSession().then(() => {
                // Create a timeline event with the transaction ID
                if (transactionData) {
                  const eventType = getEventType(transactionData.type);

                  // Use the addTimelineEvent function for consistency
                  addTimelineEvent(eventType, transactionId);

                  // Show success message and toggle stream state if needed
                  updateToast(
                    "success",
                    `${
                      transactionData.type.charAt(0).toUpperCase() +
                      transactionData.type.slice(1)
                    } action recorded successfully`
                  );

                  // Hardware state will be updated by polling automatically

                  // Refresh timeline to show latest events
                  if (timelineRef.current?.refreshTimeline) {
                    timelineRef.current?.refreshTimeline();
                  }
                }
              });
            }}
          />

          <div className="bg-white rounded-lg mb-0 px-6">
            <div className="py-4 flex justify-between items-center hidden md:flex">
              <CameraIdDisplay
                cameraId={cameraId}
                selectedCamera={selectedCamera}
                cameraAccount={cameraAccount}
                timelineRef={timelineRef}
              />
            </div>
          </div>
          <div className="px-0 pt-0 sm:px-2">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-3 relative">
                {/* IRL Apps Button - positioned next to stream */}
                {currentCameraId &&
                  unifiedCameraService.hasCamera(currentCameraId) && (
                    <div className="absolute top-2 right-4 z-50">
                      <IRLAppsButton
                        cameraId={currentCameraId}
                        walletAddress={primaryWallet?.address}
                        onEnrollmentComplete={() => {
                          updateToast("success", "Recognition token created! IRL apps are now unlocked.");

                          // Refresh timeline to show the new event
                          if (timelineRef.current?.refreshEvents) {
                            timelineRef.current?.refreshEvents();
                          }
                        }}
                      />
                    </div>
                  )}

                {/* Unified TikTok-style status bar - aligned with timeline - MOBILE ONLY */}
                <div
                  className="absolute top-2 left-4 z-40 flex items-center cursor-pointer md:hidden"
                  onClick={() => setIsMobileCameraModalOpen(true)}
                >
                  <div className="flex items-center bg-black bg-opacity-70 rounded overflow-hidden">
                    {/* Camera Status Section */}
                    {!cameraId && !cameraAccount && !selectedCamera ? (
                      <div className="bg-gray-600 text-white text-xs font-bold px-1.5 py-0.5">
                        DISCONNECTED
                      </div>
                    ) : !currentCameraStatus.isLive ? (
                      <div className="bg-gray-500 text-white text-xs font-bold px-1.5 py-0.5">
                        OFFLINE
                      </div>
                    ) : currentCameraStatus.isStreaming ? (
                      <div className="bg-red-500 text-white text-xs font-bold px-1.5 py-0.5 flex items-center gap-1">
                        <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div>
                        LIVE
                      </div>
                    ) : (
                      <div className="bg-green-500 text-white text-xs font-bold px-1.5 py-0.5">
                        ONLINE
                      </div>
                    )}

                    {/* Camera ID Section - only show when camera is selected */}
                    {(cameraId || cameraAccount || selectedCamera) && (
                      <div className="text-white text-xs px-1.5 py-0.5 border-l border-white border-opacity-20">
                        id:
                        {(
                          cameraAccount ||
                          selectedCamera?.publicKey ||
                          cameraId ||
                          ""
                        ).slice(0, 4)}
                        ...
                        {(
                          cameraAccount ||
                          selectedCamera?.publicKey ||
                          cameraId ||
                          ""
                        ).slice(-4)}
                      </div>
                    )}
                  </div>

                  {/* Check-in Status Icon */}
                  <div className="ml-2">
                    {isCheckedIn ? (
                      <CheckCircle className="w-3 h-3 text-green-400" />
                    ) : (
                      <Link2 className="w-3 h-3 text-blue-400" />
                    )}
                  </div>
                </div>

                {/* Competition Scoreboard - Mobile (right below Apps button) */}
                {currentCameraId && (
                  <div className="absolute top-10 left-4 right-4 z-50 md:hidden">
                    <CompetitionScoreboard
                      cameraId={currentCameraId}
                      walletAddress={primaryWallet?.address}
                    />
                  </div>
                )}

                {/* Mobile Timeline Overlay - positioned below status badge */}
                <div className="absolute top-10 left-2 z-30 md:hidden px-2">
                  <Timeline
                    ref={mobileTimelineRef}
                    variant="camera"
                    cameraId={cameraAccount || undefined}
                    mobileOverlay={true}
                  />
                </div>

                <StreamPlayer />

                <div className="hidden sm:flex absolute -right-14 top-0 flex-col h-full z-[45]">
                  {/* Direct buttons for desktop */}
                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectStream}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-black transition-colors rounded-xl"
                      aria-label={
                        currentCameraStatus.isStreaming
                          ? "Stop Stream"
                          : "Start Stream"
                      }
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : currentCameraStatus.isStreaming ? (
                        <StopCircle className="w-5 h-5" />
                      ) : (
                        <Play className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading
                        ? "Processing..."
                        : currentCameraStatus.isStreaming
                        ? "Stop Stream"
                        : "Start Stream"}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectPhoto}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Camera className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? "Processing..." : "Take Picture"}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectVideo}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Video className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? "Processing..." : "Record Video"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div id="camera-controls" className="flex-1 mt-2 px-2">
            <CameraControls
              onTakePicture={handleDirectPhoto}
              onRecordVideo={handleDirectVideo}
              onToggleStream={handleDirectStream}
              isLoading={loading}
              isStreaming={currentCameraStatus.isStreaming}
            />
          </div>
          <div className="max-w-3xl mt-6 md:mt-6 mx-auto flex flex-col justify-top relative">
            <div className="relative mb-12 md:mb-36">
              <div className="hidden md:flex pl-6 items-center gap-2">
                {!cameraId && !cameraAccount && !selectedCamera ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      {/* Proper prohibited symbol (🚫) */}
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-white border border-gray-400"></span>
                      <span className="absolute inset-0 flex items-center justify-center">
                        <span className="h-[1.5px] w-2 bg-gray-500 rotate-45 absolute"></span>
                        <span className="h-[1.5px] w-2 bg-gray-500 -rotate-45 absolute"></span>
                      </span>
                    </span>
                    <span className="text-gray-500 font-medium">
                      Disconnected
                    </span>
                  </div>
                ) : !currentCameraStatus.isLive ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-gray-400"></span>
                    </span>
                    <span className="text-gray-500 font-medium">Offline</span>
                  </div>
                ) : currentCameraStatus.isStreaming ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-red-500 font-medium">LIVE</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                    </span>
                    <span className="text-green-500 font-medium">Online</span>
                  </div>
                )}
              </div>
            </div>

            {/* Competition Scoreboard - floats above timeline (Desktop) */}
            {currentCameraId && (
              <div className="absolute mt-12 pb-2 px-5 left-0 w-full hidden md:block z-40">
                <CompetitionScoreboard cameraId={currentCameraId} />
              </div>
            )}

            <div className="absolute mt-12 pb-20 pl-5 left-0 w-full hidden md:block">
              <Timeline
                ref={timelineRef}
                variant="camera"
                cameraId={cameraAccount || undefined}
              />
              <div
                className="top-0 left-0 right-0 pointer-events-none"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)",
                }}
              />
            </div>

            <div className="relative md:ml-20 bg-white">
              <div className="relative px-2 sm:pl-4 sm:pr-2 md:px-4">
                <MediaGallery
                  mode="recent"
                  maxRecentItems={6}
                  cameraId={cameraAccount || undefined}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Camera Modal */}
      <CameraModal
        isOpen={isMobileCameraModalOpen}
        onClose={() => setIsMobileCameraModalOpen(false)}
        onCheckStatusChange={(newStatus: boolean) => {
          console.log("Mobile modal status change:", newStatus);
          setIsCheckedIn(newStatus);
          // Refresh timeline if needed
          if (timelineRef.current?.refreshTimeline) {
            timelineRef.current?.refreshTimeline();
          }
        }}
        camera={{
          id: cameraAccount || selectedCamera?.publicKey || "",
          owner: selectedCamera?.owner || currentCameraStatus.owner || "",
          ownerDisplayName:
            selectedCamera?.metadata?.name || "newProgramCamera",
          model: selectedCamera?.metadata?.model || "pi5",
          isLive: currentCameraStatus.isLive || false,
          isStreaming: currentCameraStatus.isStreaming || false,
          status: "ok",
          lastSeen: Date.now(),
          showDevInfo: !cameraAccount && !selectedCamera?.publicKey,
          defaultDevCamera: "EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA",
        }}
      />

      {/* Competition Start/Stop Controls - Floating at bottom */}
      {currentCameraId && (
        <CompetitionControls cameraId={currentCameraId} />
      )}
    </>
  );
}
