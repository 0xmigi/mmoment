import { useProgram } from "../anchor/setup";
import { faceProcessingService } from "./face-processing";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { PublicKey, SystemProgram, Transaction } from "@solana/web3.js";
import { Camera, X, RotateCcw, Check, Wifi } from "lucide-react";
import { useState, useRef, useCallback, useEffect } from "react";
import { useConnection } from "@solana/wallet-adapter-react";

interface PhoneSelfieEnrollmentProps {
  walletAddress?: string;
  onEnrollmentComplete?: (result: {
    success: boolean;
    error?: string;
    transactionId?: string;
  }) => void;
  onCancel?: () => void;
}

export function PhoneSelfieEnrollment({
  onEnrollmentComplete,
  onCancel,
}: PhoneSelfieEnrollmentProps) {
  const [step, setStep] = useState<
    "check-connection" | "camera" | "preview" | "processing" | "complete"
  >("check-connection");
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");
  const [qualityIssues, setQualityIssues] = useState<string[]>([]);
  const [qualityRecommendations, setQualityRecommendations] = useState<string[]>([]);
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [qualityRating, setQualityRating] = useState<string | null>(null);
  const [connectedCameraUrl, setConnectedCameraUrl] = useState<string | null>(
    null
  );

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const { connection } = useConnection();

  // Check for user session on mount
  useEffect(() => {
    findUserCurrentSession();
  }, [primaryWallet, program]);

  const findUserCurrentSession = async () => {
    if (!primaryWallet?.address) {
      setError("Please connect your wallet to enroll your face");
      return;
    }

    try {
      // Get the camera PDA from localStorage (set by IRLAppsButton)
      const storedCameraPda = localStorage.getItem('lastAccessedCameraPDA');

      if (storedCameraPda) {
        const cameraUrl = `https://${storedCameraPda}.mmoment.xyz`;

        // Test if the camera is accessible
        try {
          const testResponse = await fetch(`${cameraUrl}/api/status`, {
            method: 'GET',
            mode: 'cors'
          });

          if (testResponse.ok) {
            setConnectedCameraUrl(cameraUrl);
            setStep("camera");
            setError(null);
            return;
          }
        } catch (fetchError) {
          // Camera not accessible, continue to fallback
          console.log('[PhoneSelfieEnrollment] Stored camera not accessible:', fetchError);
        }
      }

      // Fallback: Use your main Jetson camera directly
      const mainJetsonUrl = "https://h1wonbkwjgncepeyr65xeewjfggdbospl5ubjan5vyhg.mmoment.xyz";
      try {
        const testResponse = await fetch(`${mainJetsonUrl}/api/status`, {
          method: 'GET',
          mode: 'cors'
        });

        if (testResponse.ok) {
          setConnectedCameraUrl(mainJetsonUrl);
          setStep("camera");
          setError(null);
          return;
        }
      } catch (fetchError) {
        console.log('[PhoneSelfieEnrollment] Main camera not accessible:', fetchError);
      }

      setError("No accessible camera found. Please visit a camera page first or ensure the camera is online.");
    } catch (error) {
      setError("Unable to connect to camera. Please try again later.");
    }
  };

  // Initialize camera
  useEffect(() => {
    if (step === "camera") {
      initializeCamera();
    }
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [step]);

  const initializeCamera = async () => {
    try {
      setError(null);

      // Mobile-optimized camera settings
      const constraints = {
        video: {
          facingMode: "user", // Front camera
          width: {
            min: 320,
            ideal: 720,
            max: 1920,
          },
          height: {
            min: 240,
            ideal: 1280,
            max: 1080,
          },
          aspectRatio: { ideal: 0.5625 }, // 9:16 for portrait mode
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(
        constraints
      );

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        // Ensure video plays on mobile
        videoRef.current.setAttribute("playsinline", "true");
        videoRef.current.setAttribute("autoplay", "true");
        videoRef.current.muted = true;
      }
    } catch (err) {
      console.error("Camera access error:", err);

      // Provide more specific error messages
      let errorMessage = "Unable to access camera.";
      if (err instanceof Error) {
        if (
          err.name === "NotAllowedError" ||
          err.name === "PermissionDeniedError"
        ) {
          errorMessage =
            "Camera permission denied. Please allow camera access in your browser settings.";
        } else if (
          err.name === "NotFoundError" ||
          err.name === "DevicesNotFoundError"
        ) {
          errorMessage =
            "No camera found. Please ensure your device has a camera.";
        } else if (err.name === "NotReadableError") {
          errorMessage =
            "Camera is already in use by another application. Please close other camera apps.";
        } else if (err.name === "OverconstrainedError") {
          errorMessage =
            "Camera doesn't support required settings. Try refreshing the page.";
        }
      }

      setError(errorMessage);
    }
  };

  const capturePhoto = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError("Camera not ready");
      return;
    }

    setIsCapturing(true);
    setQualityIssues([]);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Failed to get canvas context");
      }

      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to base64
      const imageData = canvas.toDataURL("image/jpeg", 0.95);

      console.log('[PhoneSelfieEnrollment] Captured image data length:', imageData.length);
      console.log('[PhoneSelfieEnrollment] Image data prefix:', imageData.substring(0, 50));
      console.log('[PhoneSelfieEnrollment] Canvas dimensions:', canvas.width, 'x', canvas.height);

      // MUST use Jetson for quality check - no fake local assessment
      setProgress("Checking image quality...");

      console.log('[PhoneSelfieEnrollment] 🚨 CRITICAL: connectedCameraUrl =', connectedCameraUrl);
      console.log('[PhoneSelfieEnrollment] 🚨 CRITICAL: typeof connectedCameraUrl =', typeof connectedCameraUrl);

      if (!connectedCameraUrl) {
        console.log('[PhoneSelfieEnrollment] 🚨 CRITICAL: connectedCameraUrl is null/undefined - THIS IS THE BUG!');
        setError("No camera connection - cannot assess image quality");
        return;
      }

      const result = await faceProcessingService.processFacialEmbedding(
        imageData,
        connectedCameraUrl,
        { requestQuality: true, encrypt: false }
      );

      if (!result.success || !result.quality) {
        setError(`Quality assessment failed: ${result.error || 'No quality data from camera'}`);
        return;
      }

      const quality = result.quality;
      console.log('[PhoneSelfieEnrollment] Jetson quality assessment:', quality);

      // Store quality information
      setQualityScore(quality.score);
      setQualityRating(quality.rating);
      setQualityIssues(quality.issues || []);
      setQualityRecommendations(quality.recommendations || []);

      setCapturedImage(imageData);
      setStep("preview");
    } catch (err) {
      console.error("Capture error:", err);
      setError("Failed to capture photo. Please try again.");
    } finally {
      setIsCapturing(false);
    }
  }, []);

  const retakePhoto = useCallback(() => {
    setCapturedImage(null);
    setQualityIssues([]);
    setQualityRecommendations([]);
    setQualityScore(null);
    setQualityRating(null);
    setStep("camera");
    setError(null);
  }, []);

  const processFaceEnrollment = useCallback(async () => {
    if (!capturedImage || !primaryWallet || !program || !connectedCameraUrl) {
      setError("Missing requirements for enrollment");
      return;
    }

    setStep("processing");
    setProgress("Sending image to camera for processing...");

    console.log('[PhoneSelfieEnrollment] Starting face enrollment processing...');
    console.log('[PhoneSelfieEnrollment] Connected camera URL:', connectedCameraUrl);
    console.log('[PhoneSelfieEnrollment] Wallet address:', primaryWallet.address);
    console.log('[PhoneSelfieEnrollment] Image data available:', !!capturedImage);

    try {
      // Send image to Jetson camera for enhanced facial embedding extraction with encryption
      console.log('[PhoneSelfieEnrollment] Calling processFacialEmbedding...');
      const result = await faceProcessingService.processFacialEmbedding(
        capturedImage,
        connectedCameraUrl,
        { encrypt: true, requestQuality: true, walletAddress: primaryWallet.address }
      );

      console.log('[PhoneSelfieEnrollment] Processing result:', {
        success: result.success,
        hasEmbedding: !!result.embedding,
        error: result.error
      });

      if (!result.success) {
        throw new Error(result.error || "Failed to process facial features");
      }

      if (!result.embedding) {
        throw new Error("No facial embedding received from camera");
      }

      console.log('[PhoneSelfieEnrollment] Enhanced processing result:', {
        hasEmbedding: !!result.embedding,
        embeddingLength: result.embedding.length,
        encrypted: result.encrypted,
        qualityScore: result.quality?.score,
        qualityRating: result.quality?.rating
      });

      // Log quality information if available
      if (result.quality) {
        console.log('[PhoneSelfieEnrollment] Quality assessment:', result.quality);
        setProgress(`Quality: ${result.quality.rating} (${result.quality.score}%) - Processing...`);
      }

      // Validate embedding length (can be 512 for high-quality or other sizes for compressed)
      if (result.embedding.length === 0) {
        throw new Error("Empty facial embedding received from camera");
      }

      setProgress("Storing encrypted facial embedding on blockchain...");

      // Get user's public key
      const userPublicKey = new PublicKey(primaryWallet.address);

      // Derive the PDA for face data storage
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("face-nft"), // CRITICAL: Must be "face-nft" not "face-embedding"
          userPublicKey.toBuffer(),
        ],
        program.programId
      );

      // Create the instruction to enroll the face
      // Convert embedding to Buffer format expected by the program
      const embeddingBuffer = Buffer.from(
        new Float32Array(result.embedding).buffer
      );

      const enrollInstruction = await program.methods
        .enrollFace(embeddingBuffer)
        .accounts({
          faceNft: faceDataPda,
          user: userPublicKey,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Create and send transaction using Dynamic wallet pattern
      const tx = new Transaction();
      tx.add(enrollInstruction);

      // Get recent blockhash
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Sign and send the transaction using Dynamic wallet
      const signer = await (primaryWallet as any).getSigner();
      const signedTx = await signer.signTransaction(tx);
      const signature = await connection.sendRawTransaction(signedTx.serialize());

      setProgress("Confirming blockchain transaction...");

      // Wait for confirmation
      await connection.confirmTransaction({ signature, ...(await connection.getLatestBlockhash()) }, "confirmed");
      const rpcUrl = program.provider.connection.rpcEndpoint;
      const response = await fetch(rpcUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "getSignatureStatuses",
          params: [[signature]],
        }),
      });

      const data = await response.json();
      const status = data.result?.value?.[0];

      if (
        status?.confirmationStatus === "confirmed" ||
        status?.confirmationStatus === "finalized"
      ) {
        setStep("complete");
        setProgress("Facial embedding successfully stored!");

        setTimeout(() => {
          onEnrollmentComplete?.({
            success: true,
            transactionId: signature,
          });
        }, 2000);
      } else {
        throw new Error("Transaction failed to confirm");
      }
    } catch (err) {
      console.error("Enrollment error:", err);
      let errorMessage = "Failed to enroll facial embedding.";

      if (err instanceof Error) {
        if (err.message.includes("already exists")) {
          errorMessage = "You already have a facial embedding stored.";
        } else if (err.message.includes("insufficient funds")) {
          errorMessage = "Insufficient SOL balance for transaction.";
        } else {
          errorMessage = err.message;
        }
      }

      setError(errorMessage);
      setStep("preview");
      onEnrollmentComplete?.({
        success: false,
        error: errorMessage,
      });
    }
  }, [
    capturedImage,
    primaryWallet,
    program,
    connectedCameraUrl,
    onEnrollmentComplete,
  ]);

  // Render based on current step
  if (step === "check-connection") {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <Wifi className="h-12 w-12 text-white/70 mx-auto mb-4" />
            <p className="text-white text-lg mb-2">Connecting to Camera</p>
            <p className="text-white/70 text-sm">Please wait...</p>
          </div>

          {error && (
            <div className="absolute bottom-20 left-4 right-4">
              <div className="bg-red-500/90 backdrop-blur-sm rounded-lg p-3">
                <p className="text-white text-sm text-center">{error}</p>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 bg-white">
          <button
            onClick={findUserCurrentSession}
            className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (step === "camera") {
    return (
      <div className="h-full flex flex-col bg-black">
        {/* Camera View - PORTRAIT aspect ratio like main camera */}
        <div className="flex-1 relative overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
            style={{
              transform: "scaleX(-1)", // Mirror the video for selfie mode
              aspectRatio: "9/16", // PORTRAIT like main camera, not landscape
            }}
          />
          <canvas ref={canvasRef} className="hidden" />

          {/* Face guide overlay - centered */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
            <div className="relative w-40 h-52 border-2 border-white/50 rounded-full">
              <div className="absolute inset-4 border border-white/20 rounded-full" />
            </div>
          </div>

          {/* Minimal status overlay */}
          <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
            <div className="bg-black/30 backdrop-blur-sm rounded-full px-3 py-1">
              <p className="text-white text-xs">Position face in oval</p>
            </div>
            {onCancel && (
              <button
                onClick={onCancel}
                className="w-8 h-8 bg-black/30 backdrop-blur-sm hover:bg-black/50 rounded-full flex items-center justify-center transition-colors"
              >
                <X className="h-4 w-4 text-white" />
              </button>
            )}
          </div>

          {/* Error overlay */}
          {error && (
            <div className="absolute bottom-16 left-4 right-4">
              <div className="bg-red-500/90 backdrop-blur-sm rounded-lg p-2">
                <p className="text-white text-xs text-center">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Floating button over camera - with bottom spacing */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/10 backdrop-blur-sm">
          <button
            onClick={capturePhoto}
            disabled={isCapturing}
            className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors flex items-center justify-center text-sm font-medium shadow-lg"
          >
            {isCapturing ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Capturing...</span>
              </div>
            ) : (
              <>
                <Camera className="h-4 w-4 mr-2" />
                Capture Face
              </>
            )}
          </button>
        </div>
      </div>
    );
  }

  if (step === "preview") {
    return (
      <div className="h-full flex flex-col bg-black">
        {/* Image Preview - PORTRAIT aspect ratio */}
        <div className="flex-1 relative overflow-hidden">
          {capturedImage && (
            <img
              src={capturedImage}
              alt="Captured selfie"
              className="w-full h-full object-cover"
              style={{
                transform: "scaleX(-1)", // Mirror the image for selfie mode
                aspectRatio: "9/16", // PORTRAIT like main camera
              }}
            />
          )}

          {/* Quality overlay at top - smaller */}
          <div className="absolute top-4 left-4 right-4">
            {qualityScore !== null && qualityRating && (
              <div className="bg-black/30 backdrop-blur-sm rounded-full px-3 py-1 flex items-center justify-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  qualityRating === 'excellent' ? 'bg-green-400' :
                  qualityRating === 'good' ? 'bg-blue-400' :
                  qualityRating === 'acceptable' ? 'bg-yellow-400' :
                  qualityRating === 'poor' ? 'bg-orange-400' :
                  'bg-red-400'
                }`} />
                <span className="text-white text-xs capitalize">
                  {qualityRating} ({qualityScore}%)
                </span>
              </div>
            )}
          </div>

          {/* Issues overlay - smaller */}
          {(qualityIssues.length > 0 || qualityRecommendations.length > 0) && (
            <div className="absolute bottom-16 left-4 right-4">
              <div className="bg-yellow-500/90 backdrop-blur-sm rounded-lg p-2">
                <p className="text-white text-xs text-center">
                  {qualityIssues.length > 0 ? qualityIssues[0] : qualityRecommendations[0]}
                </p>
              </div>
            </div>
          )}

          {/* Error overlay - smaller */}
          {error && (
            <div className="absolute bottom-16 left-4 right-4">
              <div className="bg-red-500/90 backdrop-blur-sm rounded-lg p-2">
                <p className="text-white text-xs text-center">{error}</p>
              </div>
            </div>
          )}

          {/* Cancel button - smaller */}
          {onCancel && (
            <div className="absolute top-4 right-4">
              <button
                onClick={onCancel}
                className="w-8 h-8 bg-black/30 backdrop-blur-sm hover:bg-black/50 rounded-full flex items-center justify-center transition-colors"
              >
                <X className="h-4 w-4 text-white" />
              </button>
            </div>
          )}
        </div>

        {/* Floating buttons over preview - with bottom spacing */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/10 backdrop-blur-sm">
          <div className="flex gap-3">
            <button
              onClick={retakePhoto}
              className="flex-1 bg-gray-200 text-gray-700 px-4 py-3 rounded-lg hover:bg-gray-300 transition-colors flex items-center justify-center text-sm font-medium shadow-lg"
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Retake
            </button>
            <button
              onClick={processFaceEnrollment}
              disabled={!!error}
              className="flex-1 bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors flex items-center justify-center text-sm font-medium shadow-lg"
            >
              <Check className="h-4 w-4 mr-2" />
              Create Token
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (step === "processing") {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white/50 mx-auto mb-4"></div>
            <p className="text-white text-base mb-2">Processing...</p>
            <p className="text-white/70 text-sm">
              {progress.split(' - ')[0]}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (step === "complete") {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 bg-green-600 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Check className="w-8 h-8 text-white" />
            </div>
            <p className="text-white text-xl font-semibold mb-2">
              Token Created!
            </p>
            <p className="text-white/90 text-sm text-center px-8">
              IRL apps are now unlocked across the network
            </p>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
