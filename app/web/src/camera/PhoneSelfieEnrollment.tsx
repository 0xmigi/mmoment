import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from "../anchor/setup";
import { faceProcessingService } from "./face-processing";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { PublicKey, SystemProgram, Transaction } from "@solana/web3.js";
import { Camera, X, RotateCcw, Check, AlertCircle, Wifi } from "lucide-react";
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
  const [connectedCameraId, setConnectedCameraId] = useState<string | null>(
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
    if (!primaryWallet?.address || !program || !connection) {
      setError("Please connect your wallet to enroll your face");
      return;
    }

    try {
      console.log('[PhoneSelfieEnrollment] 🔍 Checking for active user sessions...');

      const userPublicKey = new PublicKey(primaryWallet.address);

      // Get all camera accounts from the program to check sessions
      const allCameras = await program.account.cameraAccount.all();
      console.log('[PhoneSelfieEnrollment] 📋 Found cameras to check:', allCameras.length);

      let activeSession = null;
      let activeCameraPda = null;

      // Check each camera for an active session with this user
      for (const cameraAccount of allCameras) {
        const cameraPublicKey = cameraAccount.publicKey;

        // Find the session PDA for this user + camera combination
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
          const sessionAccount = await program.account.userSession.fetch(sessionPda);
          if (sessionAccount) {
            console.log('[PhoneSelfieEnrollment] ✅ Found active session with camera:', cameraPublicKey.toString());
            activeSession = sessionAccount;
            activeCameraPda = cameraPublicKey.toString();
            break; // Found an active session, use this camera
          }
        } catch (err) {
          // No session with this camera, continue checking others
          continue;
        }
      }

      if (activeSession && activeCameraPda) {
        console.log('[PhoneSelfieEnrollment] ✅ User is checked into camera:', activeCameraPda);
        const cameraUrl = `https://${activeCameraPda}.mmoment.xyz`;
        setConnectedCameraUrl(cameraUrl);
        setConnectedCameraId(activeCameraPda);
        setStep("camera");
        setError(null);
      } else {
        console.log('[PhoneSelfieEnrollment] ❌ No active sessions found');
        setError("Please check into a camera first to enroll your face. Visit a camera page and complete the check-in process.");
      }
    } catch (error) {
      console.error('[PhoneSelfieEnrollment] Error checking user sessions:', error);
      setError("Unable to verify camera session. Please ensure you're checked into a camera.");
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

      // Perform quality check using enhanced processing if camera is available
      setProgress("Checking image quality...");
      let quality;

      if (connectedCameraUrl) {
        // Use enhanced Jetson quality assessment
        const result = await faceProcessingService.processFacialEmbedding(
          imageData,
          connectedCameraUrl,
          { requestQuality: true, encrypt: false }
        );

        if (result.success && result.quality) {
          quality = result.quality;
          console.log('[PhoneSelfieEnrollment] Enhanced quality assessment:', quality);
        } else {
          // Fallback to local quality check
          quality = await faceProcessingService.analyzeImageQuality(imageData);
        }
      } else {
        // Use local quality assessment as fallback
        quality = await faceProcessingService.analyzeImageQuality(imageData);
      }

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

    try {
      // Send image to Jetson camera for enhanced facial embedding extraction with encryption
      const result = await faceProcessingService.processFacialEmbedding(
        capturedImage,
        connectedCameraUrl,
        { encrypt: true, requestQuality: true }
      );

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
      <div className="bg-gray-100 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Face Enrollment</h3>
          {onCancel && (
            <button
              onClick={onCancel}
              className="text-gray-500 hover:text-gray-700"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <div className="flex items-start">
            <Wifi className="h-5 w-5 text-yellow-600 mt-1 mr-3 flex-shrink-0" />
            <div>
              <p className="text-sm text-yellow-800 font-medium">
                Camera Connection Required
              </p>
              <p className="text-sm text-yellow-700 mt-1">
                To ensure compatibility, face enrollment must be done through a
                connected camera. Please connect to a camera first, then try
                again.
              </p>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <button
          onClick={findUserCurrentSession}
          className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          Check Session Again
        </button>
      </div>
    );
  }

  if (step === "camera") {
    return (
      <div className="bg-gray-100 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Capture Your Face</h3>
          {onCancel && (
            <button
              onClick={onCancel}
              className="text-gray-500 hover:text-gray-700"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        <div className="relative bg-black rounded-lg overflow-hidden mb-4">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto"
            style={{
              transform: "scaleX(-1)", // Mirror the video for selfie mode
              maxHeight: "400px",
              objectFit: "cover",
            }}
          />
          <canvas ref={canvasRef} className="hidden" />

          {/* Face guide overlay */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-48 h-64 border-2 border-white/50 rounded-full" />
            </div>
            <p className="absolute bottom-4 left-0 right-0 text-center text-white text-sm">
              Position your face within the oval
            </p>
          </div>
        </div>

        <p className="text-sm text-gray-600 mb-3 text-center">
          Connected to: {connectedCameraId?.replace("jetson_", "")}
        </p>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <button
          onClick={capturePhoto}
          disabled={isCapturing}
          className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors flex items-center justify-center"
        >
          {isCapturing ? (
            "Capturing..."
          ) : (
            <>
              <Camera className="h-5 w-5 mr-2" />
              Take Photo
            </>
          )}
        </button>
      </div>
    );
  }

  if (step === "preview") {
    return (
      <div className="bg-gray-100 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Review Your Photo</h3>
          {onCancel && (
            <button
              onClick={onCancel}
              className="text-gray-500 hover:text-gray-700"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        {capturedImage && (
          <div className="mb-4">
            <img
              src={capturedImage}
              alt="Captured selfie"
              className="w-full rounded-lg"
              style={{
                transform: "scaleX(-1)", // Mirror the image for selfie mode
                maxHeight: "400px",
                objectFit: "cover",
              }}
            />
          </div>
        )}

        {/* Enhanced Quality Score Display */}
        {qualityScore !== null && qualityRating && (
          <div className={`rounded-lg p-3 mb-4 ${
            qualityRating === 'excellent' ? 'bg-green-50 border border-green-200' :
            qualityRating === 'good' ? 'bg-blue-50 border border-blue-200' :
            qualityRating === 'acceptable' ? 'bg-yellow-50 border border-yellow-200' :
            qualityRating === 'poor' ? 'bg-orange-50 border border-orange-200' :
            'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  qualityRating === 'excellent' ? 'bg-green-500' :
                  qualityRating === 'good' ? 'bg-blue-500' :
                  qualityRating === 'acceptable' ? 'bg-yellow-500' :
                  qualityRating === 'poor' ? 'bg-orange-500' :
                  'bg-red-500'
                }`} />
                <span className={`text-sm font-medium capitalize ${
                  qualityRating === 'excellent' ? 'text-green-800' :
                  qualityRating === 'good' ? 'text-blue-800' :
                  qualityRating === 'acceptable' ? 'text-yellow-800' :
                  qualityRating === 'poor' ? 'text-orange-800' :
                  'text-red-800'
                }`}>
                  {qualityRating.replace('_', ' ')} Quality
                </span>
              </div>
              <span className={`text-sm font-semibold ${
                qualityRating === 'excellent' ? 'text-green-700' :
                qualityRating === 'good' ? 'text-blue-700' :
                qualityRating === 'acceptable' ? 'text-yellow-700' :
                qualityRating === 'poor' ? 'text-orange-700' :
                'text-red-700'
              }`}>
                {qualityScore}%
              </span>
            </div>
            {connectedCameraUrl && (
              <p className={`text-xs mt-1 ${
                qualityRating === 'excellent' ? 'text-green-600' :
                qualityRating === 'good' ? 'text-blue-600' :
                qualityRating === 'acceptable' ? 'text-yellow-600' :
                qualityRating === 'poor' ? 'text-orange-600' :
                'text-red-600'
              }`}>
                ✨ Enhanced assessment by Jetson camera
              </p>
            )}
          </div>
        )}

        {qualityIssues.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
            <div className="flex items-start">
              <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-sm text-yellow-800 font-medium">
                  Quality Issues Detected
                </p>
                <ul className="text-sm text-yellow-700 mt-1 list-disc list-inside">
                  {qualityIssues.map((issue, idx) => (
                    <li key={idx}>{issue}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {qualityRecommendations.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
            <div className="flex items-start">
              <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center mt-0.5 mr-2 flex-shrink-0">
                <span className="text-white text-xs font-bold">💡</span>
              </div>
              <div>
                <p className="text-sm text-blue-800 font-medium">
                  Recommendations for Better Quality
                </p>
                <ul className="text-sm text-blue-700 mt-1 list-disc list-inside">
                  {qualityRecommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <div className="flex gap-3">
          <button
            onClick={retakePhoto}
            className="flex-1 bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors flex items-center justify-center"
          >
            <RotateCcw className="h-5 w-5 mr-2" />
            Retake
          </button>
          <button
            onClick={processFaceEnrollment}
            disabled={!!error}
            className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors flex items-center justify-center"
          >
            <Check className="h-5 w-5 mr-2" />
            Use This Photo
          </button>
        </div>
      </div>
    );
  }

  if (step === "processing") {
    return (
      <div className="bg-gray-100 rounded-xl p-6">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4" />
          <p className="text-gray-700 font-medium mb-2">Processing</p>
          <p className="text-sm text-gray-500 text-center">{progress}</p>
        </div>
      </div>
    );
  }

  if (step === "complete") {
    return (
      <div className="bg-green-50 rounded-xl p-6">
        <div className="flex flex-col items-center">
          <div className="bg-green-100 rounded-full p-3 mb-4">
            <Check className="h-8 w-8 text-green-600" />
          </div>
          <p className="text-green-800 font-medium mb-2">
            Enrollment Complete!
          </p>
          <p className="text-sm text-green-700 text-center">
            Your facial embedding has been securely stored on the blockchain.
            You can now be recognized at any camera in the network.
          </p>
        </div>
      </div>
    );
  }

  return null;
}
