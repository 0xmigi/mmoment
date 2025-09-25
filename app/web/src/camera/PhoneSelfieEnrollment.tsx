import { useProgram } from "../anchor/setup";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { Camera, X, RotateCcw, Check } from "lucide-react";
import { useState, useRef, useCallback, useEffect } from "react";

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
  walletAddress,
  onEnrollmentComplete,
  onCancel,
}: PhoneSelfieEnrollmentProps) {
  const [step, setStep] = useState<
    "camera" | "preview" | "processing" | "complete"
  >("camera");
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();

  // Initialize camera
  useEffect(() => {
    initializeCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

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
        } else if (
          err.name === "NotReadableError" ||
          err.name === "TrackStartError"
        ) {
          errorMessage =
            "Camera is already in use by another app. Please close other apps using the camera.";
        } else if (err.name === "OverconstrainedError") {
          errorMessage =
            "Camera does not support the required settings. Trying default settings...";
          // Try again with basic constraints
          try {
            const basicStream = await navigator.mediaDevices.getUserMedia({
              video: { facingMode: "user" },
              audio: false,
            });
            setStream(basicStream);
            if (videoRef.current) {
              videoRef.current.srcObject = basicStream;
              videoRef.current.setAttribute("playsinline", "true");
              videoRef.current.setAttribute("autoplay", "true");
              videoRef.current.muted = true;
            }
            return; // Success with basic settings
          } catch {
            errorMessage = "Unable to access camera even with basic settings.";
          }
        }
      }
      setError(errorMessage);
    }
  };

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsCapturing(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL("image/jpeg", 0.9);
    setCapturedImage(imageData);
    setStep("preview");
    setIsCapturing(false);

    // Stop video stream
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  }, [stream]);

  const retakePhoto = () => {
    setCapturedImage(null);
    setStep("camera");
    initializeCamera();
  };

  const processAndEnroll = async () => {
    console.log("[PhoneSelfieEnrollment] Starting enrollment process...");
    console.log("[PhoneSelfieEnrollment] Wallet address:", walletAddress);
    console.log("[PhoneSelfieEnrollment] Primary wallet:", primaryWallet);
    console.log("[PhoneSelfieEnrollment] Program available:", !!program);

    if (!capturedImage) {
      setError("No image captured");
      return;
    }

    if (!walletAddress) {
      setError("No wallet address found. Please connect your wallet.");
      return;
    }

    if (!primaryWallet) {
      setError("No wallet connected. Please connect your wallet.");
      return;
    }

    if (!program) {
      setError("Solana program not loaded. Please refresh and try again.");
      return;
    }

    setStep("processing");
    setProgress("Processing facial features...");

    try {
      // Step 1: Process image to extract facial embedding
      const embedding = await processFacialEmbedding(capturedImage);

      setProgress("Preparing blockchain transaction...");

      // Step 2: Create embedding buffer for Solana
      const embeddingBuffer = Buffer.from(embedding);

      if (embeddingBuffer.length > 1024) {
        throw new Error(
          `Face embedding too large (${embeddingBuffer.length} bytes). Max allowed: 1024 bytes.`
        );
      }

      setProgress("Storing encrypted facial embedding on blockchain...");

      // Step 3: Store encrypted facial embedding on blockchain
      const userPublicKey = new PublicKey(walletAddress);
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("face-nft"), // CRITICAL: Must be "face-nft" not "face-embedding"
          userPublicKey.toBuffer(),
        ],
        program.programId
      );

      console.log(
        "[PhoneSelfieEnrollment] Face NFT PDA:",
        faceDataPda.toString()
      );

      // Dynamic handles ALL wallet operations - just use RPC
      console.log(
        "[PhoneSelfieEnrollment] Submitting transaction for user:",
        userPublicKey.toString()
      );

      let txSignature: string;
      try {
        txSignature = await program.methods
          .enrollFace(embeddingBuffer)
          .accounts({
            user: userPublicKey,
            faceNft: faceDataPda,
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        console.log(
          "[PhoneSelfieEnrollment] Transaction successful:",
          txSignature
        );
      } catch (rpcError) {
        console.error("[PhoneSelfieEnrollment] Transaction failed:", rpcError);
        const errorMessage =
          rpcError instanceof Error ? rpcError.message : "Unknown error";

        // Parse common Solana errors for better user feedback
        if (errorMessage.includes("insufficient")) {
          throw new Error(
            "Insufficient SOL balance. Please add SOL to your wallet."
          );
        } else if (errorMessage.includes("already")) {
          throw new Error("Face already enrolled. You can only enroll once.");
        } else {
          throw new Error(`Transaction failed: ${errorMessage}`);
        }
      }

      setStep("complete");
      console.log("Face enrollment successful! Transaction:", txSignature);

      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: true,
          transactionId: txSignature,
        });
      }
    } catch (err) {
      console.error("Face enrollment failed:", err);
      setError(err instanceof Error ? err.message : "Face enrollment failed");

      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: false,
          error: err instanceof Error ? err.message : "Face enrollment failed",
        });
      }
    }
  };

  const processFacialEmbedding = async (
    imageData: string
  ): Promise<number[]> => {
    // Import the face processing service
    const { faceProcessingService } = await import("./face-processing");

    const result = await faceProcessingService.processFacialEmbedding(
      imageData
    );

    if (!result.success || !result.embedding) {
      throw new Error(result.error || "Face processing failed");
    }

    return result.embedding;
  };

  const handleCancel = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (onCancel) onCancel();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-95 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white flex items-center justify-between p-4 border-b">
          <h3 className="text-lg font-semibold">Create Face ID</h3>
          <button
            onClick={handleCancel}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {step === "camera" && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Position your face in the frame and tap capture when ready.
              </p>

              <div className="relative aspect-[3/4] bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover mirror"
                  style={{ transform: "scaleX(-1)" }} // Mirror for selfie camera
                />

                {/* Face guide overlay */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="w-48 h-48 border-2 border-white rounded-full opacity-50 shadow-lg" />
                  <div className="absolute text-white text-xs bottom-4 text-center">
                    <p>Position your face within the circle</p>
                  </div>
                </div>
              </div>

              {/* Hidden canvas for capture */}
              <canvas
                ref={canvasRef}
                className="hidden"
                width={640}
                height={480}
              />

              {error && (
                <div className="text-red-600 text-sm bg-red-50 p-2 rounded">
                  {error}
                </div>
              )}

              <button
                onClick={capturePhoto}
                disabled={isCapturing || !stream}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center space-x-2"
              >
                <Camera className="h-5 w-5" />
                <span>{isCapturing ? "Capturing..." : "Capture Photo"}</span>
              </button>
            </div>
          )}

          {step === "preview" && capturedImage && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Review your photo. Does it look good?
              </p>

              <img
                src={capturedImage}
                alt="Captured selfie"
                className="w-full h-64 object-cover rounded-lg"
              />

              <div className="flex space-x-2">
                <button
                  onClick={retakePhoto}
                  className="flex-1 bg-gray-200 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-300 flex items-center justify-center space-x-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  <span>Retake</span>
                </button>

                <button
                  onClick={processAndEnroll}
                  className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 flex items-center justify-center space-x-2"
                >
                  <Check className="h-4 w-4" />
                  <span>Use Photo</span>
                </button>
              </div>
            </div>
          )}

          {step === "processing" && (
            <div className="space-y-4 text-center">
              <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-gray-600">{progress}</p>
            </div>
          )}

          {step === "complete" && (
            <div className="space-y-4 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <Check className="h-8 w-8 text-green-600" />
              </div>
              <div>
                <h4 className="text-lg font-semibold text-green-600">
                  Facial Embedding Created!
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  Your encrypted facial embedding has been securely stored
                  on-chain. You can now use CV apps on mmoment cameras.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Canvas for image processing (hidden) */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
