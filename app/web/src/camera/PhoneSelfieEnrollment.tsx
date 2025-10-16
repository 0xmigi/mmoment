import { faceProcessingService } from "./face-processing";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { Transaction } from "@solana/web3.js";
import { useConnection } from "@solana/wallet-adapter-react";
import { Camera, X, RotateCcw, Check, Wifi } from "lucide-react";
import { useState, useRef, useCallback, useEffect } from "react";

interface PhoneSelfieEnrollmentProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: (result: {
    success: boolean;
    error?: string;
    transactionId?: string;
  }) => void;
  onCancel?: () => void;
}

export function PhoneSelfieEnrollment({
  cameraId,
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
  const { connection } = useConnection();

  // Check for user session on mount
  useEffect(() => {
    findUserCurrentSession();
  }, [primaryWallet, cameraId]);

  const findUserCurrentSession = async () => {
    if (!primaryWallet?.address) {
      setError("Please connect your wallet to enroll your face");
      return;
    }

    if (!cameraId) {
      setError("No camera specified. Please access this from a camera page.");
      return;
    }

    try {
      const cameraUrl = `https://${cameraId}.mmoment.xyz`;
      setConnectedCameraUrl(cameraUrl);
      setStep("camera");
      setError(null);
    } catch (error) {
      setError("Unable to initialize camera connection.");
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

      // MUST use Jetson for quality check - no fake local assessment
      setProgress("Checking image quality...");

      console.log('[PhoneSelfieEnrollment] 🔍 DEBUG: connectedCameraUrl =', connectedCameraUrl);
      console.log('[PhoneSelfieEnrollment] 🔍 DEBUG: cameraId =', cameraId);
      console.log('[PhoneSelfieEnrollment] 🔍 DEBUG: Will call processFacialEmbedding with URL:', connectedCameraUrl);

      if (!connectedCameraUrl) {
        setError("No camera connection - cannot assess image quality");
        return;
      }

      console.log('[PhoneSelfieEnrollment] 📞 CALLING faceProcessingService.processFacialEmbedding FOR QUALITY CHECK');
      console.log('[PhoneSelfieEnrollment] 📞 primaryWallet.address =', primaryWallet?.address);
      const result = await faceProcessingService.processFacialEmbedding(
        imageData,
        connectedCameraUrl,
        { requestQuality: true, encrypt: false, walletAddress: primaryWallet?.address }
      );
      console.log('[PhoneSelfieEnrollment] 📞 RESULT from faceProcessingService:', result);

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
  }, [connectedCameraUrl, cameraId]);

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
    console.log('[PhoneSelfieEnrollment] 🚀 processFaceEnrollment called');
    console.log('[PhoneSelfieEnrollment] 🚀 Requirements check:', {
      capturedImage: !!capturedImage,
      primaryWallet: !!primaryWallet,
      connectedCameraUrl: !!connectedCameraUrl
    });

    if (!capturedImage || !primaryWallet || !connectedCameraUrl) {
      console.log('[PhoneSelfieEnrollment] ❌ Missing requirements for enrollment');
      setError("Missing requirements for enrollment");
      return;
    }

    setStep("processing");
    setProgress("Sending image to camera for processing...");

    try {
      console.log('[PhoneSelfieEnrollment] 🔍 ENROLLMENT DEBUG: connectedCameraUrl =', connectedCameraUrl);
      console.log('[PhoneSelfieEnrollment] 🔍 ENROLLMENT DEBUG: walletAddress =', primaryWallet.address);
      console.log('[PhoneSelfieEnrollment] 📞 CALLING faceProcessingService.processFacialEmbedding FOR ENROLLMENT');

      // Send image to Jetson for secure embedding extraction AND transaction building
      // Jetson will handle all biometric processing and return a pre-built transaction
      const result = await faceProcessingService.processFacialEmbedding(
        capturedImage,
        connectedCameraUrl,
        { encrypt: true, requestQuality: true, walletAddress: primaryWallet.address, buildTransaction: true }
      );

      console.log('[PhoneSelfieEnrollment] 📞 ENROLLMENT RESULT from faceProcessingService:', result);
      console.log('[PhoneSelfieEnrollment] 📞 Full result object keys:', Object.keys(result));
      console.log('[PhoneSelfieEnrollment] 📞 Result details:', {
        success: result.success,
        hasTransaction: !!result.transactionBuffer,
        hasQuality: !!result.quality,
        hasEmbedding: !!result.embedding,
        error: result.error,
        allFields: Object.keys(result).join(', ')
      });

      if (!result.success) {
        console.error('[PhoneSelfieEnrollment] ❌ Result not successful:', result.error);
        throw new Error(result.error || "Failed to process facial features");
      }

      if (!result.transactionBuffer) {
        console.error('[PhoneSelfieEnrollment] ❌ No transaction buffer found in result');
        console.error('[PhoneSelfieEnrollment] ❌ Available fields:', Object.keys(result));
        throw new Error("No transaction received from Jetson - enrollment failed. Check if buildTransaction option is working.");
      }

      console.log('[PhoneSelfieEnrollment] Enhanced processing result:', {
        hasTransaction: !!result.transactionBuffer,
        sessionId: result.sessionId,
        faceId: result.face_id,
        faceNftPda: result.face_nft_pda,
        qualityScore: result.quality?.score,
        qualityRating: result.quality?.rating
      });

      // Log quality information if available
      if (result.quality) {
        console.log('[PhoneSelfieEnrollment] Quality assessment:', result.quality);
        setProgress(`Quality: ${result.quality.rating} (${result.quality.score}%) - Processing...`);
      }

      console.log('[PhoneSelfieEnrollment] ✅ Transaction received from Jetson');
      console.log('[PhoneSelfieEnrollment] 📜 Transaction buffer type:', typeof result.transactionBuffer);
      console.log('[PhoneSelfieEnrollment] 📜 Transaction buffer preview:', result.transactionBuffer?.substring(0, 100));

      // Check if the transaction buffer is actually JSON instead of base64
      let transaction: Transaction;

      if (result.transactionBuffer.startsWith('{')) {
        console.error('[PhoneSelfieEnrollment] ❌ Jetson returned JSON instead of serialized transaction!');
        console.error('[PhoneSelfieEnrollment] ❌ JSON content:', result.transactionBuffer);
        throw new Error('Jetson API error: Transaction not properly serialized. The Jetson needs to return a base64-encoded Solana transaction, not JSON.');
      }

      setProgress("Signing blockchain transaction...");

      // Deserialize the transaction buffer from Jetson
      console.log('[PhoneSelfieEnrollment] 📜 Deserializing transaction from Jetson...');
      try {
        const txBuffer = Buffer.from(result.transactionBuffer, 'base64');
        transaction = Transaction.from(txBuffer);
        console.log('[PhoneSelfieEnrollment] 📜 Transaction deserialized successfully');
      } catch (deserializeError) {
        console.error('[PhoneSelfieEnrollment] ❌ Failed to deserialize transaction:', deserializeError);
        console.error('[PhoneSelfieEnrollment] ❌ Buffer content (first 200 chars):', result.transactionBuffer.substring(0, 200));
        throw new Error('Failed to deserialize transaction from Jetson. The API may not be returning a proper Solana transaction.');
      }

      // Sign the pre-built transaction
      console.log('[PhoneSelfieEnrollment] ✍️ Getting wallet signer...');
      let signedTx: Transaction;
      let signature: string;

      try {
        // Try to get signer first (Dynamic wallet pattern)
        if (typeof (primaryWallet as any).getSigner === 'function') {
          console.log('[PhoneSelfieEnrollment] ✍️ Wallet has getSigner method, attempting to get signer...');
          try {
            const signer = await (primaryWallet as any).getSigner();
            console.log('[PhoneSelfieEnrollment] ✍️ Wallet signer obtained');

            if (typeof signer.signTransaction === 'function') {
              console.log('[PhoneSelfieEnrollment] ✍️ Using signer.signTransaction...');
              signedTx = await signer.signTransaction(transaction);
              console.log('[PhoneSelfieEnrollment] ✍️ Transaction signed successfully via signer');
            } else {
              console.log('[PhoneSelfieEnrollment] ⚠️ Signer does not have signTransaction, falling back to direct wallet signing');
              signedTx = await (primaryWallet as any).signTransaction(transaction);
              console.log('[PhoneSelfieEnrollment] ✍️ Transaction signed successfully via wallet fallback');
            }
          } catch (signerError) {
            console.warn('[PhoneSelfieEnrollment] ⚠️ getSigner failed, trying direct wallet signTransaction:', signerError);
            signedTx = await (primaryWallet as any).signTransaction(transaction);
            console.log('[PhoneSelfieEnrollment] ✍️ Transaction signed successfully via direct wallet method');
          }
        } else {
          // Wallet doesn't have getSigner, use signTransaction directly
          console.log('[PhoneSelfieEnrollment] ✍️ Wallet does not have getSigner, using direct signTransaction...');
          signedTx = await (primaryWallet as any).signTransaction(transaction);
          console.log('[PhoneSelfieEnrollment] ✍️ Transaction signed successfully via direct wallet');
        }
      } catch (signingError) {
        console.error('[PhoneSelfieEnrollment] ❌ All signing methods failed:', signingError);
        throw new Error(`Transaction signing failed: ${signingError instanceof Error ? signingError.message : 'User rejected or signing failed'}`);
      }

      // Submit signed transaction to Solana blockchain
      console.log('[PhoneSelfieEnrollment] 🚀 Submitting transaction to Solana blockchain...');
      setProgress("Submitting to blockchain...");

      // Validate we have all required fields
      if (!result.face_id) {
        console.error('[PhoneSelfieEnrollment] ❌ Missing face_id in result:', result);
        throw new Error('Missing face_id from Jetson - cannot confirm enrollment');
      }

      if (!connection) {
        throw new Error('Solana connection not available');
      }

      try {
        // Submit the signed transaction to Solana
        console.log('[PhoneSelfieEnrollment] 📡 Sending transaction to Solana...');
        signature = await connection.sendRawTransaction(signedTx.serialize());
        console.log('[PhoneSelfieEnrollment] ✅ Transaction sent! Signature:', signature);

        // Wait for confirmation
        setProgress("Waiting for blockchain confirmation...");
        console.log('[PhoneSelfieEnrollment] ⏳ Waiting for transaction confirmation...');
        await connection.confirmTransaction(signature, 'confirmed');
        console.log('[PhoneSelfieEnrollment] ✅ Transaction confirmed on blockchain!');

        // Notify Jetson that enrollment succeeded (for local storage/cleanup)
        console.log('[PhoneSelfieEnrollment] 📞 Notifying Jetson of successful enrollment...');
        try {
          await fetch(`${connectedCameraUrl}/api/face/enroll/confirm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              wallet_address: primaryWallet.address,
              face_id: result.face_id,
              transaction_signature: signature,
              biometric_session_id: result.sessionId
            })
          });
          console.log('[PhoneSelfieEnrollment] ✅ Jetson notified successfully');
        } catch (notifyError) {
          // Non-fatal - blockchain transaction already succeeded
          console.warn('[PhoneSelfieEnrollment] ⚠️ Failed to notify Jetson (non-fatal):', notifyError);
        }

        console.log('[PhoneSelfieEnrollment] 🎉 Enrollment successful with signature:', signature);

        setStep("complete");
        setProgress("Facial embedding successfully stored!");

        setTimeout(() => {
          console.log('[PhoneSelfieEnrollment] 📞 Calling onEnrollmentComplete callback');
          onEnrollmentComplete?.({
            success: true,
            transactionId: signature,
          });
        }, 2000);

      } catch (confirmError) {
        console.error('[PhoneSelfieEnrollment] ❌ Jetson confirmation failed:', confirmError);

        // Provide specific error messages for common issues
        let errorMessage = 'Transaction confirmation failed';
        if (confirmError instanceof Error) {
          if (confirmError.message.includes('0x1778')) {
            errorMessage = 'Account already exists - you may already have a face enrollment';
          } else if (confirmError.message.includes('insufficient')) {
            errorMessage = 'Insufficient SOL balance for transaction fees';
          } else if (confirmError.message.includes('timeout')) {
            errorMessage = 'Network timeout - please try again';
          } else {
            errorMessage = `Blockchain error: ${confirmError.message}`;
          }
        }
        throw new Error(errorMessage);
      }
    } catch (err) {
      console.error('[PhoneSelfieEnrollment] ❌ Enrollment error:', err);
      let errorMessage = "Failed to enroll facial embedding.";

      if (err instanceof Error) {
        if (err.message.includes("already exists")) {
          errorMessage = "You already have a facial embedding stored.";
        } else if (err.message.includes("insufficient funds")) {
          errorMessage = "Insufficient SOL balance for transaction.";
        } else if (err.message.includes("No transaction received")) {
          errorMessage = "Jetson did not return a transaction. The API may need updating to support buildTransaction.";
        } else {
          errorMessage = err.message;
        }
      }

      console.error('[PhoneSelfieEnrollment] ❌ Setting error:', errorMessage);
      setError(errorMessage);
      setStep("preview");
      setProgress("");
      onEnrollmentComplete?.({
        success: false,
        error: errorMessage,
      });
    }
  }, [
    capturedImage,
    primaryWallet,
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
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-white/10 backdrop-blur-sm z-20">
          <button
            onClick={capturePhoto}
            disabled={isCapturing}
            className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors flex items-center justify-center text-sm font-medium shadow-lg pointer-events-auto"
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
