import { useState, useRef, useCallback, useEffect } from 'react';
import { Camera, X, RotateCcw, Check } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey, SystemProgram } from '@solana/web3.js';
import { useProgram } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';

interface PhoneSelfieEnrollmentProps {
  walletAddress?: string;
  onEnrollmentComplete?: (result: { success: boolean; error?: string; transactionId?: string }) => void;
  onCancel?: () => void;
}

export function PhoneSelfieEnrollment({
  walletAddress,
  onEnrollmentComplete,
  onCancel
}: PhoneSelfieEnrollmentProps) {
  const [step, setStep] = useState<'camera' | 'preview' | 'processing' | 'complete'>('camera');
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();

  // Initialize camera
  useEffect(() => {
    initializeCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const initializeCamera = async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user', // Front camera
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsCapturing(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImage(imageData);
    setStep('preview');
    setIsCapturing(false);

    // Stop video stream
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [stream]);

  const retakePhoto = () => {
    setCapturedImage(null);
    setStep('camera');
    initializeCamera();
  };

  const processAndEnroll = async () => {
    if (!capturedImage || !walletAddress || !primaryWallet || !program) {
      setError('Missing requirements for enrollment');
      return;
    }

    setStep('processing');
    setProgress('Processing facial features...');

    try {
      // Step 1: Process image to extract facial embedding
      const embedding = await processFacialEmbedding(capturedImage);

      setProgress('Preparing blockchain transaction...');

      // Step 2: Create embedding buffer for Solana
      const embeddingBuffer = Buffer.from(embedding);

      if (embeddingBuffer.length > 1024) {
        throw new Error(`Face embedding too large (${embeddingBuffer.length} bytes). Max allowed: 1024 bytes.`);
      }

      setProgress('Storing encrypted facial embedding on blockchain...');

      // Step 3: Store encrypted facial embedding on blockchain
      const userPublicKey = new PublicKey(walletAddress);
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('face-embedding'),
          userPublicKey.toBuffer()
        ],
        program.programId
      );

      let txSignature: string;

      if (!isSolanaWallet(primaryWallet)) {
        // Embedded wallet flow
        txSignature = await program.methods
          .enrollFace(embeddingBuffer)
          .accounts({
            user: userPublicKey,
            faceNft: faceDataPda,
            systemProgram: SystemProgram.programId
          })
          .rpc();
      } else {
        // External Solana wallet flow
        const transaction = await program.methods
          .enrollFace(embeddingBuffer)
          .accounts({
            user: userPublicKey,
            faceNft: faceDataPda,
            systemProgram: SystemProgram.programId
          })
          .transaction();

        const signedTx = await (primaryWallet as any).signTransaction(transaction);
        txSignature = await program.provider.connection.sendRawTransaction(
          signedTx.serialize()
        );

        await program.provider.connection.confirmTransaction(txSignature);
      }

      setStep('complete');
      console.log('Face enrollment successful! Transaction:', txSignature);

      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: true,
          transactionId: txSignature
        });
      }

    } catch (err) {
      console.error('Face enrollment failed:', err);
      setError(err instanceof Error ? err.message : 'Face enrollment failed');

      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: false,
          error: err instanceof Error ? err.message : 'Face enrollment failed'
        });
      }
    }
  };

  const processFacialEmbedding = async (imageData: string): Promise<number[]> => {
    // Import the face processing service
    const { faceProcessingService } = await import('./face-processing');

    const result = await faceProcessingService.processFacialEmbedding(imageData);

    if (!result.success || !result.embedding) {
      throw new Error(result.error || 'Face processing failed');
    }

    return result.embedding;
  };

  const handleCancel = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    if (onCancel) onCancel();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg max-w-md w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="text-lg font-semibold">Create Face ID</h3>
          <button onClick={handleCancel} className="p-1 hover:bg-gray-100 rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {step === 'camera' && (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Position your face in the frame and tap capture when ready.
              </p>

              <div className="relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-64 bg-black rounded-lg object-cover"
                />

                {/* Face guide overlay */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-48 h-48 border-2 border-white rounded-full opacity-50" />
                </div>
              </div>

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
                <span>{isCapturing ? 'Capturing...' : 'Capture Photo'}</span>
              </button>
            </div>
          )}

          {step === 'preview' && capturedImage && (
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

          {step === 'processing' && (
            <div className="space-y-4 text-center">
              <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-gray-600">{progress}</p>
            </div>
          )}

          {step === 'complete' && (
            <div className="space-y-4 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <Check className="h-8 w-8 text-green-600" />
              </div>
              <div>
                <h4 className="text-lg font-semibold text-green-600">Facial Embedding Created!</h4>
                <p className="text-sm text-gray-600 mt-1">
                  Your encrypted facial embedding has been securely stored on-chain. You can now use CV apps on mmoment cameras.
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