import { useProgram } from "../anchor/setup";
import { useCamera } from "./CameraProvider";
import { PhoneSelfieEnrollment } from "./PhoneSelfieEnrollment";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { PublicKey } from "@solana/web3.js";
import {
  User,
  Smartphone,
  Shield,
  CheckCircle,
  AlertCircle,
  Wifi,
} from "lucide-react";
import { useState, useEffect } from "react";

interface FacialEmbeddingManagerProps {
  walletAddress?: string;
  onComplete?: () => void;
}

export function FacialEmbeddingManager({
  walletAddress,
  onComplete,
}: FacialEmbeddingManagerProps) {
  const [hasEmbedding, setHasEmbedding] = useState<boolean | null>(null);
  const [showEnrollment, setShowEnrollment] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const { selectedCamera } = useCamera();

  // Check if user already has a facial embedding
  useEffect(() => {
    checkExistingEmbedding();
  }, [walletAddress, program]);

  const checkExistingEmbedding = async () => {
    if (!walletAddress || !program) {
      setIsChecking(false);
      return;
    }

    try {
      setIsChecking(true);

      // Check if face NFT PDA exists
      const userPublicKey = new PublicKey(walletAddress);
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("face-nft"), // MUST match the Solana program seed
          userPublicKey.toBuffer(),
        ],
        program.programId
      );

      // Try to fetch the account
      const account = await program.provider.connection.getAccountInfo(
        faceDataPda
      );
      setHasEmbedding(account !== null);
    } catch (error) {
      console.error("Error checking facial embedding:", error);
      setHasEmbedding(false);
    } finally {
      setIsChecking(false);
    }
  };

  const handleEnrollmentComplete = (result: {
    success: boolean;
    error?: string;
    transactionId?: string;
  }) => {
    if (result.success) {
      setHasEmbedding(true);
      setShowEnrollment(false);
      setError(null);
      if (onComplete) onComplete();
    } else {
      console.error("Enrollment failed:", result.error);
      setError(result.error || "Face enrollment failed");
      setShowEnrollment(false);
    }
  };

  if (isChecking) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        <span className="ml-3 text-gray-600">Checking facial embedding...</span>
      </div>
    );
  }

  if (hasEmbedding) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-6">
        <div className="flex items-start space-x-3">
          <CheckCircle className="h-6 w-6 text-green-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-green-800">
              Facial Embedding Ready
            </h3>
            <p className="text-green-700 mt-1">
              Your encrypted facial embedding is stored on-chain. You can now
              use CV apps on mmoment cameras.
            </p>
            <div className="mt-4 flex items-center text-sm text-green-600">
              <Shield className="h-4 w-4 mr-2" />
              <span>Encrypted and secure on Solana blockchain</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (showEnrollment) {
    return (
      <PhoneSelfieEnrollment
        walletAddress={walletAddress}
        onEnrollmentComplete={handleEnrollmentComplete}
        onCancel={() => setShowEnrollment(false)}
      />
    );
  }

  return (
    <div>
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-red-800">Enrollment Failed</h4>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Show connection requirement if not connected to a camera */}
      {!selectedCamera ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <div className="flex items-start space-x-3">
            <Wifi className="h-6 w-6 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-yellow-800">
                Camera Connection Required
              </h3>
              <p className="text-yellow-700 mt-1">
                To create a facial embedding, you must first connect to a
                camera. This ensures your embedding is compatible with all
                cameras in the network.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3 mb-3">
              <User className="h-6 w-6 text-blue-600" />
              <h3 className="text-base font-semibold text-blue-800">
                Create Facial Embedding
              </h3>
            </div>
            <p className="text-sm text-blue-700">
              Create a secure facial embedding to use CV apps on mmoment cameras. This only needs to be done once.
            </p>
          </div>

          <button
            onClick={() => setShowEnrollment(true)}
            disabled={!primaryWallet || !selectedCamera}
            className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center space-x-2 font-medium"
          >
            <Smartphone className="h-5 w-5" />
            <span>Create Facial Embedding</span>
          </button>

          {!primaryWallet && (
            <p className="text-xs text-gray-500 text-center">
              Please connect your wallet first
            </p>
          )}
        </div>
      )}
    </div>
  );
}
