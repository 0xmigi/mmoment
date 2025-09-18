import { useState } from 'react';
import { User } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { unifiedCameraService } from './unified-camera-service';
import { CameraActionResponse } from './camera-interface';
import { PublicKey, SystemProgram, Transaction } from '@solana/web3.js';
import { useProgram } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';

interface FaceEnrollmentButtonProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: (result: CameraActionResponse<{ enrolled: boolean; faceId: string; transactionId?: string }>) => void;
}

export function FaceEnrollmentButton({ cameraId, walletAddress, onEnrollmentComplete }: FaceEnrollmentButtonProps) {
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentStep, setEnrollmentStep] = useState<'idle' | 'preparing' | 'submitting' | 'confirming'>('idle');
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();

  const handleEnrollFace = async () => {
    // PREVENT MULTIPLE SIMULTANEOUS CALLS
    if (isEnrolling) {
      console.log('[FaceEnrollmentButton] Already enrolling, ignoring duplicate call');
      return;
    }

    if (!walletAddress || !primaryWallet || !program) {
      console.error('Missing requirements for face enrollment');
      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: false,
          error: 'Wallet or program not available'
        });
      }
      return;
    }

    console.log('[FaceEnrollmentButton] 🚀 Starting face enrollment process...');
    setIsEnrolling(true);
    setEnrollmentStep('preparing');

    try {
      console.log('[FaceEnrollmentButton] Phase 1: Preparing face enrollment transaction...');
      
      // Phase 1: Get face data from Jetson
      const prepareResult = await unifiedCameraService.prepareFaceEnrollmentTransaction(cameraId, walletAddress);
      
      console.log('[FaceEnrollmentButton] Prepare result:', prepareResult);
      
      if (!prepareResult.success || !prepareResult.data) {
        throw new Error(prepareResult.error || 'Failed to prepare face enrollment');
      }

      if (!prepareResult.data.transactionBuffer || !prepareResult.data.faceId) {
        throw new Error('Invalid response: missing transaction data');
      }

      // Parse the transaction buffer to get embedding data
      const transactionData = JSON.parse(prepareResult.data.transactionBuffer);
      const embeddingData = transactionData.args?.embedding;
      
      if (!embeddingData) {
        throw new Error('Missing embedding data from backend');
      }

      console.log('[FaceEnrollmentButton] ✅ Got embedding data from backend');
      console.log('[FaceEnrollmentButton] Embedding data type:', typeof embeddingData);
      console.log('[FaceEnrollmentButton] Embedding data length:', embeddingData.length);
      console.log('[FaceEnrollmentButton] Embedding data preview:', embeddingData.slice(0, 100));
      
      // Phase 2: Submit Anchor transaction directly (like other transactions in your app)
      setEnrollmentStep('submitting');
      console.log('[FaceEnrollmentButton] Phase 2: Submitting transaction to blockchain...');
      
      // Prepare the embedding buffer with proper error handling
      let embeddingBuffer: Buffer;
      try {
        // Check if it's already a JSON string that needs parsing
        let processedEmbeddingData = embeddingData;
        if (typeof embeddingData === 'string' && embeddingData.startsWith('{')) {
          console.log('[FaceEnrollmentButton] Embedding data appears to be JSON, parsing...');
          const parsedData = JSON.parse(embeddingData);
          processedEmbeddingData = parsedData.binaryData || parsedData.data || parsedData;
        }
        
        // Convert to buffer
        if (Array.isArray(processedEmbeddingData)) {
          console.log('[FaceEnrollmentButton] Converting array to buffer...');
          embeddingBuffer = Buffer.from(processedEmbeddingData);
        } else if (typeof processedEmbeddingData === 'string') {
          console.log('[FaceEnrollmentButton] Converting base64 string to buffer...');
          embeddingBuffer = Buffer.from(processedEmbeddingData, 'base64');
        } else {
          throw new Error(`Unexpected embedding data format: ${typeof processedEmbeddingData}`);
        }
        
        console.log('[FaceEnrollmentButton] ✅ Embedding buffer created, size:', embeddingBuffer.length);
        
        // Validate buffer size against Solana program limits
        if (embeddingBuffer.length === 0) {
          throw new Error('Embedding buffer is empty');
        }
        
        // CRITICAL: Your Solana program only accepts up to 1024 bytes!
        if (embeddingBuffer.length > 1024) {
          console.error('[FaceEnrollmentButton] ❌ CRITICAL: Embedding too large for Solana program!');
          console.error('[FaceEnrollmentButton] Backend sent:', embeddingBuffer.length, 'bytes');
          console.error('[FaceEnrollmentButton] Solana program accepts max:', 1024, 'bytes');
          throw new Error(`❌ BACKEND ERROR: Face embedding is ${embeddingBuffer.length} bytes, but Solana program only accepts up to 1024 bytes. Backend needs to compress or reduce the embedding size!`);
        }
        
        console.log('[FaceEnrollmentButton] ✅ Buffer size valid for Solana program, proceeding with transaction...');
        
      } catch (bufferError) {
        console.error('[FaceEnrollmentButton] Error creating embedding buffer:', bufferError);
        console.error('[FaceEnrollmentButton] Raw embedding data:', embeddingData);
        throw new Error(`Failed to process embedding data: ${bufferError instanceof Error ? bufferError.message : 'Unknown buffer error'}`);
      }
      
      // Generate the face NFT PDA (same as program)
      const userPublicKey = new PublicKey(walletAddress);
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('face-nft'),
          userPublicKey.toBuffer()
        ],
        program.programId
      );
      
      console.log('[FaceEnrollmentButton] Face NFT PDA:', faceDataPda.toString());
      
      // Check if it's a Solana wallet to determine signing method
      if (!isSolanaWallet(primaryWallet)) {
        // For embedded wallets - use direct .rpc() method (like other transactions)
        console.log('[FaceEnrollmentButton] Using embedded wallet transaction flow');
        
        const txSignature = await program.methods
          .enrollFace(embeddingBuffer) // Use Buffer directly for Anchor
          .accounts({
            user: userPublicKey,
            faceNft: faceDataPda,
            systemProgram: SystemProgram.programId
          })
          .rpc();
          
        console.log('[FaceEnrollmentButton] ✅ Transaction confirmed:', txSignature);
        
        // Phase 3: Confirm with backend
        setEnrollmentStep('confirming');
        console.log('[FaceEnrollmentButton] Phase 3: Confirming with backend...');
        
        const confirmationData = {
          signedTransaction: txSignature, // For embedded wallets, pass the signature
          faceId: prepareResult.data.faceId,
          biometricSessionId: prepareResult.data.metadata?.biometric_session_id
        };
        
        const confirmResult = await unifiedCameraService.confirmFaceEnrollmentTransaction(
          cameraId,
          walletAddress,
          confirmationData
        );
        
        if (confirmResult.success) {
          console.log('[FaceEnrollmentButton] ✅ Face enrollment completed successfully!');
          
          if (onEnrollmentComplete) {
            onEnrollmentComplete({
              success: true,
              data: {
                enrolled: true,
                faceId: prepareResult.data.faceId,
                transactionId: txSignature
              }
            });
          }
        } else {
          throw new Error(confirmResult.error || 'Backend confirmation failed');
        }
        
      } else {
        // For browser wallets - use manual signing (like TransactionModal.tsx)
        console.log('[FaceEnrollmentButton] Using browser wallet transaction flow');
        
        const connection = await primaryWallet.getConnection();
        
        // Build instruction
        const ix = await program.methods
          .enrollFace(embeddingBuffer)
          .accounts({
            user: userPublicKey,
            faceNft: faceDataPda,
            systemProgram: SystemProgram.programId
          })
          .instruction();
        
        // Create and send transaction
        const tx = new Transaction().add(ix);
        const { blockhash } = await connection.getLatestBlockhash();
        tx.recentBlockhash = blockhash;
        tx.feePayer = userPublicKey;
        
        const signer = await primaryWallet.getSigner();
        const signedTx = await signer.signTransaction(tx);
        const txSignature = await connection.sendRawTransaction(signedTx.serialize());
        await connection.confirmTransaction(txSignature, 'confirmed');
        
        console.log('[FaceEnrollmentButton] ✅ Transaction confirmed:', txSignature);
        
        // Phase 3: Confirm with backend
        setEnrollmentStep('confirming');
        console.log('[FaceEnrollmentButton] Phase 3: Confirming with backend...');
        
        const confirmationData = {
          signedTransaction: signedTx.serialize().toString('base64'),
          faceId: prepareResult.data.faceId,
          biometricSessionId: prepareResult.data.metadata?.biometric_session_id
        };
        
        const confirmResult = await unifiedCameraService.confirmFaceEnrollmentTransaction(
          cameraId,
          walletAddress,
          confirmationData
        );
        
        if (confirmResult.success) {
          console.log('[FaceEnrollmentButton] ✅ Face enrollment completed successfully!');
          
          if (onEnrollmentComplete) {
            onEnrollmentComplete({
              success: true,
              data: {
                enrolled: true,
                faceId: prepareResult.data.faceId,
                transactionId: txSignature
              }
            });
          }
        } else {
          throw new Error(confirmResult.error || 'Backend confirmation failed');
        }
      }
      
    } catch (error) {
      console.error('[FaceEnrollmentButton] Enrollment error:', error);
      if (onEnrollmentComplete) {
        onEnrollmentComplete({
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error during face enrollment'
        });
      }
    } finally {
      setIsEnrolling(false);
      setEnrollmentStep('idle');
    }
  };

  // Only show the button if we have a wallet address
  if (!walletAddress) {
    return null;
  }

  const getButtonText = () => {
    switch (enrollmentStep) {
      case 'preparing':
        return 'Capturing Face...';
      case 'submitting':
        return 'Minting NFT...';
      case 'confirming':
        return 'Confirming...';
      default:
        return isEnrolling ? 'Processing...' : 'Enroll Face';
    }
  };

  return (
    <button
      onClick={handleEnrollFace}
      disabled={isEnrolling || !program}
      className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-3 py-2 rounded-lg shadow-lg transition-colors"
      title="Enroll Face & Mint NFT"
    >
      <User className="w-4 h-4" />
      <span className="text-sm">
        {getButtonText()}
      </span>
    </button>
  );
} 