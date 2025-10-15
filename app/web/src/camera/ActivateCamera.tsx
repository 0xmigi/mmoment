import { useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { SystemProgram, Keypair, PublicKey } from '@solana/web3.js';
import { Camera, Loader } from 'lucide-react';
import { CONFIG } from '../core/config';
import { unifiedIpfsService } from '../storage/ipfs/unified-ipfs-service';

interface ActivateCameraProps {
  onCameraUpdate?: (params: { address: string; isLive: boolean }) => void;
  onInitialize?: () => void;
  onPhotoCapture?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info', message: string }) => void;
}

export const ActivateCamera = forwardRef<{ handleTakePicture: () => Promise<void> }, ActivateCameraProps>(
  ({ onPhotoCapture, onStatusUpdate }, ref) => {
    const { primaryWallet } = useDynamicContext();
    useConnection();
    const { program } = useProgram();
    const [loading, setLoading] = useState(false);
    const [, setShowTooltip] = useState(false);
    const buttonRef = useRef<HTMLButtonElement>(null);
    
    // Persist camera keypair
    const [cameraKeypair] = useState(() => {
      const stored = localStorage.getItem('cameraKeypair');
      if (stored) {
        const keypairData = new Uint8Array(JSON.parse(stored));
        return Keypair.fromSecretKey(keypairData);
      }
      const newKeypair = Keypair.generate();
      localStorage.setItem('cameraKeypair', JSON.stringify(Array.from(newKeypair.secretKey)));
      return newKeypair;
    });

    useImperativeHandle(ref, () => ({
      handleTakePicture
    }));

    const handleTakePicture = async () => {
      if (!primaryWallet?.address || !program) return;
      setLoading(true);

      try {
        onStatusUpdate?.({ type: 'info', message: 'Recording camera activity...' });
        
        const userPublicKey = new PublicKey(primaryWallet.address);

        // Find the session PDA
        const [sessionPda] = PublicKey.findProgramAddressSync(
          [
            Buffer.from('session'),
            userPublicKey.toBuffer(),
            cameraKeypair.publicKey.toBuffer()
          ],
          program.programId
        );

        // Derive recognition token PDA and check if it exists
        const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
          [Buffer.from('recognition-token'), userPublicKey.toBuffer()],
          program.programId
        );

        // Check if recognition token account exists
        let hasRecognitionToken = false;
        try {
          await (program.account as any).recognitionToken.fetch(recognitionTokenPda);
          hasRecognitionToken = true;
        } catch (err) {
          // No token - that's fine
        }

        // Build accounts object
        const checkInAccounts: any = {
          user: userPublicKey,
          camera: cameraKeypair.publicKey,
          session: sessionPda,
          systemProgram: SystemProgram.programId,
        };

        // Only include recognition token if it exists
        if (hasRecognitionToken) {
          checkInAccounts.recognitionToken = recognitionTokenPda;
        }

        // Use checkIn instead of recordActivity
        await program.methods.checkIn(false)
        .accounts(checkInAccounts)
        .rpc();

        onStatusUpdate?.({ type: 'info', message: 'Taking picture...' });
        const apiUrl = `${CONFIG.CAMERA_API_URL}/api/capture`;
        const captureResponse = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${primaryWallet.address}`,
          },
          cache: 'no-cache',
          referrerPolicy: 'no-referrer'
        });

        if (!captureResponse.ok) {
          const errorText = await captureResponse.text();
          console.error('Camera response error:', {
            status: captureResponse.status,
            statusText: captureResponse.statusText,
            body: errorText
          });
          throw new Error(`Camera error: ${captureResponse.status} ${captureResponse.statusText}`);
        }

        onStatusUpdate?.({ type: 'info', message: 'Processing image...' });
        const imageBlob = await captureResponse.blob();

        onStatusUpdate?.({ type: 'info', message: 'Uploading to IPFS...' });
        const results = await unifiedIpfsService.uploadFile(imageBlob, primaryWallet.address, 'image');
        
        if (results.length === 0) {
          throw new Error('Failed to upload image to any IPFS provider');
        }

        onStatusUpdate?.({ type: 'success', message: 'Picture uploaded successfully!' });
        onPhotoCapture?.();

      } catch (error) {
        console.error('Failed to take picture:', error);
        onStatusUpdate?.({ 
          type: 'error', 
          message: error instanceof Error ? error.message : 'Failed to take picture' 
        });
      } finally {
        setLoading(false);
      }
    };

    const handlePhotoClick = () => {
      console.log("PHOTO BUTTON CLICKED IN COMPONENT - direct handler");
      if (onPhotoCapture) {
        onPhotoCapture();
      } else {
        console.error("No photo capture callback provided");
        onStatusUpdate?.({ type: 'error', message: 'Photo capture callback not configured' });
      }
    };

    return (
      <button
        ref={buttonRef}
        onClick={handlePhotoClick}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        disabled={loading || !primaryWallet?.address}
        className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
      >
        {loading ? (
          <Loader className="w-5 h-5 animate-spin" />
        ) : (
          <Camera className="w-5 h-5" />
        )}
      </button>
    );
  }
);

ActivateCamera.displayName = 'ActivateCamera';