// ActivateCamera.tsx
import { useState, forwardRef, useImperativeHandle } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { SystemProgram, Keypair, PublicKey } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { pinataService } from '../services/pinata-service';
import { Camera, Loader } from 'lucide-react';
import { CONFIG } from '../config';

interface ActivateCameraProps {
  onCameraUpdate?: (params: { address: string; isLive: boolean }) => void;
  onInitialize?: () => void;
  onPhotoCapture?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info', message: string }) => void;
}

export const ActivateCamera = forwardRef<{ handleTakePicture: () => Promise<void> }, ActivateCameraProps>(({ onCameraUpdate, onInitialize, onPhotoCapture, onStatusUpdate }, ref) => {

    const { primaryWallet } = useDynamicContext();
    useConnection();
    useImperativeHandle(ref, () => ({
      handleTakePicture
    }));
    const program = useProgram();
    const [loading, setLoading] = useState(false);
    const [cameraKeypair] = useState(() => Keypair.generate());
    const [isInitialized, setIsInitialized] = useState(false);

    const initializeCamera = async () => {
      if (!primaryWallet?.address || !program || isInitialized) return;

      try {
        onStatusUpdate?.({ type: 'info', message: 'Initializing camera...' });
        await program.methods.initialize()
          .accounts({
            cameraAccount: cameraKeypair.publicKey,
            user: new PublicKey(primaryWallet.address),
            systemProgram: SystemProgram.programId,
          })
          .signers([cameraKeypair])
          .rpc();

        setIsInitialized(true);
        onCameraUpdate?.({
          address: cameraKeypair.publicKey.toString(),
          isLive: true
        });
        onInitialize?.();
      } catch (error) {
        onStatusUpdate?.({ type: 'error', message: `Failed to initialize: ${error}` });
        throw error;
      }
    };

    const handleTakePicture = async () => {
      if (!primaryWallet?.address || !program) return;
      setLoading(true);

      try {
        if (!isInitialized) {
          await initializeCamera();
        }

        onStatusUpdate?.({ type: 'info', message: 'Activating camera...' });
        await program.methods.activateCamera(new BN(100))
          .accounts({
            cameraAccount: cameraKeypair.publicKey,
            user: new PublicKey(primaryWallet.address),
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        onStatusUpdate?.({ type: 'info', message: 'Taking picture...' });
        const apiUrl = `${CONFIG.CAMERA_API_URL}/api/capture`;
        const captureResponse = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          mode: 'cors',
          credentials: 'omit'
        });

        if (!captureResponse.ok) {
          throw new Error(`Server responded with status: ${captureResponse.status}`);
        }

        onStatusUpdate?.({ type: 'info', message: 'Processing image...' });
        const imageBlob = await captureResponse.blob();

        onStatusUpdate?.({ type: 'info', message: 'Uploading to IPFS...' });
        await pinataService.uploadImage(imageBlob, primaryWallet.address);

        onStatusUpdate?.({ type: 'success', message: 'Picture uploaded successfully!' });
        onPhotoCapture?.();

      } catch (error) {
        onStatusUpdate?.({ type: 'error', message: `Error: ${error instanceof Error ? error.message : String(error)}` });
      } finally {
        setLoading(false);
      }
    };

    return (
      <div className="flex flex-col gap-4">
        <button
          onClick={handleTakePicture}
          disabled={loading || !primaryWallet?.address}
          className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-white
        ${loading ? 'bg-stone-400 cursor-not-allowed' : 'bg-stone-400 hover:bg-stone-500'}`}
        >
          {loading ? (
            <Loader className="w-5 h-5 animate-spin" />
          ) : (
            <Camera className="w-5 h-5" />
          )}
          {loading ? 'Processing...' : 'Take Picture'}
        </button>
      </div>
    );
  });

  ActivateCamera.displayName = 'ActivateCamera';