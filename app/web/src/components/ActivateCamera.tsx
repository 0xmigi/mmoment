import { useState, forwardRef, useImperativeHandle, useRef, useEffect } from 'react';
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

export const ActivateCamera = forwardRef<{ handleTakePicture: () => Promise<void> }, ActivateCameraProps>(
  ({ onCameraUpdate, onInitialize, onPhotoCapture, onStatusUpdate }, ref) => {
    const { primaryWallet } = useDynamicContext();
    useConnection();
    const program = useProgram();
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
    
    const [isInitialized, setIsInitialized] = useState(false);

    // Check initialization when wallet connects
    useEffect(() => {
      const checkInitialization = async () => {
        if (!primaryWallet?.address || !program) return;

        try {
          const account = await program.account.cameraAccount.fetch(cameraKeypair.publicKey);
          setIsInitialized(!!account);
          if (account) {
            onCameraUpdate?.({
              address: cameraKeypair.publicKey.toString(),
              isLive: true
            });
          }
        } catch {
          // Account doesn't exist, need to initialize
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
          }
        }
      };

      checkInitialization();
    }, [primaryWallet?.address, program]);

    useImperativeHandle(ref, () => ({
      handleTakePicture
    }));

    const handleTakePicture = async () => {
      if (!primaryWallet?.address || !program || !isInitialized) return;
      setLoading(true);

      try {
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
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${primaryWallet.address}`
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
      <button
        ref={buttonRef}
        onClick={handleTakePicture}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        disabled={loading || !primaryWallet?.address || !isInitialized}
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