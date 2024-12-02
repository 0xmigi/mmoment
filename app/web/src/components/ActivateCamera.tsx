import { useState } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { SystemProgram, Keypair } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { pinataService } from '../services/pinata-service';
import { CONFIG } from '../config';
import { Camera, Loader } from 'lucide-react';

interface ActivateCameraProps {
  onInitialize?: () => void;
  onPhotoCapture?: () => void;
  onCameraUpdate?: (params: { publicKey: string; isLive: boolean }) => void;
}

export function ActivateCamera({ onInitialize, onPhotoCapture, onCameraUpdate }: ActivateCameraProps) {
  const wallet = useWallet();
  const { publicKey } = wallet;
  useConnection();
  const program = useProgram();
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [cameraKeypair] = useState(() => Keypair.generate());
  const [isInitialized, setIsInitialized] = useState(false);

  const initializeCamera = async () => {
    if (!publicKey || !program || isInitialized) return;
    
    try {
      const initTx = await program.methods.initialize()
        .accounts({
          cameraAccount: cameraKeypair.publicKey,
          user: publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([cameraKeypair])
        .rpc();
      
      console.log('Camera initialized:', initTx);
      setIsInitialized(true);
      onCameraUpdate?.({
        publicKey: cameraKeypair.publicKey.toString(),
        isLive: true
      });
      onInitialize?.();
    } catch (error) {
      console.error('Failed to initialize camera:', error);
      throw error;
    }
  };

  const handleTakePicture = async () => {
    if (!publicKey || !program) return;
    setLoading(true);
    
    try {
      // Initialize camera if not already initialized
      if (!isInitialized) {
        await initializeCamera();
      }

      // Activate camera
      setStatus('Activating camera...');
      await program.methods.activateCamera(new BN(100))
        .accounts({
          cameraAccount: cameraKeypair.publicKey,
          user: publicKey,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
  
      // Call camera API
      setStatus('Taking picture...');
      const apiUrl = `${CONFIG.CAMERA_API_URL}/api/capture`;
      console.log('Calling camera API at:', apiUrl);
      
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

      setStatus('Processing image...');
      const imageBlob = await captureResponse.blob();
      
      // Upload to Pinata
      setStatus('Uploading to IPFS...');
      const url = await pinataService.uploadImage(imageBlob, publicKey.toString());
      console.log('Image uploaded to IPFS:', url);
      
      setStatus('Picture uploaded successfully!');
      onPhotoCapture?.();

    } catch (error) {
      console.error('Transaction error:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <button
        onClick={handleTakePicture}
        disabled={loading || !publicKey}
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
      
      {status && (
        <div className="text-sm text-gray-600 mt-2 p-3 bg-gray-50 rounded-lg">
          {status}
        </div>
      )}
    </div>
  );
}