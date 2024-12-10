import { useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { SystemProgram, Keypair, PublicKey } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { pinataService } from '../services/pinata-service';
import { Video, Loader } from 'lucide-react';
import { CONFIG } from '../config';

interface VideoRecorderProps {
  onVideoRecorded?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info', message: string }) => void;
}

export const VideoRecorder = forwardRef<{ startRecording: () => Promise<void> }, VideoRecorderProps>(
  ({ onVideoRecorded, onStatusUpdate }, ref) => {
    const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null);
    const CAMERA_API_BASE = CONFIG.CAMERA_API_URL;
    const { primaryWallet } = useDynamicContext();
    useConnection();
    const program = useProgram();
    const [loading, setLoading] = useState(false);
    const [timeLeft, setTimeLeft] = useState<number | null>(null);
    const [cameraKeypair] = useState(() => Keypair.generate());
    const [isInitialized, setIsInitialized] = useState(false);

    useImperativeHandle(ref, () => ({
      startRecording
    }));

    const initializeCamera = async () => {
      if (!primaryWallet?.address || !program || isInitialized) return;

      try {
        const initTx = await program.methods.initialize()
          .accounts({
            cameraAccount: cameraKeypair.publicKey,
            user: new PublicKey(primaryWallet.address),
            systemProgram: SystemProgram.programId,
          })
          .signers([cameraKeypair])
          .rpc();

        console.log('Camera initialized:', initTx);
        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize camera:', error);
        throw error;
      }
    };

    const startRecording = async () => {
      console.log("Start recording clicked", {
        hasWallet: !!primaryWallet?.address,
        hasProgram: !!program
      });
      if (!primaryWallet?.address || !program) return;
      setLoading(true);

      try {
        // Initialize if needed
        if (!isInitialized) {
          await initializeCamera();
        }

        // Activate camera
        onStatusUpdate?.({ type: 'info', message: 'Activating camera...' });
        // setStatus('Activating camera...');
        await program.methods.activateCamera(new BN(100))
          .accounts({
            cameraAccount: cameraKeypair.publicKey,
            user: new PublicKey(primaryWallet.address),
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        // Start Recording
        // setStatus('Starting recording...');
        onStatusUpdate?.({ type: 'info', message: 'Starting recording...' });
        const duration = 30;
        setTimeLeft(duration);

        const response = await fetch(`${CAMERA_API_BASE}/api/video/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ duration }),
          mode: 'cors',
          credentials: 'omit'
        });

        if (!response.ok) {
          throw new Error(`Failed to start recording: ${response.statusText}`);
        }

        const { filename } = await response.json();
        console.log('Recording started with filename:', filename);

        // Wait for recording to complete
        await new Promise<void>((resolve) => {
          let timeRemaining = duration;
          const timer = setInterval(() => {
            timeRemaining -= 1;
            setTimeLeft(timeRemaining);

            if (timeRemaining <= 0) {
              if (timer) clearInterval(timer);
              setRecordingTimer(null);
              resolve();
            }
          }, 1000);
          setRecordingTimer(timer);
        });

        // Add a small delay after recording completes
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Download and upload
        // setStatus('Downloading video...');
        onStatusUpdate?.({ type: 'info', message: 'Downloading video...' });
        const videoBlob = await downloadVideo(filename);

        // setStatus('Uploading to IPFS...');
        onStatusUpdate?.({ type: 'info', message: 'Uploading to IPFS...' });
        const ipfsUrl = await pinataService.uploadVideo(videoBlob, primaryWallet.address);
        console.log('Video uploaded to IPFS:', ipfsUrl);

        onVideoRecorded?.();
        // setStatus('Video uploaded successfully!');
        onStatusUpdate?.({ type: 'info', message: 'Video uploaded successfully!' });

      } catch (error) {
        console.error('Recording error:', error);
        // setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
        onStatusUpdate?.({ type: 'error', message: `${error instanceof Error ? error.message : String(error)}` });
      } finally {
        setLoading(false);
        setTimeLeft(null);
        if (recordingTimer) {
          clearInterval(recordingTimer);
          setRecordingTimer(null);
        }
      }
    };

    const downloadVideo = async (filename: string): Promise<Blob> => {
      const mp4filename = filename.replace('.h264', '.mp4');
      console.log('Attempting to download:', mp4filename);

      const response = await fetch(`${CAMERA_API_BASE}/api/video/download/${mp4filename}`, {
        headers: {
          'Accept': 'video/mp4',
        },
      });

      if (!response.ok) {
        console.error('Download failed:', response.status, response.statusText);
        throw new Error(`Failed to download video: ${response.statusText}`);
      }

      const blob = await response.blob();
      console.log('Downloaded video size:', blob.size);
      return blob;
    };

    useEffect(() => {
      return () => {
        if (recordingTimer) {
          clearInterval(recordingTimer);
        }
      };
    }, [recordingTimer]);

    return (
      <div className="flex flex-col gap-4">
        <button
          onClick={startRecording}
          disabled={loading || !primaryWallet?.address}
          className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-white
        ${loading ? 'bg-gray-500 cursor-not-allowed' : 'bg-gray-400 hover:bg-gray-500'}`}
        >
          {loading ? (
            <Loader className="w-5 h-5 animate-spin" />
          ) : (
            <Video className="w-5 h-5" />
          )}
          {loading && timeLeft !== null ? `Recording (${timeLeft}s)` : 'Record Video'}
        </button>
      </div>
    );
  }
);

VideoRecorder.displayName = 'VideoRecorder';