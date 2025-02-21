import { useEffect, useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { Video, Loader } from 'lucide-react';
import { CONFIG } from '../config';
import { useCamera } from './CameraProvider';
import { unifiedIpfsService } from '../services/unified-ipfs-service';

interface VideoRecorderProps {
  onVideoRecorded?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info'; message: string }) => void;
}

export const VideoRecorder = forwardRef<{ startRecording: () => Promise<void> }, VideoRecorderProps>(
  ({ onVideoRecorded, onStatusUpdate: updateToast }, ref) => {
    const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null);
    const CAMERA_API_BASE = CONFIG.CAMERA_API_URL;
    const { primaryWallet } = useDynamicContext();
    useConnection();
    const program = useProgram();
    const [loading, setLoading] = useState(false);
    const [, setShowTooltip] = useState(false);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const [, setTimeLeft] = useState<number | null>(null);
    
    // Use the shared camera context
    const { isInitialized, loading: initLoading } = useCamera();

    useImperativeHandle(ref, () => ({
      startRecording
    }));

    const startRecording = async () => {
      if (!primaryWallet?.address || !program || !isInitialized) return;
      setLoading(true);
      setTimeLeft(null);

      try {
        updateToast?.({ type: 'info', message: 'Starting recording...' });
        const duration = 30; // 30 seconds recording
        const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/video/start`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${primaryWallet.address}`
          },
          body: JSON.stringify({ duration })
        });

        if (!response.ok) {
          throw new Error('Failed to start recording');
        }

        const { filename } = await response.json();
        console.log('Recording started with filename:', filename);

        // Start countdown timer
        let timeRemaining = duration;
        const timer = setInterval(() => {
          timeRemaining -= 1;
          setTimeLeft(timeRemaining);
          if (timeRemaining <= 0) {
            clearInterval(timer);
          }
        }, 1000);
        setRecordingTimer(timer);

        // Wait for recording to complete
        await new Promise<void>((resolve) => {
          setTimeout(resolve, duration * 1000 + 2000); // Add 2 seconds buffer
        });

        updateToast?.({ type: 'info', message: 'Processing video...' });
        const videoBlob = await downloadVideo(filename);

        updateToast?.({ type: 'info', message: 'Uploading to IPFS...' });
        const results = await unifiedIpfsService.uploadFile(videoBlob, primaryWallet.address, 'video');
        
        if (results.length === 0) {
          throw new Error('Failed to upload video to any IPFS provider');
        }

        onVideoRecorded?.();
        updateToast?.({ type: 'success', message: 'Video uploaded successfully!' });

      } catch (error) {
        console.error('Recording error:', error);
        updateToast?.({ type: 'error', message: error instanceof Error ? error.message : 'Failed to record video' });
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
      <>
        <button
          ref={buttonRef}
          onClick={() => onVideoRecorded?.()}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          disabled={loading || initLoading || !isInitialized}
          className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
        >
          {loading || initLoading ? (
            <Loader className="w-5 h-5 animate-spin" />
          ) : (
            <Video className="w-5 h-5" />
          )}
        </button>
      </>
    );
  }
);

VideoRecorder.displayName = 'VideoRecorder';