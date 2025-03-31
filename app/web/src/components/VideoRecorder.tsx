import { useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../anchor/setup';
import { Video, Loader } from 'lucide-react';
import { useCamera } from './CameraProvider';

interface VideoRecorderProps {
  onVideoRecorded?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info'; message: string }) => void;
}

export const VideoRecorder = forwardRef<{ startRecording: () => Promise<void> }, VideoRecorderProps>(
  ({ onVideoRecorded, onStatusUpdate: updateToast }, ref) => {
    const { primaryWallet } = useDynamicContext();
    useConnection();
    const program = useProgram();
    const [loading, setLoading] = useState(false);
    const [, setShowTooltip] = useState(false);
    const buttonRef = useRef<HTMLButtonElement>(null);
    
    // Use the shared camera context
    const { isInitialized, loading: initLoading } = useCamera();

    useImperativeHandle(ref, () => ({
      startRecording
    }));

    const startRecording = async () => {
      console.log("VIDEO RECORD BUTTON CLICKED - starting video recording");
      if (!primaryWallet?.address || !program || !isInitialized) {
        console.log("Missing required data:", { 
          wallet: !!primaryWallet?.address, 
          program: !!program, 
          initialized: isInitialized 
        });
        return;
      }
      setLoading(true);

      try {
        console.log("Calling onVideoRecorded callback directly");
        // Simply trigger the onVideoRecorded callback which will handle the on-chain transaction
        if (onVideoRecorded) {
          onVideoRecorded();
          updateToast?.({ type: 'success', message: 'Video recording transaction sent' });
        } else {
          console.error("No onVideoRecorded callback provided");
          updateToast?.({ type: 'error', message: 'Video recording callback not configured' });
        }
      } catch (error) {
        console.error('Error triggering video recording:', error);
        updateToast?.({ type: 'error', message: error instanceof Error ? error.message : 'Failed to record video' });
      } finally {
        setLoading(false);
      }
    };

    // Handle button click directly 
    const handleButtonClick = () => {
      console.log("VIDEO BUTTON CLICKED IN COMPONENT - direct handler");
      startRecording();
    };

    return (
      <>
        <button
          ref={buttonRef}
          onClick={handleButtonClick}
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