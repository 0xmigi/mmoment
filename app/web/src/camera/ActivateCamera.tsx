import { useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Camera, Loader } from 'lucide-react';
import { CONFIG } from '../core/config';
import { unifiedIpfsService } from '../storage/ipfs/unified-ipfs-service';

interface ActivateCameraProps {
  onCameraUpdate?: (params: { address: string; isLive: boolean }) => void;
  onInitialize?: () => void;
  onPhotoCapture?: () => void;
  onStatusUpdate?: (status: { type: 'success' | 'error' | 'info', message: string }) => void;
}

// NEW PRIVACY ARCHITECTURE: Photo capture no longer requires blockchain check-in
// Sessions are managed off-chain by Jetson
export const ActivateCamera = forwardRef<{ handleTakePicture: () => Promise<void> }, ActivateCameraProps>(
  ({ onPhotoCapture, onStatusUpdate }, ref) => {
    const { primaryWallet } = useDynamicContext();
    const [loading, setLoading] = useState(false);
    const [, setShowTooltip] = useState(false);
    const buttonRef = useRef<HTMLButtonElement>(null);

    useImperativeHandle(ref, () => ({
      handleTakePicture
    }));

    const handleTakePicture = async () => {
      if (!primaryWallet?.address) return;
      setLoading(true);

      try {
        // No blockchain check-in needed - sessions are off-chain now
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