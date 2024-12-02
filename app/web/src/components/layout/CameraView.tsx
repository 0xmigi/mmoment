import { useWallet } from '@solana/wallet-adapter-react';
import { useRef, useState, useEffect, useCallback } from 'react';
import { Timeline } from '../Timeline';
import { ActivateCamera } from '../ActivateCamera';
import { VideoRecorder } from '../VideoRecorder';
import MediaGallery from '../ImageGallery';
import { CONFIG } from '../../config';

export function CameraView() {
  const { publicKey } = useWallet();
  const timelineRef = useRef<any>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);

  // Add check for camera API availability
  const checkCameraStatus = useCallback(async () => {
    if (!cameraAccount) return;
    
    try {
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/health`);
      if (response.ok) {
        setIsLive(true);
      } else {
        setIsLive(false);
      }
    } catch (error) {
      setIsLive(false);
    }
  }, [cameraAccount]);

  // Load stored camera state
  useEffect(() => {
    const storedCamera = localStorage.getItem('cameraAccount');
    if (storedCamera) {
      setCameraAccount(storedCamera);
    }
  }, []);

  // Check camera status periodically
  useEffect(() => {
    checkCameraStatus();
    const interval = setInterval(checkCameraStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkCameraStatus]);

  const handleCameraUpdate = ({ publicKey, isLive }: { publicKey: string; isLive: boolean }) => {
    setCameraAccount(publicKey);
    setIsLive(isLive);
    localStorage.setItem('cameraAccount', publicKey);
  };
  return (
    <div className="relative w-full">
      {/* <div className="max-w-3xl mt-30 mx-auto h-screen flex flex-col justify-center relative"> */}
      <div className="max-w-3xl mt-40 mx-auto flex flex-col justify-top relative">

        {/* Updated Camera ID and Live Status */}
        <div className="relative mb-40">
          <div className="flex items-center gap-2">
            {isLive && (
              <div className="flex items-center gap-2">
                <span className="relative flex h-3 w-3 -ml-1">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                </span>
                <span className="text-red-500 font-medium">LIVE</span>
              </div>
            )}
            <span className="text-sm text-gray-600">
              {cameraAccount
                ? `Camera: ${cameraAccount.slice(0, 8)}...`
                : 'No Camera Connected'
              }
            </span>
          </div>

          {/* Timeline Events with Fade - Now aligned with vertical line */}
          <div className="absolute mt-12 left-0 w-full">
            <Timeline ref={timelineRef} />
            <div
              className="top-0 left-0 right-0 pointer-events-none"
              style={{
                background: 'linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)'
              }}
            />
          </div>
        </div>

        {/* Control Frame */}
        <div className="relative ml-16 mt-8 w-[calc(100%-4rem)] bg-white p-4 z-20">
          <div className="border rounded-lg shadow-sm p-6">
            <div className="space-y-4">
              <ActivateCamera
                onCameraUpdate={handleCameraUpdate}
                onInitialize={() => {
                  timelineRef.current?.addEvent({
                    type: 'initialization',
                    timestamp: Date.now(),
                    user: { address: publicKey?.toString() || 'unknown' }
                  });
                }}
                onPhotoCapture={() => {
                  timelineRef.current?.addEvent({
                    type: 'photo_captured',
                    timestamp: Date.now(),
                    user: { address: publicKey?.toString() || 'unknown' }
                  });
                }}
              />
              <VideoRecorder
                onVideoRecorded={() => {
                  timelineRef.current?.addEvent({
                    type: 'video_recorded',
                    timestamp: Date.now(),
                    user: { address: publicKey?.toString() || 'unknown' }
                  });
                }}
              />
            </div>
          </div>
          {/* Media Gallery */}
          <div className="relative z-20">
            <MediaGallery mode="recent" maxRecentItems={6} />
          </div>
        </div>
      </div>
    </div>
  );
}