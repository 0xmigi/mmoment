
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useRef, useState, useEffect, useCallback } from 'react';
import { Timeline } from '../Timeline';
import { ActivateCamera } from '../ActivateCamera';
import { VideoRecorder } from '../VideoRecorder';
import MediaGallery from '../ImageGallery';
import { CONFIG } from '../../config';
import { MobileControls } from './MobileControls';
import { ToastMessage } from '../../types/toast';
import { ToastContainer } from '../../components/ToastContainer';

export function CameraView() {
  const { primaryWallet } = useDynamicContext();
  const timelineRef = useRef<any>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [currentToast, setCurrentToast] = useState<ToastMessage | null>(null);
  const [loading, setLoading] = useState(false);
  const activateCameraRef = useRef<{
    handleTakePicture: () => Promise<void>;
  }>(null);
  const videoRecorderRef = useRef<{
    startRecording: () => Promise<void>;
  }>(null);

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

  useEffect(() => {
    const storedCamera = localStorage.getItem('cameraAccount');
    if (storedCamera) {
      setCameraAccount(storedCamera);
    }
  }, []);

  useEffect(() => {
    checkCameraStatus();
    const interval = setInterval(checkCameraStatus, 30000);
    return () => clearInterval(interval);
  }, [checkCameraStatus]);

  const handleTakePicture = async () => {
    // Call the same functionality that's in your ActivateCamera component
    setLoading(true);
    try {
      await activateCameraRef.current?.handleTakePicture();
    } catch (error) {
      console.error('Failed to take picture:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRecordVideo = async () => {
    // Call the same functionality that's in your VideoRecorder component
    setLoading(true);
    try {
      await videoRecorderRef.current?.startRecording();
    } catch (error) {
      console.error('Failed to record video:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCameraUpdate = ({ address, isLive }: { address: string; isLive: boolean }) => {
    setCameraAccount(address);
    setIsLive(isLive);
    localStorage.setItem('cameraAccount', address);
  };

  const updateToast = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now().toString();
    setCurrentToast({ id, type, message });
  };

  const dismissToast = () => {
    setCurrentToast(null);
  };

  return (
    <>
      <MobileControls
        onTakePicture={() => {
          handleTakePicture && handleTakePicture();
        }}
        onRecordVideo={() => {
          handleRecordVideo && handleRecordVideo();
        }}
        isLoading={loading}
      />
      {/* Add just the outer scrollable container */}
      <div className="h-full overflow-y-auto pb-20">
        {/* Keep your existing layout exactly as is */}
        <div className="relative max-w-3xl mx-auto pt-8 ">
          <ToastContainer message={currentToast} onDismiss={dismissToast} />
          <div className="px-6">
            {/* Use grid to maintain consistent widths */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Wrapper for both preview and controls */}
              <div className="md:col-span-3 relative">
                {/* Preview Frame */}
                <div className="aspect-video border-2 border-gray-300 bg-gray-200 rounded-lg z-[60] relative">
                  {/* Preview Label */}
                  <div className="absolute top-4 right-4">
                    <span className="px-2 py-1 bg-red-500 text-white text-xs opacity-20 rounded-full">
                      PREVIEW
                    </span>
                  </div>
                </div>

                {/* Camera Controls - full height */}
                <div className="hidden sm:flex absolute -right-14 top-0 flex-col h-full z-[45]">
                  <div className="group h-1/2 relative">
                    <ActivateCamera
                      ref={activateCameraRef}
                      onCameraUpdate={handleCameraUpdate}
                      onInitialize={() => {
                        timelineRef.current?.addEvent({
                          type: 'initialization',
                          timestamp: Date.now(),
                          user: { address: primaryWallet?.address?.toString() || 'unknown' }
                        });
                      }}
                      onPhotoCapture={() => {
                        timelineRef.current?.addEvent({
                          type: 'photo_captured',
                          timestamp: Date.now(),
                          user: { address: primaryWallet?.address?.toString() || 'unknown' }
                        });
                      }}
                      onStatusUpdate={({ type, message }) => updateToast(type, message)}
                    />
                    {/* Hover label */}
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Processing...' : 'Take Picture'}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <VideoRecorder
                      ref={videoRecorderRef}
                      onVideoRecorded={() => {
                        timelineRef.current?.addEvent({
                          type: 'video_recorded',
                          timestamp: Date.now(),
                          user: { address: primaryWallet?.address?.toString() || 'unknown' }
                        });
                      }}
                      onStatusUpdate={({ type, message }) => updateToast(type, message)}
                    />
                    {/* Hover label */}
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Recording...' : 'Record Video'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="max-w-3xl mt-6 mx-auto flex flex-col justify-top relative">


            {/* Camera Status Header */}
            <div className="relative mb-40">
              <div className="flex pl-6 items-center gap-2">
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
            </div>

            {/* Timeline Events */}
            <div className="absolute mt-12 pl-6 left-0 w-full">
              <Timeline ref={timelineRef}
              // maxEvents={18} 
              />
              <div
                className="top-0 left-0 right-0 pointer-events-none"
                style={{
                  background: 'linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)'
                }}
              />
            </div>

            {/* Main Camera Controls Frame */}
            <div className="relative ml-16 w-[calc(100%-4rem)] bg-white">


              {/* Media Gallery */}
              <div className="relative px-6 mt-6">
                <MediaGallery mode="recent" maxRecentItems={6} />
              </div>
            </div>
          </div>
        </div>
      </div >
    </>
  );
}