import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useRef, useState, useEffect, useCallback } from 'react';
import { Timeline } from '../Timeline';
import { ActivateCamera } from '../ActivateCamera';
import { VideoRecorder } from '../VideoRecorder';
import MediaGallery from '../ImageGallery';
import { CONFIG } from '../../config';
import { MobileControls } from './MobileControls';
import { ToastMessage } from '../../types/toast';
import { ToastContainer } from '../../components/ToastContainer';
import { StreamPlayer } from '../StreamPlayer';
import { StreamControls } from '../StreamControls';
import { TransactionModal } from '../../components/headless/auth/TransactionModal';
import { useCamera } from '../CameraProvider';
import { PublicKey, SystemProgram } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { useProgram } from '../../anchor/setup';

export function CameraView() {
  const { primaryWallet } = useDynamicContext();
  const { userHasEmbeddedWallet, isSessionActive, sendOneTimeCode, createOrRestoreSession } = useEmbeddedWallet();
  const { cameraKeypair, quickActions, isInitialized } = useCamera();
  const program = useProgram();
  const timelineRef = useRef<any>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentToast, setCurrentToast] = useState<ToastMessage | null>(null);
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentAction, setCurrentAction] = useState<{
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  } | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  const [otpError, setOtpError] = useState<string | null>(null);
  const [isMobileView, setIsMobileView] = useState(window.innerWidth <= 768);

  const activateCameraRef = useRef<{
    handleTakePicture: () => Promise<void>;
  }>(null);
  const videoRecorderRef = useRef<{
    startRecording: () => Promise<void>;
  }>(null);

  // Debug the wallet type
  useEffect(() => {
    if (primaryWallet) {
      const isEmbedded = primaryWallet.connector?.name.toLowerCase() !== 'phantom';
      // Only log critical wallet info
      if (isEmbedded) {
        console.log('Using embedded wallet:', primaryWallet.connector?.name);
      }
    }
  }, [primaryWallet]);

  // Check if the user is using an embedded wallet
  const isEmbeddedWallet = userHasEmbeddedWallet();

  const checkCameraStatus = useCallback(async () => {
    if (!cameraAccount) {
      setIsLive(false);
      setIsStreaming(false);
      return;
    }

    let retryCount = 0;
    const MAX_RETRIES = 2;
    const RETRY_DELAY = 3000; // 3 seconds

    const checkWithRetry = async () => {
      try {
        const healthResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/health`);
        if (!healthResponse.ok) throw new Error('Health check failed');

        const streamResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`);
        if (!streamResponse.ok) throw new Error('Stream info check failed');

        const data = await streamResponse.json();
        setIsLive(true);
        setIsStreaming(data.isActive);
      } catch (error) {
        console.error('Failed to check camera status:', error);
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
          return checkWithRetry();
        }
        // If all retries failed, set camera as offline
        setIsLive(false);
        setIsStreaming(false);
      }
    };

    await checkWithRetry();
  }, [cameraAccount]);

  useEffect(() => {
    const storedCamera = localStorage.getItem('cameraAccount');
    if (storedCamera) {
      setCameraAccount(storedCamera);
    }
  }, []);

  useEffect(() => {
    // Initial check
    checkCameraStatus();

    // Set up polling with a longer interval (5 seconds)
    const interval = setInterval(checkCameraStatus, 5000);

    return () => clearInterval(interval);
  }, [checkCameraStatus]);

  useEffect(() => {
    const handleResize = () => {
      setIsMobileView(window.innerWidth <= 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleAction = async (type: 'photo' | 'video' | 'stream') => {
    if (!cameraKeypair || !primaryWallet?.address) {
      return;
    }

    // For embedded wallets
    if (isEmbeddedWallet && !isMobileView) {
      // Only show custom modal for desktop view
      if (!isSessionActive) {
        try {
          setIsVerifying(true);
          await sendOneTimeCode();
          return; // Exit and wait for OTP verification
        } catch (error) {
          console.error('Failed to send OTP:', error);
          updateToast('error', 'Failed to send verification code');
          return;
        }
      }

      // Show our custom modal (desktop only)
      setCurrentAction({
        type,
        cameraAccount: cameraKeypair.publicKey.toString()
      });
      setIsModalOpen(true);
      return;
    }

    // For mobile view or Phantom wallet, use direct action
    if (isMobileView || quickActions[type]) {
      if (type === 'photo') {
        await activateCameraRef.current?.handleTakePicture();
      } else if (type === 'video') {
        await videoRecorderRef.current?.startRecording();
      } else if (type === 'stream') {
        await handleStream();
      }
      return;
    }

    // Regular flow for desktop
    if (type === 'photo') {
      await activateCameraRef.current?.handleTakePicture();
    } else if (type === 'video') {
      await videoRecorderRef.current?.startRecording();
    } else if (type === 'stream') {
      await handleStream();
    }
  };

  const handleStream = async () => {
    if (!primaryWallet?.address || !program || !isInitialized) return;
    setLoading(true);

    try {
      // First activate camera on-chain
      await program.methods.activateCamera(new BN(100))
        .accounts({
          cameraAccount: cameraKeypair.publicKey,
          user: new PublicKey(primaryWallet.address),
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      // Start or stop stream based on current state
      const endpoint = isStreaming ? '/api/stream/stop' : '/api/stream/start';
      const response = await fetch(`${CONFIG.CAMERA_API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${primaryWallet.address}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to ${isStreaming ? 'stop' : 'start'} stream: ${response.statusText}`);
      }

      // Update streaming state
      setIsStreaming(!isStreaming);
      
      // Add timeline event
      timelineRef.current?.addEvent({
        type: isStreaming ? 'stream_ended' : 'stream_started',
        timestamp: Date.now(),
        user: { address: primaryWallet.address }
      });

      updateToast('success', `Stream ${isStreaming ? 'stopped' : 'started'} successfully`);
    } catch (error) {
      console.error('Stream control error:', error);
      updateToast('error', error instanceof Error ? error.message : 'Failed to control stream');
    } finally {
      setLoading(false);
    }
  };

  const handleTakePicture = async (skipTransaction = false) => {
    console.log('Starting picture capture, skipTransaction:', skipTransaction);
    setLoading(true);

    try {
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/capture`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${primaryWallet?.address}`
        }
      });

      console.log('Capture response:', response.status);

      if (!response.ok) {
        throw new Error(`Failed to take photo: ${response.statusText}`);
      }

      // Add delay before notification
      await new Promise(resolve => setTimeout(resolve, 1000));

      timelineRef.current?.addEvent({
        type: 'photo_captured',
        timestamp: Date.now(),
        user: { address: primaryWallet?.address?.toString() || 'unknown' }
      });

      updateToast('success', 'Photo captured successfully');
    } catch (error) {
      console.error('Failed to take picture:', error);
      updateToast('error', error instanceof Error ? error.message : 'Failed to take picture');
    } finally {
      setLoading(false);
    }
  };

  const handleRecordVideo = async (_skipTransaction = false) => {
    setLoading(true);
    try {
      // Call camera API after transaction is confirmed
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/video/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${primaryWallet?.address}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to start recording: ${response.statusText}`);
      }

      timelineRef.current?.addEvent({
        type: 'video_recorded',
        timestamp: Date.now(),
        user: { address: primaryWallet?.address?.toString() || 'unknown' }
      });

      updateToast('success', 'Video recording started');
    } catch (error) {
      console.error('Failed to record video:', error);
      updateToast('error', error instanceof Error ? error.message : 'Failed to record video');
    } finally {
      setLoading(false);
    }
  };

  const handleCameraUpdate = ({ address }: { address: string; isLive: boolean }) => {
    setCameraAccount(address);
    localStorage.setItem('cameraAccount', address);
  };

  const updateToast = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now().toString();
    setCurrentToast({ id, type, message });
  };

  const dismissToast = () => {
    setCurrentToast(null);
  };

  const handleTransactionSuccess = async () => {
    if (!currentAction) return;
    console.log('Transaction succeeded, handling action:', currentAction.type);

    try {
      setLoading(true);
      if (currentAction.type === 'photo') {
        await handleTakePicture(true);
      } else if (currentAction.type === 'video') {
        await handleRecordVideo(true);
      }
    } catch (error) {
      console.error('API call failed:', error);
      updateToast('error', 'Failed to complete action after transaction');
    } finally {
      setLoading(false);
      setIsModalOpen(false);
      setCurrentAction(null);
    }
  };

  const handleVerifyOtp = async (otp: string) => {
    try {
      await createOrRestoreSession({ oneTimeCode: otp });
      setIsVerifying(false);
      setOtpSent(false);
      setOtpError(null);
      // Now show the transaction modal
      setIsModalOpen(true);
    } catch (error) {
      console.error('Failed to verify OTP:', error);
      setOtpError('Invalid verification code');
    }
  };

  return (
    <>
      <MobileControls
        onTakePicture={() => handleAction('photo')}
        onRecordVideo={() => handleAction('video')}
        onToggleStream={() => handleAction('stream')}
        isLoading={loading}
        isStreaming={isStreaming}
      />
      {/* Keep your existing layout exactly as is */}
      <div className="h-full overflow-y-auto pb-40">
        {/* Keep your existing layout exactly as is */}
        <div className="relative max-w-3xl mx-auto pt-8 ">
          <ToastContainer message={currentToast} onDismiss={dismissToast} />
          <div className="bg-white rounded-lg mb-6 px-6">
            <h2 className="text-xl font-semibold">Camera</h2>
            <span className="text-[11px] text-gray-600">
              {cameraAccount
                ? `id: ${cameraAccount.slice(0, 4)}...${cameraAccount.slice(-4)}`
                : 'No Camera Connected'
              }
            </span>
          </div>
          <div className="px-2">
            {/* Use grid to maintain consistent widths */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Wrapper for both preview and controls */}
              <div className="md:col-span-3 relative">
                {/* Preview Frame */}
                <StreamPlayer />

                {/* Camera Controls - full height */}
                <div className="hidden sm:flex absolute -right-14 top-0 flex-col h-full z-[45]">
                  <StreamControls timelineRef={timelineRef} />
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
                {!isLive ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-gray-400"></span>
                    </span>
                    <span className="text-gray-500 font-medium">Offline</span>
                  </div>
                ) : isStreaming ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-red-500 font-medium">LIVE</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                    </span>
                    <span className="text-green-500 font-medium">Online</span>
                  </div>
                )}
              {/* <span className="text-sm text-gray-600">
                  {cameraAccount
                    ? `Camera: ${cameraAccount.slice(0, 8)}...`
                    : 'No Camera Connected'
                  }
                </span> */}
            </div>
          </div>

          {/* Timeline Events */}
          <div className="absolute mt-12 pl-6 left-0 w-full">
            <Timeline ref={timelineRef} variant="camera" />
            <div
              className="top-0 left-0 right-0 pointer-events-none"
              style={{
                background: 'linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)'
              }}
            />
          </div>

          {/* Main Camera Controls Frame */}
          <div className="relative md:ml-20 ml-16 bg-white">


            {/* Media Gallery */}
            <div className="relative px-6">
              <MediaGallery mode="recent" maxRecentItems={6} />
            </div>
          </div>
        </div>
      </div>
    </div >

      {/* OTP Verification Modal */ }
  {
    isVerifying && (
      <div className="fixed inset-0 z-[9999] overflow-y-auto">
        <div className="flex min-h-screen items-center justify-center px-4">
          <div
            className="fixed inset-0 bg-black bg-opacity-30 transition-opacity"
            onClick={() => {
              setIsVerifying(false);
              setOtpSent(false);
              setOtpError(null);
            }}
          />
          <div className="relative bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Verify Your Action</h2>
              <button
                onClick={() => {
                  setIsVerifying(false);
                  setOtpSent(false);
                  setOtpError(null);
                }}
                className="text-gray-400 hover:text-gray-500"
              >
                <span className="sr-only">Close</span>
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              {!otpSent ? (
                <div className="text-sm text-gray-600">
                  Sending verification code...
                </div>
              ) : (
                <form onSubmit={async (e) => {
                  e.preventDefault();
                  const otp = (e.currentTarget.elements.namedItem('otp') as HTMLInputElement).value;
                  await handleVerifyOtp(otp);
                }}>
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="otp" className="block text-sm font-medium text-gray-700">
                        Enter verification code
                      </label>
                      <input
                        type="text"
                        name="otp"
                        id="otp"
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                        placeholder="Enter code"
                        required
                      />
                    </div>
                    {otpError && (
                      <div className="text-sm text-red-600">
                        {otpError}
                      </div>
                    )}
                    <button
                      type="submit"
                      className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Verify
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  {/* Transaction Modal */ }
  <TransactionModal
    isOpen={isModalOpen}
    onClose={() => setIsModalOpen(false)}
    transactionData={currentAction || undefined}
    onSuccess={handleTransactionSuccess}
  />
    </>
  );
}
