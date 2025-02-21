import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useRef, useState, useEffect } from 'react';
import { Timeline } from '../Timeline';
import { ActivateCamera } from '../ActivateCamera';
import { VideoRecorder } from '../VideoRecorder';
import MediaGallery from '../ImageGallery';
import { CONFIG } from '../../config';
import { CameraControls } from './MobileControls';
import { ToastMessage } from '../../types/toast';
import { ToastContainer } from '../../components/ToastContainer';
import { StreamPlayer } from '../StreamPlayer';
import { StreamControls } from '../StreamControls';
import { TransactionModal } from '../../components/headless/auth/TransactionModal';
import { useCamera } from '../CameraProvider';
import { PublicKey, SystemProgram } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';
import { useProgram } from '../../anchor/setup';
import { unifiedIpfsService } from '../../services/unified-ipfs-service';
import { cameraStatus } from '../../services/camera-status';

type TimelineEventType =
  | 'photo_captured'
  | 'video_recorded'
  | 'stream_started'
  | 'stream_ended';

interface TimelineEvent {
  type: TimelineEventType;
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
  };
  timestamp: number;
  transactionId?: string;
}

export function CameraView() {
  const { primaryWallet, user } = useDynamicContext();
  useEmbeddedWallet();
  const { cameraKeypair, isInitialized } = useCamera();
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
  const [, setIsMobileView] = useState(window.innerWidth <= 768);

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

  const handleAction = async (type: 'photo' | 'video' | 'stream') => {
    if (!cameraKeypair || !primaryWallet?.address || !program || !isInitialized) {
      return;
    }

    const isEmbedded = primaryWallet.connector?.name.toLowerCase() !== 'phantom';
    if (!isEmbedded) {
      try {
        setLoading(true);

        // First sign the transaction
        const txId = await program.methods.activateCamera(new BN(100))
          .accounts({
            cameraAccount: cameraKeypair.publicKey,
            user: new PublicKey(primaryWallet.address),
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        // Then execute the camera action
        if (type === 'photo') {
          updateToast('info', 'Taking picture...');
          const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/capture`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${primaryWallet.address}`,
              'Cache-Control': 'no-cache'
            }
          });

          if (!response.ok) {
            throw new Error('Failed to capture photo');
          }

          updateToast('info', 'Processing image...');
          const imageBlob = await response.blob();

          updateToast('info', 'Uploading to IPFS...');
          const results = await unifiedIpfsService.uploadFile(imageBlob, primaryWallet.address, 'image');
          
          if (results.length === 0) {
            throw new Error('Failed to upload image to any IPFS provider');
          }

          timelineRef.current?.addEvent({
            type: 'photo_captured',
            timestamp: Date.now(),
            user: { 
              address: primaryWallet.address,
              username: user?.verifiedCredentials?.[0]?.oauthUsername,
              displayName: user?.verifiedCredentials?.[0]?.oauthDisplayName,
              pfpUrl: user?.verifiedCredentials?.[0]?.oauthAccountPhotos?.[0]
            },
            transactionId: txId
          });
          updateToast('success', 'Photo captured and uploaded successfully');
        } else if (type === 'video') {
          updateToast('info', 'Starting recording...');
          const duration = 30; // Match the original duration
          const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/video/start`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${primaryWallet.address}`,
              'Cache-Control': 'no-cache'
            },
            body: JSON.stringify({ duration })
          });

          if (!response.ok) {
            throw new Error('Failed to start recording');
          }

          const { filename } = await response.json();
          console.log('Recording started with filename:', filename);

          // Wait for recording to complete
          updateToast('info', 'Recording in progress...');
          await new Promise<void>((resolve) => {
            let timeRemaining = duration;
            const timer = setInterval(() => {
              timeRemaining -= 1;
              if (timeRemaining <= 0) {
                clearInterval(timer);
                resolve();
              }
            }, 1000);
          });

          // Add a small delay after recording completes
          await new Promise(resolve => setTimeout(resolve, 2000));

          // Download and upload
          updateToast('info', 'Processing video...');
          const mp4filename = filename.replace('.h264', '.mp4');
          const downloadResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/video/download/${mp4filename}`, {
            headers: {
              'Accept': 'video/mp4',
              'Cache-Control': 'no-cache'
            },
          });

          if (!downloadResponse.ok) {
            throw new Error('Failed to download video');
          }

          const videoBlob = await downloadResponse.blob();
          console.log('Downloaded video size:', videoBlob.size);

          updateToast('info', 'Uploading to IPFS...');
          const results = await unifiedIpfsService.uploadFile(videoBlob, primaryWallet.address, 'video');
          
          if (results.length === 0) {
            throw new Error('Failed to upload video to any IPFS provider');
          }

          timelineRef.current?.addEvent({
            type: 'video_recorded',
            timestamp: Date.now(),
            user: { 
              address: primaryWallet.address,
              username: user?.verifiedCredentials?.[0]?.oauthUsername,
              displayName: user?.verifiedCredentials?.[0]?.oauthDisplayName,
              pfpUrl: user?.verifiedCredentials?.[0]?.oauthAccountPhotos?.[0]
            },
            transactionId: txId
          });
          updateToast('success', 'Video recorded and uploaded successfully');
        } else if (type === 'stream') {
          const endpoint = isStreaming ? '/api/stream/stop' : '/api/stream/start';
          const response = await fetch(`${CONFIG.CAMERA_API_URL}${endpoint}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${primaryWallet.address}`,
              'Cache-Control': 'no-cache'
            }
          });

          if (!response.ok) {
            throw new Error(`Failed to ${isStreaming ? 'stop' : 'start'} stream`);
          }

          timelineRef.current?.addEvent({
            type: isStreaming ? 'stream_ended' : 'stream_started',
            timestamp: Date.now(),
            user: { 
              address: primaryWallet.address,
              username: user?.verifiedCredentials?.[0]?.oauthUsername,
              displayName: user?.verifiedCredentials?.[0]?.oauthDisplayName,
              pfpUrl: user?.verifiedCredentials?.[0]?.oauthAccountPhotos?.[0]
            },
            transactionId: txId
          });
        }
      } catch (error) {
        console.error('Action failed:', error);
        updateToast('error', error instanceof Error ? error.message : 'Action failed');
      } finally {
        setLoading(false);
      }
      return;
    }

    // For embedded wallets, show the transaction modal
    setCurrentAction({
      type,
      cameraAccount: cameraKeypair.publicKey.toString()
    });
    setIsModalOpen(true);
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

  const handleTransactionSuccess = async ({ transactionId }: { transactionId: string }) => {
    if (!currentAction || !timelineRef.current) return;

    try {
      setLoading(true);

      // Handle the camera action based on type
      if (currentAction.type === 'photo') {
        await activateCameraRef.current?.handleTakePicture();
      } else if (currentAction.type === 'video') {
        await videoRecorderRef.current?.startRecording();
      } else if (currentAction.type === 'stream') {
        // For stream, we need to call the API directly
        const endpoint = isStreaming ? '/api/stream/stop' : '/api/stream/start';
        const response = await fetch(`${CONFIG.CAMERA_API_URL}${endpoint}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${primaryWallet?.address}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to ${isStreaming ? 'stop' : 'start'} stream: ${response.statusText}`);
        }

        setIsStreaming(!isStreaming);
      }

      // Create the timeline event with transaction ID
      const event: TimelineEvent = {
        type: getEventType(currentAction.type),
        user: {
          address: primaryWallet?.address || '',
          username: user?.verifiedCredentials?.[0]?.oauthUsername || undefined,
          displayName: user?.verifiedCredentials?.[0]?.oauthDisplayName || undefined,
          pfpUrl: user?.verifiedCredentials?.[0]?.oauthAccountPhotos?.[0] || undefined
        },
        timestamp: Date.now(),
        transactionId
      };

      // Add the event to the timeline
      timelineRef.current.addEvent(event);

      // Show success message
      updateToast('success', 'Action completed successfully');
    } catch (error) {
      console.error('API call failed:', error);
      updateToast('error', 'Failed to complete action after transaction');
    } finally {
      setLoading(false);
      setIsModalOpen(false);
      setCurrentAction(null);
    }
  };

  // Helper function to convert action type to event type
  const getEventType = (actionType: string): TimelineEventType => {
    switch (actionType) {
      case 'photo':
        return 'photo_captured';
      case 'video':
        return 'video_recorded';
      case 'stream':
        return isStreaming ? 'stream_ended' : 'stream_started';
      default:
        return 'photo_captured';
    }
  };

  useEffect(() => {
    const unsubscribe = cameraStatus.subscribe(({ isLive: live, isStreaming: streaming }) => {
      setIsLive(live);
      setIsStreaming(streaming);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setIsMobileView(window.innerWidth <= 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <>
      <TransactionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        transactionData={currentAction || undefined}
        onSuccess={handleTransactionSuccess}
      />

      <div className="h-full overflow-y-auto pb-40">
        <div className="relative max-w-3xl mx-auto pt-8">
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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-3 relative">
                <StreamPlayer />

                <div className="hidden sm:flex absolute -right-14 top-0 flex-col h-full z-[45]">
                  <StreamControls timelineRef={timelineRef} onStreamToggle={() => handleAction('stream')} />
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
                      onPhotoCapture={() => handleAction('photo')}
                      onStatusUpdate={({ type, message }) => updateToast(type, message)}
                    />
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Processing...' : 'Take Picture'}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <VideoRecorder
                      ref={videoRecorderRef}
                      onVideoRecorded={() => handleAction('video')}
                      onStatusUpdate={({ type, message }) => updateToast(type, message)}
                    />
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Recording...' : 'Record Video'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex-1 mt-2 px-2">
                  <CameraControls
                    onTakePicture={() => handleAction('photo')}
                    onRecordVideo={() => handleAction('video')}
                    onToggleStream={() => handleAction('stream')}
                    isLoading={loading}
                    isStreaming={isStreaming}
                  />
                </div>
          <div className="max-w-3xl mt-6 mx-auto flex flex-col justify-top relative">
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
                {/* <div className="flex-1 px-2">
                  <CameraControls
                    onTakePicture={() => handleAction('photo')}
                    onRecordVideo={() => handleAction('video')}
                    onToggleStream={() => handleAction('stream')}
                    isLoading={loading}
                    isStreaming={isStreaming}
                  />
                </div> */}
              </div>
            </div>

            <div className="absolute mt-12 pb-20 pl-6 left-0 w-full">
              <Timeline ref={timelineRef} variant="camera" />
              <div
                className="top-0 left-0 right-0 pointer-events-none"
                style={{
                  background: 'linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)'
                }}
              />
            </div>

            <div className="relative md:ml-20 ml-16 bg-white">
              <div className="relative pl-4 pr-2 sm:px-4">
                <MediaGallery mode="recent" maxRecentItems={6} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
