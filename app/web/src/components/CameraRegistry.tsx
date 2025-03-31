import React, { useState } from 'react';
import { useNotifications } from './NotificationProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useCamera } from '../components/CameraProvider';
import { useProgram } from '../anchor/setup';
import { cameraRegistryService, CameraAccount } from '../services/camera-registry-service';
import { PublicKey } from '@solana/web3.js';

interface CameraRegistryProps {
  onStatusUpdate?: (status: { isRegistered: boolean; isActive: boolean }) => void;
}

export function CameraRegistry({ onStatusUpdate }: CameraRegistryProps) {
  const { primaryWallet } = useDynamicContext();
  const { cameraKeypair } = useCamera();
  const { program } = useProgram();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraData, setCameraData] = useState<CameraAccount | null>(null);
  const [isRegistered, setIsRegistered] = useState(false);
  const { addNotification } = useNotifications();
  const [cameraName, setCameraName] = useState('My Camera');
  const [cameraModel, setCameraModel] = useState('Raspberry Pi Camera');
  const [registering, setRegistering] = useState(false);
  const [activating, setActivating] = useState(false);

  const registerCamera = async (name: string, model: string, location?: [number, number]) => {
    if (!primaryWallet?.address || !cameraKeypair) {
      throw new Error('Not ready');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = new PublicKey(primaryWallet.address);
      
      const tx = await cameraRegistryService.registerCamera(
        ownerPubkey,
        cameraId,
        name,
        model,
        location
      );

      if (!tx) {
        throw new Error('Failed to register camera');
      }

      // Wait for confirmation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const data = await cameraRegistryService.getCameraAccount(
        cameraId,
        ownerPubkey
      );
      
      if (data) {
        setCameraData(data);
        setIsRegistered(true);
      }

      return tx;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const setCameraActive = async (isActive: boolean) => {
    if (!primaryWallet?.address || !cameraKeypair || !isRegistered) {
      throw new Error('Camera not registered');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = new PublicKey(primaryWallet.address);
      
      const tx = await cameraRegistryService.setCameraActive(
        ownerPubkey,
        cameraId,
        isActive
      );

      if (!tx) {
        throw new Error(`Failed to ${isActive ? 'activate' : 'deactivate'} camera`);
      }

      // Fetch the updated camera data
      const data = await cameraRegistryService.getCameraAccount(
        cameraId,
        ownerPubkey
      );
      
      if (data) {
        setCameraData(data);
      }

      return tx;
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${isActive ? 'activate' : 'deactivate'} camera`);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const recordActivity = async (type: 'photo' | 'video' | 'stream', metadata: string) => {
    if (!primaryWallet?.address || !cameraKeypair || !isRegistered) {
      throw new Error('Camera not registered');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = new PublicKey(primaryWallet.address);
      
      // Map the activity type
      let activityType;
      switch (type) {
        case 'photo':
          activityType = { photoCapture: {} };
          break;
        case 'video':
          activityType = { videoRecord: {} };
          break;
        case 'stream':
          activityType = { liveStream: {} };
          break;
        default:
          activityType = { custom: {} };
      }
      
      const tx = await cameraRegistryService.recordActivity(
        ownerPubkey,
        cameraId,
        activityType,
        metadata
      );

      if (!tx) {
        throw new Error('Failed to record activity');
      }

      return tx;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to record activity');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    if (!primaryWallet?.address || registering) return;

    try {
      setRegistering(true);
      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: false,
          isActive: false
        });
      }

      const tx = await registerCamera(cameraName, cameraModel);
      if (!tx) {
        throw new Error('Failed to register camera');
      }

      addNotification('success', 'Camera registered successfully');

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: false
        });
      }
    } catch (error) {
      const userFriendlyError = error instanceof Error ? error.message : 'Failed to register camera';
      addNotification('error', userFriendlyError);

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: false,
          isActive: false
        });
      }
    } finally {
      setRegistering(false);
    }
  };

  const handleToggleActive = async (newState: boolean) => {
    if (!primaryWallet?.address || activating) return;

    try {
      setActivating(true);
      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: newState
        });
      }

      const tx = await setCameraActive(newState);
      if (!tx) {
        throw new Error(`Failed to ${newState ? 'activate' : 'deactivate'} camera`);
      }

      addNotification('success', `Camera ${newState ? 'activated' : 'deactivated'} successfully`);

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: newState
        });
      }
    } catch (error) {
      const userFriendlyError = error instanceof Error ? error.message : `Failed to ${newState ? 'activate' : 'deactivate'} camera`;
      addNotification('error', userFriendlyError);

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: false
        });
      }
    } finally {
      setActivating(false);
    }
  };

  const handleRecordPhoto = async () => {
    if (!primaryWallet?.address || !isRegistered) return;

    try {
      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: true
        });
      }

      const tx = await recordActivity('photo', 'Photo captured');
      if (!tx) {
        throw new Error('Failed to record photo activity');
      }

      addNotification('success', 'Photo activity recorded successfully');

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: true
        });
      }
    } catch (error) {
      const userFriendlyError = error instanceof Error ? error.message : 'Failed to record photo activity';
      addNotification('error', userFriendlyError);

      if (onStatusUpdate) {
        onStatusUpdate({
          isRegistered: true,
          isActive: false
        });
      }
    }
  };

  // Record activity handlers for photo, video, and stream
  const handleRecordVideoActivity = async () => {
    try {
      addNotification('info', 'Recording video activity on blockchain...');
      
      const metadata = JSON.stringify({
        timestamp: new Date().toISOString(),
        type: 'video',
        duration: '30s',
        resolution: '1920x1080'
      });
      
      const tx = await recordActivity('video', metadata);
      
      addNotification('success', `Video activity recorded successfully! TX: ${tx.slice(0, 8)}...`);
    } catch (err) {
      console.error('Failed to record video activity:', err);
      addNotification('error', err instanceof Error ? err.message : 'Failed to record video activity');
    }
  };

  const handleRecordStreamActivity = async () => {
    try {
      addNotification('info', 'Recording stream activity on blockchain...');
      
      const metadata = JSON.stringify({
        timestamp: new Date().toISOString(),
        type: 'stream',
        duration: 'live',
        resolution: '1920x1080'
      });
      
      const tx = await recordActivity('stream', metadata);
      
      addNotification('success', `Stream activity recorded successfully! TX: ${tx.slice(0, 8)}...`);
    } catch (err) {
      console.error('Failed to record stream activity:', err);
      addNotification('error', err instanceof Error ? err.message : 'Failed to record stream activity');
    }
  };
  
  if (loading) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 mt-4">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"></div>
          <p className="text-gray-600">Loading camera registry...</p>
        </div>
      </div>
    );
  }

  if (error) {
    let errorDetails = '';
    let errorInstructions = [];
    
    // Provide more specific instructions based on error type
    if (error.includes('_bn')) {
      errorDetails = 'There was an issue reading data from the blockchain.';
      errorInstructions = [
        'Try refreshing the page',
        'Make sure your wallet is properly connected to Devnet',
        'Check that you have SOL in your wallet on Devnet'
      ];
    } else if (error.includes('network') || error.includes('connection')) {
      errorDetails = 'There was a problem connecting to the Solana network.';
      errorInstructions = [
        'Check your internet connection',
        'Make sure the Solana Devnet is operational',
        'Try again in a few minutes'
      ];
    } else if (error.includes('wallet')) {
      errorDetails = 'There was an issue with your wallet connection.';
      errorInstructions = [
        'Make sure your wallet is properly connected',
        'Try disconnecting and reconnecting your wallet',
        'Ensure your wallet is set to Devnet network'
      ];
    } else {
      errorInstructions = [
        'Make sure your wallet is properly connected and on Devnet',
        'Check that you have SOL in your wallet on Devnet',
        'Check the browser console for more detailed error messages',
        'Refresh the page and try again'
      ];
    }
    
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg mt-4">
        <p className="font-medium">Error Initializing Camera Registry</p>
        <p className="mt-2">{error}</p>
        {errorDetails && <p className="mt-1 text-sm">{errorDetails}</p>}
        
        <div className="mt-4 p-4 bg-red-100 rounded text-sm">
          <p className="font-semibold mb-2">Troubleshooting:</p>
          <ol className="list-decimal pl-4 space-y-1">
            {errorInstructions.map((instruction, idx) => (
              <li key={idx}>{instruction}</li>
            ))}
          </ol>
        </div>
        
        <div className="mt-4">
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-lg p-4 mt-4">
      <h3 className="text-lg font-medium mb-4">Blockchain Camera Registry</h3>
      
      {isRegistered ? (
        <div className="space-y-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-500">Camera Name:</div>
              <div>{cameraData?.metadata?.name}</div>
              
              <div className="text-gray-500">Camera Model:</div>
              <div>{cameraData?.metadata?.model}</div>
              
              <div className="text-gray-500">Status:</div>
              <div>
                {cameraData?.isActive ? (
                  <span className="text-green-600 inline-flex items-center">
                    <span className="w-2 h-2 bg-green-600 rounded-full mr-2"></span>
                    Active
                  </span>
                ) : (
                  <span className="text-red-600 inline-flex items-center">
                    <span className="w-2 h-2 bg-red-600 rounded-full mr-2"></span>
                    Inactive
                  </span>
                )}
              </div>
              
              <div className="text-gray-500">Registration Date:</div>
              <div>{cameraData?.metadata?.registrationDate ? 
                new Date(Number(cameraData.metadata.registrationDate) * 1000).toLocaleString() : 
                'Unknown'}</div>
              
              <div className="text-gray-500">Last Activity:</div>
              <div>{cameraData?.metadata?.lastActivity ? 
                new Date(Number(cameraData.metadata.lastActivity) * 1000).toLocaleString() : 
                'Never'}</div>
                
              <div className="text-gray-500">Wallet:</div>
              <div className="text-xs font-mono">
                {primaryWallet?.address ? 
                  `${primaryWallet.address.slice(0, 6)}...${primaryWallet.address.slice(-4)}` : 
                  'Not connected'}
              </div>
            </div>
          </div>
          
          <div className="flex justify-between">
            <button
              onClick={() => handleToggleActive(false)}
              disabled={activating}
              className={`px-4 py-2 rounded-lg text-white ${
                cameraData?.isActive 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-green-600 hover:bg-green-700'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {activating ? 'Processing...' : cameraData?.isActive ? 'Deactivate Camera' : 'Activate Camera'}
            </button>
          </div>
          
          <div className="border-t border-gray-200 pt-4 mt-4">
            <h4 className="font-medium mb-2">Record Blockchain Activity</h4>
            <p className="text-sm text-gray-500 mb-3">
              These buttons record camera activities on the blockchain without actually taking photos/videos.
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleRecordPhoto}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Record Photo Activity
              </button>
              <button
                onClick={handleRecordVideoActivity}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Record Video Activity
              </button>
              <button
                onClick={handleRecordStreamActivity}
                className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700"
              >
                Record Stream Activity
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-gray-500">
            This camera is not registered on the blockchain yet. Register it to enable blockchain-based camera control and activity tracking.
          </p>
          
          <div className="bg-blue-50 p-4 rounded-lg text-blue-700 text-sm mb-4">
            <p className="font-semibold">Connected to Solana Devnet</p>
            <p className="mt-1">Your wallet: {
              primaryWallet?.address ? 
                `${primaryWallet.address.slice(0, 10)}...${primaryWallet.address.slice(-4)}` : 
                'Not connected'
            }</p>
          </div>
          
          <div className="space-y-2">
            <div>
              <label htmlFor="camera-name" className="block text-sm font-medium text-gray-700">Camera Name</label>
              <input
                type="text"
                id="camera-name"
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>
            
            <div>
              <label htmlFor="camera-model" className="block text-sm font-medium text-gray-700">Camera Model</label>
              <input
                type="text"
                id="camera-model"
                value={cameraModel}
                onChange={(e) => setCameraModel(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>
            
            <button
              onClick={handleRegister}
              disabled={registering || !primaryWallet?.address}
              className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {registering ? 'Registering...' : 'Register Camera'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}