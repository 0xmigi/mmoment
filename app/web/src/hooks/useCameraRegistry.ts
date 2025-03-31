import { useState, useEffect, useCallback } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection, useWallet } from '@solana/wallet-adapter-react';
import { PublicKey } from '@solana/web3.js';
import { cameraRegistryService, CameraAccount } from '../services/camera-registry-service';
import { useCamera } from '../components/CameraProvider';
import { useCameraActivationProgram } from '../anchor/setup';

// Track global initialization state
let globalInitAttempted = false;

export function useCameraRegistry() {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const solanaWallet = useWallet();
  const { program, loading: programLoading, error: programError } = useCameraActivationProgram();
  const { cameraKeypair } = useCamera();
  
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraData, setCameraData] = useState<CameraAccount | null>(null);
  const [isRegistered, setIsRegistered] = useState(false);

  // Get the active wallet address
  const walletAddress = primaryWallet?.address || solanaWallet.publicKey?.toString();

  // Single initialization attempt
  useEffect(() => {
    let mounted = true;

    const initializeService = async () => {
      // Prevent any initialization if we've already tried globally
      if (globalInitAttempted) {
        return;
      }

      // Don't initialize if we're missing critical dependencies
      if (!walletAddress || !connection || !program || programLoading) {
        return;
      }

      try {
        setLoading(true);
        setError(null);

        // Mark that we've attempted initialization
        globalInitAttempted = true;

        // Initialize the service
        cameraRegistryService.useExistingProgram(program);
        
        if (mounted) {
          setInitialized(true);
        }

        // Only check camera registration if we have all required data
        if (cameraKeypair && mounted) {
          try {
            const activeWalletPublicKey = new PublicKey(walletAddress);
            const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
            
            const data = await cameraRegistryService.getCameraAccount(
              cameraId,
              activeWalletPublicKey
            );
            
            if (data && mounted) {
              setCameraData(data);
              setIsRegistered(true);
            }
          } catch (e) {
            // Silently fail for account not found
            if (e instanceof Error && !e.message.includes('Account does not exist') && mounted) {
              setError('Camera not found');
            }
          }
        }
      } catch (err) {
        if (mounted) {
          setError('Failed to initialize');
          setInitialized(false);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    // Only attempt initialization once
    if (!globalInitAttempted) {
      initializeService();
    }

    return () => {
      mounted = false;
    };
  }, [walletAddress, connection, program, programLoading, cameraKeypair]);

  // Reset state when wallet changes
  useEffect(() => {
    if (!walletAddress) {
      setInitialized(false);
      setCameraData(null);
      setIsRegistered(false);
      setError(null);
      // Reset global init state when wallet changes
      globalInitAttempted = false;
    }
  }, [walletAddress]);

  const getActivePublicKey = useCallback((): PublicKey => {
    if (!walletAddress) {
      throw new Error('No wallet connected');
    }
    return new PublicKey(walletAddress);
  }, [walletAddress]);

  // Simplified register camera function
  const registerCamera = async (name: string, model: string, location?: [number, number]) => {
    if (!walletAddress || !initialized || !cameraKeypair) {
      throw new Error('Not ready');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = getActivePublicKey();
      
      const tx = await cameraRegistryService.registerCamera(
        ownerPubkey,
        cameraId,
        name,
        model,
        location
      );

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

  // Set camera active/inactive
  const setCameraActive = async (isActive: boolean) => {
    if (!walletAddress || !initialized || !cameraKeypair || !isRegistered) {
      throw new Error('Camera not registered or service not initialized');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = getActivePublicKey();
      
      const tx = await cameraRegistryService.setCameraActive(
        ownerPubkey,
        cameraId,
        isActive
      );

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

  // Record camera activity
  const recordActivity = async (type: 'photo' | 'video' | 'stream', metadata: string) => {
    if (!walletAddress || !initialized || !cameraKeypair || !isRegistered) {
      throw new Error('Camera not registered or service not initialized');
    }

    try {
      setLoading(true);
      setError(null);

      const cameraId = `cam_${cameraKeypair.publicKey.toString().slice(0, 8)}`;
      const ownerPubkey = getActivePublicKey();
      
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

      return tx;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to record activity');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    initialized,
    error: error || programError?.message || null,
    cameraData,
    isRegistered,
    registerCamera,
    setCameraActive,
    recordActivity
  };
}