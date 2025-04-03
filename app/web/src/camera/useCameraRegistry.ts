import { useState, useEffect, useCallback } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection, useWallet } from '@solana/wallet-adapter-react';
import { PublicKey } from '@solana/web3.js';
import { cameraRegistryService, CameraAccount } from './camera-registry-service';
import { useCamera } from './CameraProvider';
import { useCameraActivationProgram } from '../anchor/setup';

// Track global initialization state
let globalInitAttempted = false;

export interface UseCameraRegistryResult {
  loading: boolean;
  initialized: boolean;
  error: string | null;
  cameraData: CameraAccount | null;
  isRegistered: boolean;
  cameraAccounts: CameraAccount[];
  refresh: () => Promise<void>;
  registerCamera: (name: string, model: string) => Promise<string | null>;
  setCameraActive: (cameraId: string, isActive: boolean) => Promise<string | null>;
  recordActivity: (type: 'photo' | 'video' | 'stream', metadata: string) => Promise<string | null>;
}

export function useCameraRegistry(): UseCameraRegistryResult {
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
  const [cameraAccounts, setCameraAccounts] = useState<CameraAccount[]>([]);

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
        cameraRegistryService.useExistingProgram(program as any);
        
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

  // Function to load camera accounts
  const loadCameraAccounts = async () => {
    if (!primaryWallet?.address) {
      setError('Wallet not connected');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      
      // Use the service to get all cameras associated with this wallet
      const ownerPublicKey = new PublicKey(primaryWallet.address);
      const cameras = await cameraRegistryService.getCamerasByOwner(ownerPublicKey);
      
      if (cameras) {
        setCameraAccounts(cameras);
      } else {
        setCameraAccounts([]);
      }
      
      setError(null);
    } catch (err) {
      console.error('Error loading camera accounts:', err);
      setError('Failed to load cameras');
    } finally {
      setLoading(false);
    }
  };

  // Register a new camera
  const registerCamera = async (name: string, model: string): Promise<string | null> => {
    if (!primaryWallet?.address) {
      setError('Wallet not connected');
      return null;
    }

    try {
      setLoading(true);
      
      // Generate a unique camera ID based on wallet and timestamp
      const shortAddress = primaryWallet.address.slice(0, 6);
      const timestamp = Date.now().toString(36);
      const cameraId = `cam_${shortAddress}_${timestamp}`;
      
      // Register the camera
      const tx = await cameraRegistryService.registerCamera(
        new PublicKey(primaryWallet.address),
        cameraId,
        name,
        model
      );
      
      // Refresh camera accounts
      await loadCameraAccounts();
      
      return tx;
    } catch (err) {
      console.error('Error registering camera:', err);
      setError('Failed to register camera');
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Set camera active status
  const setCameraActive = async (cameraId: string, isActive: boolean): Promise<string | null> => {
    if (!primaryWallet?.address) {
      setError('Wallet not connected');
      return null;
    }

    try {
      setLoading(true);
      
      // Use the service to set the camera's active status
      const tx = await cameraRegistryService.setCameraActive(
        new PublicKey(primaryWallet.address),
        cameraId,
        isActive
      );
      
      // Refresh camera accounts
      await loadCameraAccounts();
      
      return tx;
    } catch (err) {
      console.error('Error setting camera active status:', err);
      setError(`Failed to ${isActive ? 'activate' : 'deactivate'} camera`);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Record camera activity
  const recordActivity = async (type: 'photo' | 'video' | 'stream', metadata: string): Promise<string | null> => {
    if (!primaryWallet?.address || !initialized || !cameraKeypair || !isRegistered) {
      setError('Camera not registered or service not initialized');
      return null;
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

  // Load camera accounts when wallet or program changes
  useEffect(() => {
    if (program) {
      loadCameraAccounts();
    }
  }, [primaryWallet?.address, program]);

  return {
    loading,
    initialized,
    error: error || programError?.message || null,
    cameraData,
    isRegistered,
    cameraAccounts,
    refresh: loadCameraAccounts,
    registerCamera,
    setCameraActive,
    recordActivity
  };
}