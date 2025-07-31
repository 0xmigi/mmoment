/* eslint-disable @typescript-eslint/no-explicit-any */
import type React from 'react';
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Keypair, PublicKey, SystemProgram } from '@solana/web3.js';
import type { Transaction, VersionedTransaction, Connection } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { CAMERA_ACTIVATION_PROGRAM_ID, useProgram } from '../anchor/setup';

// Define types for specific structures
interface CameraAccountData {
  owner: PublicKey;
  isActive: boolean;
  activityCounter?: { toNumber: () => number };
  metadata?: {
    name: string;
    model: string;
    registrationDate?: { toNumber: () => number };
    location?: number[];
  };
  lastActivityAt?: { toNumber: () => number };
}

// We need a type that can handle both Transaction and VersionedTransaction
type SolanaTransaction = Transaction | VersionedTransaction;

// Simple cache for account info to reduce RPC calls
const accountInfoCache = new Map<string, { data: CameraData | null; timestamp: number }>();
const CACHE_TTL = 30000; // 30 seconds

// Export fetchCameraByPublicKey so it can be imported directly
export const fetchCameraByPublicKey = async (publicKey: string, connection: Connection): Promise<CameraData | null> => {
  try {
    console.log(`[CameraProvider] Fetching camera by public key: ${publicKey}`);
    
    // Check cache first
    const cameraPubkey = new PublicKey(publicKey);
    const key = cameraPubkey.toString();
    const now = Date.now();
    const cached = accountInfoCache.get(`camera_account_${key}`);
    
    if (cached && now - cached.timestamp < CACHE_TTL) {
      console.log(`[CameraProvider] Using cached camera account for ${key}`);
      return cached.data;
    }
    
    // Create a provider without a wallet (read-only)
    const dummyWallet = {
      publicKey: PublicKey.default,
      // Using the any type here explicitly since the Anchor types are complex
      signTransaction: async <T extends SolanaTransaction>(tx: T): Promise<T> => tx,
      signAllTransactions: async <T extends SolanaTransaction>(txs: T[]): Promise<T[]> => txs
    };
    
    const provider = new AnchorProvider(
      connection,
      dummyWallet,
      { commitment: 'confirmed' }
    );
    
    // Create a direct program instance
    const program = new Program(
      IDL,
      CAMERA_ACTIVATION_PROGRAM_ID,
      provider
    );
    
    console.log(`[CameraProvider] Program instance created directly with ID: ${CAMERA_ACTIVATION_PROGRAM_ID.toString()}`);
    
    // Fetch the camera account
    try {
      const cameraAccount = await program.account.cameraAccount.fetch(cameraPubkey) as unknown as CameraAccountData;
      
      if (cameraAccount) {
        console.log('[CameraProvider] Camera account found:', cameraAccount);
        const cameraData: CameraData = {
          publicKey,
          owner: cameraAccount.owner.toString(),
          isActive: cameraAccount.isActive,
          activityCounter: cameraAccount.activityCounter?.toNumber() || 0,
          metadata: {
            name: cameraAccount.metadata?.name || 'Unnamed Camera',
            model: cameraAccount.metadata?.model || 'Unknown Model',
            registrationDate: cameraAccount.metadata?.registrationDate?.toNumber() || 0,
            location: null
          }
        };
        
        // Cache the result
        accountInfoCache.set(`camera_account_${key}`, { data: cameraData, timestamp: now });
        
        return cameraData;
      }
    } catch (error) {
      console.error('[CameraProvider] Error fetching camera account:', error);
    }
    
    // If we got here, no camera was found or there was an error
    accountInfoCache.set(`camera_account_${key}`, { data: null, timestamp: now });
    return null;
  } catch (error) {
    console.error('[CameraProvider] Error fetching camera:', error);
    return null;
  }
};

export type QuickActionType = 'photo' | 'video' | 'stream';

interface QuickActions {
  photo: boolean;
  video: boolean;
  stream: boolean;
}

// Add CameraData interface
export interface CameraData {
  publicKey: string;
  owner: string;
  isActive: boolean;
  activityCounter?: number;
  lastActivityType?: number;
  lastActivityAt?: number;
  metadata: {
    name: string;
    model: string;
    registrationDate: number;
    location?: [number, number] | null;
  };
}

interface CameraContextType {
  cameraKeypair: Keypair;
  isInitialized: boolean;
  loading: boolean;
  error: string | null;
  quickActions: QuickActions;
  updateQuickAction: (type: QuickActionType, enabled: boolean) => void;
  hasQuickAction: (type: QuickActionType) => boolean;
  refreshCameraStatus: () => Promise<void>;
  // Add new properties for camera selection
  selectedCamera: CameraData | null;
  setSelectedCamera: (camera: CameraData | null) => void;
  fetchCameraById: (cameraId: string) => Promise<CameraData | null>;
  // Add global camera list refresh
  triggerCameraListRefresh: () => void;
  onCameraListRefresh: (callback: () => void) => () => void;
}

const CameraContext = createContext<CameraContextType | undefined>(undefined);

const QUICK_ACTIONS_STORAGE_KEY = 'camera_quick_actions';
const SELECTED_CAMERA_STORAGE_KEY = 'selected_camera';

export function CameraProvider({ children }: { children: React.ReactNode }) {
  const { primaryWallet } = useDynamicContext();
  const { program, loading: programLoading } = useProgram();
  const [isInitialized, setIsInitialized] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Add refresh callback system
  const [refreshCallbacks] = useState(new Set<() => void>());
  
  // Add selected camera state
  const [selectedCamera, setSelectedCamera] = useState<CameraData | null>(() => {
    const stored = localStorage.getItem(SELECTED_CAMERA_STORAGE_KEY);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (e) {
        console.error('Failed to parse stored selected camera:', e);
      }
    }
    return null;
  });
  
  // Quick actions state
  const [quickActions, setQuickActions] = useState<QuickActions>(() => {
    const stored = localStorage.getItem(QUICK_ACTIONS_STORAGE_KEY);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (e) {
        console.error('Failed to parse stored quick actions:', e);
      }
    }
    return {
      photo: false,
      video: false,
      stream: false
    };
  });
  
  // Persist camera keypair
  const [cameraKeypair] = useState(() => {
    const stored = localStorage.getItem('cameraKeypair');
    if (stored) {
      const keypairData = new Uint8Array(JSON.parse(stored));
      return Keypair.fromSecretKey(keypairData);
    }
    const newKeypair = Keypair.generate();
    localStorage.setItem('cameraKeypair', JSON.stringify(Array.from(newKeypair.secretKey)));
    return newKeypair;
  });

  // Update the selected camera and persist it
  const handleSetSelectedCamera = (camera: CameraData | null) => {
    setSelectedCamera(camera);
    if (camera) {
      localStorage.setItem(SELECTED_CAMERA_STORAGE_KEY, JSON.stringify(camera));
    } else {
      localStorage.removeItem(SELECTED_CAMERA_STORAGE_KEY);
    }
  };

  // Quick actions management
  const updateQuickAction = (type: QuickActionType, enabled: boolean) => {
    const newQuickActions = {
      ...quickActions,
      [type]: enabled
    };
    setQuickActions(newQuickActions);
    localStorage.setItem(QUICK_ACTIONS_STORAGE_KEY, JSON.stringify(newQuickActions));
  };

  const hasQuickAction = (type: QuickActionType): boolean => {
    return quickActions[type];
  };

  // Function to fetch a camera by ID (now just a wrapper around fetchCameraByPublicKey)
  const fetchCameraById = async (cameraId: string): Promise<CameraData | null> => {
    if (!program) {
      console.error('[CameraProvider] Program not available for fetchCameraById');
      return null;
    }

    try {
      // First, validate the cameraId as a PublicKey
      let publicKeyObj: PublicKey;
      try {
        publicKeyObj = new PublicKey(cameraId);
      } catch (err) {
        console.error('[CameraProvider] Invalid camera ID format:', err);
        return null;
      }
      
      console.log(`[CameraProvider] Fetching camera by ID: ${cameraId}`);
      
      // Just use the public key approach with the decoded/validated key
      const result = await fetchCameraByPublicKey(publicKeyObj.toString(), program.provider.connection);
      
      if (result) {
        setSelectedCamera(result);
        return result;
      }
      return null;
    } catch (err) {
      console.error('[CameraProvider] Error in fetchCameraById:', err);
      return null;
    }
  };

  // Function to check camera initialization status
  const checkInitialization = useCallback(async () => {
    if (!primaryWallet?.address || !program || programLoading) {
      setLoading(false);
      return;
    }

    try {
      // Check if wallet is a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        setError('Connected wallet is not a Solana wallet');
        setLoading(false);
        return;
      }

      console.log('Checking camera initialization with keypair:', cameraKeypair.publicKey.toString());
      
      // Find the registry PDA
      const [registryAddress] = await PublicKey.findProgramAddress(
        [Buffer.from('camera-registry')],
        program.programId
      );
      
      console.log('Registry address:', registryAddress.toString());

      try {
        // Try to fetch the registry account
        const registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
        console.log('Registry account found:', registryAccount);
        
        // Now check if camera account exists
        try {
          // Find the camera PDA
          const [cameraAddress] = await PublicKey.findProgramAddress(
            [
              Buffer.from('camera'),
              Buffer.from(cameraKeypair.publicKey.toString()),
              new PublicKey(primaryWallet.address).toBuffer()
            ],
            program.programId
          );
          
          console.log('Camera address:', cameraAddress.toString());
          
          // Try to fetch the camera account
          const cameraAccount = await program.account.cameraAccount.fetch(cameraAddress);
          console.log('Camera account found:', cameraAccount);
          setIsInitialized(true);
        } catch {
          console.log('Camera account not found, need to register camera');
          setIsInitialized(false);
        }
      } catch {
        console.log('Registry not initialized, initializing...');
        
        // Initialize the registry directly without explicitly using the signer
        // as the signer will be provided by the AnchorProvider
        const tx = await program.methods.initialize()
          .accounts({
            authority: new PublicKey(primaryWallet.address),
            cameraRegistry: registryAddress,
            systemProgram: SystemProgram.programId,
          })
          .rpc();
        
        console.log('Registry initialized with tx:', tx);
        setIsInitialized(true);
      }
      
      // Clear any previous errors
      setError(null);
    } catch (err) {
      console.error('Error checking initialization:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize camera');
    } finally {
      setLoading(false);
    }
  }, [primaryWallet, program, programLoading, cameraKeypair]);

  // Expose a function to refresh camera status
  const refreshCameraStatus = async () => {
    setLoading(true);
    await checkInitialization();
  };

  // Global camera list refresh functions
  const triggerCameraListRefresh = useCallback(() => {
    console.log('[CameraProvider] Triggering camera list refresh for all components');
    refreshCallbacks.forEach(callback => {
      try {
        callback();
      } catch (error) {
        console.error('[CameraProvider] Error in refresh callback:', error);
      }
    });
  }, [refreshCallbacks]);

  const onCameraListRefresh = useCallback((callback: () => void) => {
    refreshCallbacks.add(callback);
    // Return cleanup function
    return () => {
      refreshCallbacks.delete(callback);
    };
  }, [refreshCallbacks]);

  // Check initialization when wallet connects or program changes
  useEffect(() => {
    if (primaryWallet?.address && program && !programLoading) {
      checkInitialization();
    }
  }, [primaryWallet?.address, program, programLoading, checkInitialization]);

  // Add effect to clear camera when on default route
  useEffect(() => {
    // Check if we're on default route
    const isDefaultRoute = window.location.pathname === '/app' || window.location.pathname === '/app/';
    
    if (isDefaultRoute && selectedCamera) {
      console.log('[CameraProvider] On default route - clearing selected camera');
      setSelectedCamera(null);
      localStorage.removeItem('directCameraId');
      localStorage.removeItem(SELECTED_CAMERA_STORAGE_KEY);
    }
  }, [selectedCamera]);

  return (
    <CameraContext.Provider 
      value={{
        cameraKeypair,
        isInitialized,
        loading,
        error,
        quickActions,
        updateQuickAction,
        hasQuickAction,
        refreshCameraStatus,
        // Add new properties
        selectedCamera,
        setSelectedCamera: handleSetSelectedCamera,
        fetchCameraById,
        // Add global refresh functions
        triggerCameraListRefresh,
        onCameraListRefresh
      }}
    >
      {children}
    </CameraContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useCamera() {
  const context = useContext(CameraContext);
  if (context === undefined) {
    throw new Error('useCamera must be used within a CameraProvider');
  }
  return context;
}