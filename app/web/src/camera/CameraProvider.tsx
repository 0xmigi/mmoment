 
import type React from 'react';
import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Keypair, PublicKey, SystemProgram } from '@solana/web3.js';
import type { Transaction, VersionedTransaction, Connection } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { CAMERA_ACTIVATION_PROGRAM_ID, useProgram } from '../anchor/setup';
import { unifiedCameraService } from './unified-camera-service';
import { createSignedRequest } from './request-signer';
import { timelineService } from '../timeline/timeline-service';
import { useSocialProfile } from '../auth/social/useSocialProfile';
import { checkUserHasSessionChain, createUserSessionChain } from '../hooks/useUserSessionChain';

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
      const cameraAccount = await (program.account as any).cameraAccount.fetch(cameraPubkey) as unknown as CameraAccountData;
      
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
  // Unified check-in state (Phase 3 Privacy Architecture)
  isCheckedIn: boolean;
  isCheckingIn: boolean;
  checkInError: string | null;
  checkIn: () => Promise<boolean>;
  checkOut: () => Promise<boolean>;
  refreshCheckInStatus: () => Promise<void>;
  onCheckInStatusChange: (callback: (isCheckedIn: boolean) => void) => () => void;
}

const CameraContext = createContext<CameraContextType | undefined>(undefined);

const QUICK_ACTIONS_STORAGE_KEY = 'camera_quick_actions';
const SELECTED_CAMERA_STORAGE_KEY = 'selected_camera';
const SESSION_STORAGE_PREFIX = 'mmoment_session_';

export function CameraProvider({ children }: { children: React.ReactNode }) {
  const { primaryWallet } = useDynamicContext();
  const { primaryProfile } = useSocialProfile();
  const { program, loading: programLoading } = useProgram();
  // Get connection for session chain creation
  const connectionFromProgram = program?.provider?.connection;
  const [isInitialized, setIsInitialized] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add refresh callback system
  const [refreshCallbacks] = useState(new Set<() => void>());

  // Unified check-in state (Phase 3 Privacy Architecture)
  // Initialize from localStorage to preserve state across page refresh
  const [isCheckedIn, setIsCheckedIn] = useState(() => {
    // Check localStorage for any valid session
    const storedCamera = localStorage.getItem(SELECTED_CAMERA_STORAGE_KEY);
    if (!storedCamera) return false;

    try {
      const camera = JSON.parse(storedCamera);
      if (!camera?.publicKey) return false;

      // Check all possible session keys (we don't know wallet address yet)
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith(SESSION_STORAGE_PREFIX) && key.includes(camera.publicKey)) {
          const sessionData = localStorage.getItem(key);
          if (sessionData) {
            try {
              const session = JSON.parse(sessionData);
              const sessionAge = Date.now() - session.timestamp;
              const maxAge = 24 * 60 * 60 * 1000; // 24 hours
              if (sessionAge < maxAge) {
                console.log('[CameraProvider] Found valid session in localStorage, initializing isCheckedIn=true');
                return true;
              }
            } catch { /* ignore parse errors */ }
          }
        }
      }
    } catch { /* ignore parse errors */ }

    return false;
  });
  const [isCheckingIn, setIsCheckingIn] = useState(false);
  const [checkInError, setCheckInError] = useState<string | null>(null);
  const checkInCallbacksRef = useRef(new Set<(isCheckedIn: boolean) => void>());
  
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
        const registryAccount = await (program.account as any).cameraRegistry.fetch(registryAddress);
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
          const cameraAccount = await (program.account as any).cameraAccount.fetch(cameraAddress);
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

  // ============================================
  // UNIFIED CHECK-IN STATE MANAGEMENT
  // (Phase 3 Privacy Architecture - Jetson is source of truth)
  // ============================================

  // Helper to get session storage key
  const getSessionKey = useCallback((walletAddress: string, cameraId: string) => {
    return `${SESSION_STORAGE_PREFIX}${walletAddress}_${cameraId}`;
  }, []);

  // Notify all subscribers of check-in status change
  const notifyCheckInStatusChange = useCallback((newStatus: boolean) => {
    console.log(`[CameraProvider] Notifying ${checkInCallbacksRef.current.size} subscribers of check-in status: ${newStatus}`);
    checkInCallbacksRef.current.forEach(callback => {
      try {
        callback(newStatus);
      } catch (error) {
        console.error('[CameraProvider] Error in check-in status callback:', error);
      }
    });
  }, []);

  // Subscribe to check-in status changes
  const onCheckInStatusChange = useCallback((callback: (isCheckedIn: boolean) => void) => {
    checkInCallbacksRef.current.add(callback);
    return () => {
      checkInCallbacksRef.current.delete(callback);
    };
  }, []);

  // Refresh check-in status from Jetson (source of truth)
  const refreshCheckInStatus = useCallback(async () => {
    const cameraId = selectedCamera?.publicKey;
    const walletAddress = primaryWallet?.address;

    if (!cameraId || !walletAddress) {
      // Don't set isCheckedIn=false here - wallet might still be loading
      // Only clear state when we explicitly know user is not checked in
      console.log('[CameraProvider] Cannot refresh check-in status - missing camera or wallet (not clearing state)');
      return;
    }

    console.log(`[CameraProvider] Refreshing check-in status for ${walletAddress.slice(0, 8)}... at camera ${cameraId.slice(0, 8)}...`);

    try {
      const result = await unifiedCameraService.getSessionStatus(cameraId, walletAddress);

      if (result.success && result.data) {
        const newStatus = result.data.isCheckedIn;
        console.log(`[CameraProvider] Jetson reports isCheckedIn: ${newStatus}`);

        // Update state
        setIsCheckedIn(newStatus);

        // Update localStorage to match Jetson state
        const sessionKey = getSessionKey(walletAddress, cameraId);
        if (newStatus) {
          // Ensure localStorage reflects checked-in state
          const existingSession = localStorage.getItem(sessionKey);
          if (!existingSession) {
            localStorage.setItem(sessionKey, JSON.stringify({
              timestamp: Date.now(),
              cameraId,
              walletAddress
            }));
          }
        } else {
          // Clear localStorage if Jetson says not checked in
          localStorage.removeItem(sessionKey);
        }

        // Notify subscribers
        notifyCheckInStatusChange(newStatus);
      } else {
        console.log('[CameraProvider] Failed to get session status from Jetson, checking localStorage');

        // Fallback to localStorage check
        const sessionKey = getSessionKey(walletAddress, cameraId);
        const storedSession = localStorage.getItem(sessionKey);

        if (storedSession) {
          try {
            const session = JSON.parse(storedSession);
            const sessionAge = Date.now() - session.timestamp;
            const maxAge = 24 * 60 * 60 * 1000; // 24 hours

            if (sessionAge < maxAge) {
              setIsCheckedIn(true);
              notifyCheckInStatusChange(true);
              return;
            } else {
              localStorage.removeItem(sessionKey);
            }
          } catch (e) {
            console.error('[CameraProvider] Failed to parse session:', e);
            localStorage.removeItem(sessionKey);
          }
        }

        setIsCheckedIn(false);
        notifyCheckInStatusChange(false);
      }
    } catch (error) {
      console.error('[CameraProvider] Error refreshing check-in status:', error);
      setIsCheckedIn(false);
      notifyCheckInStatusChange(false);
    }
  }, [selectedCamera?.publicKey, primaryWallet?.address, getSessionKey, notifyCheckInStatusChange]);

  // Check in to the currently selected camera
  // This also ensures user has a session chain (creates one if needed)
  const checkIn = useCallback(async (): Promise<boolean> => {
    const cameraId = selectedCamera?.publicKey;
    const walletAddress = primaryWallet?.address;

    if (!cameraId || !walletAddress || !primaryWallet) {
      console.error('[CameraProvider] Cannot check in - missing camera or wallet');
      setCheckInError('No camera selected or wallet not connected');
      return false;
    }

    console.log(`[CameraProvider] Starting check-in for ${walletAddress.slice(0, 8)}... at camera ${cameraId.slice(0, 8)}...`);
    setIsCheckingIn(true);
    setCheckInError(null);

    try {
      // Step 1: Check if user has a session chain (needed for storing access keys at checkout)
      if (program && connectionFromProgram && isSolanaWallet(primaryWallet)) {
        const userPubkey = new PublicKey(walletAddress);
        const sessionChainStatus = await checkUserHasSessionChain(userPubkey, program);

        if (!sessionChainStatus.exists) {
          console.log('[CameraProvider] User has no session chain, creating one...');

          // Create session chain - this will prompt for wallet signature
          const signature = await createUserSessionChain(
            userPubkey,
            primaryWallet,
            program,
            connectionFromProgram
          );

          if (!signature) {
            // User may have cancelled the signature or it failed
            // Don't block check-in, but log it - backend will queue keys
            console.warn('[CameraProvider] Session chain creation failed or was cancelled. Proceeding with check-in anyway.');
          } else {
            console.log('[CameraProvider] Session chain created successfully');
          }
        } else {
          console.log('[CameraProvider] User already has session chain');
        }
      }

      // Step 2: Create signed request using Ed25519
      const signedParams = await createSignedRequest(primaryWallet);
      if (!signedParams) {
        throw new Error('Failed to sign check-in request. Please try again.');
      }

      // Step 3: Call Jetson check-in endpoint
      const result = await unifiedCameraService.checkin(cameraId, {
        ...signedParams,
        display_name: primaryProfile?.displayName,
        username: primaryProfile?.username
      });

      if (result.success && result.data) {
        console.log('[CameraProvider] Check-in successful:', result.data);

        // Update state
        setIsCheckedIn(true);

        // Store session in localStorage
        const sessionKey = getSessionKey(walletAddress, cameraId);
        localStorage.setItem(sessionKey, JSON.stringify({
          sessionId: result.data.session_id,
          timestamp: Date.now(),
          cameraId,
          walletAddress
        }));

        // Clear timeline for new session
        timelineService.clearForNewSession();

        // Notify subscribers
        notifyCheckInStatusChange(true);

        // Set jetson-camera's currentSession locally (required for takePhoto)
        // We use setSession instead of connect() to avoid a second API call to /api/session/connect
        // which would fail since a session already exists from /api/checkin
        await unifiedCameraService.setSession(cameraId, {
          sessionId: result.data.session_id,
          walletAddress: walletAddress,
          cameraPda: cameraId,
          timestamp: Date.now(),
          isActive: true
        });

        return true;
      } else {
        const errorMsg = result.error || 'Check-in failed';
        console.error('[CameraProvider] Check-in failed:', errorMsg);
        setCheckInError(errorMsg);
        return false;
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Check-in failed';
      console.error('[CameraProvider] Check-in error:', error);
      setCheckInError(errorMsg);
      return false;
    } finally {
      setIsCheckingIn(false);
    }
  }, [selectedCamera?.publicKey, primaryWallet, primaryProfile, getSessionKey, notifyCheckInStatusChange, program, connectionFromProgram]);

  // Check out from the currently selected camera
  const checkOut = useCallback(async (): Promise<boolean> => {
    const cameraId = selectedCamera?.publicKey;
    const walletAddress = primaryWallet?.address;

    if (!cameraId || !walletAddress) {
      console.error('[CameraProvider] Cannot check out - missing camera or wallet');
      setCheckInError('No camera selected or wallet not connected');
      return false;
    }

    console.log(`[CameraProvider] Starting check-out for ${walletAddress.slice(0, 8)}... from camera ${cameraId.slice(0, 8)}...`);
    setIsCheckingIn(true); // Reuse loading state
    setCheckInError(null);

    try {
      // Call Jetson checkout endpoint
      const result = await unifiedCameraService.checkout(cameraId, {
        wallet_address: walletAddress,
        transaction_signature: '' // Empty in Phase 3 architecture
      });

      if (result.success) {
        console.log('[CameraProvider] Check-out successful:', result.data);

        // Update state
        setIsCheckedIn(false);

        // Clear session from localStorage
        const sessionKey = getSessionKey(walletAddress, cameraId);
        localStorage.removeItem(sessionKey);

        // End timeline session
        timelineService.endSession();

        // Notify subscribers
        notifyCheckInStatusChange(false);

        return true;
      } else {
        const errorMsg = result.error || 'Check-out failed';
        console.error('[CameraProvider] Check-out failed:', errorMsg);
        setCheckInError(errorMsg);
        return false;
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Check-out failed';
      console.error('[CameraProvider] Check-out error:', error);
      setCheckInError(errorMsg);
      return false;
    } finally {
      setIsCheckingIn(false);
    }
  }, [selectedCamera?.publicKey, primaryWallet?.address, getSessionKey, notifyCheckInStatusChange]);

  // Auto-refresh check-in status when camera or wallet changes
  useEffect(() => {
    if (selectedCamera?.publicKey && primaryWallet?.address) {
      console.log('[CameraProvider] Camera or wallet changed, refreshing check-in status');
      refreshCheckInStatus();
    }
    // Don't clear check-in state when wallet is loading (e.g., page refresh)
    // State is initialized from localStorage and will be verified with Jetson once wallet loads
    // Only clear state explicitly when:
    // 1. User checks out (checkOut function)
    // 2. Jetson confirms user is not checked in (refreshCheckInStatus)
    // 3. Camera is explicitly cleared/changed to a different camera
  }, [selectedCamera?.publicKey, primaryWallet?.address, refreshCheckInStatus]);

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
        onCameraListRefresh,
        // Unified check-in state (Phase 3 Privacy Architecture)
        isCheckedIn,
        isCheckingIn,
        checkInError,
        checkIn,
        checkOut,
        refreshCheckInStatus,
        onCheckInStatusChange
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