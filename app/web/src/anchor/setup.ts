import { useState, useEffect, useRef } from 'react';
import { AnchorProvider, Program, Idl, setProvider } from '@coral-xyz/anchor';
import { PublicKey, Keypair } from '@solana/web3.js';
import { useConnection, useAnchorWallet } from '@solana/wallet-adapter-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { IDL } from './idl';
import type { CameraNetwork } from './idl';

// Updated program ID to match the one in lib.rs
export const CAMERA_ACTIVATION_PROGRAM_ID = new PublicKey("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL");

// Camera Network Program ID (same as the activation program ID)
export const CAMERA_NETWORK_PROGRAM_ID = new PublicKey("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL");

// Global cache to prevent multiple initializations
let globalProgramInstance: Program<CameraNetwork> | null = null;
let lastWalletAddress: string | null = null;
let setupInProgress = false;
let lastSetupAttempt = 0;
const SETUP_COOLDOWN_MS = 5000; // 5 seconds cooldown between setup attempts

/**
 * Hook to use the Camera Activation program
 */
export function useCameraActivationProgram() {
  const { connection } = useConnection();
  const dynamicContext = useDynamicContext();
  const { primaryWallet } = dynamicContext;
  
  const [program, setProgram] = useState<Program<CameraNetwork> | null>(globalProgramInstance);
  const [loading, setLoading] = useState<boolean>(!globalProgramInstance);
  const [error, setError] = useState<Error | null>(null);
  
  // Get wallet address from Dynamic
  const walletAddress = primaryWallet?.address;
  
  // Check if wallet is connected
  const isWalletConnected = !!walletAddress;
  
  const setupAttemptedRef = useRef(false);

  useEffect(() => {
    let mounted = true;
    
    const setupProgram = async () => {
      // Don't setup if already in progress or if we've recently attempted
      const now = Date.now();
      if (setupInProgress || (now - lastSetupAttempt < SETUP_COOLDOWN_MS)) {
        return;
      }
      
      // Don't setup if we already have a program instance for this wallet
      if (globalProgramInstance && walletAddress === lastWalletAddress) {
        if (mounted) {
          setProgram(globalProgramInstance);
          setLoading(false);
          setError(null);
        }
        return;
      }
      
      // Mark that we're attempting setup
      setupInProgress = true;
      lastSetupAttempt = now;
      setupAttemptedRef.current = true;
      
      try {
        if (!isWalletConnected || !primaryWallet) {
          console.log('No wallet connected, skipping program setup');
          if (mounted) {
            setLoading(false);
          }
          setupInProgress = false;
          return;
        }

        // Check if it's a Solana wallet
        if (!isSolanaWallet(primaryWallet)) {
          throw new Error('This is not a Solana wallet');
        }

        if (!connection) {
          throw new Error('No connection available');
        }

        console.log('Setting up program with wallet address:', walletAddress);
        console.log('Using program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());

        // Use a dummy wallet for Anchor that won't actually sign
        // We'll handle signing separately in the application
        const dummyWallet = {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: any) => tx,
          signAllTransactions: async (txs: any) => txs,
        } as any; // Type assertion to avoid TypeScript errors

        // Create provider with the connection and wallet adapter
        const provider = new AnchorProvider(
          connection, 
          dummyWallet,
          { commitment: "confirmed" }
        );

        console.log('Created AnchorProvider with connection:', connection.rpcEndpoint);
        
        // Create program instance
        const prog = new Program(
          IDL as any,
          CAMERA_ACTIVATION_PROGRAM_ID,
          provider
        );
        console.log('Successfully created program instance');
        
        // Update global cache
        globalProgramInstance = prog;
        lastWalletAddress = walletAddress;
        
        if (mounted) {
          setProgram(prog);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to setup program:', err);
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setLoading(false);
        }
      } finally {
        setupInProgress = false;
      }
    };
    
    // Only attempt setup if we haven't already or if wallet address changed
    if (!setupAttemptedRef.current || walletAddress !== lastWalletAddress) {
      setupProgram();
    }
    
    return () => { mounted = false; };
  }, [connection, walletAddress, primaryWallet, dynamicContext]);

  return { program, loading, error, isWalletConnected };
}

/**
 * Custom hook for setting up the Camera Network Program with a connected wallet
 */
export const useCameraNetworkProgram = () => {
  const { connection } = useConnection();
  const wallet = useAnchorWallet();
  
  const [programId] = useState(CAMERA_NETWORK_PROGRAM_ID);
  const [program, setProgram] = useState<Program<CameraNetwork> | null>(null);
  const [registryAddress, setRegistryAddress] = useState<PublicKey | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const setupProgram = async () => {
      try {
        setLoading(true);
        setError(null);

        if (!wallet) {
          console.log("[useCameraNetworkProgram] No wallet connected");
          setLoading(false);
          return;
        }

        console.log("[useCameraNetworkProgram] Wallet connected:", wallet.publicKey.toString());

        // Create provider
        const provider = new AnchorProvider(
          connection,
          wallet,
          AnchorProvider.defaultOptions()
        );
        setProvider(provider);

        // Initialize program
        const program = new Program(IDL as Idl, programId, provider) as unknown as Program<CameraNetwork>;
        setProgram(program);
        console.log("[useCameraNetworkProgram] Program initialized");

        // Find PDA for the registry
        const [registryPDA] = PublicKey.findProgramAddressSync(
          [Buffer.from("camera-registry")],
          programId
        );
        setRegistryAddress(registryPDA);
        console.log("[useCameraNetworkProgram] Registry PDA:", registryPDA.toString());

        setLoading(false);
      } catch (error) {
        console.error("[useCameraNetworkProgram] Error setting up program:", error);
        setError(error as Error);
        setLoading(false);
      }
    };

    setupProgram();
  }, [connection, wallet, programId]);

  return {
    program,
    programId,
    registryAddress,
    loading,
    error,
    connected: !!wallet,
    walletAddress: wallet?.publicKey || null,
  };
};

/**
 * Initializes a program without requiring a connected wallet adapter
 * Use for read-only interactions or when manually signing transactions
 */
export const useCameraNetworkProgramWithoutWallet = (walletKeyPair?: Keypair) => {
  const { connection } = useConnection();
  const [program, setProgram] = useState<Program<CameraNetwork> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const setupProgram = async () => {
      try {
        setLoading(true);
        setError(null);

        // Create an "empty" wallet adapter - will need manual transaction signing
        const wallet = walletKeyPair
          ? {
              publicKey: walletKeyPair.publicKey,
              signTransaction: async (tx: any) => {
                tx.partialSign(walletKeyPair);
                return tx;
              },
              signAllTransactions: async (txs: any[]) => {
                return txs.map((tx) => {
                  tx.partialSign(walletKeyPair);
                  return tx;
                });
              },
            }
          : {
              publicKey: PublicKey.default,
              signTransaction: async () => {
                throw new Error("Wallet not initialized");
              },
              signAllTransactions: async () => {
                throw new Error("Wallet not initialized");
              },
            };

        // Create provider
        const provider = new AnchorProvider(
          connection,
          wallet as any,
          AnchorProvider.defaultOptions()
        );
        setProvider(provider);

        // Initialize program
        const program = new Program(
          IDL as Idl,
          CAMERA_NETWORK_PROGRAM_ID,
          provider
        ) as unknown as Program<CameraNetwork>;
        setProgram(program);

        setLoading(false);
      } catch (error) {
        console.error("[useProgram] Setup error:", error);
        setError(error as Error);
        setLoading(false);
      }
    };

    setupProgram();
  }, [connection, walletKeyPair]);

  return {
    program,
    loading,
    error,
    connected: !!walletKeyPair,
    walletAddress: walletKeyPair?.publicKey || null,
  };
};

/**
 * Helper to find a camera account PDA
 */
export const findCameraPDA = (cameraName: string, owner: PublicKey) => {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("camera"), owner.toBuffer(), Buffer.from(cameraName)],
    CAMERA_NETWORK_PROGRAM_ID
  );
};

/**
 * Helper to find a user session PDA
 */
export const findSessionPDA = (user: PublicKey, camera: PublicKey) => {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("session"), user.toBuffer(), camera.toBuffer()],
    CAMERA_NETWORK_PROGRAM_ID
  );
};

/**
 * Helper to find a recognition token PDA
 */
export const findRecognitionTokenPDA = (user: PublicKey) => {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("recognition-token"), user.toBuffer()],
    CAMERA_NETWORK_PROGRAM_ID
  );
};

/**
 * @deprecated Use findRecognitionTokenPDA instead
 */
export const findFaceDataPDA = findRecognitionTokenPDA;

/**
 * Alias for useCameraActivationProgram for backward compatibility
 */
export function useProgram() {
  const { connection } = useConnection();
  const { primaryWallet } = useDynamicContext();
  const [program, setProgram] = useState<Program<CameraNetwork> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let isMounted = true;
    console.log('[useProgram] Effect triggered, initializing program');
    console.log('[useProgram] Connection:', !!connection, connection?.rpcEndpoint);
    console.log('[useProgram] Wallet:', !!primaryWallet, primaryWallet?.address);
    
    const setupProgram = async () => {
      try {
        if (!connection) {
          console.error('[useProgram] No connection available');
          if (isMounted) {
            setLoading(false);
            setError(new Error('No connection available'));
          }
          return;
        }
        
        if (!primaryWallet?.address) {
          console.error('[useProgram] No wallet connected');
          if (isMounted) {
            setLoading(false);
            setError(new Error('No wallet connected'));
          }
          return;
        }

        console.log('[useProgram] Setting up program with wallet:', primaryWallet.address);
        console.log('[useProgram] Using program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());
        
        try {
          // Create an empty wallet adapter - we'll sign transactions manually
          const emptyWallet = {
            publicKey: new PublicKey(primaryWallet.address),
            signTransaction: async (tx: any) => tx,
            signAllTransactions: async (txs: any) => txs,
          };

          // Create the provider
          console.log('[useProgram] Creating provider with connection:', connection.rpcEndpoint);
          const provider = new AnchorProvider(
            connection,
            emptyWallet as any,
            { commitment: 'confirmed' }
          );

          // Create the program
          console.log('[useProgram] Creating program instance');
          const programInstance = new Program(
            IDL as any,
            CAMERA_ACTIVATION_PROGRAM_ID,
            provider
          );

          console.log('[useProgram] Program initialized successfully:', programInstance.programId.toString());
          console.log('[useProgram] IDL methods available:', 
            Object.keys(programInstance.methods)
              .filter(key => typeof programInstance.methods[key] === 'function')
          );
          
          if (isMounted) {
            setProgram(programInstance);
            setError(null);
          }
        } catch (err) {
          console.error('[useProgram] Error in initialization:', err);
          if (err instanceof Error) {
            console.error('[useProgram] Error name:', err.name);
            console.error('[useProgram] Error message:', err.message);
            console.error('[useProgram] Error stack:', err.stack);
          }
          
          if (isMounted) {
            setError(err instanceof Error ? err : new Error('Unknown error setting up program'));
          }
          throw err; // Rethrow to see full error in console
        }
      } catch (err) {
        console.error('[useProgram] Error setting up program:', err);
        if (isMounted) {
          setError(err instanceof Error ? err : new Error('Unknown error setting up program'));
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    setupProgram();
    
    return () => {
      isMounted = false;
    };
  }, [connection, primaryWallet?.address]);

  return { program, loading, error };
} 