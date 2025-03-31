import { useState, useEffect, useRef } from 'react';
import { AnchorProvider, Program } from '@coral-xyz/anchor';
import { PublicKey, Transaction } from '@solana/web3.js';
import { useConnection } from '@solana/wallet-adapter-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { IDL, MySolanaProject } from './idl';

// Program ID on devnet
export const CAMERA_ACTIVATION_PROGRAM_ID = new PublicKey('7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4');

// Global cache to prevent multiple initializations
let globalProgramInstance: Program<MySolanaProject> | null = null;
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
  
  const [program, setProgram] = useState<Program<MySolanaProject> | null>(globalProgramInstance);
  const [loading, setLoading] = useState<boolean>(!globalProgramInstance);
  const [error, setError] = useState<Error | null>(null);
  
  // Get wallet address from Dynamic
  const walletAddress = primaryWallet?.address;
  
  // Check if wallet is connected
  const isWalletConnected = !!walletAddress;
  
  const setupAttemptedRef = useRef(false);

  useEffect(() => {
    let mounted = true;
    
    // Log wallet connection status for debugging
    // console.log('Wallet connection status:', {
    //   dynamicWallet: !!primaryWallet?.address,
    //   walletAddress,
    //   isWalletConnected
    // });
    
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

        console.log('Setting up program with wallet address:', walletAddress);
        console.log('Using program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());

        // Get the connection from the wallet
        const walletConnection = await primaryWallet.getConnection();
        console.log('Connection established to:', walletConnection.rpcEndpoint);

        // Get the signer
        const signer = await primaryWallet.getSigner();
        console.log('Signer obtained');

        // Create provider with the wallet's connection and signer
        const provider = new AnchorProvider(
          walletConnection,
          {
            publicKey: new PublicKey(primaryWallet.address),
            signTransaction: async (tx: Transaction) => {
              try {
                // Use the signer to sign the transaction
                const signedTx = await signer.signTransaction(tx);
                return signedTx;
              } catch (err) {
                console.error('Error signing transaction:', err);
                throw err;
              }
            },
            signAllTransactions: async (txs: Transaction[]) => {
              try {
                // Use the signer to sign all transactions
                const signedTxs = await Promise.all(txs.map(tx => signer.signTransaction(tx)));
                return signedTxs;
              } catch (err) {
                console.error('Error signing multiple transactions:', err);
                throw err;
              }
            },
          } as any,
          { commitment: "confirmed" }
        );

        console.log('Created AnchorProvider with connection:', walletConnection.rpcEndpoint);
        
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
 * Alias for useCameraActivationProgram for backward compatibility
 */
export function useProgram() {
  const { connection } = useConnection();
  const { primaryWallet } = useDynamicContext();
  const [program, setProgram] = useState<Program<MySolanaProject> | null>(null);
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