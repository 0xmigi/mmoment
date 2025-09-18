import { Program, AnchorProvider, Idl } from '@coral-xyz/anchor';
import { PublicKey, Connection } from '@solana/web3.js';
import { useConnection, useAnchorWallet } from '@solana/wallet-adapter-react';
import { useEffect, useState } from 'react';
import { IDL } from '../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';

// Re-export useful constants
export { CAMERA_ACTIVATION_PROGRAM_ID };

// Hook to get an Anchor program
export function useAnchorProgram<IDLType extends Idl = typeof IDL>(
  programId: PublicKey = CAMERA_ACTIVATION_PROGRAM_ID,
  idl: IDLType = IDL as unknown as IDLType
) {
  const { connection } = useConnection();
  const wallet = useAnchorWallet();
  const [program, setProgram] = useState<Program<IDLType> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;
    
    const initProgram = async () => {
      try {
        if (!wallet) {
          if (mounted) {
            setProgram(null);
            setLoading(false);
          }
          return;
        }
        
        setLoading(true);
        const provider = new AnchorProvider(
          connection,
          wallet,
          { commitment: 'confirmed' }
        );
        
        const program = new Program(idl, programId, provider);
        
        if (mounted) {
          setProgram(program);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        console.error('Error initializing Anchor program:', err);
        if (mounted) {
          setError(err instanceof Error ? err : new Error('Unknown error'));
          setLoading(false);
        }
      }
    };

    initProgram();
    
    return () => {
      mounted = false;
    };
  }, [connection, wallet, programId, idl]);

  return { program, loading, error };
}

// Get a read-only program for public access
export function getReadOnlyProgram<IDLType extends Idl = typeof IDL>(
  connection: Connection,
  programId: PublicKey = CAMERA_ACTIVATION_PROGRAM_ID,
  idl: IDLType = IDL as unknown as IDLType
) {
  // Create a provider without a wallet (read-only)
  const provider = new AnchorProvider(
    connection,
    {
      publicKey: PublicKey.default,
      signTransaction: async (tx: any) => tx,
      signAllTransactions: async (txs: any) => txs,
    } as any,
    { commitment: 'confirmed' }
  );
  
  // Create a program instance
  return new Program(idl, programId, provider);
} 