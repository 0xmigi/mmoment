import { useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useProgram } from '../anchor/setup';
import { Keypair, PublicKey, SystemProgram } from '@solana/web3.js';

export function useCameraInit() {
  const { primaryWallet } = useDynamicContext();
  const program = useProgram();
  const [isInitialized, setIsInitialized] = useState(false);
  const [cameraAccount] = useState(() => {
    // Store keypair in localStorage to persist across refreshes
    const stored = localStorage.getItem('cameraKeypair');
    if (stored) {
      const keypairData = new Uint8Array(JSON.parse(stored));
      return Keypair.fromSecretKey(keypairData);
    }
    const newKeypair = Keypair.generate();
    localStorage.setItem('cameraKeypair', JSON.stringify(Array.from(newKeypair.secretKey)));
    return newKeypair;
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkInitialization = async () => {
      if (!primaryWallet?.address || !program) {
        setLoading(false);
        return;
      }

      try {
        // Try to fetch the camera account
        const account = await program.account.cameraAccount.fetch(cameraAccount.publicKey);
        setIsInitialized(!!account);
      } catch {
        // Account doesn't exist, need to initialize
        try {
          await program.methods.initialize()
            .accounts({
              cameraAccount: cameraAccount.publicKey,
              user: new PublicKey(primaryWallet.address),
              systemProgram: SystemProgram.programId,
            })
            .signers([cameraAccount])
            .rpc();
          setIsInitialized(true);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to initialize camera');
        }
      }
      setLoading(false);
    };

    checkInitialization();
  }, [primaryWallet?.address, program]);

  return {
    isInitialized,
    cameraAccount,
    loading,
    error
  };
}