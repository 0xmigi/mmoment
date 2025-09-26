import { useState, useEffect } from 'react';
import { useProgram } from '../anchor/setup';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey } from '@solana/web3.js';

export interface FacialEmbeddingStatus {
  hasEmbedding: boolean;
  isLoading: boolean;
  error: string | null;
  lastChecked: Date | null;
}

export function useFacialEmbeddingStatus(): FacialEmbeddingStatus {
  const [status, setStatus] = useState<FacialEmbeddingStatus>({
    hasEmbedding: false,
    isLoading: true,
    error: null,
    lastChecked: null,
  });

  const { program } = useProgram();
  const { primaryWallet } = useDynamicContext();

  const checkEmbeddingStatus = async () => {
    if (!primaryWallet?.address || !program) {
      setStatus(prev => ({
        ...prev,
        isLoading: false,
        error: 'Wallet or program not available',
        lastChecked: new Date(),
      }));
      return;
    }

    try {
      setStatus(prev => ({ ...prev, isLoading: true, error: null }));

      const userPublicKey = new PublicKey(primaryWallet.address);

      // Derive the PDA for face data storage (matching PhoneSelfieEnrollment)
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("face-nft"), // Must match the seed in enrollment
          userPublicKey.toBuffer(),
        ],
        program.programId
      );

      console.log('[useFacialEmbeddingStatus] Checking for face NFT at PDA:', faceDataPda.toString());

      // Try to fetch the face data account
      const faceAccount = await program.account.faceData.fetch(faceDataPda);

      if (faceAccount) {
        console.log('[useFacialEmbeddingStatus] ✅ Face embedding found:', faceAccount);
        setStatus({
          hasEmbedding: true,
          isLoading: false,
          error: null,
          lastChecked: new Date(),
        });
      } else {
        console.log('[useFacialEmbeddingStatus] ❌ No face embedding found');
        setStatus({
          hasEmbedding: false,
          isLoading: false,
          error: null,
          lastChecked: new Date(),
        });
      }
    } catch (error) {
      console.log('[useFacialEmbeddingStatus] ❌ No face embedding account found (expected if not enrolled)');
      // If account doesn't exist, it means user hasn't enrolled yet
      setStatus({
        hasEmbedding: false,
        isLoading: false,
        error: null,
        lastChecked: new Date(),
      });
    }
  };

  useEffect(() => {
    checkEmbeddingStatus();
  }, [primaryWallet?.address, program]);

  // Re-check periodically (every 30 seconds)
  useEffect(() => {
    const interval = setInterval(() => {
      if (!status.isLoading) {
        checkEmbeddingStatus();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [status.isLoading]);

  return status;
}