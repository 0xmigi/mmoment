import { useState, useEffect } from 'react';
import { useProgram } from '../anchor/setup';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey } from '@solana/web3.js';

export interface FacialEmbeddingStatus {
  hasEmbedding: boolean;
  isLoading: boolean;
  error: string | null;
  lastChecked: Date | null;
  refetch: () => Promise<void>;
}

export function useFacialEmbeddingStatus(): FacialEmbeddingStatus {
  const [status, setStatus] = useState<Omit<FacialEmbeddingStatus, 'refetch'>>({
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

      // Derive the PDA for recognition token storage (matching PhoneSelfieEnrollment)
      const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("recognition-token"), // Updated seed
          userPublicKey.toBuffer(),
        ],
        program.programId
      );

      console.log('[useFacialEmbeddingStatus] 🔍 Checking for recognition token...');
      console.log('[useFacialEmbeddingStatus] 🔍 Wallet:', primaryWallet.address);
      console.log('[useFacialEmbeddingStatus] 🔍 Program ID:', program.programId.toString());
      console.log('[useFacialEmbeddingStatus] 🔍 PDA:', recognitionTokenPda.toString());

      // Try to fetch the recognition token account
      const faceAccount = await (program.account as any).recognitionToken.fetch(recognitionTokenPda);
      console.log('[useFacialEmbeddingStatus] 🔍 Raw account data:', faceAccount);

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
      console.log('[useFacialEmbeddingStatus] ❌ Error fetching face embedding:', error);
      console.log('[useFacialEmbeddingStatus] ❌ Error details:', error instanceof Error ? error.message : String(error));
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

  return {
    ...status,
    refetch: checkEmbeddingStatus,
  };
}