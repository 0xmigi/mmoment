import React, { createContext, useContext, useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useProgram } from '../anchor/setup';
import { Keypair, PublicKey, SystemProgram } from '@solana/web3.js';

export type QuickActionType = 'photo' | 'video' | 'custom';

interface QuickActions {
  photo: boolean;
  video: boolean;
  custom: boolean;
}

interface CameraContextType {
  cameraKeypair: Keypair;
  isInitialized: boolean;
  loading: boolean;
  error: string | null;
  quickActions: QuickActions;
  updateQuickAction: (type: QuickActionType, enabled: boolean) => void;
  hasQuickAction: (type: QuickActionType) => boolean;
}

const CameraContext = createContext<CameraContextType | undefined>(undefined);

const QUICK_ACTIONS_STORAGE_KEY = 'camera_quick_actions';

export function CameraProvider({ children }: { children: React.ReactNode }) {
  const { primaryWallet } = useDynamicContext();
  const program = useProgram();
  const [isInitialized, setIsInitialized] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
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
      custom: false
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

  // Check initialization when wallet connects
  useEffect(() => {
    const checkInitialization = async () => {
      if (!primaryWallet?.address || !program) {
        setLoading(false);
        return;
      }

      try {
        const account = await program.account.cameraAccount.fetch(cameraKeypair.publicKey);
        setIsInitialized(!!account);
      } catch {
        // Account doesn't exist, need to initialize
        try {
          await program.methods.initialize()
            .accounts({
              cameraAccount: cameraKeypair.publicKey,
              user: new PublicKey(primaryWallet.address),
              systemProgram: SystemProgram.programId,
            })
            .signers([cameraKeypair])
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

  return (
    <CameraContext.Provider 
      value={{
        cameraKeypair,
        isInitialized,
        loading,
        error,
        quickActions,
        updateQuickAction,
        hasQuickAction
      }}
    >
      {children}
    </CameraContext.Provider>
  );
}

export function useCamera() {
  const context = useContext(CameraContext);
  if (context === undefined) {
    throw new Error('useCamera must be used within a CameraProvider');
  }
  return context;
}