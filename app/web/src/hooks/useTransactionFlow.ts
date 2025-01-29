import { useState } from 'react';
import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';

type CameraActionType = 'photo' | 'video' | 'stream' | 'initialize';

interface TransactionData {
  type: CameraActionType;
  cameraAccount: string;
}

export const useTransactionFlow = () => {
  const { primaryWallet } = useDynamicContext();
  const { userHasEmbeddedWallet } = useEmbeddedWallet();
  const [showTransactionModal, setShowTransactionModal] = useState(false);
  const [pendingTransaction, setPendingTransaction] = useState<TransactionData | null>(null);

  const initiateCameraAction = (type: CameraActionType, cameraAccount: string) => {
    if (!primaryWallet) {
      throw new Error('No wallet connected');
    }

    // For embedded wallets, we need to show our UI first
    if (userHasEmbeddedWallet()) {
      setPendingTransaction({ type, cameraAccount });
      setShowTransactionModal(true);
      return;
    }

    // For EOA wallets like Phantom, we can proceed directly with their UI
    // Just set the transaction data, the parent component should handle the actual transaction
    setPendingTransaction({ type, cameraAccount });
    // Don't show our modal for EOA wallets
    setShowTransactionModal(false);
  };

  const closeTransactionModal = () => {
    setShowTransactionModal(false);
    setPendingTransaction(null);
  };

  return {
    showTransactionModal,
    pendingTransaction,
    initiateCameraAction,
    closeTransactionModal,
    isEmbeddedWallet: userHasEmbeddedWallet(),
  };
}; 