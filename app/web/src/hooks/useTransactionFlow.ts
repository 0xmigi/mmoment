import { useState } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

type CameraActionType = 'photo' | 'video' | 'stream' | 'initialize';

interface TransactionData {
  type: CameraActionType;
  cameraAccount: string;
}

export const useTransactionFlow = () => {
  const { primaryWallet } = useDynamicContext();
  const [showTransactionModal, setShowTransactionModal] = useState(false);
  const [pendingTransaction, setPendingTransaction] = useState<TransactionData | null>(null);

  const initiateCameraAction = (type: CameraActionType, cameraAccount: string) => {
    if (!primaryWallet) {
      throw new Error('No wallet connected');
    }

    setPendingTransaction({ type, cameraAccount });
    setShowTransactionModal(true);
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
  };
}; 