import { Dialog } from '@headlessui/react';
import { X } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../../../anchor/setup';
import { SystemProgram, PublicKey, ComputeBudgetProgram, Transaction } from '@solana/web3.js';
import { useState } from 'react';
import { isSolanaWallet } from '@dynamic-labs/solana';

interface TransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactionData?: {
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  };
  onSuccess?: (data: { transactionId: string; cameraId: string }) => void;
}

export const TransactionModal: React.FC<TransactionModalProps> = ({
  isOpen,
  onClose,
  transactionData,
  onSuccess
}) => {
  const { primaryWallet } = useDynamicContext();
  useConnection();
  const { program } = useProgram();
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fee] = useState<number>(100); // Default fee in lamports

  const handleConfirmTransaction = async () => {
    if (!primaryWallet?.address || !program || !transactionData) {
      setError('Wallet not connected or missing data');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const cameraPublicKey = new PublicKey(transactionData.cameraAccount);
      setStatus('Preparing transaction...');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      // Create activity type based on transaction type
      let activityType;
      switch (transactionData.type) {
        case 'photo':
          activityType = { photoCapture: {} };
          break;
        case 'video':
          activityType = { videoRecord: {} };
          break;
        case 'stream':
          activityType = { liveStream: {} };
          break;
        default:
          activityType = { custom: {} };
      }

      // Create metadata with timestamp and other relevant info
      const metadata = JSON.stringify({
        timestamp: new Date().toISOString(),
        action: `${transactionData.type}_capture`,
        userAddress: primaryWallet.address,
        cameraId: transactionData.cameraAccount
      });

      // Create the recordActivity instruction
      const ix = await program.methods
        .recordActivity({
          activityType,
          metadata
        })
        .accounts({
          owner: new PublicKey(primaryWallet.address),
          camera: cameraPublicKey,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
      
      // Add priority fee
      const addPriorityFee = ComputeBudgetProgram.setComputeUnitPrice({
        microLamports: 375000, // 0.375 lamports per compute unit
      });
      
      // Set compute unit limit
      const modifyComputeUnits = ComputeBudgetProgram.setComputeUnitLimit({
        units: 200000,
      });
      
      // Create new transaction and add instructions
      const transaction = new Transaction();
      transaction.add(addPriorityFee, modifyComputeUnits, ix);
      
      setStatus('Signing transaction...');
      
      // Get the signer
      const signer = await primaryWallet.getSigner();
      
      // Set recent blockhash and fee payer
      const walletConnection = await primaryWallet.getConnection();
      transaction.recentBlockhash = (await walletConnection.getLatestBlockhash()).blockhash;
      transaction.feePayer = new PublicKey(primaryWallet.address);
      
      // Sign and send the transaction
      const result = await signer.signAndSendTransaction(transaction);
      const signature = result.signature;
      
      setStatus('Transaction confirmed');
      
      // Pass back the transaction signature and camera ID
      onSuccess?.({
        transactionId: signature,
        cameraId: transactionData.cameraAccount
      });
      onClose();
    } catch (err) {
      console.error('Transaction error:', err);
      setError(err instanceof Error ? err.message : 'Transaction failed');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const formatActionType = (type: string) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  };

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Confirm {transactionData?.type || ''} Action
            </Dialog.Title>
            {!loading && (
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            )}
          </div>

          {/* Transaction Content */}
          <div className="p-3 space-y-4">
            {/* Transaction Details */}
            <div className="space-y-2">
              <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                <div>
                  <div className="text-xs font-medium text-gray-500">Action</div>
                  <div className="text-sm text-gray-900">
                    {formatActionType(transactionData?.type || '')}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                <div>
                  <div className="text-xs font-medium text-gray-500">Network</div>
                  <div className="text-sm text-gray-900">Solana Devnet</div>
                </div>
              </div>

              <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                <div>
                  <div className="text-xs font-medium text-gray-500">Fee</div>
                  <div className="text-sm text-gray-900">{fee} lamports</div>
                </div>
              </div>
            </div>

            {/* Status Messages */}
            {status && (
              <div className="bg-blue-50 text-blue-700 px-2 py-1.5 rounded-lg text-xs flex items-center">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current mr-2" />
                {status}
              </div>
            )}

            {error && (
              <div className="bg-red-50 text-red-700 px-2 py-1.5 rounded-lg text-xs">
                {error}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleConfirmTransaction}
                disabled={loading}
                className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Confirm'}
              </button>
              {!loading && (
                <button
                  onClick={onClose}
                  className="flex-1 bg-gray-100 text-gray-700 py-2 px-3 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}; 