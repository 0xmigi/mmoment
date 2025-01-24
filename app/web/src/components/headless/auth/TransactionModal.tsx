import React, { useState } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram } from '../../../anchor/setup';
import { SystemProgram, PublicKey } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';

interface TransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactionData?: {
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  };
  onSuccess?: () => void;
}

export const TransactionModal: React.FC<TransactionModalProps> = ({
  isOpen,
  onClose,
  transactionData,
  onSuccess
}) => {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const program = useProgram();
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfirmTransaction = async () => {
    if (!primaryWallet?.address || !program || !transactionData) {
      setError('Wallet not connected or missing data');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const cameraPublicKey = new PublicKey(transactionData.cameraAccount);
      setStatus('Activating camera...');
      
      // Send the transaction using program's rpc method
      const signature = await program.methods.activateCamera(new BN(100))
        .accounts({
          cameraAccount: cameraPublicKey,
          user: new PublicKey(primaryWallet.address),
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      // Wait for confirmation
      await connection.confirmTransaction(signature);

      setStatus('Transaction confirmed');
      onSuccess?.();
      onClose();
    } catch (err) {
      console.error('Transaction error:', err);
      setError(err instanceof Error ? err.message : 'Transaction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`fixed inset-0 z-[9999] overflow-y-auto ${isOpen ? 'block' : 'hidden'}`}>
      <div className="flex min-h-screen items-center justify-center px-4">
        {/* Backdrop */}
        <div 
          className="fixed inset-0 bg-black bg-opacity-30 transition-opacity"
          onClick={!loading ? onClose : undefined}
        />

        {/* Modal */}
        <div className="relative bg-white rounded-lg shadow-xl max-w-md w-full p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">
              Confirm Transaction
            </h2>
            {!loading && (
              <button 
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500"
              >
                <span className="sr-only">Close</span>
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Network</span>
              <span className="font-medium">Solana Mainnet</span>
            </div>

            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Network Fee</span>
              <span className="font-medium">~0.00001 SOL</span>
            </div>

            {status && (
              <div className="text-sm text-blue-600">
                {status}
              </div>
            )}

            {error && (
              <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleConfirmTransaction}
                disabled={loading}
                className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Confirm'}
              </button>
              {!loading && (
                <button
                  onClick={onClose}
                  className="flex-1 bg-gray-100 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 