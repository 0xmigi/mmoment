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
  useConnection();
  const program = useProgram();
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fee, setFee] = useState<number>(100); // Default fee in lamports

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
      
      // Simulate a brief delay to show the signing process
      await new Promise(resolve => setTimeout(resolve, 1000));
      setStatus('Signing transaction...');
      
      // Send the transaction
      await program.methods.activateCamera(new BN(fee))
        .accounts({
          cameraAccount: cameraPublicKey,
          user: new PublicKey(primaryWallet.address),
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      // Add a small delay to show confirmation
      await new Promise(resolve => setTimeout(resolve, 500));
      setStatus('Transaction confirmed');
      
      // Add another small delay before closing
      await new Promise(resolve => setTimeout(resolve, 500));
      onSuccess?.();
      onClose();
    } catch (err) {
      console.error('Transaction error:', err);
      setError(err instanceof Error ? err.message : 'Transaction failed');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] overflow-y-auto bg-black bg-opacity-50 backdrop-blur-sm">
      <div className="flex min-h-screen items-center justify-center px-4">
        <div className="relative bg-white rounded-xl shadow-2xl max-w-md w-full p-6 transform transition-all duration-200 ease-out scale-100 animate-in fade-in slide-in-from-bottom-4">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-xl font-semibold">
                Confirm {transactionData?.type || ''} Action
              </h2>
              <p className="text-sm text-gray-500 mt-1">
                Review and approve the transaction
              </p>
            </div>
            {!loading && (
              <button 
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500 transition-colors"
              >
                <span className="sr-only">Close</span>
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          <div className="space-y-6">
            {/* Transaction Details */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Action</span>
                <span className="font-medium capitalize">{transactionData?.type || 'Unknown'}</span>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Network</span>
                <span className="font-medium">Solana Devnet</span>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">Account</span>
                <span className="font-medium font-mono text-xs">
                  {primaryWallet?.address ? `${primaryWallet.address.slice(0, 4)}...${primaryWallet.address.slice(-4)}` : 'Not connected'}
                </span>
              </div>
            </div>

            {/* Fee Adjustment */}
            <div className="space-y-2">
              <label htmlFor="fee" className="block text-sm font-medium text-gray-700">
                Transaction Fee (lamports)
              </label>
              <input
                type="number"
                id="fee"
                value={fee}
                onChange={(e) => setFee(Number(e.target.value))}
                min="0"
                step="1"
                disabled={loading}
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>

            {status && (
              <div className="bg-blue-50 text-blue-700 p-3 rounded-lg text-sm flex items-center">
                <svg className="animate-spin -ml-1 mr-3 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {status}
              </div>
            )}

            {error && (
              <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button
                onClick={handleConfirmTransaction}
                disabled={loading}
                className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {loading ? 'Processing...' : 'Confirm Transaction'}
              </button>
              {!loading && (
                <button
                  onClick={onClose}
                  className="flex-1 bg-gray-100 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-200 transition-colors font-medium"
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