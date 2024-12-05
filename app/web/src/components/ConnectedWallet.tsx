import React, { useState } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import { ConnectedSolanaWallet } from '@privy-io/react-auth';
import { PublicKey, Transaction, Connection, SystemProgram } from '@solana/web3.js';

interface ConnectedWalletProps {
  wallet: ConnectedSolanaWallet;
}

export const ConnectedWallet: React.FC<ConnectedWalletProps> = ({ wallet }) => {
  const [showSendTransaction, setShowSendTransaction] = useState(false);
  usePrivy();

  const handleSendTransaction = async () => {
    try {
      const connection = new Connection('https://api.devnet.solana.com');
      const { blockhash } = await connection.getLatestBlockhash();

      const transaction = new Transaction({
        recentBlockhash: blockhash,
        feePayer: new PublicKey(wallet.address),
      }).add(
        SystemProgram.transfer({
          fromPubkey: new PublicKey(wallet.address),
          toPubkey: new PublicKey(wallet.address), // Change this to recipient
          lamports: 100000, // 0.0001 SOL
        }),
      );

      const txHash = await wallet.sendTransaction!(transaction, connection);
      console.log('Transaction sent:', txHash);
    } catch (error) {
      console.error('Transaction failed:', error);
    }
  };

  return (
    <div className="flex items-center gap-4">
      <div className="text-sm text-gray-600">
        {wallet.address.slice(0, 4)}...{wallet.address.slice(-4)}
      </div>
      {showSendTransaction && (
        <div className="absolute top-16 right-4 mt-2 bg-white p-4 rounded-lg shadow-lg border">
          <h3 className="text-sm font-medium mb-2">Send Transaction</h3>
          <button
            onClick={handleSendTransaction}
            className="text-sm px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700"
          >
            Confirm
          </button>
          <button
            onClick={() => setShowSendTransaction(false)}
            className="text-sm px-3 py-1 ml-2 bg-gray-200 rounded hover:bg-gray-300"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};