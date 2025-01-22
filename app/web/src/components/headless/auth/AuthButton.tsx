import { useState } from 'react';
import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { WalletModal } from './WalletModal';

export const HeadlessAuthButton = () => {
  const { primaryWallet, handleLogOut } = useDynamicContext();
  useEmbeddedWallet();
  const [showDropdown, setShowDropdown] = useState(false);
  const [showWalletModal, setShowWalletModal] = useState(false);

  if (!primaryWallet?.address) {
    return (
      <>
        <button
          onClick={() => setShowWalletModal(true)}
          className="px-6 py-2 bg-[#e7eeff] text-black rounded-lg hover:bg-[#a5bafc] transition-colors"
        >
          Connect Wallet
        </button>
        <WalletModal 
          isOpen={showWalletModal} 
          onClose={() => setShowWalletModal(false)} 
        />
      </>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="px-4 py-2 bg-[#e7eeff] text-black rounded-lg hover:bg-[#a5bafc] transition-colors flex items-center gap-2"
      >
        <span>{primaryWallet.address.slice(0, 4)}...{primaryWallet.address.slice(-4)}</span>
        <svg
          className={`w-4 h-4 transition-transform ${showDropdown ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {showDropdown && (
        <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-100 py-1">
          <button
            onClick={() => {
              handleLogOut();
              setShowDropdown(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50"
          >
            Disconnect
          </button>
        </div>
      )}
    </div>
  );
}; 