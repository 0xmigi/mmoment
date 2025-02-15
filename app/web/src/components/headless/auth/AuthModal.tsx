import React, { useState } from 'react';
import { useWalletOptions, WalletOption, useConnectWithOtp, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { WalletIcon } from '@dynamic-labs/wallet-book';
import { Dialog } from '@headlessui/react';
import { X } from 'lucide-react';
import { HeadlessSocialLogin } from './HeadlessSocialLogin';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AuthModal({ isOpen, onClose }: AuthModalProps) {
  const { walletOptions, selectWalletOption } = useWalletOptions();
  const { connectWithEmail, verifyOneTimePassword } = useConnectWithOtp();
  const { createEmbeddedWallet, userHasEmbeddedWallet } = useEmbeddedWallet();
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showOtpInput, setShowOtpInput] = useState(false);
  const [otp, setOtp] = useState('');

  if (!isOpen) return null;

  const handleWalletSelect = async (wallet: WalletOption) => {
    try {
      setIsLoading(true);
      setError('');
      await selectWalletOption(wallet.key);
      onClose();
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      setError('Failed to connect wallet. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      setError('');
      await connectWithEmail(email);
      setShowOtpInput(true);
    } catch (error) {
      console.error('Failed to send verification code:', error);
      setError('Failed to send verification code. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      setError('');
      await verifyOneTimePassword(otp);
      
      if (!userHasEmbeddedWallet) {
        await createEmbeddedWallet();
      }

      onClose();
    } catch (error) {
      console.error('Failed to verify code:', error);
      setError('Invalid verification code. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Filter to show only Phantom wallet
  const phantomWallet = walletOptions.find(wallet => 
    wallet.name.toLowerCase() === 'phantom'
  ) as WalletOption | undefined;

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
              Sign In
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Sign In Content */}
          <div className="p-3 space-y-4">
            {error && (
              <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}

            {showOtpInput ? (
              <form onSubmit={handleOtpSubmit} className="space-y-4">
                <div>
                  <input
                    type="text"
                    id="otp"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value)}
                    disabled={isLoading}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    placeholder="Enter verification code"
                    required
                  />
                  <p className="mt-2 text-xs text-gray-500">
                    We sent a code to your email address
                  </p>
                </div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Verifying...' : 'Verify'}
                </button>
              </form>
            ) : (
              <div className="space-y-3">
                {/* Email at the top */}
                <form onSubmit={handleEmailSubmit}>
                  <input
                    type="email"
                    id="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isLoading}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    placeholder="name@example.com"
                    required
                  />
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full mt-2 bg-blue-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? 'Please wait...' : 'Continue with Email'}
                  </button>
                </form>

                <div className="relative my-6">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-200"></div>
                  </div>
                  <div className="relative flex justify-center text-xs">
                    <span className="px-2 bg-white text-gray-500">or continue with</span>
                  </div>
                </div>

                {/* Other sign-in options */}
                {phantomWallet && (
                  <button
                    onClick={() => handleWalletSelect(phantomWallet)}
                    disabled={isLoading}
                    className="w-full flex items-center p-3 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <div className="w-6 h-6 mr-3">
                      <WalletIcon walletKey={phantomWallet.key} />
                    </div>
                    <span className="flex-1 text-left text-sm font-medium">Phantom</span>
                  </button>
                )}

                {/* Farcaster */}
                <HeadlessSocialLogin onSuccess={onClose} />
              </div>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
} 