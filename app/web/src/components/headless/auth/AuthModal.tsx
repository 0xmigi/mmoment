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

  const modalContent = (
    <div className="space-y-6">
      {error && (
        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {showOtpInput ? (
        <form onSubmit={handleOtpSubmit} className="space-y-4">
          <div>
            <label htmlFor="otp" className="block text-sm font-medium text-gray-700">
              Enter verification code
            </label>
            <input
              type="text"
              id="otp"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              disabled={isLoading}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              placeholder="Enter code"
              required
            />
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Verifying...' : 'Verify'}
          </button>
        </form>
      ) : (
        <>
          <div>
            <h3 className="text-sm font-medium mb-2">Continue with a wallet</h3>
            <div className="space-y-2">
              {phantomWallet && (
                <button
                  onClick={() => handleWalletSelect(phantomWallet)}
                  disabled={isLoading}
                  className="w-full flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors border border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="w-8 h-8 mr-3">
                    <WalletIcon walletKey={phantomWallet.key} />
                  </div>
                  <span className="flex-1 text-left font-medium">{phantomWallet.name}</span>
                  {phantomWallet.isInstalledOnBrowser && (
                    <span className="text-sm text-green-600">Installed</span>
                  )}
                </button>
              )}
            </div>
          </div>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">OR</span>
            </div>
          </div>

          <HeadlessSocialLogin onSuccess={onClose} />

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">
                or continue with email
              </span>
            </div>
          </div>

          <form onSubmit={handleEmailSubmit}>
            <div className="space-y-2">
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Enter your email
              </label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                placeholder="name@example.com"
                required
              />
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Please wait...' : 'Continue with Email'}
              </button>
            </div>
          </form>
        </>
      )}
    </div>
  );

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="mx-auto max-w-sm rounded-xl bg-white p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <Dialog.Title className="text-lg font-medium">
              Sign In
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          <div className="space-y-4">
            {modalContent}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
} 