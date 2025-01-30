import React, { useState } from 'react';
import { useWalletOptions, WalletOption, useDynamicContext, useConnectWithOtp, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { WalletIcon } from '@dynamic-labs/wallet-book';

interface WalletModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const WalletModal = ({ isOpen, onClose }: WalletModalProps) => {
  const { walletOptions, selectWalletOption } = useWalletOptions();
  const { connectWithEmail, verifyOneTimePassword } = useConnectWithOtp();
  useDynamicContext();
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
      
      // Create embedded wallet if user doesn't have one
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
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center px-4">
        {/* Backdrop */}
        <div 
          className="fixed inset-0 bg-black bg-opacity-30 transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative bg-white rounded-lg shadow-xl max-w-md w-full p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">
              {showOtpInput ? 'Verify your email' : 'Log in or sign up'}
            </h2>
            <button 
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500"
            >
              <span className="sr-only">Close</span>
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
              {error}
            </div>
          )}

          <div className="space-y-4">
            {showOtpInput ? (
              <form onSubmit={handleOtpSubmit}>
                <div className="space-y-2">
                  <label htmlFor="otp" className="block text-sm font-medium text-gray-700">
                    Enter verification code
                  </label>
                  <input
                    type="text"
                    id="otp"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value)}
                    disabled={isLoading}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    placeholder="Enter code"
                    required
                  />
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isLoading ? 'Verifying...' : 'Verify'}
                  </button>
                </div>
              </form>
            ) : (
              <>
                <div>
                  <h3 className="text-sm font-medium mb-2">Continue with a wallet</h3>
                  <div className="space-y-2">
                    {phantomWallet && (
                      <button
                        key={phantomWallet.key}
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
                      {isLoading ? 'Please wait...' : 'Continue'}
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}; 