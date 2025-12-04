import React, { useState, useEffect } from 'react';
import { getPinataCredentials, updatePinataCredentials } from './config';

/**
 * Component for managing Pinata credentials in development mode
 */
export function PinataSettings() {
  const [isOpen, setIsOpen] = useState(false);
  const [jwt, setJwt] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [isSaved, setIsSaved] = useState(false);
  const [usePipeStorage, setUsePipeStorage] = useState(false);

  // Load credentials on mount
  useEffect(() => {
    const creds = getPinataCredentials();
    setJwt(creds.PINATA_JWT || '');
    setApiKey(creds.PINATA_API_KEY || '');
    setApiSecret(creds.PINATA_API_SECRET || '');

    // Load storage preference
    const storageType = localStorage.getItem('mmoment_storage_type');
    setUsePipeStorage(storageType === 'pipe');
  }, []);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updatePinataCredentials({
      PINATA_JWT: jwt,
      PINATA_API_KEY: apiKey,
      PINATA_API_SECRET: apiSecret
    });

    // Save storage preference
    localStorage.setItem('mmoment_storage_type', usePipeStorage ? 'pipe' : 'pinata');

    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 3000);
  };

  const handleStorageToggle = () => {
    const newValue = !usePipeStorage;
    setUsePipeStorage(newValue);
    localStorage.setItem('mmoment_storage_type', newValue ? 'pipe' : 'pinata');
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 3000);
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <button
          onClick={() => setIsOpen(true)}
          className="bg-primary hover:bg-primary text-white rounded-full p-2 shadow-lg"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z"></path>
            <path d="M12 14v2"></path>
            <path d="M12 8v2"></path>
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Storage Settings</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-500 hover:text-gray-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18"></path>
              <path d="m6 6 12 12"></path>
            </svg>
          </button>
        </div>
        
        <div className="mb-4 p-3 bg-yellow-50 text-yellow-800 rounded border border-yellow-200">
          <p className="text-sm">
            These credentials are stored in your browser's localStorage for development purposes.
            They are not committed to your source code.
          </p>
        </div>

        <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Storage Provider</h3>
              <p className="text-sm text-gray-600 mt-1">
                {usePipeStorage ? 'Using Pipe Network (User-owned storage)' : 'Using Pinata IPFS'}
              </p>
            </div>
            <button
              type="button"
              onClick={handleStorageToggle}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                usePipeStorage ? 'bg-green-500' : 'bg-gray-200'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  usePipeStorage ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          {usePipeStorage && (
            <div className="mt-3 text-sm text-green-700 bg-green-50 p-2 rounded">
              Gallery will fetch content from user's Pipe storage account
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className={usePipeStorage ? 'opacity-50 pointer-events-none' : ''}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pinata JWT Token
            </label>
            <input
              type="text"
              value={jwt}
              onChange={(e) => setJwt(e.target.value)}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-primary focus:border-primary"
              placeholder="eyJhbGciOiJIUzI1NiIsInR5..."
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pinata API Key
            </label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-primary focus:border-primary"
              placeholder="ba5bee0fc39a35e442b6..."
            />
          </div>
          
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Pinata API Secret
            </label>
            <input
              type="text"
              value={apiSecret}
              onChange={(e) => setApiSecret(e.target.value)}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-primary focus:border-primary"
              placeholder="your_api_secret_here..."
            />
          </div>

          <div className="flex items-center justify-between">
            <button
              type="submit"
              className="bg-primary hover:bg-primary text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-50"
            >
              Save Credentials
            </button>
            
            {isSaved && (
              <span className="text-green-600 text-sm ml-2">
                ✓ Settings saved
              </span>
            )}
          </div>
        </form>
      </div>
    </div>
  );
} 