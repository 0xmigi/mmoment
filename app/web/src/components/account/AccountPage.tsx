import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';

interface FarcasterCredential {
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

export function AccountPage() {
  const { primaryWallet, handleLogOut, user } = useDynamicContext();
  const { revealWalletKey } = useEmbeddedWallet();
  const navigate = useNavigate();
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [showBackupOptions, setShowBackupOptions] = useState(false);

  const handleSignOut = async () => {
    try {
      await handleLogOut();
      navigate('/app');
    } catch (err) {
      console.error('Failed to sign out:', err);
    }
  };

  const handleExportWallet = async (type: 'recoveryPhrase' | 'privateKey') => {
    setIsExporting(true);
    setExportError(null);
    try {
      await revealWalletKey({
        htmlContainerId: 'wallet-export-container',
        type,
      });
    } catch (error) {
      console.error('Failed to export wallet:', error);
      setExportError('Failed to export wallet. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  if (!primaryWallet?.address) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto mb-4" />
          <p className="text-gray-600">Loading account...</p>
        </div>
      </div>
    );
  }

  const isEmbeddedWallet = primaryWallet.connector?.name.toLowerCase() !== 'phantom';
  const farcasterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is FarcasterCredential => 
      cred?.oauthProvider?.toLowerCase() === 'farcaster'
  );

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Account Settings</h1>
        </div>

        {/* Profile Section - Show if connected with Farcaster */}
        {farcasterCred && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Profile</h2>
            <div className="flex items-center gap-4">
              {farcasterCred.oauthAccountPhotos?.[0] && (
                <img
                  src={farcasterCred.oauthAccountPhotos[0]}
                  alt="Profile"
                  className="w-16 h-16 rounded-full"
                />
              )}
              <div>
                <p className="font-medium text-lg">{farcasterCred.oauthDisplayName}</p>
                <p className="text-sm text-gray-500">{farcasterCred.oauthUsername}</p>
              </div>
            </div>
          </div>
        )}

        {/* Connected Wallet Section */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Connected Wallet</h2>
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium text-gray-700">Wallet Address</p>
              <p className="text-[10px] text-gray-500 font-mono">{primaryWallet.address}</p>
            </div>

            {/* Wallet Export Section - Only for embedded wallets */}
            {isEmbeddedWallet && (
              <div className="pt-4">
                {!showBackupOptions ? (
                  <button
                    onClick={() => setShowBackupOptions(true)}
                    className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    Back up Wallet
                  </button>
                ) : (
                  <div className="space-y-3">
                    <button
                      onClick={() => handleExportWallet('privateKey')}
                      disabled={isExporting}
                      className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
                    >
                      {isExporting ? 'Exporting...' : 'Export Private Key'}
                    </button>
                    <button
                      onClick={() => handleExportWallet('recoveryPhrase')}
                      disabled={isExporting}
                      className="w-full px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
                    >
                      {isExporting ? 'Exporting...' : 'Export Recovery Phrase'}
                    </button>
                    <button
                      onClick={() => setShowBackupOptions(false)}
                      className="w-full px-4 py-2.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {exportError && (
                  <p className="text-sm text-red-600 mt-3">{exportError}</p>
                )}

                <div 
                  id="wallet-export-container" 
                  className="mt-4 p-4 bg-gray-50 rounded-lg font-mono text-sm break-all"
                />
              </div>
            )}
          </div>
        </div>

        {/* Authentication Info */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Authentication Method</h2>
          <div className="space-y-2">
            <p className="text-sm text-gray-700">
              Connected with: <span className="font-medium">{primaryWallet.connector?.name}</span>
            </p>
          </div>
        </div>

        {/* Sign Out Button */}
        <div className="bg-white rounded-lg shadow p-6">
          <button
            onClick={handleSignOut}
            className="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
          >
            Sign Out
          </button>
        </div>
      </div>
    </div>
  );
} 
