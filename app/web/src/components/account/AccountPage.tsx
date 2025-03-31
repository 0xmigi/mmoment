import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { User, Copy, Check } from 'lucide-react';
import { CameraRegistry } from '../CameraRegistry';

interface FarcasterCredential {
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

interface StatusMessage {
  type: 'success' | 'error' | 'info';
  message: string;
}

export function AccountPage() {
  const { primaryWallet, handleLogOut, user } = useDynamicContext();
  const { revealWalletKey } = useEmbeddedWallet();
  const navigate = useNavigate();
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [showBackupOptions, setShowBackupOptions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);

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

  const handleCopyAddress = async () => {
    if (!primaryWallet?.address) return;
    
    try {
      await navigator.clipboard.writeText(primaryWallet.address);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy address:', err);
    }
  };

  const handleStatusUpdate = (status: StatusMessage) => {
    setStatusMessage(status);
    // Auto-clear status messages after 5 seconds
    setTimeout(() => setStatusMessage(null), 5000);
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
    <div className="min-h-screen bg-white">
      <div className="max-w-3xl mx-auto pt-8 px-4">
        <div className="bg-white rounded-lg mb-6">
          <h1 className="text-xl font-semibold">Account</h1>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div className={`mb-4 p-3 rounded-lg ${
            statusMessage.type === 'success' ? 'bg-green-50 text-green-700' : 
            statusMessage.type === 'error' ? 'bg-red-50 text-red-700' : 
            'bg-blue-50 text-blue-700'
          }`}>
            <p>{statusMessage.message}</p>
          </div>
        )}

        {/* Account Details */}
        <div className="space-y-3">
          {/* Core Identity Section */}
          <div className="bg-gray-50 rounded-xl px-4 py-4">
            <div className="text-sm">
              <div className="flex items-start justify-between">
                <div>
                  <div className="font-medium mb-1">Primary Identity</div>
                  <div className="text-gray-500 text-xs mb-3">
                    {farcasterCred ? 'pulled from Farcaster' : 'pulled from Wallet (default)'}
                  </div>
                </div>
                {farcasterCred?.oauthAccountPhotos?.[0] ? (
                  <img
                    src={farcasterCred.oauthAccountPhotos[0]}
                    alt="Profile"
                    className="w-12 h-12 rounded-full"
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center">
                    <User className="w-6 h-6 text-gray-400" />
                  </div>
                )}
              </div>
              <div className="mt-3">
                <div className="font-medium">Display Name</div>
                <div className="text-gray-500">
                  {farcasterCred?.oauthDisplayName || `${primaryWallet.address.slice(0, 6)}...${primaryWallet.address.slice(-4)}`}
                </div>
              </div>
            </div>
          </div>

          {/* Camera Registry Section */}
          <CameraRegistry />

          {/* Linked Identities Section */}
          <div className="bg-gray-50 rounded-xl px-4 py-4">
            <div className="text-sm">
              <div className="font-medium mb-3">Linked Identities</div>
              
              {/* Farcaster Identity */}
              <div className="flex items-center justify-between py-2">
                <div>
                  <div className="font-medium text-gray-700">Farcaster</div>
                  <div className="text-gray-500">
                    {farcasterCred?.oauthUsername ? farcasterCred.oauthUsername : 'Not connected'}
                  </div>
                </div>
                {!farcasterCred && (
                  <button className="text-blue-600 text-sm hover:text-blue-700 transition-colors">
                    Connect
                  </button>
                )}
              </div>

              {/* Email Identity */}
              <div className="flex items-center justify-between py-2 border-t border-gray-100">
                <div>
                  <div className="font-medium text-gray-700">Email</div>
                  <div className="text-gray-500">
                    {user?.email || 'Not connected'}
                  </div>
                </div>
                {!user?.email && (
                  <button className="text-blue-600 text-sm hover:text-blue-700 transition-colors">
                    Connect
                  </button>
                )}
              </div>

              {/* Wallet Identity */}
              <div className="flex items-center justify-between py-2 border-t border-gray-100">
                <div className="w-full">
                  <div className="flex items-center justify-between mb-0.5">
                    <div className="font-medium text-gray-700">Solana Wallet</div>
                    <button
                      onClick={handleCopyAddress}
                      className="p-1 hover:bg-gray-200 rounded-md transition-colors"
                      title="Copy address"
                    >
                      {copied ? (
                        <Check className="w-3 h-3 text-green-600" />
                      ) : (
                        <Copy className="w-3 h-3 text-gray-400" />
                      )}
                    </button>
                  </div>
                  <div className="text-gray-500 font-mono text-xs">
                    {primaryWallet.address}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Wallet Backup Section */}
          {isEmbeddedWallet && (
            <div className="bg-gray-50 rounded-xl px-4 py-4">
              <div className="text-sm">
                <div className="font-medium mb-2">Wallet Backup</div>
                {!showBackupOptions ? (
                  <button
                    onClick={() => setShowBackupOptions(true)}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
                  >
                    Back up Wallet
                  </button>
                ) : (
                  <div className="space-y-2">
                    <button
                      onClick={() => handleExportWallet('privateKey')}
                      disabled={isExporting}
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isExporting ? 'Exporting...' : 'Export Private Key'}
                    </button>
                    <button
                      onClick={() => handleExportWallet('recoveryPhrase')}
                      disabled={isExporting}
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isExporting ? 'Exporting...' : 'Export Recovery Phrase'}
                    </button>
                    <button
                      onClick={() => setShowBackupOptions(false)}
                      className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-300 transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                )}
                {exportError && (
                  <p className="text-sm text-red-600 mt-2">{exportError}</p>
                )}
                <div 
                  id="wallet-export-container" 
                  className="mt-2 p-4 bg-gray-100 rounded-lg font-mono text-xs break-all"
                />
              </div>
            </div>
          )}

          {/* Sign Out Button */}
          <div className="bg-gray-50 rounded-xl px-4 py-4">
            <button
              onClick={handleSignOut}
              className="w-full px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 
