import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { Copy, User, LogOut, KeyRound, Check, Globe, Lock } from 'lucide-react';
import { PipeStorageSection } from './PipeStorageSection';

// Define interfaces
interface SocialCredential {
  format: string;
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

interface StatusMessage {
  message: string;
  type: 'success' | 'error' | 'info';
}

export function AccountPage() {
  const { primaryWallet, handleLogOut, user } = useDynamicContext();
  const { revealWalletKey } = useEmbeddedWallet();
  const navigate = useNavigate();
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);
  const [copied, setCopied] = useState(false);
  const [showBackupOptions, setShowBackupOptions] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleSignOut = async () => {
    try {
      await handleLogOut();
      navigate('/');
    } catch (err) {
      console.error('Failed to sign out:', err);
    }
  };

  const handleCopyAddress = async () => {
    if (!primaryWallet?.address) return;
    
    try {
      await navigator.clipboard.writeText(primaryWallet.address);
      setCopied(true);
      setStatusMessage({ type: 'info', message: 'Wallet address copied!' });
      setTimeout(() => {
        setCopied(false);
        setStatusMessage(null);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy address:', err);
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
      setStatusMessage({ 
        type: 'success', 
        message: `${type === 'privateKey' ? 'Private key' : 'Recovery phrase'} displayed.` 
      });
    } catch (error) {
      console.error('Failed to export wallet:', error);
      setExportError('Failed to export wallet. Please try again.');
      setStatusMessage({ type: 'error', message: 'Failed to export wallet.' });
    } finally {
      setIsExporting(false);
    }
  };

  if (!primaryWallet?.address) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-gray-600">Loading...</div>
      </div>
    );
  }

  // Get social credentials
  const socialCreds = user?.verifiedCredentials?.filter(
    (cred: any): cred is SocialCredential => cred.format === 'oauth'
  ) || [];
  
  // Find specific social providers
  const twitterCred = socialCreds.find(cred => cred.oauthProvider === 'twitter');
  const farcasterCred = socialCreds.find(cred => cred.oauthProvider === 'farcaster');
  
  // Prioritize credentials (Farcaster > Twitter > none)
  const primarySocialCred = farcasterCred || twitterCred;
  const primarySocialProvider = farcasterCred ? 'Farcaster' : twitterCred ? 'X / Twitter' : null;
  
  // Prepare the profile image and display name
  const profileImageUrl = primarySocialCred?.oauthAccountPhotos?.[0];
  const displayName = primarySocialCred?.oauthDisplayName || primaryWallet.address.slice(0, 6) + '...' + primaryWallet.address.slice(-4);
  
  // Check if the wallet is an embedded wallet (not Phantom)
  const isEmbeddedWallet = primaryWallet.connector?.name.toLowerCase() !== 'phantom';

  // Define identity items for the branching display
  const identities = [
    {
      id: 'twitter',
      label: 'X / Twitter',
      value: twitterCred?.oauthUsername,
      connected: !!twitterCred,
      isPublic: true,
      icon: <Globe className="w-3 h-3 mr-1" />
    },
    {
      id: 'farcaster',
      label: 'Farcaster',
      value: farcasterCred?.oauthUsername,
      connected: !!farcasterCred,
      isPublic: true,
      icon: <Globe className="w-3 h-3 mr-1" />
    },
    {
      id: 'email',
      label: 'Email',
      value: user?.email,
      connected: !!user?.email,
      isPublic: false,
      icon: <Lock className="w-3 h-3 mr-1" />
    },
    {
      id: 'wallet',
      label: 'Solana Wallet',
      value: primaryWallet.address,
      shortValue: `${primaryWallet.address.slice(0, 6)}...${primaryWallet.address.slice(-4)}`,
      connected: true,
      isPublic: true,
      isWallet: true,
      icon: <Globe className="w-3 h-3 mr-1" />
    }
  ].filter(item => item.connected || ['farcaster', 'email', 'twitter'].includes(item.id));

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto pt-8 px-4">
        <div className="bg-white mb-6">
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

        {/* Identity Section */}
        <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-6">
          <h2 className="text-lg font-medium mb-6 sm:mb-8">Identity</h2>
          
          {/* Profile Picture & Identity Tree - RIGHT ALIGNED, mobile responsive */}
          <div className="relative mb-10">
            {/* Profile Container - right-aligned on ALL screen sizes */}
            <div className="flex justify-end mb-8">
              <div className="relative">
                {/* Profile Avatar */}
                {profileImageUrl ? (
                  <img 
                    src={profileImageUrl} 
                    alt={displayName} 
                    className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border-2 border-gray-200"
                  />
                ) : (
                  <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gray-200 flex items-center justify-center">
                    <User className="w-8 h-8 sm:w-10 sm:h-10 text-gray-400" />
                  </div>
                )}
                
                {/* Vertical line coming down from profile - centered and extended */}
                <div className="absolute left-1/2 top-full w-0.5 h-[350px] bg-gray-300 transform -translate-x-1/2"></div>
              </div>
            </div>
            
            {/* Profile name - now left-aligned like other identity items */}
            <div className="mb-6 mr-[100px] sm:mr-[135px]">
              <div className="font-medium text-gray-800">{displayName}</div>
              <div className="text-sm text-gray-600">
                {primarySocialProvider || 'Wallet Address'}
              </div>
            </div>
            
            {/* Identity Tree Structure - styled to match the profile popup */}
            <div className="flex flex-col relative">
              {/* Identity Items - adjusted for mobile while maintaining alignment */}
              <div className="space-y-6 sm:space-y-7">
                {identities.map((identity, idx) => (
                  <div key={idx} className="relative mr-[100px] sm:mr-[135px]">
                    {/* Horizontal connector to main stem */}
                    <div className="absolute right-[-40px] sm:right-[-68px] top-[12px] w-[40px] sm:w-[68px] h-0.5 bg-gray-300"></div>
                    
                    {/* Identity Content - matching popup style */}
                    <div className="flex justify-between items-start pr-3 sm:pr-8">
                      {/* Identity Info - LEFT ALIGNED - matching popup style */}
                      <div className="pl-0">
                        <div className="font-medium text-gray-800">{identity.label}</div>
                        {identity.connected ? (
                          identity.isWallet ? (
                            <div className="text-sm text-gray-600 font-mono">{identity.shortValue}</div>
                          ) : (
                            <>
                              <div className="text-sm text-gray-600">
                                {identity.id === 'twitter' && '@'}{identity.value}
                              </div>
                              <div className="text-xs text-gray-400 mt-1">
                                {identity.isPublic ? 'Public' : 'Private'}
                              </div>
                            </>
                          )
                        ) : (
                          <div className="text-sm text-gray-500">Not connected</div>
                        )}
                      </div>
                      
                      {/* Action Button - RIGHT ALIGNED - Removed Connect buttons, kept only Copy for wallet */}
                      <div className="ml-auto">
                        {identity.connected && identity.isWallet && (
                          <button
                            onClick={handleCopyAddress}
                            className="text-gray-400 hover:text-gray-600 transition-colors p-1"
                            title="Copy address"
                          >
                            {copied ? 
                              <Check className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" /> : 
                              <Copy className="w-4 h-4 sm:w-5 sm:h-5" />
                            }
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Pipe Storage Section */}
        <PipeStorageSection />

        {/* Wallet Backup Section - responsive padding */}
        {isEmbeddedWallet && (
          <div className="bg-gray-50 rounded-xl px-4 py-4 mb-4">
            <div className="text-sm">
              <div className="font-medium mb-3">Wallet Backup</div>
              <button
                onClick={() => setShowBackupOptions(!showBackupOptions)}
                className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                <KeyRound className="w-4 h-4" />
                Back up Wallet
              </button>
              {showBackupOptions && (
                <div className="mt-4 space-y-2">
                  <button
                    onClick={() => handleExportWallet('recoveryPhrase')}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors disabled:opacity-50"
                  >
                    Show Recovery Phrase
                  </button>
                  <button
                    onClick={() => handleExportWallet('privateKey')}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors disabled:opacity-50"
                  >
                    Show Private Key
                  </button>
                </div>
              )}
              {exportError && (
                <p className="text-sm text-red-600 mt-2">{exportError}</p>
              )}
              <div 
                id="wallet-export-container" 
                className="mt-2 p-3 sm:p-4 bg-gray-100 rounded-lg font-mono text-xs break-all"
              />
            </div>
          </div>
        )}

        {/* Sign Out Button - responsive padding */}
        <div className="bg-gray-50 rounded-xl px-4 py-4 mb-8">
          <button
            onClick={handleSignOut}
            className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </div>
    </div>
  );
} 
