import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Download } from 'lucide-react';

interface ProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
    farcasterUsername?: string;
    bio?: string;
    verifiedCredentials?: {
      oauthProvider: string;
      oauthDisplayName?: string;
      oauthUsername?: string;
      oauthAccountPhotos?: string[];
    }[];
    walletAddress?: string;
  };
  action?: {
    type: string;
    timestamp: number;
    transactionId?: string;
    mediaUrl?: string;
  };
}

export function ProfileModal({ isOpen, onClose, user, action }: ProfileModalProps) {
  if (!isOpen) return null;

  // Get social identity credentials
  const farcasterCred = user?.verifiedCredentials?.find(cred => cred.oauthProvider === 'farcaster');
  const twitterCred = user?.verifiedCredentials?.find(cred => cred.oauthProvider === 'twitter');
  
  // Prioritize Farcaster, then Twitter
  const primarySocialCred = farcasterCred || twitterCred;
  
  // Get display information
  const displayName = user.displayName || primarySocialCred?.oauthDisplayName || primarySocialCred?.oauthUsername;
  const profileImage = user.pfpUrl || primarySocialCred?.oauthAccountPhotos?.[0];
  
  // Only use wallet as fallback
  const displayIdentity = displayName || `${user.address.slice(0, 4)}...${user.address.slice(-4)}`;

  const handleWarpcastClick = () => {
    if (farcasterCred?.oauthUsername) {
      window.open(`https://warpcast.com/${farcasterCred.oauthUsername.replace('@', '')}`, '_blank');
    }
  };
  
  const handleTwitterClick = () => {
    if (twitterCred?.oauthUsername) {
      window.open(`https://twitter.com/${twitterCred.oauthUsername.replace('@', '')}`, '_blank');
    }
  };

  const handleExplorerClick = () => {
    if (action?.transactionId) {
      window.open(`https://solscan.io/tx/${action.transactionId}?cluster=devnet`, '_blank');
    }
  };

  const handleDownload = () => {
    if (action?.mediaUrl) {
      const link = document.createElement('a');
      link.href = action.mediaUrl;
      link.download = `${action.type}_${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      second: 'numeric',
      hour12: true
    }).format(date);
  };


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
              Profile
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Profile Content */}
          <div className="p-3 space-y-4">
            {/* Core Identity Section */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gray-100 overflow-hidden">
                {profileImage && (
                  <img
                    src={profileImage}
                    alt={displayIdentity}
                    className="w-full h-full object-cover"
                  />
                )}
              </div>
              <div>
                <div className="text-sm font-medium">
                  {displayIdentity}
                </div>
              </div>
            </div>

            {/* Connected Accounts */}
            <div className="space-y-2">
              {/* Farcaster Account */}
              {farcasterCred && (
                <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-gray-700">Farcaster</span>
                      {farcasterCred === primarySocialCred && (
                        <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">source</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      @{farcasterCred.oauthUsername?.replace('@', '')}
                    </div>
                  </div>
                  <button 
                    onClick={handleWarpcastClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                  >
                    View <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              )}
              
              {/* Twitter Account */}
              {twitterCred && (
                <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-gray-700">X / Twitter</span>
                      {primarySocialCred === twitterCred && (
                        <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">source</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      @{twitterCred.oauthUsername?.replace('@', '')}
                    </div>
                  </div>
                  <button 
                    onClick={handleTwitterClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                  >
                    View <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>

            {/* Action Details */}
            {action && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-gray-500">Action</div>
                <div className="bg-gray-50 px-2 py-2 rounded-lg">
                  <div className="space-y-2">
                    <div>
                      <div className="text-sm font-medium text-gray-900 capitalize">
                        {action.type.replace(/_/g, ' ')}
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatDate(action.timestamp)}
                      </div>
                    </div>
                    <div className="flex items-center justify-between pt-1 border-t border-gray-200/75">
                      <div className="text-[10px] font-mono text-gray-500">
                        {action.transactionId ? `${action.transactionId.slice(0, 8)}...${action.transactionId.slice(-8)}` : 'Processing...'}
                      </div>
                      <div className="flex items-center gap-2">
                        {action.mediaUrl && (
                          <button
                            onClick={handleDownload}
                            className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                          >
                            Download <Download className="w-3 h-3" />
                          </button>
                        )}
                        {action.transactionId && (
                          <button
                            onClick={handleExplorerClick}
                            className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                          >
                            View Tx <ExternalLink className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
} 