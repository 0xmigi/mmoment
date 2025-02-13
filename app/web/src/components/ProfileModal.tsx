import { Dialog } from '@headlessui/react';
import { User, X, ExternalLink } from 'lucide-react';

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
  };
  action?: {
    type: string;
    timestamp: number;
    transactionId?: string;
  };
}

export function ProfileModal({ isOpen, onClose, user, action }: ProfileModalProps) {
  if (!isOpen) return null;

  const handleWarpcastClick = () => {
    if (user.farcasterUsername) {
      window.open(`https://warpcast.com/${user.farcasterUsername.replace('@', '')}`, '_blank');
    }
  };

  const handleExplorerClick = () => {
    if (action?.transactionId) {
      window.open(`https://explorer.solana.com/tx/${action.transactionId}?cluster=devnet`, '_blank');
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

  const formatTxHash = (hash: string) => {
    return `${hash.slice(0, 8)}...${hash.slice(-8)}`;
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
            <div className="flex items-start justify-between">
              <div>
                <div className="text-sm font-medium mb-0.5">Name</div>
                <div className="text-sm text-gray-900">
                  {user.displayName || user.username || `${user.address.slice(0, 6)}...${user.address.slice(-4)}`}
                </div>
              </div>
              {user.pfpUrl ? (
                <img
                  src={user.pfpUrl}
                  alt={user.displayName || 'Profile'}
                  className="w-12 h-12 rounded-full"
                />
              ) : (
                <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center">
                  <User className="w-6 h-6 text-gray-400" />
                </div>
              )}
            </div>

            {/* Connected Account */}
            <div className="space-y-2">
              <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-gray-700">Farcaster</span>
                    <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">source</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {user.farcasterUsername ? user.farcasterUsername : 'Not connected'}
                  </div>
                </div>
                {user.farcasterUsername && (
                  <button 
                    onClick={handleWarpcastClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                  >
                    View <ExternalLink className="w-3 h-3" />
                  </button>
                )}
              </div>
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
                    {action.transactionId && (
                      <div className="flex items-center justify-between pt-1 border-t border-gray-200/75">
                        <div className="text-[10px] font-mono text-gray-500">
                          {formatTxHash(action.transactionId)}
                        </div>
                        <button
                          onClick={handleExplorerClick}
                          className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                        >
                          View Tx <ExternalLink className="w-3 h-3" />
                        </button>
                      </div>
                    )}
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