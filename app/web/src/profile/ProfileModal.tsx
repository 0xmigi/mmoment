import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Trophy, Users, Clock, Gift } from 'lucide-react';
import { CVActivityMetadata } from '../timeline/timeline-types';

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
    provider?: string; // From backend profile (e.g., 'farcaster', 'twitter')
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
    cvActivity?: CVActivityMetadata;
    /** Future: Prize/bounty earned for completing this action */
    prize?: {
      amount: number;
      currency: string;
      from?: string; // Who issued the prize
    };
  };
}

export function ProfileModal({ isOpen, onClose, user, action }: ProfileModalProps) {
  if (!isOpen) return null;

  // Get social identity credentials (from current user's verifiedCredentials)
  const farcasterCred = user?.verifiedCredentials?.find(cred => cred.oauthProvider === 'farcaster');
  const twitterCred = user?.verifiedCredentials?.find(cred => cred.oauthProvider === 'twitter');

  // Check if we have backend profile data (for other users)
  const hasBackendProfile = user.provider && user.username;
  const backendProvider = user.provider?.toLowerCase();

  // Prioritize Farcaster, then Twitter (from verifiedCredentials)
  const primarySocialCred = farcasterCred || twitterCred;
  
  // Get display information
  const displayName = user.displayName || primarySocialCred?.oauthDisplayName || primarySocialCred?.oauthUsername;
  const profileImage = user.pfpUrl || primarySocialCred?.oauthAccountPhotos?.[0];
  
  // Only use wallet as fallback
  const displayIdentity = displayName || `${user.address.slice(0, 4)}...${user.address.slice(-4)}`;

  const handleExplorerClick = () => {
    if (action?.transactionId) {
      window.open(`https://solscan.io/tx/${action.transactionId}?cluster=devnet`, '_blank');
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

  // Format app name nicely (pushup -> Push-ups)
  const formatAppName = (appName: string): string => {
    const mapping: Record<string, string> = {
      'pushup': 'Push-ups',
      'pullup': 'Pull-ups',
      'squat': 'Squats',
      'situp': 'Sit-ups',
      'jumping_jack': 'Jumping Jacks',
    };
    return mapping[appName] || appName.charAt(0).toUpperCase() + appName.slice(1).replace(/_/g, ' ');
  };

  // Get rank suffix (1st, 2nd, 3rd, etc.)
  const getRankSuffix = (rank: number): string => {
    if (rank === 1) return 'st';
    if (rank === 2) return 'nd';
    if (rank === 3) return 'rd';
    return 'th';
  };

  // Get rank color for styling
  const getRankColor = (rank: number): string => {
    if (rank === 1) return 'text-yellow-600 bg-yellow-50';
    if (rank === 2) return 'text-gray-500 bg-gray-100';
    if (rank === 3) return 'text-amber-700 bg-amber-50';
    return 'text-gray-600 bg-gray-50';
  };

  // Check if this is a CV activity
  const isCVActivity = action?.type === 'cv_activity' && action?.cvActivity;


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
              {/* Farcaster Account - from verifiedCredentials OR backend */}
              {(farcasterCred || (hasBackendProfile && backendProvider === 'farcaster')) && (
                <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-gray-700">Farcaster</span>
                      {(farcasterCred === primarySocialCred || (hasBackendProfile && backendProvider === 'farcaster')) && (
                        <span className="text-[10px] text-primary bg-primary-light px-1.5 py-0.5 rounded">source</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      @{farcasterCred?.oauthUsername?.replace('@', '') || user.username?.replace('@', '')}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      const username = farcasterCred?.oauthUsername || user.username;
                      if (username) {
                        window.open(`https://farcaster.xyz/${username.replace('@', '')}`, '_blank');
                      }
                    }}
                    className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center gap-1"
                  >
                    View <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              )}

              {/* Twitter Account - from verifiedCredentials OR backend */}
              {(twitterCred || (hasBackendProfile && backendProvider === 'twitter')) && (
                <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-gray-700">X / Twitter</span>
                      {(primarySocialCred === twitterCred || (hasBackendProfile && backendProvider === 'twitter')) && (
                        <span className="text-[10px] text-primary bg-primary-light px-1.5 py-0.5 rounded">source</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      @{twitterCred?.oauthUsername?.replace('@', '') || user.username?.replace('@', '')}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      const username = twitterCred?.oauthUsername || user.username;
                      if (username) {
                        window.open(`https://twitter.com/${username.replace('@', '')}`, '_blank');
                      }
                    }}
                    className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center gap-1"
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
                    {/* CV Activity - Rich Display */}
                    {isCVActivity && action.cvActivity ? (
                      <>
                        {/* Main activity summary */}
                        <div className="flex items-start gap-2">
                          <div className="flex-1">
                            <div className="text-sm font-medium text-gray-900">
                              Completed {action.cvActivity.user_stats?.reps ?? 0} {formatAppName(action.cvActivity.app_name)}
                            </div>
                            <div className="text-xs text-gray-500">
                              {formatDate(action.timestamp)}
                            </div>
                          </div>
                          {/* Rank badge for competitions */}
                          {action.cvActivity.participant_count > 1 && (
                            <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getRankColor(action.cvActivity.results?.find(r => r.stats?.reps === action.cvActivity?.user_stats?.reps)?.rank ?? 1)}`}>
                              <Trophy className="w-3 h-3" />
                              {(() => {
                                const rank = action.cvActivity.results?.find(r => r.stats?.reps === action.cvActivity?.user_stats?.reps)?.rank ?? 1;
                                return `${rank}${getRankSuffix(rank)}`;
                              })()}
                            </div>
                          )}
                        </div>

                        {/* Competition details */}
                        {action.cvActivity.participant_count > 1 && (
                          <div className="flex items-center gap-3 text-xs text-gray-500 pt-1">
                            <div className="flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              <span>{action.cvActivity.participant_count} participants</span>
                            </div>
                            {action.cvActivity.duration_seconds && (
                              <div className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                <span>{Math.round(action.cvActivity.duration_seconds)}s</span>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Prize/Bounty (future) */}
                        {action.prize && (
                          <div className="flex items-center gap-2 pt-1 text-xs">
                            <Gift className="w-3 h-3 text-green-600" />
                            <span className="text-green-700 font-medium">
                              Earned {action.prize.amount} {action.prize.currency}
                              {action.prize.from && <span className="text-gray-500"> from {action.prize.from}</span>}
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      /* Standard action display for non-CV activities */
                      <div>
                        <div className="text-sm font-medium text-gray-900 capitalize">
                          {action.type.replace(/_/g, ' ')}
                        </div>
                        <div className="text-xs text-gray-500">
                          {formatDate(action.timestamp)}
                        </div>
                      </div>
                    )}

                    {/* Transaction link - only show for on-chain actions (check-outs, financialized actions) */}
                    {(action.type === 'check_out' || action.type === 'auto_check_out' || action.prize) && action.transactionId && (
                      <div className="flex items-center justify-between pt-1 border-t border-gray-200/75">
                        <div className="text-[10px] font-mono text-gray-500">
                          {action.transactionId.slice(0, 8)}...{action.transactionId.slice(-8)}
                        </div>
                        <button
                          onClick={handleExplorerClick}
                          className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center gap-1"
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