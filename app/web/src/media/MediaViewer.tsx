import { Dialog } from '@headlessui/react';
import { X, ArrowUpRight, Download, Trash2 } from 'lucide-react';
import { IPFSMedia } from '../storage/ipfs/ipfs-service';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useState } from 'react';
import { unifiedIpfsService } from '../storage/ipfs/unified-ipfs-service';

// Copy the necessary types
type TimelineEventType =
  | 'initialization'
  | 'user_connected'
  | 'photo_captured'
  | 'video_recorded'
  | 'stream_started'
  | 'stream_ended';

interface TimelineUser {
  address: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
}

interface TimelineEvent {
  id: string;
  type: TimelineEventType;
  user: TimelineUser;
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
}

interface MediaViewerProps {
  isOpen: boolean;
  onClose: () => void;
  media: IPFSMedia | null;
  event?: TimelineEvent;
  onDelete?: (mediaId: string) => void;
}

export default function MediaViewer({ isOpen, onClose, media, event, onDelete }: MediaViewerProps) {
  const { user, primaryWallet } = useDynamicContext();
  const [currentVideoUrl, setCurrentVideoUrl] = useState('');
  const [videoErrorCount, setVideoErrorCount] = useState(0);
  const [deleting, setDeleting] = useState(false);

  if (!media) return null;

  // Add debugging for video media
  if (media.type === 'video' && !currentVideoUrl) {
    console.log('Video details:', {
      url: media.url,
      mimeType: media.mimeType,
      backupUrls: media.backupUrls,
      directUrl: media.directUrl
    });
    
    // Initialize with the best URL
    setCurrentVideoUrl(media.directUrl || media.url);
  }

  // Get social identity from user's verified credentials
  const farcasterCred = user?.verifiedCredentials?.find(cred => 
    cred.oauthProvider === 'farcaster'
  );
  
  const twitterCred = user?.verifiedCredentials?.find(cred => 
    cred.oauthProvider === 'twitter'
  );
  
  // Prioritize Farcaster over Twitter
  const primarySocialCred = farcasterCred || twitterCred;
  const socialProvider = farcasterCred ? 'Farcaster' : twitterCred ? 'X / Twitter' : null;
  
  // Get display information
  const displayName = primarySocialCred?.oauthDisplayName || event?.user.displayName;
  const username = primarySocialCred?.oauthUsername || event?.user.username;
  const profileImage = primarySocialCred?.oauthAccountPhotos?.[0] || event?.user.pfpUrl;
  
  // Only use wallet address as fallback if no social identity
  const displayIdentity = displayName || `${media.walletAddress.slice(0, 4)}...${media.walletAddress.slice(-4)}`;

  // Use event's transaction ID if available, fallback to media's transaction ID
  const transactionId = event?.transactionId || media.transactionId;

  // Function to handle media download
  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Function to handle media deletion
  const handleDelete = async (mediaId: string) => {
    if (!primaryWallet?.address) return;
    
    try {
      setDeleting(true);
      
      const success = await unifiedIpfsService.deleteMedia(mediaId, primaryWallet.address);
      
      if (success) {
        if (onDelete) {
          onDelete(mediaId);
        }
        onClose();
      }
    } catch (err) {
      console.error('Delete error:', err);
    } finally {
      setDeleting(false);
    }
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
        <Dialog.Panel className="mx-auto w-full sm:w-[480px] md:w-[640px] rounded-xl bg-white shadow-xl">
          {/* Header with buttons */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Media
            </Dialog.Title>
            <div className="flex items-center space-x-1">
              <button
                onClick={() => handleDownload(media.url, `${media.id}.${media.type === 'video' ? 'mp4' : 'jpg'}`)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
                title="Download"
              >
                <Download className="w-4 h-4 text-blue-500" />
              </button>
              <button
                onClick={() => handleDelete(media.id)}
                disabled={deleting}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors disabled:opacity-50"
                title="Delete"
              >
                <Trash2 className={`w-4 h-4 text-red-500 ${deleting ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
                title="Close"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>
          </div>

          {/* Media Content */}
          <div className="space-y-4">
            {/* Media preview - removed padding */}
            <div className="w-full">
              {media.type === 'video' ? (
                <video
                  src={currentVideoUrl || media.url}
                  className="w-full max-h-[60vh] object-contain"
                  controls
                  autoPlay
                  playsInline
                  onError={(e) => {
                    const video = e.currentTarget;
                    console.error(`Video error loading ${video.src} (attempt ${videoErrorCount + 1})`);
                    
                    // Try to find the next URL to try
                    const allUrls = [
                      media.directUrl, 
                      ...media.backupUrls, 
                      media.url
                    ].filter(Boolean) as string[];
                    
                    // Find the current URL's index
                    const currentIndex = allUrls.indexOf(currentVideoUrl || media.url);
                    const nextIndex = (currentIndex + 1) % allUrls.length;
                    
                    // Try the next URL if we haven't cycled through all options yet
                    if (nextIndex !== currentIndex && videoErrorCount < allUrls.length) {
                      const nextUrl = allUrls[nextIndex];
                      console.log(`Trying alternative URL (${videoErrorCount + 1}/${allUrls.length}): ${nextUrl}`);
                      setCurrentVideoUrl(nextUrl);
                      setVideoErrorCount(prev => prev + 1);
                    } else {
                      console.error("All video URLs failed to load");
                    }
                  }}
                />
              ) : (
                <img
                  src={media.url}
                  alt="Media preview"
                  className="w-full max-h-[60vh] object-contain"
                />
              )}
            </div>

            {/* Info sections with padding */}
            <div className="px-3 pb-3 space-y-4">
              {/* Core Identity Section */}
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gray-100 overflow-hidden">
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
                  {socialProvider && (
                    <div className="text-xs text-gray-500">
                      {username && `@${username.replace('@', '')}`}
                      {' '}
                      <span className="text-[10px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded ml-1">
                        {socialProvider}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Action Details */}
              <div>
                <div className="text-[13px] font-medium text-gray-900">Action</div>
                <div className="mt-1 bg-gray-50 px-3 py-2 rounded-lg">
                  <div className="text-sm text-gray-600">
                    {event?.type === 'video_recorded' ? 'Video Recorded' : 
                     event?.type === 'photo_captured' ? 'Photo Captured' : 
                     event?.type === 'stream_started' ? 'Stream Started' : 
                     event?.type === 'stream_ended' ? 'Stream Ended' : 'Photo Captured'}
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {new Date(media.timestamp).toLocaleString()}
                  </div>
                  {(event?.cameraId || media.cameraId) && (
                    <div className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                      <span>Camera: </span>
                      <span className="font-mono">
                        {`${(event?.cameraId || media.cameraId || '').slice(0, 4)}...${(event?.cameraId || media.cameraId || '').slice(-4)}`}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Transaction */}
              {transactionId && (
                <div>
                  <div className="text-[13px] font-medium text-gray-900">Transaction</div>
                  <div className="mt-1 bg-gray-50 px-3 py-2 rounded-lg flex items-center justify-between">
                    <span className="text-xs font-mono text-gray-600 truncate max-w-[180px]">
                      {`${transactionId.slice(0, 8)}...${transactionId.slice(-8)}`}
                    </span>
                    <a
                      href={`https://solscan.io/tx/${transactionId}?cluster=devnet`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                    >
                      View Tx <ArrowUpRight className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}