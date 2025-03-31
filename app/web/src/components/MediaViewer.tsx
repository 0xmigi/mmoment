import { Dialog } from '@headlessui/react';
import { X, ArrowUpRight } from 'lucide-react';
import { IPFSMedia } from '../services/ipfs-service';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useState } from 'react';

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
}

export default function MediaViewer({ isOpen, onClose, media, event }: MediaViewerProps) {
  const { user } = useDynamicContext();
  const [currentVideoUrl, setCurrentVideoUrl] = useState('');
  const [videoErrorCount, setVideoErrorCount] = useState(0);

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

  // Get Farcaster identity from user's verified credentials
  const farcasterCred = user?.verifiedCredentials?.find(cred => 
    cred.oauthProvider === 'farcaster'
  );
  
  const displayName = farcasterCred?.oauthDisplayName || farcasterCred?.oauthUsername;
  const profileImage = farcasterCred?.oauthAccountPhotos?.[0];
  
  // Only use wallet address as fallback if no Farcaster identity
  const displayIdentity = displayName || `${media.walletAddress.slice(0, 4)}...${media.walletAddress.slice(-4)}`;

  // Use event's transaction ID if available, fallback to media's transaction ID
  const transactionId = event?.transactionId || media.transactionId;

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
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Media
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
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
                  <div className="text-xs text-gray-500">
                    Farcaster {farcasterCred && <span className="text-blue-500">source</span>}
                  </div>
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