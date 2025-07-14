import { Dialog } from '@headlessui/react';
import { X, ArrowUpRight, Download, Trash2 } from 'lucide-react';
import { IPFSMedia } from '../storage/ipfs/ipfs-service';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useState } from 'react';
import { unifiedIpfsService } from '../storage/ipfs/unified-ipfs-service';
import { TimelineEvent } from '../timeline/timeline-types';

interface MediaViewerProps {
  isOpen: boolean;
  onClose: () => void;
  media: IPFSMedia | null;
  event?: TimelineEvent;
  onDelete?: (mediaId: string) => void;
}

export default function MediaViewer({ isOpen, onClose, media, event, onDelete }: MediaViewerProps) {
  const { user, primaryWallet } = useDynamicContext();
  const [deleting, setDeleting] = useState(false);

  if (!media) return null;

  // Get social identity from user's verified credentials
  const farcasterCred = user?.verifiedCredentials?.find(cred => 
    cred.oauthProvider === 'farcaster'
  );
  
  const twitterCred = user?.verifiedCredentials?.find(cred => 
    cred.oauthProvider === 'twitter'
  );
  
  // Prioritize Farcaster over Twitter
  const primarySocialCred = farcasterCred || twitterCred;
  
  // Get social identity from event first, user's verified credentials as fallback
  // This lets us display profile info correctly for other users
  const displayName = event?.user.displayName || primarySocialCred?.oauthDisplayName;
  const username = event?.user.username || primarySocialCred?.oauthUsername;
  const profileImage = event?.user.pfpUrl || primarySocialCred?.oauthAccountPhotos?.[0];
  
  // Determine social provider from event data or current user credentials
  const socialProvider = (() => {
    // If we have a username that includes farcaster.xyz, it's Farcaster
    if (username?.includes('farcaster.xyz')) return 'Farcaster';
    // If username has a Twitter domain
    if (username?.includes('twitter.com')) return 'X / Twitter';
    // Use the provider from credentials as a fallback
    if (farcasterCred) return 'Farcaster';
    if (twitterCred) return 'X / Twitter';
    return null;
  })();
  
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
    if (!primaryWallet?.address || !media) return;
    
    try {
      setDeleting(true);
      
      let success = false;

      if (media.provider === 'jetson') {
        // Handle Jetson video deletion from localStorage
        const jetsonVideos = JSON.parse(localStorage.getItem('jetson-videos') || '[]');
        const filteredVideos = jetsonVideos.filter((video: any) => video.id !== mediaId);
        localStorage.setItem('jetson-videos', JSON.stringify(filteredVideos));
        success = true;
        console.log('Deleted Jetson video from localStorage:', mediaId);
      } else {
        // Handle IPFS media deletion with proper timing
        console.log('🗑️ Starting IPFS media deletion for:', mediaId);
        
        // Add a small delay to allow UI to show loading state
        await new Promise(resolve => setTimeout(resolve, 500));
        
        success = await unifiedIpfsService.deleteMedia(mediaId, primaryWallet.address);
        console.log('🗑️ IPFS media deletion result:', mediaId, 'success:', success);
        
        // Add another small delay before UI updates to ensure backend operations complete
        if (success) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      
      if (success) {
        console.log('✅ MEDIA DELETION SUCCESS (MediaViewer):', mediaId);
        if (onDelete) {
          onDelete(mediaId);
        }
        onClose();
      } else {
        console.error('❌ MEDIA DELETION FAILED (MediaViewer):', mediaId);
      }
    } catch (err) {
      console.error('🗑️ MEDIA DELETION ERROR (MediaViewer):', err);
      // Note: We're not showing a toast here since the deletion might still succeed
      // The error could be from timing issues, not actual failure
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
                  key={media.id}
                  src={media.url}
                  className="w-full max-h-[60vh] object-contain"
                  controls
                  autoPlay
                  playsInline
                />
              ) : (
                <img
                  src={media.url}
                  alt="Media preview"
                  className="w-full max-h-[60vh] object-contain"
                  onError={(e) => {
                    const img = e.target as HTMLImageElement;
                    if (media.backupUrls?.length) {
                      const currentIndex = media.backupUrls.indexOf(img.src);
                      if (currentIndex < media.backupUrls.length - 1) {
                        img.src = media.backupUrls[currentIndex + 1];
                      }
                    }
                  }}
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