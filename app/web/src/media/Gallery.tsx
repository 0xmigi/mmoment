import { useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Image, Video } from 'lucide-react';
import MediaViewer from './MediaViewer';
import { unifiedIpfsService } from '../storage/ipfs/unified-ipfs-service';
import { IPFSMedia } from '../storage/ipfs/ipfs-service';

interface MediaGalleryProps {
  mode?: 'recent' | 'archive';
  maxRecentItems?: number;
  cameraId?: string;
}

export default function MediaGallery({ mode = 'recent', maxRecentItems = 5, cameraId }: MediaGalleryProps) {
  const { primaryWallet } = useDynamicContext();
  const [media, setMedia] = useState<IPFSMedia[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [, setDeleting] = useState<string | null>(null);
  const [selectedMedia, setSelectedMedia] = useState<IPFSMedia | null>(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);

  // Add click handler for media items
  const handleMediaClick = (media: IPFSMedia) => {
    console.log("Viewing media with transaction ID:", media.transactionId || "none");
    setSelectedMedia(media);
    setIsViewerOpen(true);
  };

  useEffect(() => {
    let isSubscribed = true;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 5000; // 5 seconds
    const POLL_INTERVAL = 10000; // 10 seconds

    const fetchMediaWithRetry = async () => {
      if (!primaryWallet?.address) {
        if (isSubscribed) {
          setMedia([]);
          setLoading(false);
        }
        return;
      }

      try {
        setError(null);
        
        // Get ALL media from IPFS only - no localStorage shortcuts!
        const allMedia = await unifiedIpfsService.getMediaForWallet(primaryWallet.address);

        if (!isSubscribed) return;

        // Reset retry count on success
        retryCount = 0;

        // Sort by timestamp, newest first
        const sortedMedia = allMedia.sort((a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );

        // Filter by cameraId if provided
        let cameraFilteredMedia = sortedMedia;
        if (cameraId) {
          // Filter by cameraId stored in IPFS metadata
          cameraFilteredMedia = sortedMedia.filter(item => {
            return item.cameraId === cameraId;
          });
        }

        // Filter based on mode
        let filteredMedia;
        if (mode === 'recent') {
          filteredMedia = cameraFilteredMedia.slice(0, maxRecentItems);
        } else {
          // Archive mode should show ALL media
          filteredMedia = cameraFilteredMedia;
        }

        setMedia(filteredMedia);
      } catch (err) {
        console.error('Failed to fetch media:', err);
        
        // Only retry if we haven't exceeded MAX_RETRIES
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          setTimeout(fetchMediaWithRetry, RETRY_DELAY);
        } else {
          if (isSubscribed) {
            setError('Failed to load media. Please try again later.');
          }
        }
      } finally {
        if (isSubscribed) {
          setLoading(false);
        }
      }
    };

    // Initial fetch
    fetchMediaWithRetry();

    // Set up polling with a longer interval
    const interval = setInterval(fetchMediaWithRetry, POLL_INTERVAL);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [primaryWallet?.address, mode, maxRecentItems, cameraId]);

  const handleDelete = async (mediaId: string) => {
    if (!primaryWallet?.address) return;

    try {
      setDeleting(mediaId);
      setError(null);

      console.log('🗑️ Starting IPFS media deletion for:', mediaId);
      
      // Add a small delay to allow UI to show loading state
      await new Promise(resolve => setTimeout(resolve, 500));

      // All media is in IPFS - delete from IPFS only
      const success = await unifiedIpfsService.deleteMedia(mediaId, primaryWallet.address);
      console.log('🗑️ IPFS media deletion result:', mediaId, 'success:', success);

      if (success) {
        console.log('✅ MEDIA DELETION SUCCESS (Gallery):', mediaId);
        // Add a small delay before updating UI to ensure backend operations complete
        await new Promise(resolve => setTimeout(resolve, 1000));
        setMedia(current => current.filter(m => m.id !== mediaId));
      } else {
        console.error('❌ MEDIA DELETION FAILED (Gallery):', mediaId);
        setError('Failed to delete media. Please try again.');
      }
    } catch (err) {
      console.error('🗑️ MEDIA DELETION ERROR (Gallery):', err);
      // Only show error message if deletion actually failed
      // Don't show error for timing issues or temporary failures
      if (err instanceof Error && err.message.includes('Failed to delete')) {
        setError('Failed to delete media. Please try again.');
      } else {
        console.warn('🗑️ Deletion error might be temporary, not showing user error message');
      }
    } finally {
      setDeleting(null);
    }
  };


  if (!primaryWallet?.address) {
    return (
      <div className="text-center py-8">
        Please connect your wallet to view your media
      </div>
    );
  }

  const title = mode === 'recent'
    ? `Your recents`
    : `Gallery (${media.length})`;

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-xl text-left font-bold text-gray-800 mb-6">{title}</h2>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-left py-4">Loading your media...</div>
      ) : media.length === 0 ? (
        <div className="text-left py-4 text-gray-500">
          {mode === 'recent' ? 'No media in current session' : 'No archived media found'}
        </div>
      ) : (
        <div 
          className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3"
          data-testid="media-gallery"
        >
          {media.map((item) => (
            <div
              key={item.id}
              className="relative group cursor-pointer aspect-square"
              onClick={() => handleMediaClick(item)}
            >
              <div className="absolute inset-0 rounded-lg overflow-hidden">
                {item.type === 'video' ? (
                  // biome-ignore lint/a11y/useMediaCaption: <explanation>
                  <video
                    src={item.url}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      console.error('IPFS video playback error:', {
                        mediaId: item.id,
                        url: item.url,
                        mimeType: item.mimeType,
                        error: e.currentTarget.error
                      });
                      
                      // Try backup URLs if available
                      const video = e.currentTarget as HTMLVideoElement;
                      if (item.backupUrls?.length) {
                        const currentIndex = item.backupUrls.indexOf(video.src);
                        if (currentIndex < item.backupUrls.length - 1) {
                          console.log('Trying backup URL:', item.backupUrls[currentIndex + 1]);
                          video.src = item.backupUrls[currentIndex + 1];
                        }
                      }
                    }}
                    onLoadStart={(_e) => {
                      console.log('Video load started:', {
                        mediaId: item.id,
                        url: item.url,
                        mimeType: item.mimeType
                      });
                    }}
                    onCanPlay={(e) => {
                      console.log('Video can play:', {
                        mediaId: item.id,
                        duration: e.currentTarget.duration,
                        videoWidth: e.currentTarget.videoWidth,
                        videoHeight: e.currentTarget.videoHeight
                      });
                    }}
                  />
                ) : (
                  <img
                    src={item.url}
                    alt={`Captured moment`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                    onError={(e) => {
                      const img = e.target as HTMLImageElement;
                      if (item.backupUrls?.length) {
                        const currentIndex = item.backupUrls.indexOf(img.src);
                        if (currentIndex < item.backupUrls.length - 1) {
                          img.src = item.backupUrls[currentIndex + 1];
                        }
                      }
                    }}
                  />
                )}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
                <div className="absolute top-2 left-2">
                  {item.type === 'video' ? (
                    <Video className="w-4 h-4 text-white drop-shadow" />
                  ) : (
                    <Image className="w-4 h-4 text-white drop-shadow" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      <MediaViewer
        isOpen={isViewerOpen}
        onClose={() => {
          setIsViewerOpen(false);
          setSelectedMedia(null);
        }}
        media={selectedMedia}
        onDelete={(mediaId) => handleDelete(mediaId)}
      />
    </div>
  );
}