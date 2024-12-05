import { useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Trash2, Download, Image, Video } from 'lucide-react';
import { pinataService } from '../services/pinata-service';

interface MediaItem {
  id: string;
  url: string;
  type: 'image' | 'video';
  timestamp: string;
  walletAddress: string;
  backupUrls?: string[];
}

interface MediaGalleryProps {
  mode?: 'recent' | 'archive';
  maxRecentItems?: number;
}

export default function MediaGallery({ mode = 'recent', maxRecentItems = 5 }: MediaGalleryProps) {
  const { primaryWallet } = useDynamicContext();
  const [media, setMedia] = useState<MediaItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchMedia = async () => {
    if (!primaryWallet?.address) {
      setMedia([]);
      setLoading(false);
      return;
    }

    try {
      setError(null);
      const walletAddress = primaryWallet.address;
      const allMedia = await pinataService.getMediaForWallet(walletAddress);

      console.log('Mode:', mode);
      console.log('Total media items:', allMedia.length);

      // Sort by timestamp, newest first
      const sortedMedia = allMedia.sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );

      console.log('Sorted media length:', sortedMedia.length);

      // Filter based on mode
      let filteredMedia;
      if (mode === 'recent') {
        filteredMedia = sortedMedia.slice(0, maxRecentItems);
        console.log('Recent media (first 5):', filteredMedia.length);
      } else {
        filteredMedia = sortedMedia.slice(maxRecentItems);
        console.log('Archived media (after 5):', filteredMedia.length);
      }

      setMedia(filteredMedia);
    } catch (err) {
      console.error('Failed to fetch media:', err);
      setError('Failed to load media. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMedia();
    const interval = setInterval(fetchMedia, 5000);
    return () => clearInterval(interval);
  }, [primaryWallet?.address, mode, maxRecentItems]);

  const handleDelete = async (mediaId: string) => {
    if (!primaryWallet?.address) return;

    try {
      setDeleting(mediaId);
      setError(null);

      const success = await pinataService.deleteMedia(mediaId, primaryWallet.address);

      if (success) {
        setMedia(current => current.filter(m => m.id !== mediaId));
      } else {
        setError('Failed to delete media. Please try again.');
      }
    } catch (err) {
      console.error('Delete error:', err);
      setError('Failed to delete media. Please try again.');
    } finally {
      setDeleting(null);
    }
  };

  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!primaryWallet?.address) {
    return (
      <div className="text-center py-8">
        Please connect your wallet to view your media
      </div>
    );
  }

  // Rest of your component remains exactly the same from here
  const title = mode === 'recent'
    ? `Current Session (${media.length})`
    : `Previous Sessions (${media.length})`;

  return (
    <div className="max-w-4xl mx-auto p-4">
      {/* Rest of your JSX remains unchanged */}
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
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {media.map((item) => (
            <div key={item.id} className="bg-white rounded-lg shadow-sm relative group">
              <div className="relative h-48">
                {item.type === 'video' ? (
                  // biome-ignore lint/a11y/useMediaCaption: <explanation>
                  <video
                    src={item.url}
                    className="absolute inset-0 w-full h-full object-cover rounded-t-lg"
                    controls
                  />
                ) : (
                  <img
                    src={item.url}
                    alt={`Captured moment`}
                    className="absolute inset-0 w-full h-full object-cover rounded-t-lg"
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
                <div className="absolute top-2 right-2 flex gap-2">
                  <button
                    onClick={() => handleDownload(item.url, `${item.id}.${item.type === 'video' ? 'mp4' : 'jpg'}`)}
                    className="p-2 rounded-full bg-white/80 hover:bg-white
                      opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                  >
                    <Download className="w-5 h-5 text-blue-500" />
                  </button>
                  <button
                    onClick={() => handleDelete(item.id)}
                    disabled={deleting === item.id}
                    className="p-2 rounded-full bg-white/80 hover:bg-white
                      opacity-0 group-hover:opacity-100 transition-opacity duration-200
                      disabled:opacity-50"
                  >
                    <Trash2
                      className={`w-5 h-5 text-red-500 ${deleting === item.id ? 'animate-spin' : ''}`}
                    />
                  </button>
                </div>
                <div className="absolute top-2 left-2">
                  {item.type === 'video' ? (
                    <Video className="w-5 h-5 text-white drop-shadow" />
                  ) : (
                    <Image className="w-5 h-5 text-white drop-shadow" />
                  )}
                </div>
              </div>
              <div className="p-2">
                <p className="text-sm text-gray-600">{item.timestamp}</p>
                <p className="text-xs text-gray-400 truncate">ID: {item.id}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}