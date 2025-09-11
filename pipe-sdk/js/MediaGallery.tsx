/**
 * React component for MMOMENT web app - Display user's camera captures
 * This would go in app/web/src/components/
 */

import React, { useState, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';

// Import the Pipe SDK (would be npm package in production)
// import PipeSDKBrowser from 'pipe-sdk-browser';

interface MediaItem {
  filename: string;
  timestamp: string;
  encrypted: boolean;
  url?: string;
}

interface PipeCredentials {
  userId: string;
  userAppKey: string;
}

export const MediaGallery: React.FC = () => {
  const { publicKey } = useWallet();
  const [media, setMedia] = useState<MediaItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [credentials, setCredentials] = useState<PipeCredentials | null>(null);
  const [sdk, setSdk] = useState<any>(null);

  // Initialize SDK
  useEffect(() => {
    // In production: import PipeSDKBrowser from 'pipe-sdk-browser';
    const PipeSDKBrowser = (window as any).PipeSDKBrowser;
    if (PipeSDKBrowser) {
      setSdk(new PipeSDKBrowser());
    }
  }, []);

  // Load user credentials when wallet connects
  useEffect(() => {
    if (publicKey) {
      loadUserCredentials(publicKey.toString());
    }
  }, [publicKey]);

  // Load credentials from backend or local storage
  const loadUserCredentials = async (walletAddress: string) => {
    try {
      // In production: Fetch from your backend
      const response = await fetch(`/api/pipe/credentials/${walletAddress}`);
      const creds = await response.json();
      setCredentials(creds);

      // Load user's media list
      await loadUserMedia(creds);
    } catch (error) {
      console.error('Failed to load credentials:', error);
    }
  };

  // Load list of user's media files
  const loadUserMedia = async (creds: PipeCredentials) => {
    setLoading(true);
    try {
      // In production: Get file list from your backend
      const response = await fetch(`/api/media/${publicKey?.toString()}`);
      const files = await response.json();

      // Convert to MediaItem format
      const mediaItems: MediaItem[] = files.map((file: any) => ({
        filename: file.filename,
        timestamp: file.created_at,
        encrypted: file.filename.endsWith('.enc'),
      }));

      setMedia(mediaItems);
    } catch (error) {
      console.error('Failed to load media:', error);
    } finally {
      setLoading(false);
    }
  };

  // Download and decrypt a specific media file
  const loadMediaContent = async (item: MediaItem) => {
    if (!sdk || !credentials) return;

    try {
      // Download and decrypt
      const imageData = await sdk.downloadAndDecrypt(
        credentials.userId,
        credentials.userAppKey,
        item.filename,
        null  // Use auto-generated password
      );

      // Create displayable URL
      const url = sdk.createImageUrl(imageData);

      // Update media item with URL
      setMedia(prev => prev.map(m =>
        m.filename === item.filename
          ? { ...m, url }
          : m
      ));
    } catch (error) {
      console.error(`Failed to load ${item.filename}:`, error);
    }
  };

  // Share a photo publicly
  const sharePhoto = async (item: MediaItem) => {
    if (!credentials) return;

    try {
      // Create public link via backend
      const response = await fetch('/api/media/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: credentials.userId,
          userAppKey: credentials.userAppKey,
          filename: item.filename,
        }),
      });

      const { publicLink } = await response.json();

      // If encrypted, add password to URL fragment
      if (item.encrypted && sdk) {
        const password = await sdk.generateUserPassword(credentials.userId);
        const shareUrl = `${publicLink}#${password.substring(0, 16)}`;

        // Copy to clipboard
        await navigator.clipboard.writeText(shareUrl);
        alert('Share link copied to clipboard!');
      }
    } catch (error) {
      console.error('Failed to share:', error);
    }
  };

  if (!publicKey) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-500">Connect your wallet to view your photos</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-6">Your Camera Captures</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {media.map((item) => (
          <div key={item.filename} className="bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Image container */}
            <div className="aspect-w-16 aspect-h-9 bg-gray-200">
              {item.url ? (
                <img
                  src={item.url}
                  alt="Camera capture"
                  className="object-cover w-full h-full"
                />
              ) : (
                <button
                  onClick={() => loadMediaContent(item)}
                  className="w-full h-full flex items-center justify-center hover:bg-gray-300 transition"
                >
                  <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <span className="ml-2">Load Photo</span>
                </button>
              )}
            </div>

            {/* Info section */}
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">
                  {new Date(item.timestamp).toLocaleDateString()}
                </span>
                {item.encrypted && (
                  <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">
                    🔒 Encrypted
                  </span>
                )}
              </div>

              {/* Actions */}
              <div className="flex space-x-2">
                {item.url && (
                  <>
                    <a
                      href={item.url}
                      download={item.filename}
                      className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                    >
                      Download
                    </a>
                    <button
                      onClick={() => sharePhoto(item)}
                      className="flex-1 px-3 py-2 bg-gray-200 text-gray-800 text-sm rounded hover:bg-gray-300"
                    >
                      Share
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {media.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No photos yet. Visit a MMOMENT camera to get started!</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// PUBLIC PHOTO VIEWER COMPONENT
// ============================================================================

interface PublicPhotoViewerProps {
  hash: string;
}

export const PublicPhotoViewer: React.FC<PublicPhotoViewerProps> = ({ hash }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsPassword, setNeedsPassword] = useState(false);
  const [password, setPassword] = useState('');

  useEffect(() => {
    // Check URL for password in fragment
    const urlPassword = window.location.hash.substring(1);
    if (urlPassword) {
      loadPhotoWithPassword(urlPassword);
    } else {
      checkIfEncrypted();
    }
  }, [hash]);

  const checkIfEncrypted = async () => {
    // Try to load without password first
    try {
      const sdk = new (window as any).PipeSDKBrowser();
      const data = await sdk.publicDownload(hash);

      // Check if it's encrypted (has metadata)
      try {
        sdk.unpackEncryptedFile(data);
        setNeedsPassword(true);
      } catch {
        // Not encrypted, display directly
        const url = sdk.createImageUrl(data);
        setImageUrl(url);
      }
    } catch (error) {
      console.error('Failed to load photo:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPhotoWithPassword = async (pwd: string) => {
    try {
      const sdk = new (window as any).PipeSDKBrowser();
      const data = await sdk.publicDownload(hash);
      const { encryptedData, metadata } = sdk.unpackEncryptedFile(data);
      const decrypted = await sdk.decrypt(encryptedData, pwd, metadata);
      const url = sdk.createImageUrl(decrypted);
      setImageUrl(url);
      setNeedsPassword(false);
    } catch (error) {
      alert('Wrong password!');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>;
  }

  if (needsPassword) {
    return (
      <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">This photo is encrypted</h3>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter password"
          className="w-full px-3 py-2 border rounded mb-4"
        />
        <button
          onClick={() => loadPhotoWithPassword(password)}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Decrypt Photo
        </button>
      </div>
    );
  }

  return imageUrl ? (
    <img src={imageUrl} alt="Shared photo" className="max-w-full rounded-lg shadow-lg" />
  ) : (
    <p>Failed to load photo</p>
  );
};
