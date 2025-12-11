/**
 * Walrus Gallery Service
 *
 * Fetches user's Walrus files from backend and listens for real-time updates.
 * Mirrors the pattern from pipe-gallery-service.ts.
 */

import { CONFIG } from '../../core/config';

export interface WalrusAccessGrant {
  pubkey: string;
  encryptedKey: string;  // base64-encoded encrypted AES key
}

export interface WalrusGalleryItem {
  id: string;              // blobId
  blobId: string;          // Walrus blob ID
  name: string;
  url: string;             // Direct Walrus gateway URL (encrypted content)
  type: 'image' | 'video';
  mimeType: string;
  timestamp: number;
  backupUrls?: string[];
  walletAddress?: string;
  provider: 'walrus';
  cameraId?: string;
  // Encryption metadata
  encrypted: boolean;
  nonce?: string;          // base64-encoded AES-GCM nonce
  originalSize?: number;
  encryptedSize?: number;
  accessGrants?: WalrusAccessGrant[];
  suiOwner?: string;
  // Decrypted state (set by client after decryption)
  decryptedUrl?: string;   // Object URL after decryption
  metadata?: {
    camera?: string;
    location?: string;
  };
  // Local pending state (for photos not yet uploaded to Walrus)
  isPending?: boolean;
  localUrl?: string;       // Local camera URL for pending photos
  jobId?: number;          // Upload job ID for tracking
}

class WalrusGalleryService {
  private backendUrl: string;
  private galleryUpdateListeners: Set<() => void> = new Set();
  private socket: any = null;
  // Pending photos not yet uploaded to Walrus (keyed by jobId or filename)
  private pendingPhotos: Map<string, WalrusGalleryItem> = new Map();

  constructor() {
    this.backendUrl = CONFIG.BACKEND_URL;
    this.initializeWebSocket();
  }

  /**
   * Add a pending local photo that hasn't been uploaded to Walrus yet.
   * This shows the photo immediately in the gallery while upload happens in background.
   */
  addPendingPhoto(photo: {
    filename: string;
    localUrl: string;       // Full URL to fetch the photo from camera (e.g., https://camera.mmoment.xyz/api/photos/filename.jpg)
    blob?: Blob;            // Optional blob for immediate display
    jobId?: number;
    walletAddress: string;
    cameraId?: string;
    timestamp?: number;
    type?: 'image' | 'video';
  }): void {
    const pendingItem: WalrusGalleryItem = {
      id: `pending-${photo.filename}`,
      blobId: `pending-${photo.filename}`,
      name: photo.filename,
      url: photo.localUrl,
      type: photo.type || 'image',
      mimeType: photo.type === 'video' ? 'video/mp4' : 'image/jpeg',
      timestamp: photo.timestamp || Date.now(),
      walletAddress: photo.walletAddress,
      provider: 'walrus',
      cameraId: photo.cameraId,
      encrypted: false,  // Local photos are not encrypted
      isPending: true,
      localUrl: photo.localUrl,
      jobId: photo.jobId,
      // Set decryptedUrl so Gallery can display it immediately
      decryptedUrl: photo.blob ? URL.createObjectURL(photo.blob) : photo.localUrl,
    };

    console.log(`📸 [Walrus] Adding pending photo: ${photo.filename}`);
    this.pendingPhotos.set(photo.filename, pendingItem);
    this.notifyGalleryUpdate();
  }

  /**
   * Remove a pending photo (called when upload completes or fails)
   */
  removePendingPhoto(filename: string): void {
    if (this.pendingPhotos.has(filename)) {
      const item = this.pendingPhotos.get(filename);
      // Revoke object URL if we created one
      if (item?.decryptedUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(item.decryptedUrl);
      }
      this.pendingPhotos.delete(filename);
      console.log(`📸 [Walrus] Removed pending photo: ${filename}`);
    }
  }

  /**
   * Get all pending photos
   */
  getPendingPhotos(): WalrusGalleryItem[] {
    return Array.from(this.pendingPhotos.values());
  }

  /**
   * Clear all pending photos (e.g., on logout)
   */
  clearPendingPhotos(): void {
    // Revoke all object URLs
    this.pendingPhotos.forEach(item => {
      if (item.decryptedUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(item.decryptedUrl);
      }
    });
    this.pendingPhotos.clear();
  }

  /**
   * Initialize WebSocket connection for real-time upload notifications
   */
  private initializeWebSocket() {
    import('socket.io-client').then(({ io }) => {
      this.socket = io(this.backendUrl, {
        transports: ['websocket', 'polling'],
        autoConnect: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      this.socket.on('connect', () => {
        console.log('📡 [Walrus] Connected to WebSocket for gallery updates');
      });

      // Listen for Walrus upload completion events
      this.socket.on('walrus:upload:complete', (data: any) => {
        console.log('📸 [Walrus] New media uploaded:', data);
        this.notifyGalleryUpdate();
      });

      this.socket.on('disconnect', () => {
        console.log('📡 [Walrus] Disconnected from WebSocket');
      });
    }).catch(error => {
      console.warn('[Walrus] Failed to initialize WebSocket:', error);
    });
  }

  /**
   * Subscribe to gallery update events
   */
  onGalleryUpdate(callback: () => void): () => void {
    this.galleryUpdateListeners.add(callback);
    return () => {
      this.galleryUpdateListeners.delete(callback);
    };
  }

  /**
   * Notify all listeners that gallery should refresh
   */
  private notifyGalleryUpdate() {
    this.galleryUpdateListeners.forEach(listener => {
      try {
        listener();
      } catch (error) {
        console.error('[Walrus] Error in gallery update listener:', error);
      }
    });
  }

  /**
   * Fetch user's files from Walrus storage via backend
   */
  async getUserFiles(walletAddress: string, includeShared: boolean = true): Promise<WalrusGalleryItem[]> {
    try {
      console.log(`📦 [Walrus] Fetching gallery for wallet: ${walletAddress.slice(0, 8)}...`);

      const url = `${this.backendUrl}/api/walrus/gallery/${walletAddress}?includeShared=${includeShared}`;
      const response = await fetch(url);

      if (!response.ok) {
        console.error('[Walrus] Failed to fetch gallery:', response.statusText);
        return [];
      }

      const data = await response.json();

      if (!data.success) {
        console.warn('[Walrus] Gallery request unsuccessful:', data.error);
        return [];
      }

      // Backend returns "items" array with all files (owned + shared)
      const allFiles = data.items || [];
      const ownedCount = data.ownedCount || 0;
      const sharedCount = data.sharedCount || 0;

      console.log(`✅ [Walrus] Found ${ownedCount} owned + ${sharedCount} shared files`);

      // Transform backend response to gallery items
      // Backend returns: id, blobId, url, type, cameraId, timestamp, encrypted, nonce, accessGrants
      const walrusFiles = allFiles.map((item: any): WalrusGalleryItem => {
        const isVideo = item.type === 'video' || item.fileType === 'video';

        return {
          id: item.blobId || item.id,
          blobId: item.blobId || item.id,
          name: `${(item.blobId || item.id).slice(0, 8)}_${item.type || 'photo'}`,
          url: item.url || item.downloadUrl,  // Direct Walrus aggregator URL
          type: isVideo ? 'video' : 'image',
          mimeType: isVideo ? 'video/mp4' : 'image/jpeg',
          timestamp: item.timestamp || Date.now(),
          backupUrls: [],
          walletAddress: item.walletAddress,
          provider: 'walrus',
          cameraId: item.cameraId,
          // Encryption metadata
          encrypted: item.encrypted !== false,  // Default to true
          nonce: item.nonce,
          originalSize: item.originalSize,
          encryptedSize: item.encryptedSize,
          accessGrants: item.accessGrants,
          suiOwner: item.suiOwner,
          metadata: {
            camera: item.cameraId,
          },
        };
      });

      // Include pending local photos (not yet uploaded to Walrus)
      // Filter to only include pending photos for this wallet
      const pendingForWallet = this.getPendingPhotos().filter(
        p => p.walletAddress === walletAddress
      );

      if (pendingForWallet.length > 0) {
        console.log(`📸 [Walrus] Including ${pendingForWallet.length} pending local photos`);
      }

      // Merge pending photos with Walrus files, sorted by timestamp (newest first)
      return [...pendingForWallet, ...walrusFiles].sort((a, b) => b.timestamp - a.timestamp);
    } catch (error) {
      console.error('[Walrus] Error fetching gallery:', error);
      // Still return pending photos even if Walrus fetch fails
      return this.getPendingPhotos();
    }
  }

  /**
   * Get a specific file's metadata and access key
   */
  async getFileWithAccessKey(
    blobId: string,
    walletAddress: string
  ): Promise<{ file: WalrusGalleryItem | null; encryptedKey: string | null }> {
    try {
      // First get file metadata
      const fileResponse = await fetch(`${this.backendUrl}/api/walrus/file/${blobId}`);

      if (!fileResponse.ok) {
        console.error('[Walrus] Failed to fetch file:', fileResponse.statusText);
        return { file: null, encryptedKey: null };
      }

      const fileData = await fileResponse.json();

      if (!fileData.success) {
        return { file: null, encryptedKey: null };
      }

      // Then get user's access key
      const accessResponse = await fetch(
        `${this.backendUrl}/api/walrus/access-key/${blobId}/${walletAddress}`
      );

      let encryptedKey: string | null = null;

      if (accessResponse.ok) {
        const accessData = await accessResponse.json();
        if (accessData.success && accessData.encryptedKey) {
          encryptedKey = accessData.encryptedKey;
        }
      }

      const item = fileData.file;
      const isVideo = item.fileType === 'video';

      const file: WalrusGalleryItem = {
        id: item.blobId,
        blobId: item.blobId,
        name: `${item.blobId.slice(0, 8)}_${item.fileType}`,
        url: item.downloadUrl,
        type: isVideo ? 'video' : 'image',
        mimeType: isVideo ? 'video/mp4' : 'image/jpeg',
        timestamp: item.timestamp || Date.now(),
        provider: 'walrus',
        cameraId: item.cameraId,
        encrypted: true,
        nonce: item.nonce,
        originalSize: item.originalSize,
        encryptedSize: item.encryptedSize,
        accessGrants: item.accessGrants,
        suiOwner: item.suiOwner,
        walletAddress: item.walletAddress,
      };

      return { file, encryptedKey };
    } catch (error) {
      console.error('[Walrus] Error fetching file with access key:', error);
      return { file: null, encryptedKey: null };
    }
  }

  /**
   * Get recent files with pagination
   */
  async getRecentFiles(
    walletAddress: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<WalrusGalleryItem[]> {
    const allFiles = await this.getUserFiles(walletAddress);

    return allFiles
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(offset, offset + limit);
  }

  /**
   * Search files by camera or other metadata
   */
  async searchFiles(
    walletAddress: string,
    query: string
  ): Promise<WalrusGalleryItem[]> {
    const allFiles = await this.getUserFiles(walletAddress);

    return allFiles.filter(file => {
      const cameraMatch = file.cameraId?.toLowerCase().includes(query.toLowerCase());
      const blobMatch = file.blobId.toLowerCase().includes(query.toLowerCase());
      return cameraMatch || blobMatch;
    });
  }
}

// Export singleton instance
export const walrusGalleryService = new WalrusGalleryService();
