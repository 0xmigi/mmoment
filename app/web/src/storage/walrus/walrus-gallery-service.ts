/**
 * Walrus Gallery Service
 *
 * Manages a unified media cache where items track their upload status via job_id.
 * Uses Jetson's SQLite upload queue as source of truth for upload status.
 */

import { CONFIG } from '../../core/config';
import { unifiedCameraService } from '../../camera/unified-camera-service';

export interface WalrusAccessGrant {
  pubkey: string;
  encryptedKey: string;  // base64-encoded encrypted AES key
}

export interface WalrusGalleryItem {
  id: string;              // Unique identifier (blobId when backed up, job-{jobId} when pending)
  blobId: string;          // Walrus blob ID (empty when pending)
  name: string;
  url: string;             // Current source URL (local when pending, Walrus when backed up)
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
  // Upload tracking (from Jetson's SQLite queue)
  backedUp: boolean;       // true = stored in Walrus, false = local only
  jobId?: number;          // Upload job ID from Jetson
  localUrl?: string;       // Original local camera URL (preserved for display)
}

class WalrusGalleryService {
  private backendUrl: string;
  private galleryUpdateListeners: Set<() => void> = new Set();
  private socket: any = null;

  // Unified media cache keyed by jobId (for pending) or blobId (for backed up)
  private mediaCache: Map<string, WalrusGalleryItem> = new Map();

  // Active polling intervals for pending uploads
  private pollIntervals: Map<number, NodeJS.Timeout> = new Map();

  constructor() {
    this.backendUrl = CONFIG.BACKEND_URL;
    this.initializeWebSocket();
  }

  /**
   * Add a local photo from capture response.
   * Uses jobId from Jetson's upload queue for tracking.
   * If jobId is not provided, creates a temporary local-only item.
   */
  addLocalPhoto(photo: {
    filename: string;
    localUrl: string;
    blob?: Blob;
    jobId?: number;  // Optional - if not provided, won't track upload status
    walletAddress: string;
    cameraId?: string;
    timestamp?: number;
    type?: 'image' | 'video';
  }): void {
    // Use jobId if available, otherwise generate a temporary ID
    const hasJobId = photo.jobId !== undefined && photo.jobId !== null;
    const cacheKey = hasJobId ? `job-${photo.jobId}` : `local-${Date.now()}-${photo.filename}`;

    const item: WalrusGalleryItem = {
      id: cacheKey,
      blobId: '',  // Will be filled when upload completes
      name: photo.filename,
      url: photo.localUrl,
      type: photo.type || 'image',
      mimeType: photo.type === 'video' ? 'video/mp4' : 'image/jpeg',
      timestamp: photo.timestamp || Date.now(),
      walletAddress: photo.walletAddress,
      provider: 'walrus',
      cameraId: photo.cameraId,
      encrypted: false,  // Local photos not encrypted yet
      backedUp: false,
      jobId: photo.jobId,
      localUrl: photo.localUrl,
      decryptedUrl: photo.blob ? URL.createObjectURL(photo.blob) : photo.localUrl,
    };

    console.log(`📸 [Walrus] Adding local ${photo.type || 'photo'}: ${photo.filename}${hasJobId ? ` (job #${photo.jobId})` : ' (no job tracking)'}`);
    this.mediaCache.set(cacheKey, item);
    this.notifyGalleryUpdate();

    // Start polling for upload completion only if we have a jobId
    if (hasJobId && photo.jobId !== undefined) {
      this.startPollingJobStatus(photo.jobId, photo.cameraId);
    }
  }

  /**
   * Poll Jetson's upload queue to check when job completes
   */
  private startPollingJobStatus(jobId: number, cameraId?: string): void {
    // Don't create duplicate polling
    if (this.pollIntervals.has(jobId)) return;

    const pollInterval = setInterval(async () => {
      try {
        const cameraApiUrl = cameraId ? unifiedCameraService.getCameraApiUrl(cameraId) : null;
        if (!cameraApiUrl) {
          console.warn(`[Walrus] No camera API URL for job #${jobId}`);
          return;
        }

        const response = await fetch(`${cameraApiUrl}/api/upload-status/${jobId}`);
        if (!response.ok) return;

        const data = await response.json();
        if (!data.success) return;

        console.log(`📸 [Walrus] Job #${jobId} status: ${data.status}`);

        if (data.status === 'completed' && data.blob_id) {
          // Upload complete! Update the item
          this.markJobAsBackedUp(jobId, {
            blobId: data.blob_id,
            downloadUrl: data.download_url,
          });

          // Stop polling
          this.stopPollingJobStatus(jobId);
        } else if (data.status === 'failed') {
          console.error(`[Walrus] Job #${jobId} failed: ${data.error}`);
          // Keep local URL, stop polling
          this.stopPollingJobStatus(jobId);
        }
      } catch (error) {
        console.error(`[Walrus] Error polling job #${jobId}:`, error);
      }
    }, 3000); // Poll every 3 seconds

    this.pollIntervals.set(jobId, pollInterval);
    console.log(`📸 [Walrus] Started polling for job #${jobId}`);
  }

  private stopPollingJobStatus(jobId: number): void {
    const interval = this.pollIntervals.get(jobId);
    if (interval) {
      clearInterval(interval);
      this.pollIntervals.delete(jobId);
      console.log(`📸 [Walrus] Stopped polling for job #${jobId}`);
    }
  }

  /**
   * Update item when upload completes (via polling or WebSocket)
   */
  private markJobAsBackedUp(jobId: number, data: { blobId: string; downloadUrl: string }): void {
    const cacheKey = `job-${jobId}`;
    const item = this.mediaCache.get(cacheKey);

    if (!item) {
      console.warn(`[Walrus] No cached item for job #${jobId}`);
      return;
    }

    // Update item in place - keep using local URL for display
    // The Walrus URL is stored but we prefer local/decrypted for playback
    const updatedItem: WalrusGalleryItem = {
      ...item,
      id: data.blobId,
      blobId: data.blobId,
      // Keep the local URL for display - prefer decryptedUrl > localUrl > walrus URL
      url: item.decryptedUrl || item.localUrl || data.downloadUrl,
      encrypted: true,
      backedUp: true,
      // Store Walrus URL as backup
      backupUrls: [data.downloadUrl, ...(item.backupUrls || [])],
    };

    // Re-key by blobId instead of jobId
    this.mediaCache.delete(cacheKey);
    this.mediaCache.set(data.blobId, updatedItem);

    console.log(`✅ [Walrus] Job #${jobId} backed up: ${data.blobId.slice(0, 12)}...`);
    this.notifyGalleryUpdate();
  }

  /**
   * Clear local items and stop all polling
   */
  clearLocalItems(): void {
    // Stop all polling
    for (const [, interval] of this.pollIntervals.entries()) {
      clearInterval(interval);
    }
    this.pollIntervals.clear();

    // Clear non-backed-up items
    for (const [key, item] of this.mediaCache.entries()) {
      if (!item.backedUp) {
        if (item.decryptedUrl?.startsWith('blob:')) {
          URL.revokeObjectURL(item.decryptedUrl);
        }
        this.mediaCache.delete(key);
      }
    }
  }

  /**
   * Initialize WebSocket for real-time upload notifications
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

      // Listen for Walrus upload completion events from backend
      this.socket.on('walrus:upload:complete', (data: any) => {
        console.log('📸 [Walrus] Upload complete notification:', data);

        // Try to match by timestamp + cameraId if we don't have jobId
        // This handles cases where we missed the initial capture
        if (data.blobId && data.downloadUrl) {
          // Check if we already have this item
          if (!this.mediaCache.has(data.blobId)) {
            // Look for matching pending item by timestamp
            for (const [, item] of this.mediaCache.entries()) {
              if (!item.backedUp && item.cameraId === data.cameraId) {
                const timeDiff = Math.abs(item.timestamp - (data.timestamp || 0));
                if (timeDiff < 5000) {  // 5 second tolerance
                  this.markJobAsBackedUp(item.jobId!, {
                    blobId: data.blobId,
                    downloadUrl: data.downloadUrl,
                  });
                  break;
                }
              }
            }
          }
        }

        this.notifyGalleryUpdate();
      });

      this.socket.on('disconnect', () => {
        console.log('📡 [Walrus] Disconnected from WebSocket');
      });
    }).catch(error => {
      console.warn('[Walrus] Failed to initialize WebSocket:', error);
    });
  }

  onGalleryUpdate(callback: () => void): () => void {
    this.galleryUpdateListeners.add(callback);
    return () => {
      this.galleryUpdateListeners.delete(callback);
    };
  }

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
   * Fetch user's files from backend and merge with local cache
   */
  async getUserFiles(walletAddress: string, includeShared: boolean = true): Promise<WalrusGalleryItem[]> {
    try {
      console.log(`📦 [Walrus] Fetching gallery for wallet: ${walletAddress.slice(0, 8)}...`);

      const url = `${this.backendUrl}/api/walrus/gallery/${walletAddress}?includeShared=${includeShared}`;
      const response = await fetch(url);

      if (!response.ok) {
        console.error('[Walrus] Failed to fetch gallery:', response.statusText);
        return this.getCachedItemsForWallet(walletAddress);
      }

      const data = await response.json();

      if (!data.success) {
        console.warn('[Walrus] Gallery request unsuccessful:', data.error);
        return this.getCachedItemsForWallet(walletAddress);
      }

      const allFiles = data.items || [];
      console.log(`✅ [Walrus] Found ${data.ownedCount || 0} owned + ${data.sharedCount || 0} shared files`);

      // Process backend files
      for (const item of allFiles) {
        const isVideo = item.type === 'video' || item.fileType === 'video';
        const blobId = item.blobId || item.id;

        // Check if we already have this item (possibly from local capture)
        const existingByBlob = this.mediaCache.get(blobId);

        // Also check if we have a pending item for the same timestamp/camera
        let matchingPendingKey: string | null = null;
        for (const [key, cached] of this.mediaCache.entries()) {
          if (!cached.backedUp && cached.cameraId === item.cameraId) {
            const timeDiff = Math.abs(cached.timestamp - (item.timestamp || 0));
            if (timeDiff < 5000) {
              matchingPendingKey = key;
              break;
            }
          }
        }

        if (matchingPendingKey) {
          // Upgrade pending item to backed up - keep local URL for display
          const pending = this.mediaCache.get(matchingPendingKey)!;
          const walrusUrl = item.url || item.downloadUrl;
          const upgraded: WalrusGalleryItem = {
            ...pending,
            id: blobId,
            blobId: blobId,
            // Prefer local URL for display, Walrus URL as backup
            url: pending.decryptedUrl || pending.localUrl || walrusUrl,
            encrypted: item.encrypted !== false,
            backedUp: true,
            nonce: item.nonce,
            suiOwner: item.suiOwner,
            accessGrants: item.accessGrants,
            backupUrls: [walrusUrl, ...(pending.backupUrls || [])],
          };
          this.mediaCache.delete(matchingPendingKey);
          this.mediaCache.set(blobId, upgraded);
          // Stop polling if still active
          if (pending.jobId) this.stopPollingJobStatus(pending.jobId);
          console.log(`📸 [Walrus] Upgraded pending item to backed-up: ${pending.name}`);
        } else if (!existingByBlob) {
          // New item from backend
          const newItem: WalrusGalleryItem = {
            id: blobId,
            blobId: blobId,
            name: `${blobId.slice(0, 8)}_${item.type || 'photo'}`,
            url: item.url || item.downloadUrl,
            type: isVideo ? 'video' : 'image',
            mimeType: isVideo ? 'video/mp4' : 'image/jpeg',
            timestamp: item.timestamp || Date.now(),
            backupUrls: [],
            walletAddress: item.walletAddress || walletAddress,
            provider: 'walrus',
            cameraId: item.cameraId,
            encrypted: item.encrypted !== false,
            nonce: item.nonce,
            originalSize: item.originalSize,
            encryptedSize: item.encryptedSize,
            accessGrants: item.accessGrants,
            suiOwner: item.suiOwner,
            backedUp: true,
            metadata: { camera: item.cameraId },
          };
          this.mediaCache.set(blobId, newItem);
        }
      }

      return this.getCachedItemsForWallet(walletAddress);
    } catch (error) {
      console.error('[Walrus] Error fetching gallery:', error);
      return this.getCachedItemsForWallet(walletAddress);
    }
  }

  private getCachedItemsForWallet(walletAddress: string): WalrusGalleryItem[] {
    return Array.from(this.mediaCache.values())
      .filter(item => item.walletAddress === walletAddress)
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  async getFileWithAccessKey(
    blobId: string,
    walletAddress: string
  ): Promise<{ file: WalrusGalleryItem | null; encryptedKey: string | null }> {
    try {
      const fileResponse = await fetch(`${this.backendUrl}/api/walrus/file/${blobId}`);
      if (!fileResponse.ok) return { file: null, encryptedKey: null };

      const fileData = await fileResponse.json();
      if (!fileData.success) return { file: null, encryptedKey: null };

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
        backedUp: true,
      };

      return { file, encryptedKey };
    } catch (error) {
      console.error('[Walrus] Error fetching file with access key:', error);
      return { file: null, encryptedKey: null };
    }
  }

  async getRecentFiles(
    walletAddress: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<WalrusGalleryItem[]> {
    const allFiles = await this.getUserFiles(walletAddress);
    return allFiles.slice(offset, offset + limit);
  }

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

  /**
   * Delete a file from Walrus gallery
   * This removes it from the backend database (soft delete)
   * The actual blob may remain on Walrus until its epochs expire
   */
  async deleteFile(blobId: string, walletAddress: string): Promise<boolean> {
    try {
      console.log(`🗑️ [Walrus] Deleting blob ${blobId.slice(0, 12)}... for wallet ${walletAddress.slice(0, 8)}...`);

      const response = await fetch(`${this.backendUrl}/api/walrus/delete/${blobId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ walletAddress }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('[Walrus] Delete failed:', errorData.error || response.statusText);
        return false;
      }

      const data = await response.json();

      if (data.success) {
        // Remove from local cache
        const item = this.mediaCache.get(blobId);
        if (item) {
          // Revoke decrypted URL if exists
          if (item.decryptedUrl?.startsWith('blob:')) {
            URL.revokeObjectURL(item.decryptedUrl);
          }
          this.mediaCache.delete(blobId);
        }

        console.log(`✅ [Walrus] Blob ${blobId.slice(0, 12)}... deleted successfully`);
        this.notifyGalleryUpdate();
        return true;
      }

      return false;
    } catch (error) {
      console.error('[Walrus] Error deleting file:', error);
      return false;
    }
  }

  // Legacy compatibility
  addPendingPhoto = this.addLocalPhoto.bind(this);
  removePendingPhoto(_filename: string): void {
    console.log('[Walrus] removePendingPhoto deprecated - items managed via jobId');
  }
  getPendingPhotos(): WalrusGalleryItem[] {
    return Array.from(this.mediaCache.values()).filter(item => !item.backedUp);
  }
  clearPendingPhotos = this.clearLocalItems.bind(this);
}

export const walrusGalleryService = new WalrusGalleryService();
