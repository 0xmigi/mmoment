export interface IPFSMetadata {
  name: string;
  keyvalues: {
    walletAddress: string;
    timestamp: string;
    isDeleted: string;
    type: string;  // 'image' or 'video'
    mimeType: string;
    transactionId?: string;
    cameraId?: string;
  };
}

export interface IPFSProvider {
  name: string;
  gateway: string;
  uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video', metadata?: { transactionId?: string; cameraId?: string }): Promise<string>;
  getMediaForWallet(walletAddress: string): Promise<Array<IPFSMedia>>;
  deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean>;
  checkPinStatus(ipfsHash: string): Promise<boolean>;
  repin(ipfsHash: string): Promise<boolean>;
}

export interface IPFSMedia {
  id: string;
  url: string;
  type: 'image' | 'video';
  mimeType: string;
  walletAddress: string;
  timestamp: string;
  backupUrls: string[];
  provider: string;
  transactionId?: string;
  cameraId?: string;
  directUrl?: string;
}

export class IPFSService {
  private primaryProvider?: IPFSProvider;
  private backupProvider?: IPFSProvider;

  setPrimaryProvider(provider: IPFSProvider) {
    this.primaryProvider = provider;
  }

  setBackupProvider(provider: IPFSProvider) {
    this.backupProvider = provider;
  }

  async uploadFile(
    blob: Blob, 
    walletAddress: string, 
    type: 'image' | 'video', 
    options?: { directUrl?: string; transactionId?: string; cameraId?: string }
  ): Promise<IPFSMedia[]> {
    if (!this.primaryProvider) {
      throw new Error('No primary provider configured');
    }

    try {
      // Try primary provider first
      const url = await this.primaryProvider.uploadFile(blob, walletAddress, type, {
        transactionId: options?.transactionId,
        cameraId: options?.cameraId
      });
      const ipfsHash = url.split('/').pop()!;
      
      // Get backup URLs
      const backupUrls = this.getBackupUrls(ipfsHash);
      
      // Add direct URL as the first backup URL if provided (useful for videos)
      if (options?.directUrl && type === 'video') {
        if (!backupUrls.includes(options.directUrl)) {
          backupUrls.unshift(options.directUrl);
        }
      }
      
      // Determine MIME type more accurately for videos
      let mimeType = 'image/jpeg';
      if (type === 'video') {
        // Check the actual blob type first
        if (blob.type && blob.type.includes('video')) {
          mimeType = blob.type;
        } else if (blob.type && blob.type.includes('quicktime')) {
          mimeType = 'video/quicktime';
        } else {
          // Default to mp4 for unknown video types
          mimeType = 'video/mp4';
        }
      }
      
      return [{
        id: ipfsHash,
        url,
        type,
        mimeType,
        walletAddress,
        timestamp: new Date().toISOString(),
        backupUrls,
        provider: this.primaryProvider.name,
        directUrl: options?.directUrl,
        transactionId: options?.transactionId,
        cameraId: options?.cameraId
      }];
    } catch (error) {
      console.error(`Failed to upload to primary provider:`, error);
      
      // If primary fails and we have a backup, try that
      if (this.backupProvider) {
        try {
          const url = await this.backupProvider.uploadFile(blob, walletAddress, type, {
            transactionId: options?.transactionId,
            cameraId: options?.cameraId
          });
          const ipfsHash = url.split('/').pop()!;
          
          // Get backup URLs
          const backupUrls = this.getBackupUrls(ipfsHash);
          
          // Add direct URL as the first backup URL if provided (useful for videos)
          if (options?.directUrl && type === 'video') {
            if (!backupUrls.includes(options.directUrl)) {
              backupUrls.unshift(options.directUrl);
            }
          }
          
          // Determine MIME type more accurately for videos
          let mimeType = 'image/jpeg';
          if (type === 'video') {
            // Check the actual blob type first
            if (blob.type && blob.type.includes('video')) {
              mimeType = blob.type;
            } else if (blob.type && blob.type.includes('quicktime')) {
              mimeType = 'video/quicktime';
            } else {
              // Default to mp4 for unknown video types
              mimeType = 'video/mp4';
            }
          }
          
          return [{
            id: ipfsHash,
            url,
            type,
            mimeType,
            walletAddress,
            timestamp: new Date().toISOString(),
            backupUrls,
            provider: this.backupProvider.name,
            directUrl: options?.directUrl,
            transactionId: options?.transactionId,
            cameraId: options?.cameraId
          }];
        } catch (backupError) {
          console.error(`Failed to upload to backup provider:`, backupError);
        }
      }
      
      return [];
    }
  }

  private getBackupUrls(ipfsHash: string): string[] {
    const urls = [];
    if (this.primaryProvider) {
      urls.push(`${this.primaryProvider.gateway}/ipfs/${ipfsHash}`);
    }
    if (this.backupProvider) {
      urls.push(`${this.backupProvider.gateway}/ipfs/${ipfsHash}`);
    }
    urls.push(
      `https://ipfs.io/ipfs/${ipfsHash}`,
      `https://gateway.ipfs.io/ipfs/${ipfsHash}`
    );
    return urls;
  }

  async getMediaForWallet(walletAddress: string): Promise<IPFSMedia[]> {
    if (!this.primaryProvider) {
      throw new Error('No primary provider configured');
    }

    try {
      // Try to get media from primary provider
      return await this.primaryProvider.getMediaForWallet(walletAddress);
    } catch (error) {
      console.error(`Failed to fetch media from primary provider:`, error);
      
      // If primary fails and we have a backup, try that
      if (this.backupProvider) {
        try {
          return await this.backupProvider.getMediaForWallet(walletAddress);
        } catch (backupError) {
          console.error(`Failed to fetch media from backup provider:`, backupError);
        }
      }
      
      return [];
    }
  }

  async deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean> {
    if (!this.primaryProvider) {
      throw new Error('No primary provider configured');
    }

    try {
      // Try to delete from primary provider
      return await this.primaryProvider.deleteMedia(ipfsHash, walletAddress);
    } catch (error) {
      console.error(`Failed to delete from primary provider:`, error);
      
      // If primary fails and we have a backup, try that
      if (this.backupProvider) {
        try {
          return await this.backupProvider.deleteMedia(ipfsHash, walletAddress);
        } catch (backupError) {
          console.error(`Failed to delete from backup provider:`, backupError);
        }
      }
      
      return false;
    }
  }

  async checkAndRepinMedia(ipfsHash: string): Promise<void> {
    if (!this.primaryProvider || !this.backupProvider) return;

    try {
      // Check if the media is pinned in primary provider
      const isPinnedInPrimary = await this.primaryProvider.checkPinStatus(ipfsHash);
      
      if (!isPinnedInPrimary) {
        // Check if it's available in backup
        const isPinnedInBackup = await this.backupProvider.checkPinStatus(ipfsHash);
        
        if (isPinnedInBackup) {
          // Try to repin to primary from backup
          await this.primaryProvider.repin(ipfsHash);
        }
      }
    } catch (error) {
      console.error(`Failed to check/repin media:`, error);
    }
  }
} 