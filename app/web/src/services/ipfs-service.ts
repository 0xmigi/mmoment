export interface IPFSMetadata {
  name: string;
  keyvalues: {
    walletAddress: string;
    timestamp: string;
    isDeleted: string;
    type: string;  // 'image' or 'video'
    mimeType: string;
  };
}

export interface IPFSProvider {
  name: string;
  gateway: string;
  uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<string>;
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

  async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<IPFSMedia[]> {
    if (!this.primaryProvider) {
      throw new Error('No primary provider configured');
    }

    try {
      // Try primary provider first
      const url = await this.primaryProvider.uploadFile(blob, walletAddress, type);
      const ipfsHash = url.split('/').pop()!;
      
      return [{
        id: ipfsHash,
        url,
        type,
        mimeType: type === 'video' ? 'video/mp4' : 'image/jpeg',
        walletAddress,
        timestamp: new Date().toISOString(),
        backupUrls: this.getBackupUrls(ipfsHash),
        provider: this.primaryProvider.name
      }];
    } catch (error) {
      console.error(`Failed to upload to primary provider:`, error);
      
      // If primary fails and we have a backup, try that
      if (this.backupProvider) {
        try {
          const url = await this.backupProvider.uploadFile(blob, walletAddress, type);
          const ipfsHash = url.split('/').pop()!;
          
          return [{
            id: ipfsHash,
            url,
            type,
            mimeType: type === 'video' ? 'video/mp4' : 'image/jpeg',
            walletAddress,
            timestamp: new Date().toISOString(),
            backupUrls: this.getBackupUrls(ipfsHash),
            provider: this.backupProvider.name
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
      `https://cloudflare-ipfs.com/ipfs/${ipfsHash}`,
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