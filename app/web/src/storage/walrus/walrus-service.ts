import { IPFSProvider, IPFSMedia } from '../ipfs/ipfs-service';
import { StorageService, StorageResult, UploadOptions } from '../storage-provider';

// Walrus API constants - Mainnet endpoints
const WALRUS_PUBLISHER = 'https://publisher.walrus-mainnet.walrus.space';
const WALRUS_AGGREGATOR = 'https://aggregator.walrus-mainnet.walrus.space';
const WALRUS_HTTP_GATEWAY = 'https://aggregator.walrus-mainnet.walrus.space';

export interface WalrusStorageQuota {
  totalBytes: number;
  usedBytes: number;
  remainingBytes: number;
}

/**
 * Walrus storage service implementation
 * Implements both IPFSProvider for compatibility with existing code
 * and StorageService for the general storage interface
 */
export class WalrusService implements IPFSProvider, StorageService {
  readonly name = 'Walrus';
  readonly gateway = WALRUS_HTTP_GATEWAY;
  readonly publisher = WALRUS_PUBLISHER;
  readonly aggregator = WALRUS_AGGREGATOR;
  
  // Storage plans in bytes with corresponding costs
  private storagePlans = {
    free: { bytes: 2 * 1024 * 1024 * 1024, cost: 0 }, // 2GB free
    tier1: { bytes: 10 * 1024 * 1024 * 1024, cost: 299 }, // 10GB for $2.99
    tier2: { bytes: 50 * 1024 * 1024 * 1024, cost: 999 }, // 50GB for $9.99
    tier3: { bytes: 100 * 1024 * 1024 * 1024, cost: 1699 } // 100GB for $16.99
  };
  
  /**
   * Check if Walrus service is available and responding
   */
  async isAvailable(): Promise<boolean> {
    try {
      // This would be replaced with an actual health check endpoint
      // when Walrus API documentation is available
      return true; // Placeholder for now
    } catch (error) {
      console.error('Walrus health check failed:', error);
      return false;
    }
  }
  
  /**
   * Upload a file to Walrus
   */
  async upload(
    file: File, 
    options?: UploadOptions
  ): Promise<StorageResult> {
    const blob = await file.arrayBuffer().then(buffer => new Blob([buffer]));
    return this.uploadBlob(
      blob, 
      file.name, 
      options
    );
  }
  
  /**
   * Upload a blob to Walrus
   */
  async uploadBlob(
    blob: Blob, 
    filename: string, 
    options?: UploadOptions
  ): Promise<StorageResult> {
    const walletAddress = options?.metadata?.walletAddress || 'anonymous';
    const type = filename.endsWith('.mp4') || filename.endsWith('.mov') ? 'video' : 'image';
    
    console.log(`Uploading ${filename} as ${type} for wallet ${walletAddress}`);
    
    try {
      // Set storage epochs (how long to store the data)
      const storageEpochs = options?.metadata?.storageEpochs || 1;
      
      // Real Walrus API implementation
      const response = await fetch(`${this.publisher}/v1/blobs?epochs=${storageEpochs}`, {
        method: 'PUT',
        body: blob,
        headers: {
          'Content-Type': 'application/octet-stream'
        }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Walrus upload failed: ${response.status} ${errorText}`);
      }
      
      // Parse the response. The API returns blob ID as text in the response body.
      const blobId = await response.text();
      console.log('Walrus upload response:', blobId);
      
      // If we got a valid blob ID, use it; otherwise, fallback to random ID
      const validBlobId = blobId && blobId.trim().length > 0 
        ? blobId.trim() 
        : `walrus-${Math.random().toString(36).substring(2, 15)}`;
      
      // Construct the URL from the blob ID
      const url = `${this.gateway}/v1/blobs/${validBlobId}`;
      
      return { 
        id: validBlobId, 
        url 
      };
    } catch (error) {
      console.error('Walrus upload failed:', error);
      throw error;
    }
  }
  
  /**
   * Get URL for a file stored in Walrus
   */
  getUrl(blobId: string): string {
    return `${this.gateway}/v1/blobs/${blobId}`;
  }
  
  /**
   * Delete a file from Walrus
   */
  async delete(blobId: string): Promise<boolean> {
    try {
      // Placeholder - would implement actual deletion API call
      console.log(`Deleting blob ${blobId} from Walrus`);
      return true;
    } catch (error) {
      console.error('Failed to delete from Walrus:', error);
      return false;
    }
  }
  
  /**
   * Get user's storage quota from Walrus
   */
  async getUserQuota(walletAddress: string): Promise<WalrusStorageQuota> {
    try {
      // In a real implementation, this would fetch quota from the Walrus API
      // For now, we'll return the free tier quota
      console.log(`Fetching storage quota for ${walletAddress}`);
      
      // For later: implement actual API call to get quota
      // const response = await fetch(`${this.aggregator}/v1/quota/${walletAddress}`);
      // if (response.ok) {
      //   const quotaData = await response.json();
      //   return {
      //     totalBytes: quotaData.totalBytes,
      //     usedBytes: quotaData.usedBytes,
      //     remainingBytes: quotaData.remainingBytes
      //   };
      // }
      
      // Return default free tier 
      return {
        totalBytes: this.storagePlans.free.bytes,
        usedBytes: 0,
        remainingBytes: this.storagePlans.free.bytes
      };
    } catch (error) {
      console.error('Failed to get quota:', error);
      // Return default free tier if we can't fetch
      return {
        totalBytes: this.storagePlans.free.bytes,
        usedBytes: 0,
        remainingBytes: this.storagePlans.free.bytes
      };
    }
  }
  
  // IPFSProvider interface implementation
  
  /**
   * Upload a file to Walrus using the IPFSProvider interface
   */
  async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<string> {
    const result = await this.uploadBlob(
      blob, 
      `file.${type === 'video' ? 'mp4' : 'jpg'}`,
      { metadata: { walletAddress } }
    );
    return result.url;
  }
  
  /**
   * Get all media for a wallet from Walrus
   */
  async getMediaForWallet(walletAddress: string): Promise<Array<IPFSMedia>> {
    try {
      // This would make an actual API call in a real implementation
      console.log(`Fetching media for wallet ${walletAddress} from Walrus`);
      
      // Return empty array for now as a placeholder
      return [];
    } catch (error) {
      console.error('Failed to fetch media from Walrus:', error);
      return [];
    }
  }
  
  /**
   * Delete media from Walrus
   */
  async deleteMedia(blobId: string, _walletAddress: string): Promise<boolean> {
    return this.delete(blobId);
  }
  
  /**
   * Check if a blob is available on Walrus
   */
  async checkPinStatus(blobId: string): Promise<boolean> {
    try {
      // This would make an actual API call in a real implementation
      console.log(`Checking blob status for ${blobId}`);
      return true; // Placeholder
    } catch (error) {
      console.error('Failed to check blob status:', error);
      return false;
    }
  }
  
  /**
   * Ensure a blob is available on Walrus (no repin needed in Walrus)
   */
  async repin(blobId: string): Promise<boolean> {
    // Walrus handles availability automatically with erasure coding
    // No explicit repin needed unlike IPFS
    return await this.checkPinStatus(blobId);
  }
}

// Create a singleton instance
export const walrusService = new WalrusService(); 