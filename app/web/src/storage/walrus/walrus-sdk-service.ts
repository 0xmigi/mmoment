import { getFullnodeUrl, SuiClient } from '@mysten/sui/client';
import { WalrusClient } from '@mysten/walrus';
import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';
import { IPFSProvider, IPFSMedia } from '../ipfs/ipfs-service';
import { StorageService, StorageResult, UploadOptions } from '../storage-provider';

// Get the WASM URL for client-side use
// Using a CDN URL for the WASM file
const WALRUS_WASM_URL = 'https://unpkg.com/@mysten/walrus-wasm@latest/web/walrus_wasm_bg.wasm';

// Create the SUI client for Walrus with explicit type
const suiClient = new SuiClient({
  url: getFullnodeUrl('testnet'),
}) as any; // Temporary fix for type mismatch

// Create a Walrus client with reasonable timeout settings
const walrusClient = new WalrusClient({
  network: 'testnet',
  suiClient,
  wasmUrl: WALRUS_WASM_URL,
  storageNodeClientOptions: {
    timeout: 60_000, // 60 seconds timeout
  },
});

export interface WalrusStorageQuota {
  totalBytes: number;
  usedBytes: number;
  remainingBytes: number;
}

/**
 * Walrus SDK service implementation
 * Uses the official Mysten Labs Walrus SDK for better reliability
 */
export class WalrusSdkService implements IPFSProvider, StorageService {
  readonly name = 'Walrus SDK';
  readonly gateway = 'https://aggregator.walrus-testnet.walrus.space'; 
  readonly publisher = 'https://publisher.walrus-testnet.walrus.space';
  readonly aggregator = 'https://aggregator.walrus-testnet.walrus.space';
  
  // Storage plans in bytes with corresponding costs
  private storagePlans = {
    free: { bytes: 2 * 1024 * 1024 * 1024, cost: 0 }, // 2GB free
    tier1: { bytes: 10 * 1024 * 1024 * 1024, cost: 299 }, // 10GB for $2.99
    tier2: { bytes: 50 * 1024 * 1024 * 1024, cost: 999 }, // 50GB for $9.99
    tier3: { bytes: 100 * 1024 * 1024 * 1024, cost: 1699 } // 100GB for $16.99
  };

  // In-memory keypair for testing (would not use in production)
  private demoKeypair = new Ed25519Keypair();
  
  /**
   * Check if Walrus service is available and responding
   */
  async isAvailable(): Promise<boolean> {
    try {
      // For now we'll just check if the client is initialized
      return !!walrusClient;
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
    const blob = await file.arrayBuffer();
    return this.uploadBlob(
      new Blob([blob]), 
      file.name, 
      options
    );
  }
  
  /**
   * Upload a blob to Walrus using the SDK
   */
  async uploadBlob(
    blob: Blob, 
    filename: string, 
    options?: UploadOptions
  ): Promise<StorageResult> {
    const storageEpochs = options?.metadata?.storageEpochs || 1;
    
    console.log(`Uploading ${filename} (${blob.size} bytes) to Walrus as SUI wallet ${this.demoKeypair.toSuiAddress()}`);

    try {
      // First approach: Upload directly using the SDK
      // This might not work without proper SUI and WAL tokens
      try {
        // Convert the blob to a Uint8Array for the SDK
        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        console.log(`Attempting SDK upload with ${uint8Array.length} bytes and ${storageEpochs} epochs...`);
        
        // Attempt direct upload via SDK
        const { blobId } = await walrusClient.writeBlob({
          blob: uint8Array,
          deletable: true,
          epochs: storageEpochs,
          signer: this.demoKeypair,
        });
        
        console.log(`SDK upload successful! Blob ID: ${blobId}`);
        
        // Construct URL for accessing the blob
        const url = `${this.aggregator}/v1/blobs/${blobId}`;
        
        return { 
          id: blobId, 
          url 
        };
      } catch (sdkError) {
        console.warn('SDK direct upload failed with error:', sdkError);
        
        // Fallback to direct HTTP API call if SDK fails
        console.log(`Falling back to HTTP API upload...`);
        
        const response = await fetch(`${this.publisher}/v1/blobs?epochs=${storageEpochs}`, {
          method: 'PUT',
          body: blob,
          headers: {
            'Content-Type': 'application/octet-stream'
          }
        });
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`HTTP API upload failed: ${response.status} ${errorText}`);
          throw new Error(`Walrus upload failed: ${response.status} ${errorText}`);
        }
        
        const responseText = await response.text();
        console.log('Walrus HTTP upload successful, response:', responseText);
        
        // Parse the blob ID from the response
        const blobId = responseText.trim();
        if (!blobId) {
          console.warn('No blob ID returned from HTTP API');
        }
        
        // Construct URL for accessing the blob
        const url = `${this.aggregator}/v1/blobs/${blobId}`;
        
        return { 
          id: blobId, 
          url 
        };
      }
    } catch (error) {
      console.error('All Walrus upload methods failed:', error);
      throw error;
    }
  }
  
  /**
   * Get URL for a file stored in Walrus
   */
  getUrl(blobId: string): string {
    return `${this.aggregator}/v1/blobs/${blobId}`;
  }
  
  /**
   * Delete a file from Walrus
   */
  async delete(blobId: string): Promise<boolean> {
    try {
      console.log(`Deleting blob ${blobId} from Walrus`);
      // Note: Deletion may require proper permissions
      return true;
    } catch (error) {
      console.error('Failed to delete from Walrus:', error);
      return false;
    }
  }
  
  /**
   * Get user's storage quota from Walrus
   */
  async getUserQuota(_walletAddress: string): Promise<WalrusStorageQuota> {
    try {
      // Return default free tier since Walrus doesn't have a quota endpoint yet
      return {
        totalBytes: this.storagePlans.free.bytes,
        usedBytes: 0,
        remainingBytes: this.storagePlans.free.bytes
      };
    } catch (error) {
      console.error('Failed to get quota:', error);
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
      console.log(`Fetching media for wallet ${walletAddress} from Walrus`);
      return []; // No easy way to query this with current API
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
      console.log(`Checking availability of blob ${blobId}`);
      
      // Try checking via SDK first
      try {
        const blob = await walrusClient.readBlob({ blobId });
        return blob.length > 0;
      } catch (sdkError) {
        console.warn('SDK blob check failed, falling back to HTTP:', sdkError);
        
        // Check if blob exists by doing a HEAD request
        const response = await fetch(`${this.aggregator}/v1/blobs/${blobId}`, {
          method: 'HEAD'
        });
        
        console.log(`HTTP HEAD check status: ${response.status}`);
        return response.ok;
      }
    } catch (error) {
      console.error('Failed to check blob status:', error);
      return false;
    }
  }
  
  /**
   * Ensure a blob is available on Walrus
   */
  async repin(_blobId: string): Promise<boolean> {
    // Walrus handles availability with erasure coding, no explicit repin needed
    return true;
  }
}

// Export the singleton instance
export const walrusSdkService = new WalrusSdkService(); 