import { IPFSProvider, IPFSMedia } from './ipfs-service';
import { CONFIG } from '../../core/config';

export class FilebaseService implements IPFSProvider {
  readonly name = 'Filebase';
  readonly gateway = 'https://ipfs.filebase.io';

  async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<string> {
    try {
      const formData = new FormData();
      formData.append('file', blob);
      formData.append('walletAddress', walletAddress);
      formData.append('type', type);

      const response = await fetch(`${CONFIG.BACKEND_URL}/api/media/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const data = await response.json();
      return data.url;
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  }

  async getMediaForWallet(walletAddress: string): Promise<Array<IPFSMedia>> {
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/media/${walletAddress}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch media');
      }

      const data = await response.json();
      return data.media;
    } catch (error) {
      console.error('Failed to fetch media:', error);
      return [];
    }
  }

  async deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean> {
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/media/${ipfsHash}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ walletAddress })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to delete media');
      }

      return true;
    } catch (error) {
      console.error('Failed to delete media:', error);
      return false;
    }
  }

  async checkPinStatus(ipfsHash: string): Promise<boolean> {
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/media/${ipfsHash}/status`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to check pin status');
      }

      const data = await response.json();
      return data.isPinned;
    } catch (error) {
      console.error('Failed to check pin status:', error);
      return false;
    }
  }

  async repin(ipfsHash: string): Promise<boolean> {
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/media/${ipfsHash}/repin`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to repin');
      }

      return true;
    } catch (error) {
      console.error('Failed to repin:', error);
      return false;
    }
  }
}

export const filebaseService = new FilebaseService(); 