import axios from 'axios';
import { IPFSProvider, IPFSMedia, IPFSMetadata } from './ipfs-service';

const PINATA_JWT = import.meta.env.VITE_PINATA_JWT;

const pinataApi = axios.create({
  headers: {
    'Authorization': `Bearer ${PINATA_JWT}`,
    'Content-Type': 'application/json'
  }
});

interface PinataResponse {
  rows: Array<{
    ipfs_pin_hash: string;
    metadata: IPFSMetadata;
    date_pinned: string;
  }>;
}

export class PinataService implements IPFSProvider {
  readonly name = 'Pinata';
  readonly gateway = 'https://gateway.pinata.cloud';

  async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<string> {
    if (!PINATA_JWT) {
      throw new Error('Pinata JWT is not configured');
    }

    try {
      const formData = new FormData();
      const filename = `${walletAddress}_${Date.now()}.${type === 'video' ? 'mp4' : 'jpg'}`;
      formData.append('file', blob, filename);

      const metadata: IPFSMetadata = {
        name: filename,
        keyvalues: {
          walletAddress: walletAddress,
          timestamp: Date.now().toString(),
          isDeleted: 'false',
          type: type,
          mimeType: type === 'video' ? 'video/mp4' : 'image/jpeg'
        }
      };
      formData.append('pinataMetadata', JSON.stringify(metadata));
      
      const res = await pinataApi.post(
        "https://api.pinata.cloud/pinning/pinFileToIPFS",
        formData,
        {
          headers: {
            'Content-Type': `multipart/form-data`,
          }
        }
      );
  
      return `${this.gateway}/ipfs/${res.data.IpfsHash}`;
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  }

  async getMediaForWallet(walletAddress: string): Promise<Array<IPFSMedia>> {
    if (!PINATA_JWT) {
      throw new Error('Pinata JWT is not configured');
    }

    try {
      const response = await pinataApi.get<PinataResponse>(
        'https://api.pinata.cloud/data/pinList',
        {
          params: {
            status: 'pinned',
            metadata: JSON.stringify({
              keyvalues: {
                walletAddress: { value: walletAddress, op: 'eq' },
                isDeleted: { value: 'false', op: 'eq' }
              }
            })
          }
        }
      );
  
      return response.data.rows
        .filter(pin => pin.metadata?.keyvalues?.walletAddress === walletAddress)
        .map(pin => ({
          id: pin.ipfs_pin_hash,
          url: `${this.gateway}/ipfs/${pin.ipfs_pin_hash}`,
          type: pin.metadata.keyvalues.type as 'image' | 'video',
          mimeType: pin.metadata.keyvalues.mimeType,
          walletAddress: pin.metadata.keyvalues.walletAddress,
          timestamp: new Date(pin.date_pinned).toISOString(),
          backupUrls: [
            `https://ipfs.io/ipfs/${pin.ipfs_pin_hash}`,
            `https://cloudflare-ipfs.com/ipfs/${pin.ipfs_pin_hash}`,
            `https://gateway.ipfs.io/ipfs/${pin.ipfs_pin_hash}`
          ],
          provider: this.name
        }));
    } catch (error) {
      console.error('Failed to fetch media:', error);
      throw error;
    }
  }

  async deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean> {
    if (!PINATA_JWT) {
      throw new Error('Pinata JWT is not configured');
    }

    try {
      const response = await pinataApi.get<PinataResponse>(
        'https://api.pinata.cloud/data/pinList',
        {
          params: {
            status: 'pinned',
            hash: ipfsHash
          }
        }
      );

      const pin = response.data.rows[0];
      if (!pin || pin.metadata?.keyvalues?.walletAddress !== walletAddress) {
        return false;
      }

      await pinataApi.delete(
        `https://api.pinata.cloud/pinning/unpin/${ipfsHash}`
      );
      
      return true;
    } catch (error) {
      console.error('Failed to delete media:', error);
      return false;
    }
  }

  async checkPinStatus(ipfsHash: string): Promise<boolean> {
    if (!PINATA_JWT) {
      throw new Error('Pinata JWT is not configured');
    }

    try {
      const response = await pinataApi.get<PinataResponse>(
        'https://api.pinata.cloud/data/pinList',
        {
          params: {
            status: 'pinned',
            hash: ipfsHash
          }
        }
      );

      return response.data.rows.length > 0;
    } catch (error) {
      console.error('Failed to check pin status:', error);
      return false;
    }
  }

  async repin(ipfsHash: string): Promise<boolean> {
    if (!PINATA_JWT) {
      throw new Error('Pinata JWT is not configured');
    }

    try {
      await pinataApi.post(
        'https://api.pinata.cloud/pinning/pinByHash',
        {
          hashToPin: ipfsHash
        }
      );
      return true;
    } catch (error) {
      console.error('Failed to repin:', error);
      return false;
    }
  }
}

export const pinataService = new PinataService();