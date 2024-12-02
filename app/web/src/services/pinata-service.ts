import axios from 'axios';

const PINATA_JWT = import.meta.env.VITE_PINATA_JWT || "";

const pinataApi = axios.create({
  headers: {
    'Authorization': `Bearer ${PINATA_JWT}`,
    'Content-Type': 'application/json'
  }
});

interface PinataMetadata {
  name: string;
  keyvalues: {
    walletAddress: string;
    timestamp: string;
    isDeleted: string;
    type: string;  // 'image' or 'video'
    mimeType: string; // Added for explicit MIME type
  };
}

interface PinataResponse {
  rows: Array<{
    ipfs_pin_hash: string;
    metadata: PinataMetadata;
    date_pinned: string;
  }>;
}

export class PinataService {
  async uploadImage(imageBlob: Blob, walletAddress: string): Promise<string> {
    return this.uploadFile(imageBlob, walletAddress, 'image');
  }

  async uploadVideo(videoBlob: Blob, walletAddress: string): Promise<string> {
    // Ensure we're sending an MP4 blob
    const mp4Blob = new Blob([videoBlob], { type: 'video/mp4' });
    return this.uploadFile(mp4Blob, walletAddress, 'video');
  }

  private async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video'): Promise<string> {
    try {
      console.log(`Starting ${type} upload for wallet:`, walletAddress);
      
      const formData = new FormData();
      const filename = `${walletAddress}_${Date.now()}.${type === 'video' ? 'mov' : 'jpg'}`;
      formData.append('file', blob, filename);

      const metadata: PinataMetadata = {
        name: filename,
        keyvalues: {
          walletAddress: walletAddress,
          timestamp: Date.now().toString(),
          isDeleted: 'false',
          type: type,
          mimeType: type === 'video' ? 'video/quicktime' : 'image/jpeg'
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
  
      console.log('Upload successful:', res.data);
      return `https://gateway.pinata.cloud/ipfs/${res.data.IpfsHash}`;
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  }

  async getMediaForWallet(walletAddress: string): Promise<Array<any>> {
    try {
      console.log('Fetching media for wallet:', walletAddress);
      
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
  
      const filteredMedia = response.data.rows.filter(pin => 
        pin.metadata?.keyvalues?.walletAddress === walletAddress
      );

      return filteredMedia.map(pin => ({
        id: pin.ipfs_pin_hash,
        url: `https://gateway.pinata.cloud/ipfs/${pin.ipfs_pin_hash}`,
        type: pin.metadata.keyvalues.type || 'image',
        mimeType: pin.metadata.keyvalues.mimeType || 'image/jpeg',
        walletAddress: pin.metadata.keyvalues.walletAddress,
        timestamp: new Date(pin.date_pinned).toLocaleString(),
        backupUrls: [
          `https://ipfs.io/ipfs/${pin.ipfs_pin_hash}`,
          `https://cloudflare-ipfs.com/ipfs/${pin.ipfs_pin_hash}`,
          `https://gateway.ipfs.io/ipfs/${pin.ipfs_pin_hash}`
        ]
      }));
    } catch (error) {
      console.error('Failed to fetch media:', error);
      throw error;
    }
  }

  async deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean> {
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
}

export const pinataService = new PinataService();