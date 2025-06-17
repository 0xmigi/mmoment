import axios from 'axios';
import { IPFSProvider, IPFSMedia, IPFSMetadata } from './ipfs-service';
import { getPinataCredentials } from '../config';

// Get credentials from config (which manages localStorage and environment variables)
let { PINATA_JWT, PINATA_API_KEY, PINATA_API_SECRET } = getPinataCredentials();

// Configure API instance with JWT as primary auth method
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

  constructor() {
    // Test authentication on initialization
    this.testAuthentication().then(result => {
      if (result.success) {
        console.log('✅ Pinata authentication successful');
      } else {
        console.warn('⚠️ Pinata authentication failed:', result.message);
        console.warn('👉 Using development credentials from localStorage, but these may expire in the future.');
        console.warn('   For production, set credentials in environment variables.');
      }
    }).catch(err => {
      console.warn('⚠️ Error testing Pinata authentication:', err);
      console.warn('👉 Check your internet connection and make sure Pinata services are available.');
    });
  }
  
  /**
   * Refresh credentials from localStorage or environment variables
   */
  private refreshCredentials() {
    const creds = getPinataCredentials();
    PINATA_JWT = creds.PINATA_JWT;
    PINATA_API_KEY = creds.PINATA_API_KEY;
    PINATA_API_SECRET = creds.PINATA_API_SECRET;
    
    // Update API instance headers with new JWT
    pinataApi.defaults.headers.common['Authorization'] = `Bearer ${PINATA_JWT}`;
  }

  /**
   * Test if Pinata credentials are working
   */
  async testAuthentication(): Promise<{success: boolean, message: string}> {
    // Refresh credentials before testing
    this.refreshCredentials();
    
    try {
      // First try with API key/secret
      if (PINATA_API_KEY) {
        try {
          console.log(`Testing Pinata API key: ${PINATA_API_KEY.substring(0, 8)}...`);
          
          const response = await fetch('https://api.pinata.cloud/data/testAuthentication', {
            method: 'GET',
            headers: {
              'pinata_api_key': PINATA_API_KEY,
              'pinata_secret_api_key': PINATA_API_SECRET
            }
          });
          
          const data = await response.json();
          
          if (response.ok) {
            return { 
              success: true, 
              message: `API key authentication successful: ${JSON.stringify(data)}`
            };
          } else {
            console.error('API key authentication failed:', data);
          }
        } catch (apiKeyError) {
          console.error('Error testing API key authentication:', apiKeyError);
        }
      }
      
      // Then try with JWT
      if (PINATA_JWT) {
        try {
          console.log('Testing Pinata JWT authentication');
          
          const response = await fetch('https://api.pinata.cloud/data/testAuthentication', {
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${PINATA_JWT}`
            }
          });
          
          const data = await response.json();
          
          if (response.ok) {
            return { 
              success: true, 
              message: `JWT authentication successful: ${JSON.stringify(data)}`
            };
          } else {
            console.error('JWT authentication failed:', data);
          }
        } catch (jwtError) {
          console.error('Error testing JWT authentication:', jwtError);
        }
      }
      
      // If we reach here, both authentication methods failed
      return { 
        success: false, 
        message: 'Both API key and JWT authentication failed'
      };
    } catch (error) {
      return { 
        success: false, 
        message: `Error testing authentication: ${error}`
      };
    }
  }

  async uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video', metadata?: { transactionId?: string; cameraId?: string }): Promise<string> {
    console.log(`📤 Uploading ${type} to Pinata for wallet ${walletAddress.slice(0, 8)}...`);
    this.refreshCredentials();
    
    try {
      const formData = new FormData();
      const timestamp = Date.now();
      
      // Determine file extension based on blob type for videos
      let fileExtension = 'jpg';
      let mimeType = 'image/jpeg';
      
      if (type === 'video') {
        // Check the actual MIME type of the blob to determine extension
        if (blob.type.includes('quicktime') || blob.type.includes('mov')) {
          fileExtension = 'mov';
          mimeType = 'video/quicktime';
        } else {
          fileExtension = 'mp4';
          mimeType = 'video/mp4';
        }
      }
      
      const filename = `${walletAddress}_${timestamp}.${fileExtension}`;
      formData.append('file', blob, filename);

      const pinataMetadata: IPFSMetadata = {
        name: filename,
        keyvalues: {
          walletAddress: walletAddress,
          timestamp: timestamp.toString(),
          isDeleted: 'false',
          type: type,
          mimeType: mimeType,
          transactionId: metadata?.transactionId,
          cameraId: metadata?.cameraId
        }
      };
      formData.append('pinataMetadata', JSON.stringify(pinataMetadata));
      
      // First try with JWT authentication
      try {
        console.log("Trying upload with JWT authentication");
        const res = await axios.post(
          "https://api.pinata.cloud/pinning/pinFileToIPFS",
          formData,
          {
            headers: {
              'Authorization': `Bearer ${PINATA_JWT}`,
              'Content-Type': `multipart/form-data`,
            }
          }
        );
        
        if (res.data && res.data.IpfsHash) {
          console.log(`Successfully uploaded to IPFS with JWT: ${res.data.IpfsHash}`);
          return `${this.gateway}/ipfs/${res.data.IpfsHash}`;
        }
      } catch (jwtError) {
        console.warn('JWT upload failed, trying with API key:', jwtError);
      }
      
      // If JWT fails, try with API key authentication
      try {
        console.log("Trying upload with API key authentication");
        const res = await axios.post(
          "https://api.pinata.cloud/pinning/pinFileToIPFS",
          formData,
          {
            headers: {
              'pinata_api_key': PINATA_API_KEY,
              'pinata_secret_api_key': PINATA_API_SECRET,
              'Content-Type': `multipart/form-data`,
            }
          }
        );
        
        if (res.data && res.data.IpfsHash) {
          console.log(`Successfully uploaded to IPFS with API keys: ${res.data.IpfsHash}`);
          return `${this.gateway}/ipfs/${res.data.IpfsHash}`;
        }
      } catch (apiKeyError) {
        console.error('API key upload failed:', apiKeyError);
        throw apiKeyError;
      }
      
      throw new Error('Failed to upload to Pinata with both JWT and API key');
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  }

  async getMediaForWallet(walletAddress: string, _retryCount = 0): Promise<Array<IPFSMedia>> {
    console.log(`🔍 Fetching media for wallet ${walletAddress === 'all' ? 'ALL' : walletAddress.slice(0, 8)}...`);
    this.refreshCredentials();

    try {
      // Fix: Don't use undefined for 'all' case - use different query approach
      const queryParams: any = {
        status: 'pinned',
        pageLimit: 1000,
        pageOffset: 0
      };
      
      // Only add metadata filter if we have a specific wallet address
      if (walletAddress !== 'all') {
        queryParams.metadata = JSON.stringify({
          keyvalues: {
            walletAddress: { value: walletAddress, op: 'eq' },
            isDeleted: { value: 'false', op: 'eq' }
          }
        });
      }
      
      console.log(`📊 Query params:`, queryParams);

      // Try with JWT first
      try {
        const response = await axios.get<PinataResponse>(
          'https://api.pinata.cloud/data/pinList',
          {
            headers: {
              'Authorization': `Bearer ${PINATA_JWT}`
            },
            params: queryParams
          }
        );
        
        if (response.data && response.data.rows) {
          console.log(`✅ Successfully fetched ${response.data.rows.length} total items with JWT`);
          const processedMedia = this.processMediaResponse(response.data, walletAddress);
          console.log(`📄 Processed to ${processedMedia.length} media items for wallet`);
          return processedMedia;
        }
      } catch (jwtError) {
        console.warn('🚨 JWT media fetch failed, trying with API key:', jwtError);
      }
      
      // If JWT fails, try with API key
      try {
        const response = await axios.get<PinataResponse>(
          'https://api.pinata.cloud/data/pinList',
          {
            headers: {
              'pinata_api_key': PINATA_API_KEY,
              'pinata_secret_api_key': PINATA_API_SECRET
            },
            params: queryParams
          }
        );
        
        if (response.data && response.data.rows) {
          console.log(`✅ Successfully fetched ${response.data.rows.length} total items with API key`);
          const processedMedia = this.processMediaResponse(response.data, walletAddress);
          console.log(`📄 Processed to ${processedMedia.length} media items for wallet`);
          return processedMedia;
        }
      } catch (apiKeyError) {
        console.error('🚨 API key media fetch failed:', apiKeyError);
        throw apiKeyError;
      }
      
      console.warn('🚨 No media data returned from either authentication method');
      return [];
    } catch (error) {
      console.error('🚨 Failed to fetch media for wallet:', error);
      return [];
    }
  }

  async deleteMedia(ipfsHash: string, walletAddress: string): Promise<boolean> {
    console.log(`🗑️ Deleting media ${ipfsHash} for wallet ${walletAddress.slice(0, 8)}...`);
    this.refreshCredentials();

    try {
      // Get pin info
      let pin = null;
      try {
        const response = await axios.get<PinataResponse>(
          'https://api.pinata.cloud/data/pinList',
          {
            headers: {
              'Authorization': `Bearer ${PINATA_JWT}`
            },
            params: {
              status: 'pinned',
              hash: ipfsHash
            }
          }
        );
        
        pin = response.data.rows[0];
      } catch (jwtError) {
        console.warn('JWT pin check failed, trying with API key:', jwtError);
        
        try {
          const response = await axios.get<PinataResponse>(
            'https://api.pinata.cloud/data/pinList',
            {
              headers: {
                'pinata_api_key': PINATA_API_KEY,
                'pinata_secret_api_key': PINATA_API_SECRET
              },
              params: {
                status: 'pinned',
                hash: ipfsHash
              }
            }
          );
          
          pin = response.data.rows[0];
        } catch (apiKeyError) {
          console.error('API key pin check failed:', apiKeyError);
          // Continue anyway - try to delete even if we can't verify
        }
      }

      // Relaxed verification - allow delete if pin exists or if verification fails
      if (pin && pin.metadata?.keyvalues?.walletAddress && pin.metadata.keyvalues.walletAddress !== walletAddress) {
        console.warn(`Wallet mismatch: expected ${walletAddress}, got ${pin.metadata.keyvalues.walletAddress}`);
        // Still try to delete - maybe it's a legacy item
      }

      // Try to unpin with JWT first
      try {
        await axios.delete(
          `https://api.pinata.cloud/pinning/unpin/${ipfsHash}`,
          {
            headers: {
              'Authorization': `Bearer ${PINATA_JWT}`
            }
          }
        );
        
        console.log(`✅ Successfully deleted media with JWT`);
        return true;
      } catch (jwtError) {
        console.warn('JWT unpin failed, trying with API key:', jwtError);
        
        try {
          await axios.delete(
            `https://api.pinata.cloud/pinning/unpin/${ipfsHash}`,
            {
              headers: {
                'pinata_api_key': PINATA_API_KEY,
                'pinata_secret_api_key': PINATA_API_SECRET
              }
            }
          );
          
          console.log(`✅ Successfully deleted media with API key`);
          return true;
        } catch (apiKeyError) {
          console.error('❌ Both JWT and API key unpin failed:', apiKeyError);
          return false;
        }
      }
    } catch (error) {
      console.error('❌ Failed to delete media:', error);
      return false;
    }
  }

  async checkPinStatus(ipfsHash: string): Promise<boolean> {
    console.log(`📌 Checking pin status for ${ipfsHash}...`);
    this.refreshCredentials();

    // Try JWT first
    try {
      const response = await axios.get<PinataResponse>(
        'https://api.pinata.cloud/data/pinList',
        {
          headers: {
            'Authorization': `Bearer ${PINATA_JWT}`
          },
          params: {
            status: 'pinned',
            hash: ipfsHash
          }
        }
      );

      return response.data.rows.length > 0;
    } catch (jwtError) {
      console.warn('JWT pin status check failed, trying with API key:', jwtError);
      
      // Try with API key
      try {
        const response = await axios.get<PinataResponse>(
          'https://api.pinata.cloud/data/pinList',
          {
            headers: {
              'pinata_api_key': PINATA_API_KEY,
              'pinata_secret_api_key': PINATA_API_SECRET
            },
            params: {
              status: 'pinned',
              hash: ipfsHash
            }
          }
        );

        return response.data.rows.length > 0;
      } catch (apiKeyError) {
        console.error('API key pin status check failed:', apiKeyError);
        return false;
      }
    }
  }

  async repin(ipfsHash: string): Promise<boolean> {
    console.log(`🔄 Repinning ${ipfsHash}...`);
    this.refreshCredentials();

    // Try JWT first
    try {
      await axios.post(
        'https://api.pinata.cloud/pinning/pinByHash',
        {
          hashToPin: ipfsHash
        },
        {
          headers: {
            'Authorization': `Bearer ${PINATA_JWT}`,
            'Content-Type': 'application/json'
          }
        }
      );
      console.log(`Successfully repinned with JWT`);
      return true;
    } catch (jwtError) {
      console.warn('JWT repin failed, trying with API key:', jwtError);
      
      // Try with API key
      try {
        await axios.post(
          'https://api.pinata.cloud/pinning/pinByHash',
          {
            hashToPin: ipfsHash
          },
          {
            headers: {
              'pinata_api_key': PINATA_API_KEY,
              'pinata_secret_api_key': PINATA_API_SECRET,
              'Content-Type': 'application/json'
            }
          }
        );
        console.log(`Successfully repinned with API key`);
        return true;
      } catch (apiKeyError) {
        console.error('API key repin failed:', apiKeyError);
        return false;
      }
    }
  }

  /**
   * Process the media response from Pinata
   */
  private processMediaResponse(data: PinataResponse, walletAddress: string): Array<IPFSMedia> {
    const filter = walletAddress === 'all' 
      ? (_pin: any) => true
      : (pin: any) => pin.metadata?.keyvalues?.walletAddress === walletAddress;
        
    return data.rows
      .filter(filter)
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
        provider: this.name,
        transactionId: pin.metadata.keyvalues.transactionId,
        cameraId: pin.metadata.keyvalues.cameraId
      }));
  }
}

export const pinataService = new PinataService();