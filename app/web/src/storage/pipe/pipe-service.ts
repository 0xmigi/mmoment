/**
 * Pipe Network storage service for MMOMENT web client
 * 
 * Handles downloading and decrypting media from users' Pipe storage accounts.
 * Integrates with existing MMOMENT storage provider pattern.
 */

export interface PipeMedia {
  id: string;
  url: string;
  type: 'image' | 'video';
  walletAddress: string;
  timestamp: string;
  encrypted: boolean;
  provider: 'pipe';
  fileSize?: number;
  metadata?: {
    cameraId?: string;
    captureType?: string;
  };
}

export interface PipeCredentials {
  userId: string;
  userAppKey: string;
}

export interface PipeStorageProvider {
  name: string;
  uploadFile(blob: Blob, walletAddress: string, type: 'image' | 'video', metadata?: any): Promise<string>;
  getMediaForWallet(walletAddress: string): Promise<PipeMedia[]>;
  deleteMedia(fileId: string, walletAddress: string): Promise<boolean>;
  downloadFile(fileId: string, walletAddress: string): Promise<Blob>;
}

export class PipeService implements PipeStorageProvider {
  readonly name = 'Pipe Network';
  
  private credentials: PipeCredentials | null = null;
  private baseUrl = 'https://us-east-00-firestarter.pipenetwork.com';

  constructor() {
    this.initializeCredentials();
  }

  private async initializeCredentials() {
    try {
      // Try to get credentials from your backend/API
      const response = await fetch('/api/pipe/credentials');
      if (response.ok) {
        this.credentials = await response.json();
        console.log('✅ Pipe credentials loaded');
      }
    } catch (error) {
      console.warn('⚠️ Pipe credentials not available:', error);
      console.info('💡 To enable Pipe storage: Set up Pipe credentials in backend');
    }
  }

  /**
   * Create or get Pipe account for a wallet address
   */
  async createOrGetAccount(walletAddress: string): Promise<PipeCredentials | null> {
    try {
      console.log(`🔄 Creating/getting Pipe account for ${walletAddress.slice(0, 8)}...`);
      
      // Call backend to create Pipe account using wallet address as username
      const response = await fetch('/api/pipe/create-account', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          walletAddress,
        }),
      });

      if (response.ok) {
        const credentials = await response.json();
        this.credentials = credentials;
        console.log('✅ Pipe account created/retrieved successfully');
        return credentials;
      } else {
        console.error('❌ Failed to create Pipe account');
        return null;
      }
    } catch (error) {
      console.error('❌ Error creating Pipe account:', error);
      return null;
    }
  }

  isAvailable(): boolean {
    return this.credentials !== null;
  }

  async uploadFile(
    blob: Blob, 
    _walletAddress: string, 
    type: 'image' | 'video',
    metadata?: { cameraId?: string; timestamp?: string }
  ): Promise<string> {
    if (!this.credentials) {
      throw new Error('Pipe credentials not available');
    }

    const formData = new FormData();
    
    // Generate MMOMENT filename
    const timestamp = metadata?.timestamp || new Date().toISOString();
    const cameraId = metadata?.cameraId || 'web';
    const extension = type === 'video' ? 'mp4' : 'jpg';
    const filename = `mmoment_${type}_${cameraId}_${timestamp}.${extension}`;
    
    formData.append('file', blob, filename);

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      headers: {
        'X-User-Id': this.credentials.userId,
        'X-User-App-Key': this.credentials.userAppKey,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Pipe upload failed: ${error}`);
    }

    const result = await response.text(); // Pipe returns filename as string
    console.log(`✅ Uploaded to Pipe: ${result}`);
    
    return result;
  }

  async getMediaForWallet(walletAddress: string): Promise<PipeMedia[]> {
    if (!this.credentials) {
      console.warn('Pipe credentials not available');
      return [];
    }

    try {
      // TODO: Implement file listing via Pipe API
      // For now, return empty array since Pipe API doesn't have list endpoint
      // This would need to be tracked in your backend or via metadata
      
      console.log(`📋 Listing Pipe files for ${walletAddress.slice(0, 8)}...`);
      
      // In a real implementation, you'd need to:
      // 1. Store file metadata in your backend when uploads happen
      // 2. Query your backend for user's files
      // 3. Return them in the PipeMedia format
      
      return [];
      
    } catch (error) {
      console.error('Failed to get Pipe media:', error);
      return [];
    }
  }

  async downloadFile(_fileId: string, _walletAddress: string): Promise<Blob> {
    if (!this.credentials) {
      throw new Error('Pipe credentials not available');
    }

    const response = await fetch(`${this.baseUrl}/download`, {
      method: 'GET',
      headers: {
        'X-User-Id': this.credentials.userId,
        'X-User-App-Key': this.credentials.userAppKey,
      },
      // Note: Pipe uses query params for download
      // You'd add ?file_name=${fileId} to the URL
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Pipe download failed: ${error}`);
    }

    return response.blob();
  }

  async deleteMedia(fileId: string, _walletAddress: string): Promise<boolean> {
    if (!this.credentials) {
      throw new Error('Pipe credentials not available');
    }

    try {
      const response = await fetch(`${this.baseUrl}/deleteFile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': this.credentials.userId,
          'X-User-App-Key': this.credentials.userAppKey,
        },
        body: JSON.stringify({
          file_name: fileId,
        }),
      });

      if (!response.ok) {
        console.error(`Failed to delete ${fileId} from Pipe`);
        return false;
      }

      console.log(`✅ Deleted ${fileId} from Pipe`);
      return true;

    } catch (error) {
      console.error('Error deleting from Pipe:', error);
      return false;
    }
  }

  async checkBalance(): Promise<{ sol: number; pipe: number }> {
    if (!this.credentials) {
      return { sol: 0, pipe: 0 };
    }

    try {
      // Check SOL balance
      const solResponse = await fetch(`${this.baseUrl}/checkWallet`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': this.credentials.userId,
          'X-User-App-Key': this.credentials.userAppKey,
        },
        body: JSON.stringify({}),
      });

      let solBalance = 0;
      if (solResponse.ok) {
        const solData = await solResponse.json();
        solBalance = solData.balance_sol || 0;
      }

      // Check PIPE token balance
      const pipeResponse = await fetch(`${this.baseUrl}/getCustomTokenBalance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: this.credentials.userId,
          user_app_key: this.credentials.userAppKey,
        }),
      });

      let pipeBalance = 0;
      if (pipeResponse.ok) {
        const pipeData = await pipeResponse.json();
        pipeBalance = pipeData.ui_amount || 0;
      }

      return { sol: solBalance, pipe: pipeBalance };

    } catch (error) {
      console.error('Error checking Pipe balance:', error);
      return { sol: 0, pipe: 0 };
    }
  }

  async swapSolForPipe(solAmount: number): Promise<{ success: boolean; tokensReceived?: number; error?: string }> {
    if (!this.credentials) {
      return { success: false, error: 'Pipe credentials not available' };
    }

    try {
      console.log(`🔄 Swapping ${solAmount} SOL for PIPE tokens...`);

      const response = await fetch(`${this.baseUrl}/swapSolForPipe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Id': this.credentials.userId,
          'X-User-App-Key': this.credentials.userAppKey,
        },
        body: JSON.stringify({
          amount_sol: solAmount,
        }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Swap failed: ${error}`);
      }

      const result = await response.json();
      console.log(`✅ Swap successful: ${result.tokens_minted} PIPE tokens received`);

      return {
        success: true,
        tokensReceived: result.tokens_minted || 0,
      };

    } catch (error) {
      console.error('Error swapping SOL for PIPE:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Swap failed',
      };
    }
  }
}

// Export singleton instance
export const pipeService = new PipeService();