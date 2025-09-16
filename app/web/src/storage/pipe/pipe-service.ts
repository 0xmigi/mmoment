/**
 * Pipe Network storage service for MMOMENT web client
 *
 * Handles downloading and decrypting media from users' Pipe storage accounts.
 * Integrates with existing MMOMENT storage provider pattern.
 */

export interface PipeMedia {
  id: string;
  url: string;
  type: "image" | "video";
  walletAddress: string;
  timestamp: string;
  encrypted: boolean;
  provider: "pipe";
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
  uploadFile(
    blob: Blob,
    walletAddress: string,
    type: "image" | "video",
    metadata?: any
  ): Promise<string>;
  getMediaForWallet(walletAddress: string): Promise<PipeMedia[]>;
  deleteMedia(fileId: string, walletAddress: string): Promise<boolean>;
  downloadFile(fileId: string, walletAddress: string): Promise<Blob>;
}

export class PipeService implements PipeStorageProvider {
  readonly name = "Pipe Network";

  private credentials: PipeCredentials | null = null;
  private baseUrl = "https://us-west-00-firestarter.pipenetwork.com";
  private authTokens: { access_token?: string; refresh_token?: string } = {};
  private initPromise: Promise<void>;

  constructor() {
    this.initPromise = this.initializeCredentials();
  }

  private async initializeCredentials() {
    // Credentials are now loaded when user creates/connects their Pipe account
    // No automatic fallback credentials
    console.log("🔄 Pipe service initialized - credentials will be loaded when user connects");
  }

  /**
   * Load credentials for a specific wallet address
   */
  async loadCredentialsForWallet(walletAddress: string): Promise<void> {
    try {
      const response = await fetch(`/api/pipe/credentials?wallet=${encodeURIComponent(walletAddress)}`);
      if (response.ok) {
        this.credentials = await response.json();
        console.log(`✅ Pipe credentials loaded for ${walletAddress.slice(0, 8)}...`);
      } else {
        console.log(`ℹ️ No existing Pipe account for ${walletAddress.slice(0, 8)}...`);
        this.credentials = null;
      }
    } catch (error) {
      console.error("Failed to load Pipe credentials:", error);
      this.credentials = null;
    }
  }

  /**
   * Create or get Pipe account for a wallet address
   */
  async createOrGetAccount(
    walletAddress: string
  ): Promise<PipeCredentials | null> {
    try {
      console.log(
        `🔄 Creating/getting Pipe account for ${walletAddress.slice(0, 8)}...`
      );

      // Call backend to create Pipe account using wallet address as username
      const response = await fetch("/api/pipe/create-account", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          walletAddress,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        this.credentials = result;
        if (result.existing) {
          console.log("✅ Existing Pipe account retrieved");
        } else {
          console.log("✅ New Pipe account created");
        }
        return result;
      } else {
        console.error("❌ Failed to create/get Pipe account");
        return null;
      }
    } catch (error) {
      console.error("❌ Error creating Pipe account:", error);
      return null;
    }
  }

  async ensureInitialized(): Promise<void> {
    await this.initPromise;
  }

  isAvailable(): boolean {
    return this.credentials !== null;
  }

  getCredentials(): PipeCredentials | null {
    return this.credentials;
  }

  async uploadFile(
    blob: Blob,
    _walletAddress: string,
    type: "image" | "video",
    metadata?: { cameraId?: string; timestamp?: string }
  ): Promise<string> {
    if (!this.credentials) {
      throw new Error("Pipe credentials not available");
    }

    const formData = new FormData();

    // Generate MMOMENT filename
    const timestamp = metadata?.timestamp || new Date().toISOString();
    const cameraId = metadata?.cameraId || "web";
    const extension = type === "video" ? "mp4" : "jpg";
    const filename = `mmoment_${type}_${cameraId}_${timestamp}.${extension}`;

    formData.append("file", blob, filename);

    const uploadUrl = new URL(`${this.baseUrl}/priorityUpload`);
    uploadUrl.searchParams.append("user_id", this.credentials.userId);
    uploadUrl.searchParams.append("user_app_key", this.credentials.userAppKey);

    const response = await fetch(uploadUrl.toString(), {
      method: "POST",
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
      console.warn("Pipe credentials not available");
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
      console.error("Failed to get Pipe media:", error);
      return [];
    }
  }

  async downloadFile(_fileId: string, _walletAddress: string): Promise<Blob> {
    if (!this.credentials) {
      throw new Error("Pipe credentials not available");
    }

    const downloadUrl = new URL(`${this.baseUrl}/download-stream`);
    downloadUrl.searchParams.append("user_id", this.credentials.userId);
    downloadUrl.searchParams.append(
      "user_app_key",
      this.credentials.userAppKey
    );
    downloadUrl.searchParams.append("file_name", _fileId);

    const response = await fetch(downloadUrl.toString(), {
      method: "GET",
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Pipe download failed: ${error}`);
    }

    return response.blob();
  }

  async deleteMedia(fileId: string, _walletAddress: string): Promise<boolean> {
    if (!this.credentials) {
      throw new Error("Pipe credentials not available");
    }

    try {
      const response = await fetch(`${this.baseUrl}/deleteFile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: this.credentials.userId,
          user_app_key: this.credentials.userAppKey,
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
      console.error("Error deleting from Pipe:", error);
      return false;
    }
  }

  async checkBalance(walletAddress: string): Promise<{ sol: number; pipe: number }> {
    if (!this.credentials) {
      return { sol: 0, pipe: 0 };
    }

    try {
      // Check SOL balance via backend proxy to avoid CORS issues
      const solResponse = await fetch(`/api/pipe/proxy/checkWallet`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": this.credentials.userId,
          "X-User-App-Key": this.credentials.userAppKey,
          "X-Wallet-Address": walletAddress,
        },
        body: JSON.stringify({}),
      });

      let solBalance = 0;
      if (solResponse.ok) {
        const solData = await solResponse.json();
        solBalance = solData.balance_sol || 0;
        console.log("✅ SOL balance:", solBalance);
      } else {
        console.error("Failed to get SOL balance:", await solResponse.text());
      }

      // Check PIPE token balance via backend proxy
      let pipeBalance = 0;
      try {
        const pipeResponse = await fetch(`/api/pipe/proxy/checkCustomToken`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-User-Id": this.credentials.userId,
            "X-User-App-Key": this.credentials.userAppKey,
            "X-Wallet-Address": walletAddress,
          },
          body: JSON.stringify({
            token_mint: "35mhJor7qTD212YXdLkB8sRzTbaYRXmTzHTCFSDP5voJ",
          }),
        });

        if (pipeResponse.ok) {
          const pipeData = await pipeResponse.json();
          pipeBalance = pipeData.ui_amount || 0;
          console.log("✅ PIPE balance:", pipeBalance);
        } else {
          console.log("Failed to get PIPE balance:", await pipeResponse.text());
        }
      } catch (error) {
        console.error("Failed to get PIPE balance:", error);
        pipeBalance = 0;
      }

      return { sol: solBalance, pipe: pipeBalance };
    } catch (error) {
      console.error("Error checking Pipe balance:", error);
      return { sol: 0, pipe: 0 };
    }
  }

  async exchangeSolForPipe(
    solAmount: number,
    walletAddress: string
  ): Promise<{ success: boolean; tokensReceived?: number; error?: string }> {
    try {
      console.log(`🔄 Swapping ${solAmount} SOL for PIPE tokens...`);

      // Try without token_mint parameter first
      const response = await fetch(`/api/pipe/proxy/exchangeSolForTokens`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Wallet-Address": walletAddress,
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
      console.log(`✅ Swap successful:`, result);

      // Handle different possible response formats
      const tokensReceived = result.tokens_minted || result.pipe_tokens || result.amount || 0;

      return {
        success: true,
        tokensReceived,
      };
    } catch (error) {
      console.error("Error swapping SOL for PIPE:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Swap failed",
      };
    }
  }
}

// Export singleton instance
export const pipeService = new PipeService();
