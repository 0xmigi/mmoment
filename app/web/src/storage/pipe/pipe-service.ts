/**
 * Pipe Network storage service for MMOMENT web client
 *
 * Handles downloading and decrypting media from users' Pipe storage accounts.
 * Integrates with existing MMOMENT storage provider pattern.
 */

import { CONFIG } from '../../core/config';

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
  // Use us-west-01 for consistency with backend (mainnet endpoint)
  private baseUrl = "https://us-west-01-firestarter.pipenetwork.com";
  private initPromise: Promise<void>;
  // Upload retry configuration
  private readonly maxRetries = 3;
  private readonly retryDelayMs = 1000;

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
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/credentials?wallet=${encodeURIComponent(walletAddress)}`);
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
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/create-account`, {
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

    // Generate MMOMENT filename
    const timestamp = metadata?.timestamp || new Date().toISOString();
    const cameraId = metadata?.cameraId || "web";
    const extension = type === "video" ? "mp4" : "jpg";
    const filename = `mmoment_${type}_${cameraId}_${timestamp}.${extension}`;

    // Retry logic for resilient uploads
    let lastError: Error | null = null;
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const formData = new FormData();
        formData.append("file", blob, filename);

        const uploadUrl = new URL(`${this.baseUrl}/priorityUpload`);
        uploadUrl.searchParams.append("user_id", this.credentials.userId);
        uploadUrl.searchParams.append("user_app_key", this.credentials.userAppKey);

        // Add timeout via AbortController (2 min for videos, 30s for images)
        const controller = new AbortController();
        const timeoutMs = type === "video" ? 120000 : 30000;
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        const response = await fetch(uploadUrl.toString(), {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`Pipe upload failed (${response.status}): ${error}`);
        }

        const result = await response.text(); // Pipe returns filename as string
        console.log(`✅ Uploaded to Pipe: ${result} (attempt ${attempt})`);

        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on abort (timeout)
        if (lastError.name === "AbortError") {
          throw new Error(`Upload timed out after ${type === "video" ? "2 minutes" : "30 seconds"}`);
        }

        console.warn(`⚠️ Upload attempt ${attempt}/${this.maxRetries} failed:`, lastError.message);

        if (attempt < this.maxRetries) {
          // Exponential backoff
          const delay = this.retryDelayMs * Math.pow(2, attempt - 1);
          console.log(`   Retrying in ${delay}ms...`);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error("Upload failed after all retries");
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

    console.log(`🔄 Downloading file directly from Pipe: ${_fileId.slice(0, 20)}...`);

    // Try POST method first (matches backend pattern)
    try {
      const response = await fetch(`${this.baseUrl}/download`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: this.credentials.userId,
          user_app_key: this.credentials.userAppKey,
          file_name: _fileId,
        }),
      });

      if (response.ok) {
        console.log(`✅ POST download successful, decoding base64...`);
        const base64Data = await response.text();
        // Convert base64 to blob
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return new Blob([bytes]);
      }
    } catch (postError) {
      console.warn(`⚠️ POST download failed, trying fallback...`);
    }

    // Fallback to GET method with headers
    const response = await fetch(`${this.baseUrl}/download-stream`, {
      method: "GET",
      headers: {
        "X-User-Id": this.credentials.userId,
        "X-User-App-Key": this.credentials.userAppKey,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Pipe download failed: ${error}`);
    }

    console.log(`✅ GET download successful`);
    return response.blob();
  }

  async deleteMedia(fileId: string, walletAddress: string): Promise<boolean> {
    if (!this.credentials) {
      throw new Error("Pipe credentials not available");
    }

    try {
      console.log(`🗑️ Deleting file ${fileId} from Pipe...`);

      // Use backend endpoint which uses the new SDK
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/delete/${encodeURIComponent(walletAddress)}/${encodeURIComponent(fileId)}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const error = await response.json();
        console.error(`Failed to delete ${fileId} from Pipe:`, error);
        return false;
      }

      const result = await response.json();
      console.log(`✅ Deleted ${fileId} from Pipe:`, result);
      return true;
    } catch (error) {
      console.error("Error deleting from Pipe:", error);
      return false;
    }
  }

  /**
   * Create a public share link for a file
   */
  async createShareLink(
    fileId: string,
    walletAddress: string,
    options?: { title?: string; description?: string }
  ): Promise<{ success: boolean; shareUrl?: string; linkHash?: string; error?: string }> {
    if (!this.credentials) {
      return {
        success: false,
        error: "Pipe credentials not available",
      };
    }

    try {
      console.log(`🔗 Creating share link for ${fileId}...`);

      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/share/${encodeURIComponent(walletAddress)}/${encodeURIComponent(fileId)}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: options?.title,
          description: options?.description,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        console.error("Failed to create share link:", error);
        return {
          success: false,
          error: error.error || "Failed to create share link",
        };
      }

      const result = await response.json();
      console.log(`✅ Share link created:`, result.shareUrl);

      return {
        success: true,
        shareUrl: result.shareUrl,
        linkHash: result.linkHash,
      };
    } catch (error) {
      console.error("Error creating share link:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Failed to create share link",
      };
    }
  }

  /**
   * Delete a public share link
   */
  async deleteShareLink(linkHash: string, walletAddress: string): Promise<boolean> {
    if (!this.credentials) {
      throw new Error("Pipe credentials not available");
    }

    try {
      console.log(`🗑️ Deleting share link ${linkHash}...`);

      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/share/${encodeURIComponent(walletAddress)}/${encodeURIComponent(linkHash)}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const error = await response.json();
        console.error("Failed to delete share link:", error);
        return false;
      }

      console.log(`✅ Share link deleted`);
      return true;
    } catch (error) {
      console.error("Error deleting share link:", error);
      return false;
    }
  }

  async checkBalance(walletAddress: string): Promise<{ sol: number; pipe: number }> {
    if (!this.credentials) {
      return { sol: 0, pipe: 0 };
    }

    try {
      // Check SOL balance via backend proxy to avoid CORS issues
      const solResponse = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/proxy/checkWallet`, {
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
        const pipeResponse = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/proxy/checkCustomToken`, {
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
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/pipe/proxy/exchangeSolForTokens`, {
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
