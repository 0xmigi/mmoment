/**
 * Firestarter SDK - Client library for Pipe Network integration
 *
 * Provides developers with a simple way to integrate Pipe Network decentralized storage
 * into their applications, following the same patterns as Firestarter GUI.
 *
 * Key Features:
 * - Wallet-to-Pipe account mapping
 * - Direct Pipe Network API calls
 * - Blake3 hash calculation for content addressing
 * - Client-side upload history tracking
 * - JWT token management
 * - Multi-user support
 */

// Main SDK exports
export { FirestarterClient } from './client';
export { UserManager } from './user-manager';
export { UploadHistory } from './upload-history';

// Import classes and types for internal use
import { FirestarterClient } from './client';
import { UserManager } from './user-manager';
import { UploadHistory, FileSystemStorage } from './upload-history';
import type { PipeConfig, PipeUser, UploadOptions, UploadResult, FileRecord } from './types';

// Types
export type {
  PipeConfig,
  PipeUser,
  UploadOptions,
  UploadResult,
  FileRecord,
  TokenBalance,
  WalletBalance,
  PipeError,
} from './types';

// Errors
export {
  PipeApiError,
  PipeValidationError,
  PipeSessionError,
  PipeStorageError,
} from './errors';

// Main SDK class - this is what developers will use
export class FirestarterSDK {
  private client: FirestarterClient;
  private userManager: UserManager;
  private uploadHistory: UploadHistory;

  constructor(config: PipeConfig = {}) {
    this.client = new FirestarterClient(config);
    this.userManager = new UserManager(this.client);
    // Use persistent file storage for upload history in Node.js environments
    this.uploadHistory = new UploadHistory(new FileSystemStorage('./firestarter-upload-history.json'));

    // Give the client access to upload history for download filename lookup
    (this.client as any).uploadHistory = this.uploadHistory;
  }

  /**
   * Create or get existing Pipe account for a wallet
   */
  async createUserAccount(walletAddress: string): Promise<PipeUser> {
    return this.userManager.createOrGetUser(walletAddress);
  }

  /**
   * Upload file to Pipe Network for a specific user
   */
  async uploadFile(
    walletAddress: string,
    data: Buffer,
    fileName: string,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    const user = await this.userManager.getUser(walletAddress);
    const result = await this.client.upload(user, data, fileName, options);

    // Track upload locally
    await this.uploadHistory.recordUpload({
      fileId: result.fileId,
      originalFileName: fileName,
      storedFileName: result.fileName,
      userId: walletAddress,
      uploadedAt: result.uploadedAt,
      size: result.size,
      blake3Hash: result.blake3Hash,
      metadata: options.metadata || {},
    });

    return result;
  }

  /**
   * List files for a user
   */
  async listUserFiles(walletAddress: string): Promise<FileRecord[]> {
    return this.uploadHistory.getUserFiles(walletAddress);
  }

  /**
   * Get user's Pipe wallet balance
   */
  async getUserBalance(walletAddress: string): Promise<{ sol: number; pipe: number; publicKey: string }> {
    const user = await this.userManager.getUser(walletAddress);
    const [solBalance, pipeBalance] = await Promise.all([
      this.client.checkSolBalance(user),
      this.client.checkPipeBalance(user),
    ]);

    return {
      sol: solBalance.balanceSol,
      pipe: pipeBalance.uiAmount,
      publicKey: solBalance.publicKey,
    };
  }

  /**
   * Download file from Pipe Network
   */
  async downloadFile(walletAddress: string, fileId: string): Promise<Buffer> {
    const user = await this.userManager.getUser(walletAddress);
    return this.client.download(user, fileId);
  }

  /**
   * Exchange SOL for PIPE tokens
   */
  async exchangeSolForPipe(walletAddress: string, amountSol: number): Promise<number> {
    const user = await this.userManager.getUser(walletAddress);
    return this.client.exchangeSolForPipe(user, amountSol);
  }
}

// Convenience function for quick setup
export function createFirestarterSDK(config?: PipeConfig): FirestarterSDK {
  return new FirestarterSDK(config);
}