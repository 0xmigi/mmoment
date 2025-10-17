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
export { FirestarterClient } from './client';
export { UserManager } from './user-manager';
export { UploadHistory } from './upload-history';
import type { PipeConfig, PipeUser, UploadOptions, UploadResult, FileRecord } from './types';
export type { PipeConfig, PipeUser, UploadOptions, UploadResult, FileRecord, TokenBalance, WalletBalance, PipeError, } from './types';
export { PipeApiError, PipeValidationError, PipeSessionError, PipeStorageError, } from './errors';
export declare class FirestarterSDK {
    private client;
    private userManager;
    private uploadHistory;
    constructor(config?: PipeConfig);
    /**
     * Create or get existing Pipe account for a wallet
     */
    createUserAccount(walletAddress: string): Promise<PipeUser>;
    /**
     * Upload file to Pipe Network for a specific user
     */
    uploadFile(walletAddress: string, data: Buffer, fileName: string, options?: UploadOptions): Promise<UploadResult>;
    /**
     * List files for a user
     */
    listUserFiles(walletAddress: string): Promise<FileRecord[]>;
    /**
     * Get user's Pipe wallet balance
     */
    getUserBalance(walletAddress: string): Promise<{
        sol: number;
        pipe: number;
        publicKey: string;
    }>;
    /**
     * Download file from Pipe Network
     */
    downloadFile(walletAddress: string, fileId: string): Promise<Buffer>;
    /**
     * Exchange SOL for PIPE tokens
     */
    exchangeSolForPipe(walletAddress: string, amountSol: number): Promise<number>;
}
export declare function createFirestarterSDK(config?: PipeConfig): FirestarterSDK;
//# sourceMappingURL=index.d.ts.map