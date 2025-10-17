/**
 * FirestarterClient - Direct Pipe Network API integration
 *
 * Handles direct communication with Pipe Network endpoints,
 * following the same patterns as your existing frontend PipeService
 * and the Firestarter GUI architecture.
 */
import { PipeConfig, PipeUser, UploadOptions, UploadResult, TokenBalance, WalletBalance } from './types';
interface PipeUserAuth {
    userId: string;
    userAppKey: string;
    username: string;
    password: string;
    accessToken?: string;
    refreshToken?: string;
    tokenExpiry?: number;
    createdAt: number;
}
export declare class FirestarterClient {
    private baseUrl;
    private api;
    private userAccounts;
    constructor(config?: PipeConfig);
    /**
     * Create new Pipe user account - EXACT copy of your working backend logic
     */
    createUser(username: string, password?: string): Promise<PipeUser>;
    /**
     * Get auth headers for a user - EXACT copy of your backend auth logic
     */
    private getAuthHeaders;
    /**
     * Upload file to Pipe Network for a specific user with JWT auth
     */
    upload(user: PipeUser, data: Buffer, fileName: string, options?: UploadOptions): Promise<UploadResult>;
    /**
     * Download file from Pipe Network using user_app_key authentication
     * Following the exact Firestarter GUI pattern with user_app_key
     *
     * @param user PipeUser with auth credentials
     * @param fileIdentifier Either Blake3 hash (will lookup storedFileName) or actual storedFileName
     */
    download(user: PipeUser, fileIdentifier: string): Promise<Buffer>;
    /**
     * Delete file from Pipe Network
     * Follows the same pattern as your frontend PipeService.deleteMedia()
     */
    deleteFile(user: PipeUser, fileName: string): Promise<boolean>;
    /**
     * Find user account by wallet address (like your backend does)
     */
    private findUserByWallet;
    /**
     * Get stored account by username (for UserManager integration)
     */
    getStoredAccount(username: string): PipeUserAuth | null;
    /**
     * Check SOL balance for a user - using exact backend auth flow
     */
    checkSolBalance(user: PipeUser): Promise<WalletBalance>;
    /**
     * Check PIPE token balance for a user - using exact backend auth flow
     */
    checkPipeBalance(user: PipeUser): Promise<TokenBalance>;
    /**
     * Exchange SOL for PIPE tokens with JWT auth
     */
    exchangeSolForPipe(user: PipeUser, amountSol: number): Promise<number>;
    /**
     * Generate deterministic password for a user (matches your backend implementation)
     */
    private generateUserPassword;
    /**
     * Calculate Blake3 hash for content addressing
     * This provides the same content addressing used by Firestarter GUI
     */
    private calculateBlake3Hash;
    /**
     * Get API base URL
     */
    getBaseUrl(): string;
    /**
     * Update configuration
     */
    updateConfig(config: Partial<PipeConfig>): void;
}
export {};
//# sourceMappingURL=client.d.ts.map