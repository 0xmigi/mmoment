import { PipeConfig, PipeUser, UploadOptions, UploadResult, FileRecord } from './types';
interface UserSession {
    user: PipeUser;
    createdAt: Date;
    lastActivity: Date;
    metadata: Record<string, any>;
}
export declare class SessionManager {
    private client;
    private tracker;
    private sessions;
    private maxIdleTime;
    constructor(config?: PipeConfig, storagePath?: string);
    /**
     * Set maximum idle time before session cleanup
     */
    setMaxIdleTime(milliseconds: number): void;
    /**
     * Get or create a session for a user (identified by wallet address)
     */
    getOrCreateSession(userId: string): Promise<UserSession>;
    /**
     * Create a new session
     */
    private createSession;
    /**
     * Upload file for a specific user
     */
    uploadForUser(userId: string, data: Buffer, fileName: string, options?: UploadOptions): Promise<UploadResult>;
    /**
     * Upload camera capture for user
     */
    uploadCameraCapture(userId: string, imageData: Buffer, captureType?: string): Promise<UploadResult>;
    /**
     * Download file for a specific user
     */
    downloadForUser(userId: string, fileName: string, priority?: boolean): Promise<Buffer>;
    /**
     * List all files uploaded by a user
     */
    listUserFiles(userId: string): Promise<FileRecord[]>;
    /**
     * Get a specific file record
     */
    getUserFile(fileId: string): Promise<FileRecord | null>;
    /**
     * Search user's files by filename pattern
     */
    searchUserFiles(userId: string, pattern: string): Promise<FileRecord[]>;
    /**
     * Get recent files for a user (with limit)
     */
    getRecentUserFiles(userId: string, limit: number): Promise<FileRecord[]>;
    /**
     * Get user balance information
     */
    getUserBalance(userId: string): Promise<{
        sol: number;
        pipe: number;
        publicKey: string;
    }>;
    /**
     * Exchange SOL for PIPE tokens
     */
    exchangeSolForPipe(userId: string, amountSol: number): Promise<number>;
    /**
     * Create a public link for a file
     */
    createPublicLink(userId: string, fileName: string): Promise<string>;
    /**
     * Get upload statistics
     */
    getUploadStats(): Promise<any>;
    /**
     * Clean up old upload records
     */
    cleanupOldUploads(olderThanDays: number): Promise<number>;
    /**
     * Clean up inactive sessions
     */
    cleanupInactiveSessions(): number;
    /**
     * Get number of active sessions
     */
    getActiveSessionCount(): number;
    /**
     * Get session info for monitoring
     */
    getSessionInfo(userId: string): any;
    /**
     * Check if session is still valid
     */
    private isSessionValid;
    /**
     * Get MIME type from filename
     */
    private getMimeType;
}
export {};
//# sourceMappingURL=session-manager.d.ts.map