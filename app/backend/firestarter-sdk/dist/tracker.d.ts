import { FileRecord } from './types';
export interface UploadHistory {
    files: Record<string, FileRecord>;
    userFiles: Record<string, string[]>;
    lastUpdated: Date;
}
export declare class UploadTracker {
    private storagePath;
    private cache;
    constructor(storagePath: string);
    /**
     * Record a successful upload
     */
    recordUpload(record: FileRecord): Promise<void>;
    /**
     * Get all files for a user
     */
    getUserFiles(userId: string): Promise<FileRecord[]>;
    /**
     * Get a specific file by ID
     */
    getFile(fileId: string): Promise<FileRecord | null>;
    /**
     * Search files by filename pattern
     */
    searchFiles(userId: string, pattern: string): Promise<FileRecord[]>;
    /**
     * Get recent files for a user (with limit)
     */
    getRecentUserFiles(userId: string, limit: number): Promise<FileRecord[]>;
    /**
     * Get upload statistics
     */
    getStats(): Promise<{
        totalFiles: number;
        totalUsers: number;
        totalSize: number;
        lastUpdated: Date;
    }>;
    /**
     * Clean up old records
     */
    cleanupOldRecords(olderThanDays: number): Promise<number>;
    /**
     * Load history from disk (with caching)
     */
    private loadHistory;
    /**
     * Save history to disk
     */
    private saveHistory;
    /**
     * Clear the cache (useful for testing)
     */
    clearCache(): void;
}
//# sourceMappingURL=tracker.d.ts.map