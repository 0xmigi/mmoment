/**
 * UploadHistory - Client-side upload tracking for the Firestarter SDK
 *
 * Tracks file uploads locally to provide listing functionality,
 * since Pipe Network doesn't have a native file listing API.
 * This connects with your existing frontend patterns.
 */
import { FileRecord } from './types';
declare global {
    var window: any;
    var localStorage: any;
}
interface UploadHistoryStorage {
    getItem(key: string): string | null;
    setItem(key: string, value: string): void;
    removeItem(key: string): void;
}
export declare class UploadHistory {
    private storage;
    private readonly storageKey;
    constructor(storage?: UploadHistoryStorage);
    /**
     * Record a new file upload
     * This is called automatically by the FirestarterSDK after successful uploads
     */
    recordUpload(record: FileRecord): Promise<void>;
    /**
     * Get upload history for a specific user (wallet address)
     * Returns files sorted by upload date (most recent first)
     */
    getUserFiles(userId: string): Promise<FileRecord[]>;
    /**
     * Get a specific file record by fileId
     */
    getFileRecord(fileId: string): Promise<FileRecord | null>;
    /**
     * Get a file record by Blake3 hash
     * This is needed for downloads since we store Blake3 hash as fileId
     */
    getFileByHash(blake3Hash: string): Promise<FileRecord | null>;
    /**
     * Remove a file record from history
     * This is called when files are deleted from Pipe Network
     */
    removeFileRecord(fileId: string): Promise<boolean>;
    /**
     * Get upload statistics for a user
     */
    getUserStats(userId: string): Promise<{
        totalFiles: number;
        totalSize: number;
        oldestUpload?: Date;
        newestUpload?: Date;
    }>;
    /**
     * Search files by filename or metadata
     */
    searchFiles(userId: string, query: string): Promise<FileRecord[]>;
    /**
     * Clear all upload history for a specific user
     */
    clearUserHistory(userId: string): Promise<number>;
    /**
     * Clear all upload history (use with caution)
     */
    clearAllHistory(): Promise<void>;
    /**
     * Export upload history as JSON
     */
    exportHistory(userId?: string): Promise<string>;
    /**
     * Import upload history from JSON
     */
    importHistory(jsonData: string, merge?: boolean): Promise<number>;
    /**
     * Get the raw upload history array
     */
    private getUploadHistory;
    /**
     * Save the upload history array
     */
    private saveUploadHistory;
}
export declare class BrowserStorage implements UploadHistoryStorage {
    getItem(key: string): string | null;
    setItem(key: string, value: string): void;
    removeItem(key: string): void;
}
export declare class FileSystemStorage implements UploadHistoryStorage {
    private filePath;
    constructor(filePath?: string);
    getItem(_key: string): string | null;
    setItem(_key: string, value: string): void;
    removeItem(_key: string): void;
}
export {};
//# sourceMappingURL=upload-history.d.ts.map