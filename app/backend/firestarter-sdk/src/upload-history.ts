/**
 * UploadHistory - Client-side upload tracking for the Firestarter SDK
 *
 * Tracks file uploads locally to provide listing functionality,
 * since Pipe Network doesn't have a native file listing API.
 * This connects with your existing frontend patterns.
 */

import { FileRecord } from './types';
import { PipeStorageError } from './errors';

// Global declarations for browser compatibility
declare global {
  var window: any;
  var localStorage: any;
}

interface UploadHistoryStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

// Default storage adapter for browser/Node.js
class DefaultStorage implements UploadHistoryStorage {
  private data: Map<string, string> = new Map();

  getItem(key: string): string | null {
    return this.data.get(key) || null;
  }

  setItem(key: string, value: string): void {
    this.data.set(key, value);
  }

  removeItem(key: string): void {
    this.data.delete(key);
  }
}

export class UploadHistory {
  private storage: UploadHistoryStorage;
  private readonly storageKey = 'firestarter_upload_history';

  constructor(storage?: UploadHistoryStorage) {
    // Use provided storage or default in-memory storage
    this.storage = storage || new DefaultStorage();
  }

  /**
   * Record a new file upload
   * This is called automatically by the FirestarterSDK after successful uploads
   */
  async recordUpload(record: FileRecord): Promise<void> {
    try {
      const history = await this.getUploadHistory();

      // Add new record to the beginning of the array (most recent first)
      history.unshift({
        ...record,
        uploadedAt: record.uploadedAt || new Date(),
      });

      // Keep only the last 1000 uploads per user to prevent unlimited growth
      const userRecords = history.filter(r => r.userId === record.userId);
      if (userRecords.length > 1000) {
        // Remove older records for this user
        const otherUsers = history.filter(r => r.userId !== record.userId);
        const recentUserRecords = userRecords.slice(0, 1000);
        const updatedHistory = [...recentUserRecords, ...otherUsers];
        await this.saveUploadHistory(updatedHistory);
      } else {
        await this.saveUploadHistory(history);
      }

      console.log(`📝 Recorded upload: ${record.originalFileName} for user ${record.userId}`);
    } catch (error) {
      console.error('Failed to record upload:', error);
      throw new PipeStorageError(`Failed to record upload: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get upload history for a specific user (wallet address)
   * Returns files sorted by upload date (most recent first)
   */
  async getUserFiles(userId: string): Promise<FileRecord[]> {
    try {
      const history = await this.getUploadHistory();
      return history
        .filter(record => record.userId === userId)
        .sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime());
    } catch (error) {
      console.error('Failed to get user files:', error);
      return [];
    }
  }

  /**
   * Get a specific file record by fileId
   */
  async getFileRecord(fileId: string): Promise<FileRecord | null> {
    try {
      const history = await this.getUploadHistory();
      return history.find(record => record.fileId === fileId) || null;
    } catch (error) {
      console.error('Failed to get file record:', error);
      return null;
    }
  }

  /**
   * Get a file record by Blake3 hash
   * This is needed for downloads since we store Blake3 hash as fileId
   */
  async getFileByHash(blake3Hash: string): Promise<FileRecord | null> {
    try {
      const history = await this.getUploadHistory();
      return history.find(record => record.blake3Hash === blake3Hash) || null;
    } catch (error) {
      console.error('Failed to get file by hash:', error);
      return null;
    }
  }

  /**
   * Remove a file record from history
   * This is called when files are deleted from Pipe Network
   */
  async removeFileRecord(fileId: string): Promise<boolean> {
    try {
      const history = await this.getUploadHistory();
      const initialLength = history.length;
      const updatedHistory = history.filter(record => record.fileId !== fileId);

      if (updatedHistory.length < initialLength) {
        await this.saveUploadHistory(updatedHistory);
        console.log(`🗑️ Removed file record: ${fileId}`);
        return true;
      }

      return false;
    } catch (error) {
      console.error('Failed to remove file record:', error);
      return false;
    }
  }

  /**
   * Get upload statistics for a user
   */
  async getUserStats(userId: string): Promise<{
    totalFiles: number;
    totalSize: number;
    oldestUpload?: Date;
    newestUpload?: Date;
  }> {
    try {
      const userFiles = await this.getUserFiles(userId);

      if (userFiles.length === 0) {
        return { totalFiles: 0, totalSize: 0 };
      }

      const totalSize = userFiles.reduce((sum, file) => sum + (file.size || 0), 0);
      const uploadDates = userFiles.map(file => new Date(file.uploadedAt));

      return {
        totalFiles: userFiles.length,
        totalSize,
        oldestUpload: new Date(Math.min(...uploadDates.map(d => d.getTime()))),
        newestUpload: new Date(Math.max(...uploadDates.map(d => d.getTime()))),
      };
    } catch (error) {
      console.error('Failed to get user stats:', error);
      return { totalFiles: 0, totalSize: 0 };
    }
  }

  /**
   * Search files by filename or metadata
   */
  async searchFiles(userId: string, query: string): Promise<FileRecord[]> {
    try {
      const userFiles = await this.getUserFiles(userId);
      const lowercaseQuery = query.toLowerCase();

      return userFiles.filter(file =>
        file.originalFileName.toLowerCase().includes(lowercaseQuery) ||
        file.storedFileName.toLowerCase().includes(lowercaseQuery) ||
        Object.values(file.metadata || {}).some(value =>
          String(value).toLowerCase().includes(lowercaseQuery)
        )
      );
    } catch (error) {
      console.error('Failed to search files:', error);
      return [];
    }
  }

  /**
   * Clear all upload history for a specific user
   */
  async clearUserHistory(userId: string): Promise<number> {
    try {
      const history = await this.getUploadHistory();
      const initialLength = history.length;
      const updatedHistory = history.filter(record => record.userId !== userId);

      await this.saveUploadHistory(updatedHistory);
      const removedCount = initialLength - updatedHistory.length;

      console.log(`🗑️ Cleared ${removedCount} records for user ${userId}`);
      return removedCount;
    } catch (error) {
      console.error('Failed to clear user history:', error);
      return 0;
    }
  }

  /**
   * Clear all upload history (use with caution)
   */
  async clearAllHistory(): Promise<void> {
    try {
      this.storage.removeItem(this.storageKey);
      console.log('🗑️ Cleared all upload history');
    } catch (error) {
      console.error('Failed to clear all history:', error);
      throw new PipeStorageError(`Failed to clear history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Export upload history as JSON
   */
  async exportHistory(userId?: string): Promise<string> {
    try {
      const history = await this.getUploadHistory();
      const dataToExport = userId
        ? history.filter(record => record.userId === userId)
        : history;

      return JSON.stringify(dataToExport, null, 2);
    } catch (error) {
      console.error('Failed to export history:', error);
      throw new PipeStorageError(`Failed to export history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Import upload history from JSON
   */
  async importHistory(jsonData: string, merge = true): Promise<number> {
    try {
      const importedRecords: FileRecord[] = JSON.parse(jsonData);

      if (!Array.isArray(importedRecords)) {
        throw new Error('Invalid data format: expected array of FileRecord objects');
      }

      if (merge) {
        const existingHistory = await this.getUploadHistory();
        const combined = [...importedRecords, ...existingHistory];

        // Remove duplicates based on fileId
        const unique = combined.filter((record, index, arr) =>
          arr.findIndex(r => r.fileId === record.fileId) === index
        );

        await this.saveUploadHistory(unique);
        return importedRecords.length;
      } else {
        await this.saveUploadHistory(importedRecords);
        return importedRecords.length;
      }
    } catch (error) {
      console.error('Failed to import history:', error);
      throw new PipeStorageError(`Failed to import history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get the raw upload history array
   */
  private async getUploadHistory(): Promise<FileRecord[]> {
    try {
      const historyJson = this.storage.getItem(this.storageKey);
      if (!historyJson) {
        return [];
      }

      return JSON.parse(historyJson);
    } catch (error) {
      console.warn('Failed to parse upload history, starting fresh:', error);
      return [];
    }
  }

  /**
   * Save the upload history array
   */
  private async saveUploadHistory(history: FileRecord[]): Promise<void> {
    try {
      const historyJson = JSON.stringify(history);
      this.storage.setItem(this.storageKey, historyJson);
    } catch (error) {
      throw new PipeStorageError(`Failed to save upload history: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

// Browser localStorage adapter
export class BrowserStorage implements UploadHistoryStorage {
  getItem(key: string): string | null {
    if (typeof window !== 'undefined' && window.localStorage) {
      return window.localStorage.getItem(key);
    }
    return null;
  }

  setItem(key: string, value: string): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(key, value);
    }
  }

  removeItem(key: string): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(key);
    }
  }
}

// Node.js file system adapter
export class FileSystemStorage implements UploadHistoryStorage {
  private filePath: string;

  constructor(filePath = './firestarter-upload-history.json') {
    this.filePath = filePath;
  }

  getItem(_key: string): string | null {
    try {
      const fs = require('fs');
      if (fs.existsSync(this.filePath)) {
        return fs.readFileSync(this.filePath, 'utf8');
      }
    } catch (error) {
      console.warn('Failed to read upload history file:', error);
    }
    return null;
  }

  setItem(_key: string, value: string): void {
    try {
      const fs = require('fs');
      fs.writeFileSync(this.filePath, value, 'utf8');
    } catch (error) {
      console.error('Failed to write upload history file:', error);
    }
  }

  removeItem(_key: string): void {
    try {
      const fs = require('fs');
      if (fs.existsSync(this.filePath)) {
        fs.unlinkSync(this.filePath);
      }
    } catch (error) {
      console.error('Failed to remove upload history file:', error);
    }
  }
}