"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UploadTracker = void 0;
const fs_1 = require("fs");
const path_1 = require("path");
const errors_1 = require("./errors");
class UploadTracker {
    constructor(storagePath) {
        this.cache = null;
        this.storagePath = storagePath;
    }
    /**
     * Record a successful upload
     */
    async recordUpload(record) {
        const history = await this.loadHistory();
        // Add to files map
        history.files[record.fileId] = record;
        // Add to user index
        if (!history.userFiles[record.userId]) {
            history.userFiles[record.userId] = [];
        }
        history.userFiles[record.userId].push(record.fileId);
        // Update timestamp
        history.lastUpdated = new Date();
        // Save and update cache
        await this.saveHistory(history);
        this.cache = history;
    }
    /**
     * Get all files for a user
     */
    async getUserFiles(userId) {
        const history = await this.loadHistory();
        const fileIds = history.userFiles[userId] || [];
        const files = [];
        for (const fileId of fileIds) {
            const record = history.files[fileId];
            if (record) {
                files.push(record);
            }
        }
        // Sort by upload time, newest first
        return files.sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime());
    }
    /**
     * Get a specific file by ID
     */
    async getFile(fileId) {
        const history = await this.loadHistory();
        return history.files[fileId] || null;
    }
    /**
     * Search files by filename pattern
     */
    async searchFiles(userId, pattern) {
        const files = await this.getUserFiles(userId);
        const patternLower = pattern.toLowerCase();
        return files.filter(file => file.originalFileName.toLowerCase().includes(patternLower) ||
            file.storedFileName.toLowerCase().includes(patternLower));
    }
    /**
     * Get recent files for a user (with limit)
     */
    async getRecentUserFiles(userId, limit) {
        const files = await this.getUserFiles(userId);
        return files.slice(0, limit);
    }
    /**
     * Get upload statistics
     */
    async getStats() {
        const history = await this.loadHistory();
        const totalFiles = Object.keys(history.files).length;
        const totalUsers = Object.keys(history.userFiles).length;
        const totalSize = Object.values(history.files).reduce((sum, file) => sum + file.size, 0);
        return {
            totalFiles,
            totalUsers,
            totalSize,
            lastUpdated: history.lastUpdated,
        };
    }
    /**
     * Clean up old records
     */
    async cleanupOldRecords(olderThanDays) {
        const history = await this.loadHistory();
        const cutoff = new Date(Date.now() - olderThanDays * 24 * 60 * 60 * 1000);
        let removedCount = 0;
        // Remove old files
        const filesToKeep = {};
        for (const [fileId, record] of Object.entries(history.files)) {
            if (new Date(record.uploadedAt) >= cutoff) {
                filesToKeep[fileId] = record;
            }
            else {
                removedCount++;
            }
        }
        history.files = filesToKeep;
        // Rebuild user index
        history.userFiles = {};
        for (const [fileId, record] of Object.entries(history.files)) {
            if (!history.userFiles[record.userId]) {
                history.userFiles[record.userId] = [];
            }
            history.userFiles[record.userId].push(fileId);
        }
        if (removedCount > 0) {
            history.lastUpdated = new Date();
            await this.saveHistory(history);
            this.cache = history;
        }
        return removedCount;
    }
    /**
     * Load history from disk (with caching)
     */
    async loadHistory() {
        if (this.cache) {
            return this.cache;
        }
        try {
            const data = await fs_1.promises.readFile(this.storagePath, 'utf-8');
            const parsed = JSON.parse(data);
            // Convert date strings back to Date objects
            const history = {
                files: {},
                userFiles: parsed.userFiles || {},
                lastUpdated: new Date(parsed.lastUpdated || Date.now()),
            };
            // Convert file dates
            for (const [fileId, record] of Object.entries(parsed.files || {})) {
                history.files[fileId] = {
                    ...record,
                    uploadedAt: new Date(record.uploadedAt),
                };
            }
            this.cache = history;
            return history;
        }
        catch (error) {
            if (error.code === 'ENOENT') {
                // File doesn't exist, create empty history
                const emptyHistory = {
                    files: {},
                    userFiles: {},
                    lastUpdated: new Date(),
                };
                this.cache = emptyHistory;
                return emptyHistory;
            }
            throw new errors_1.PipeStorageError(`Failed to load upload history: ${error.message}`);
        }
    }
    /**
     * Save history to disk
     */
    async saveHistory(history) {
        try {
            // Ensure directory exists
            await fs_1.promises.mkdir((0, path_1.dirname)(this.storagePath), { recursive: true });
            // Write to temp file first, then rename (atomic operation)
            const tempPath = `${this.storagePath}.tmp`;
            await fs_1.promises.writeFile(tempPath, JSON.stringify(history, null, 2));
            await fs_1.promises.rename(tempPath, this.storagePath);
        }
        catch (error) {
            throw new errors_1.PipeStorageError(`Failed to save upload history: ${error.message}`);
        }
    }
    /**
     * Clear the cache (useful for testing)
     */
    clearCache() {
        this.cache = null;
    }
}
exports.UploadTracker = UploadTracker;
//# sourceMappingURL=tracker.js.map