"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionManager = void 0;
const client_1 = require("./client");
const tracker_1 = require("./tracker");
const errors_1 = require("./errors");
const path_1 = require("path");
const os_1 = require("os");
class SessionManager {
    constructor(config = {}, storagePath) {
        this.sessions = new Map();
        this.maxIdleTime = 3600000; // 1 hour in milliseconds
        this.client = new client_1.PipeClient(config);
        // Default storage path: ~/.mmoment/pipe-uploads.json
        const defaultPath = (0, path_1.join)((0, os_1.homedir)(), '.mmoment', 'pipe-uploads.json');
        this.tracker = new tracker_1.UploadTracker(storagePath || defaultPath);
    }
    /**
     * Set maximum idle time before session cleanup
     */
    setMaxIdleTime(milliseconds) {
        this.maxIdleTime = milliseconds;
    }
    /**
     * Get or create a session for a user (identified by wallet address)
     */
    async getOrCreateSession(userId) {
        // Check if session exists and is still valid
        const existing = this.sessions.get(userId);
        if (existing && this.isSessionValid(existing)) {
            existing.lastActivity = new Date();
            return existing;
        }
        // Create new session
        return this.createSession(userId);
    }
    /**
     * Create a new session
     */
    async createSession(userId) {
        if (!userId || userId.length < 8) {
            throw new errors_1.PipeValidationError('User ID must be at least 8 characters');
        }
        // Generate username from userId (for deterministic account creation)
        const username = `mmoment_${userId.slice(0, 16)}`;
        let user;
        try {
            // Try to create user (might already exist)
            user = await this.client.createUser(username);
        }
        catch (error) {
            if (error.message?.includes('exists') || error.status === 409) {
                // User already exists, create placeholder
                // In production, you'd retrieve stored credentials
                user = {
                    userId: username,
                    userAppKey: '', // Would be retrieved from secure storage
                    username,
                };
            }
            else {
                throw error;
            }
        }
        // Create session
        const session = {
            user,
            createdAt: new Date(),
            lastActivity: new Date(),
            metadata: {},
        };
        // Store in map
        this.sessions.set(userId, session);
        return session;
    }
    /**
     * Upload file for a specific user
     */
    async uploadForUser(userId, data, fileName, options = {}) {
        const session = await this.getOrCreateSession(userId);
        session.lastActivity = new Date();
        // Upload to Pipe
        const result = await this.client.upload(session.user, data, fileName, options);
        // Record the upload in our tracker
        const record = {
            fileId: result.fileId,
            originalFileName: fileName,
            storedFileName: result.fileName,
            userId,
            uploadedAt: result.uploadedAt,
            size: result.size,
            mimeType: this.getMimeType(fileName),
            blake3Hash: result.blake3Hash,
            metadata: options.metadata || {},
        };
        await this.tracker.recordUpload(record);
        return result;
    }
    /**
     * Upload camera capture for user
     */
    async uploadCameraCapture(userId, imageData, captureType = 'photo') {
        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `mmoment_${captureType}_${timestamp}.jpg`;
        const options = {
            priority: captureType === 'video', // Videos get priority
            fileName,
            metadata: {
                captureType,
                camera: 'jetson',
                timestamp,
            },
        };
        return this.uploadForUser(userId, imageData, fileName, options);
    }
    /**
     * Download file for a specific user
     */
    async downloadForUser(userId, fileName, priority = false) {
        const session = await this.getOrCreateSession(userId);
        session.lastActivity = new Date();
        return this.client.download(session.user, fileName, priority);
    }
    /**
     * List all files uploaded by a user
     */
    async listUserFiles(userId) {
        return this.tracker.getUserFiles(userId);
    }
    /**
     * Get a specific file record
     */
    async getUserFile(fileId) {
        return this.tracker.getFile(fileId);
    }
    /**
     * Search user's files by filename pattern
     */
    async searchUserFiles(userId, pattern) {
        return this.tracker.searchFiles(userId, pattern);
    }
    /**
     * Get recent files for a user (with limit)
     */
    async getRecentUserFiles(userId, limit) {
        return this.tracker.getRecentUserFiles(userId, limit);
    }
    /**
     * Get user balance information
     */
    async getUserBalance(userId) {
        const session = await this.getOrCreateSession(userId);
        session.lastActivity = new Date();
        const [solBalance, pipeBalance] = await Promise.all([
            this.client.checkSolBalance(session.user),
            this.client.checkPipeBalance(session.user),
        ]);
        return {
            sol: solBalance.balanceSol,
            pipe: pipeBalance.uiAmount,
            publicKey: solBalance.publicKey,
        };
    }
    /**
     * Exchange SOL for PIPE tokens
     */
    async exchangeSolForPipe(userId, amountSol) {
        const session = await this.getOrCreateSession(userId);
        session.lastActivity = new Date();
        return this.client.exchangeSolForPipe(session.user, amountSol);
    }
    /**
     * Create a public link for a file
     */
    async createPublicLink(userId, fileName) {
        const session = await this.getOrCreateSession(userId);
        session.lastActivity = new Date();
        return this.client.createPublicLink(session.user, fileName);
    }
    /**
     * Get upload statistics
     */
    async getUploadStats() {
        return this.tracker.getStats();
    }
    /**
     * Clean up old upload records
     */
    async cleanupOldUploads(olderThanDays) {
        return this.tracker.cleanupOldRecords(olderThanDays);
    }
    /**
     * Clean up inactive sessions
     */
    cleanupInactiveSessions() {
        const now = Date.now();
        let cleanedUp = 0;
        for (const [userId, session] of this.sessions.entries()) {
            if (now - session.lastActivity.getTime() > this.maxIdleTime) {
                this.sessions.delete(userId);
                cleanedUp++;
            }
        }
        return cleanedUp;
    }
    /**
     * Get number of active sessions
     */
    getActiveSessionCount() {
        return this.sessions.size;
    }
    /**
     * Get session info for monitoring
     */
    getSessionInfo(userId) {
        const session = this.sessions.get(userId);
        if (!session) {
            return null;
        }
        return {
            userId: session.user.userId,
            username: session.user.username,
            createdAt: session.createdAt,
            lastActivity: session.lastActivity,
            idleTime: Date.now() - session.lastActivity.getTime(),
            metadata: session.metadata,
        };
    }
    /**
     * Check if session is still valid
     */
    isSessionValid(session) {
        const idleTime = Date.now() - session.lastActivity.getTime();
        return idleTime < this.maxIdleTime;
    }
    /**
     * Get MIME type from filename
     */
    getMimeType(fileName) {
        const ext = fileName.split('.').pop()?.toLowerCase();
        switch (ext) {
            case 'jpg':
            case 'jpeg':
                return 'image/jpeg';
            case 'png':
                return 'image/png';
            case 'gif':
                return 'image/gif';
            case 'webp':
                return 'image/webp';
            case 'mp4':
                return 'video/mp4';
            case 'webm':
                return 'video/webm';
            case 'mov':
                return 'video/quicktime';
            default:
                return 'application/octet-stream';
        }
    }
}
exports.SessionManager = SessionManager;
//# sourceMappingURL=session-manager.js.map