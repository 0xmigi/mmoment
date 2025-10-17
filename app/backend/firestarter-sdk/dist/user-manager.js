"use strict";
/**
 * UserManager - Handles wallet to Pipe account mapping
 *
 * Manages the relationship between user wallets and their Pipe Network accounts,
 * following the same pattern as your existing frontend PipeService and PipeWalletBridge.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.UserManager = void 0;
const errors_1 = require("./errors");
class UserManager {
    constructor(client) {
        this.sessions = new Map();
        this.client = client;
    }
    /**
     * Create or get existing Pipe account for a wallet address
     * This mirrors your existing PipeService.createOrGetAccount()
     */
    async createOrGetUser(walletAddress) {
        if (!walletAddress || walletAddress.length < 8) {
            throw new errors_1.PipeValidationError('Wallet address must be at least 8 characters');
        }
        // Check if we already have a session for this wallet
        const existing = this.sessions.get(walletAddress);
        if (existing && this.isSessionValid(existing)) {
            existing.lastActivity = new Date();
            return existing.user;
        }
        // Create new Pipe account using deterministic username
        // Use first 20 characters of wallet address as username
        const username = walletAddress.slice(0, 20);
        let user;
        try {
            // Try to create user account (might already exist)
            user = await this.client.createUser(username);
            console.log(`✅ Created new Pipe user: ${username}`);
        }
        catch (error) {
            if (error.message?.includes('exists') || error.status === 409) {
                // User already exists - this is fine
                console.log(`ℹ️ User ${username} already exists`);
                // Create placeholder user object
                // In a real implementation, you'd retrieve the stored credentials
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
        // Create and store session
        const session = {
            user,
            createdAt: new Date(),
            lastActivity: new Date(),
            walletAddress,
        };
        this.sessions.set(walletAddress, session);
        return user;
    }
    /**
     * Get existing user session
     */
    async getUser(walletAddress) {
        // First check if we have an authenticated account in the client
        const username = `mmoment_${walletAddress.slice(0, 16)}`;
        const authenticatedAccount = this.client.getStoredAccount(username);
        if (authenticatedAccount) {
            // Return user from authenticated account data
            return {
                userId: authenticatedAccount.userId,
                userAppKey: authenticatedAccount.userAppKey,
                username: authenticatedAccount.username,
            };
        }
        const session = this.sessions.get(walletAddress);
        if (!session || !this.isSessionValid(session)) {
            // No valid session, create new one
            return this.createOrGetUser(walletAddress);
        }
        session.lastActivity = new Date();
        return session.user;
    }
    /**
     * Check if user has a valid session
     */
    hasUser(walletAddress) {
        const session = this.sessions.get(walletAddress);
        return session ? this.isSessionValid(session) : false;
    }
    /**
     * Get session info for monitoring
     */
    getSessionInfo(walletAddress) {
        const session = this.sessions.get(walletAddress);
        if (!session) {
            return null;
        }
        return {
            userId: session.user.userId,
            username: session.user.username,
            walletAddress: session.walletAddress,
            createdAt: session.createdAt,
            lastActivity: session.lastActivity,
            idleTime: Date.now() - session.lastActivity.getTime(),
        };
    }
    /**
     * Clean up inactive sessions
     */
    cleanupInactiveSessions(maxIdleTimeMs = 3600000) {
        const now = Date.now();
        let cleanedUp = 0;
        for (const [walletAddress, session] of this.sessions.entries()) {
            if (now - session.lastActivity.getTime() > maxIdleTimeMs) {
                this.sessions.delete(walletAddress);
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
     * Clear all sessions
     */
    clearAllSessions() {
        this.sessions.clear();
    }
    /**
     * Check if session is still valid
     */
    isSessionValid(session) {
        const maxIdleTime = 3600000; // 1 hour
        const idleTime = Date.now() - session.lastActivity.getTime();
        return idleTime < maxIdleTime;
    }
    /**
     * Generate deterministic password for a user (matches your backend implementation)
     */
    generateUserPassword(userId) {
        // This would use crypto in a real implementation
        // For now, return a placeholder
        return `password_${userId}`;
    }
}
exports.UserManager = UserManager;
//# sourceMappingURL=user-manager.js.map