/**
 * UserManager - Handles wallet to Pipe account mapping
 *
 * Manages the relationship between user wallets and their Pipe Network accounts,
 * following the same pattern as your existing frontend PipeService and PipeWalletBridge.
 */
import { FirestarterClient } from './client';
import { PipeUser } from './types';
export declare class UserManager {
    private client;
    private sessions;
    constructor(client: FirestarterClient);
    /**
     * Create or get existing Pipe account for a wallet address
     * This mirrors your existing PipeService.createOrGetAccount()
     */
    createOrGetUser(walletAddress: string): Promise<PipeUser>;
    /**
     * Get existing user session
     */
    getUser(walletAddress: string): Promise<PipeUser>;
    /**
     * Check if user has a valid session
     */
    hasUser(walletAddress: string): boolean;
    /**
     * Get session info for monitoring
     */
    getSessionInfo(walletAddress: string): any;
    /**
     * Clean up inactive sessions
     */
    cleanupInactiveSessions(maxIdleTimeMs?: number): number;
    /**
     * Get number of active sessions
     */
    getActiveSessionCount(): number;
    /**
     * Clear all sessions
     */
    clearAllSessions(): void;
    /**
     * Check if session is still valid
     */
    private isSessionValid;
    /**
     * Generate deterministic password for a user (matches your backend implementation)
     */
    private generateUserPassword;
}
//# sourceMappingURL=user-manager.d.ts.map