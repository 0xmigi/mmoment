"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.FirestarterSDK = exports.PipeStorageError = exports.PipeSessionError = exports.PipeValidationError = exports.PipeApiError = exports.UploadHistory = exports.UserManager = exports.FirestarterClient = void 0;
exports.createFirestarterSDK = createFirestarterSDK;
// Main SDK exports
var client_1 = require("./client");
Object.defineProperty(exports, "FirestarterClient", { enumerable: true, get: function () { return client_1.FirestarterClient; } });
var user_manager_1 = require("./user-manager");
Object.defineProperty(exports, "UserManager", { enumerable: true, get: function () { return user_manager_1.UserManager; } });
var upload_history_1 = require("./upload-history");
Object.defineProperty(exports, "UploadHistory", { enumerable: true, get: function () { return upload_history_1.UploadHistory; } });
// Import classes and types for internal use
const client_2 = require("./client");
const user_manager_2 = require("./user-manager");
const upload_history_2 = require("./upload-history");
// Errors
var errors_1 = require("./errors");
Object.defineProperty(exports, "PipeApiError", { enumerable: true, get: function () { return errors_1.PipeApiError; } });
Object.defineProperty(exports, "PipeValidationError", { enumerable: true, get: function () { return errors_1.PipeValidationError; } });
Object.defineProperty(exports, "PipeSessionError", { enumerable: true, get: function () { return errors_1.PipeSessionError; } });
Object.defineProperty(exports, "PipeStorageError", { enumerable: true, get: function () { return errors_1.PipeStorageError; } });
// Main SDK class - this is what developers will use
class FirestarterSDK {
    constructor(config = {}) {
        this.client = new client_2.FirestarterClient(config);
        this.userManager = new user_manager_2.UserManager(this.client);
        // Use persistent file storage for upload history in Node.js environments
        this.uploadHistory = new upload_history_2.UploadHistory(new upload_history_2.FileSystemStorage('./firestarter-upload-history.json'));
        // Give the client access to upload history for download filename lookup
        this.client.uploadHistory = this.uploadHistory;
    }
    /**
     * Create or get existing Pipe account for a wallet
     */
    async createUserAccount(walletAddress) {
        return this.userManager.createOrGetUser(walletAddress);
    }
    /**
     * Upload file to Pipe Network for a specific user
     */
    async uploadFile(walletAddress, data, fileName, options = {}) {
        const user = await this.userManager.getUser(walletAddress);
        const result = await this.client.upload(user, data, fileName, options);
        // Track upload locally
        await this.uploadHistory.recordUpload({
            fileId: result.fileId,
            originalFileName: fileName,
            storedFileName: result.fileName,
            userId: walletAddress,
            uploadedAt: result.uploadedAt,
            size: result.size,
            blake3Hash: result.blake3Hash,
            metadata: options.metadata || {},
        });
        return result;
    }
    /**
     * List files for a user
     */
    async listUserFiles(walletAddress) {
        return this.uploadHistory.getUserFiles(walletAddress);
    }
    /**
     * Get user's Pipe wallet balance
     */
    async getUserBalance(walletAddress) {
        const user = await this.userManager.getUser(walletAddress);
        const [solBalance, pipeBalance] = await Promise.all([
            this.client.checkSolBalance(user),
            this.client.checkPipeBalance(user),
        ]);
        return {
            sol: solBalance.balanceSol,
            pipe: pipeBalance.uiAmount,
            publicKey: solBalance.publicKey,
        };
    }
    /**
     * Download file from Pipe Network
     */
    async downloadFile(walletAddress, fileId) {
        const user = await this.userManager.getUser(walletAddress);
        return this.client.download(user, fileId);
    }
    /**
     * Exchange SOL for PIPE tokens
     */
    async exchangeSolForPipe(walletAddress, amountSol) {
        const user = await this.userManager.getUser(walletAddress);
        return this.client.exchangeSolForPipe(user, amountSol);
    }
}
exports.FirestarterSDK = FirestarterSDK;
// Convenience function for quick setup
function createFirestarterSDK(config) {
    return new FirestarterSDK(config);
}
//# sourceMappingURL=index.js.map