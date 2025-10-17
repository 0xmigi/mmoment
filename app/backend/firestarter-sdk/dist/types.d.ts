export interface PipeConfig {
    baseUrl?: string;
    timeout?: number;
}
export interface PipeUser {
    userId: string;
    userAppKey: string;
    username?: string;
    solanaPubkey?: string;
}
export interface UploadOptions {
    priority?: boolean;
    fileName?: string;
    metadata?: Record<string, any>;
}
export interface UploadResult {
    fileId: string;
    fileName: string;
    size: number;
    uploadedAt: Date;
    blake3Hash?: string;
    userId?: string;
    metadata?: Record<string, any>;
}
export interface FileRecord {
    fileId: string;
    originalFileName: string;
    storedFileName: string;
    userId: string;
    uploadedAt: Date;
    size: number;
    mimeType?: string;
    blake3Hash?: string;
    metadata: Record<string, any>;
}
export interface TokenBalance {
    balance: number;
    uiAmount: number;
    decimals?: number;
    tokenMint?: string;
}
export interface WalletBalance {
    balanceSol: number;
    publicKey: string;
    userId?: string;
}
export interface PipeError extends Error {
    status?: number;
    code?: string;
}
//# sourceMappingURL=types.d.ts.map