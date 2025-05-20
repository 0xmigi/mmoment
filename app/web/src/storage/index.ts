// Storage Module Exports

// Provider
export * from './storage-provider';
export * from './storage-service';
export * from './ipfs';

// Configuration
export { getPinataCredentials, updatePinataCredentials } from './config';

// UI Components
export { PinataSettings } from './PinataSettings';

// Fix for ambiguous exports
// Export types from walrus/index.ts
export type { WalrusStorageQuota } from './walrus';
// Export the service implementation from walrus/index.ts
export { walrusService } from './walrus';
// Export the service implementation from walrus-sdk-service.ts
export { walrusSdkService } from './walrus/walrus-sdk-service';
// Export other types from the SDK service
export type { } from './walrus/walrus-sdk-service';

// Re-export common interfaces
export type { UploadOptions, StorageResult, StorageService } from './storage-provider';
export type { StorageProvider } from './storage-provider';

// Re-export main services
import { unifiedIpfsService } from './ipfs/unified-ipfs-service';

export {
  unifiedIpfsService
}; 