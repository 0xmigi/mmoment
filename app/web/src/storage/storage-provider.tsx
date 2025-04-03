import React, { createContext, useContext, useState, useCallback } from 'react';
import { unifiedIpfsService } from './ipfs/unified-ipfs-service';
import { pinataService } from './ipfs/pinata-service';
import { filebaseService } from './ipfs/filebase-service';

console.log('Storage provider initializing with IPFS services');

// Common interfaces for storage operations
export interface UploadOptions {
  pin?: boolean;
  metadata?: Record<string, any>;
  onProgress?: (progress: number) => void;
  directUrl?: string;
}

export interface StorageResult {
  id: string;           // Unique identifier (CID for IPFS, UUID for others)
  url: string;          // URL to access the content
  metadata?: any;       // Any additional metadata
}

// Storage service interface - all storage providers must implement this
export interface StorageService {
  name: string;
  upload: (file: File, options?: UploadOptions) => Promise<StorageResult>;
  uploadBlob: (blob: Blob, filename: string, options?: UploadOptions) => Promise<StorageResult>;
  getUrl: (fileId: string) => string;
  delete?: (fileId: string) => Promise<boolean>;
  isAvailable: () => Promise<boolean>;
}

// All available storage providers
export type StorageProvider = 'ipfs' | 'pinata' | 'filebase' | 'walrus';

// Storage context type
interface StorageContextType {
  service: StorageService;
  activeProvider: StorageProvider;
  setProvider: (provider: StorageProvider) => void;
  isLoading: boolean;
  error: string | null;
}

// Create storage context
export const StorageContext = createContext<StorageContextType | null>(null);

// Storage context provider component
export function StorageProvider({ 
  children,
  defaultProvider = 'ipfs'
}: { 
  children: React.ReactNode;
  defaultProvider?: StorageProvider;
}) {
  const [activeProvider, setActiveProvider] = useState<StorageProvider>(defaultProvider);
  const [service, setService] = useState<StorageService>(() => getStorageService(defaultProvider));
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Switch storage service implementation
  const setProvider = useCallback(async (provider: StorageProvider) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Get the new service
      const newService = getStorageService(provider);
      
      // Check if service is available
      const isAvailable = await newService.isAvailable();
      if (!isAvailable) {
        throw new Error(`Storage provider ${provider} is not available`);
      }
      
      // Update state
      setActiveProvider(provider);
      setService(newService);
      
      // Save preference to localStorage
      localStorage.setItem('preferredStorageProvider', provider);
      
    } catch (err) {
      console.error(`Failed to switch to storage provider ${provider}:`, err);
      setError(err instanceof Error ? err.message : 'Failed to switch storage provider');
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  return (
    <StorageContext.Provider value={{ 
      service, 
      activeProvider, 
      setProvider,
      isLoading,
      error
    }}>
      {children}
    </StorageContext.Provider>
  );
}

// Helper to get storage service implementation
function getStorageService(provider: StorageProvider): StorageService {
  // Import storage services
  switch(provider) {
    case 'ipfs':
      return {
        name: 'IPFS',
        upload: async (file, options) => {
          const results = await unifiedIpfsService.uploadFile(
            file, 
            options?.metadata?.walletAddress || 'anonymous', 
            'image',
            options
          );
          return results[0] || { id: 'error', url: '' };
        },
        uploadBlob: async (blob, filename, options) => {
          const results = await unifiedIpfsService.uploadFile(
            blob, 
            options?.metadata?.walletAddress || 'anonymous', 
            filename.endsWith('.mp4') || filename.endsWith('.mov') ? 'video' : 'image',
            options
          );
          return results[0] || { id: 'error', url: '' };
        },
        getUrl: (fileId) => `https://ipfs.io/ipfs/${fileId}`,
        isAvailable: async () => true
      } as StorageService;
    
    case 'pinata':
      return {
        name: 'Pinata',
        upload: async (file, options) => {
          const url = await pinataService.uploadFile(
            file, 
            options?.metadata?.walletAddress || 'anonymous', 
            'image'
          );
          const id = url.split('/').pop() || 'error';
          return { id, url };
        },
        uploadBlob: async (blob, filename, options) => {
          const url = await pinataService.uploadFile(
            blob, 
            options?.metadata?.walletAddress || 'anonymous', 
            filename.endsWith('.mp4') || filename.endsWith('.mov') ? 'video' : 'image'
          );
          const id = url.split('/').pop() || 'error';
          return { id, url };
        },
        getUrl: (fileId) => `https://gateway.pinata.cloud/ipfs/${fileId}`,
        isAvailable: async () => true
      } as StorageService;
    
    case 'filebase':
      return {
        name: 'Filebase',
        upload: async (file, options) => {
          const url = await filebaseService.uploadFile(
            file, 
            options?.metadata?.walletAddress || 'anonymous', 
            'image'
          );
          const id = url.split('/').pop() || 'error';
          return { id, url };
        },
        uploadBlob: async (blob, filename, options) => {
          const url = await filebaseService.uploadFile(
            blob, 
            options?.metadata?.walletAddress || 'anonymous', 
            filename.endsWith('.mp4') || filename.endsWith('.mov') ? 'video' : 'image'
          );
          const id = url.split('/').pop() || 'error';
          return { id, url };
        },
        getUrl: (fileId) => `https://ipfs.filebase.io/ipfs/${fileId}`,
        isAvailable: async () => true
      } as StorageService;
    
    case 'walrus':
      // This will be implemented in the future
      return {
        name: 'Walrus',
        upload: async () => ({ id: 'not-implemented', url: '' }),
        uploadBlob: async () => ({ id: 'not-implemented', url: '' }),
        getUrl: () => '',
        isAvailable: async () => false
      } as StorageService;
    
    default:
      // Default to IPFS
      return {
        name: 'IPFS',
        upload: async (file, options) => {
          const results = await unifiedIpfsService.uploadFile(
            file, 
            options?.metadata?.walletAddress || 'anonymous', 
            'image'
          );
          return results[0] || { id: 'error', url: '' };
        },
        uploadBlob: async (blob, filename, options) => {
          const results = await unifiedIpfsService.uploadFile(
            blob, 
            options?.metadata?.walletAddress || 'anonymous', 
            filename.endsWith('.mp4') || filename.endsWith('.mov') ? 'video' : 'image',
            options
          );
          return results[0] || { id: 'error', url: '' };
        },
        getUrl: (fileId) => `https://ipfs.io/ipfs/${fileId}`,
        isAvailable: async () => true
      } as StorageService;
  }
}

// Hook to use storage
export function useStorage() {
  const context = useContext(StorageContext);
  if (!context) {
    throw new Error('useStorage must be used within a StorageProvider');
  }
  return context;
} 