import React, { useEffect, useState } from 'react';
import { walrusSdkService, WalrusStorageQuota } from '../../storage/walrus/walrus-sdk-service';

interface WalrusStorageInfoProps {
  walletAddress: string;
}

export const WalrusStorageInfo: React.FC<WalrusStorageInfoProps> = ({ walletAddress }) => {
  const [quota, setQuota] = useState<WalrusStorageQuota | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    async function fetchQuota() {
      try {
        setLoading(true);
        setError(null);
        const quotaData = await walrusSdkService.getUserQuota(walletAddress);
        setQuota(quotaData);
      } catch (err) {
        console.error('Failed to fetch Walrus quota:', err);
        setError('Could not fetch storage quota');
      } finally {
        setLoading(false);
      }
    }
    
    fetchQuota();
  }, [walletAddress]);
  
  // Format bytes to human-readable format
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    
    return parseFloat((bytes / Math.pow(1024, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  // Calculate usage percentage
  const getUsagePercentage = (): number => {
    if (!quota || quota.totalBytes === 0) return 0;
    return Math.round((quota.usedBytes / quota.totalBytes) * 100);
  };
  
  return (
    <div className="border rounded-lg p-4 bg-white shadow">
      <h2 className="text-lg font-semibold mb-4">Walrus Storage</h2>
      
      {loading && (
        <div className="text-center text-gray-500 py-4">
          Loading storage information...
        </div>
      )}
      
      {error && (
        <div className="text-red-500 py-2">
          {error}
        </div>
      )}
      
      {quota && !loading && (
        <div>
          <div className="mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium">
                {formatBytes(quota.usedBytes)} used of {formatBytes(quota.totalBytes)}
              </span>
              <span className="text-sm font-medium">
                {getUsagePercentage()}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-purple-600 h-2.5 rounded-full" 
                style={{ width: `${getUsagePercentage()}%` }}
              ></div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Available</p>
              <p className="font-medium">{formatBytes(quota.remainingBytes)}</p>
            </div>
            <div>
              <p className="text-gray-500">Total</p>
              <p className="font-medium">{formatBytes(quota.totalBytes)}</p>
            </div>
          </div>
          
          <div className="mt-4 text-xs text-gray-500">
            <p>Using Walrus testnet SDK: {walrusSdkService.aggregator}</p>
          </div>
        </div>
      )}
    </div>
  );
}; 