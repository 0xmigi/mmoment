import React, { useState } from 'react';
import { walrusService } from '../../storage/walrus';
import { WalrusStorageInfo } from '../settings/WalrusStorageInfo';

interface WalrusExampleProps {
  walletAddress: string;
}

export const WalrusExample: React.FC<WalrusExampleProps> = ({ walletAddress }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ id: string; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [storageEpochs, setStorageEpochs] = useState(1);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };
  
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
    
    try {
      setUploading(true);
      setError(null);
      
      // Upload to Walrus with storage epochs using the service
      const result = await walrusService.upload(file, {
        metadata: { 
          walletAddress,
          storageEpochs
        }
      });
      
      setUploadResult(result);
    } catch (err) {
      setError('Failed to upload: ' + (err instanceof Error ? err.message : String(err)));
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };
  
  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Walrus Storage Demo</h1>
      
      <div className="mb-6 text-sm bg-primary-light p-4 rounded border border-primary-light">
        <p className="font-medium text-primary mb-2">About Walrus Storage</p>
        <p className="text-primary">
          This demo uploads files to the Walrus testnet using Mysten's official SDK. 
          Files are stored for the specified number of storage epochs.
          The publisher endpoint used is: <code className="bg-primary-light px-1">{walrusService.publisher}</code>
        </p>
      </div>
      
      {/* Storage info */}
      <div className="mb-8">
        <WalrusStorageInfo walletAddress={walletAddress} />
      </div>
      
      {/* Upload form */}
      <div className="p-4 border rounded-lg bg-white shadow mb-6">
        <h2 className="text-lg font-semibold mb-4">Upload to Walrus</h2>
        
        {error && (
          <div className="mb-4 p-2 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select File
          </label>
          <input
            type="file"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded file:border-0
              file:text-sm file:font-semibold
              file:bg-purple-50 file:text-purple-700
              hover:file:bg-purple-100"
          />
          {file && (
            <div className="mt-2 text-sm text-gray-500">
              Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </div>
          )}
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Storage Epochs
          </label>
          <input
            type="number"
            min="1"
            max="10"
            value={storageEpochs}
            onChange={(e) => setStorageEpochs(Math.max(1, parseInt(e.target.value) || 1))}
            className="border rounded p-2 w-20"
          />
          <span className="ml-2 text-sm text-gray-500">
            How long to store the file
          </span>
        </div>
        
        <button
          onClick={handleUpload}
          disabled={uploading || !file}
          className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 disabled:opacity-50"
        >
          {uploading ? 'Uploading...' : 'Upload to Walrus'}
        </button>
      </div>
      
      {/* Upload result */}
      {uploadResult && (
        <div className="p-4 border rounded-lg bg-white shadow">
          <h2 className="text-lg font-semibold mb-4">Upload Successful!</h2>
          
          <div className="grid grid-cols-1 gap-2">
            <div>
              <span className="font-medium">File ID:</span> {uploadResult.id}
            </div>
            <div>
              <span className="font-medium">URL:</span>{' '}
              <a
                href={uploadResult.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                {uploadResult.url}
              </a>
            </div>
            <div className="mt-4">
              <button
                onClick={() => window.open(uploadResult.url, '_blank')}
                className="bg-primary text-white px-3 py-1 rounded text-sm hover:bg-primary-hover"
              >
                View File
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 