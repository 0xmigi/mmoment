import { useState, useEffect } from 'react';
import { useProgram } from '../anchor/setup';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { PublicKey } from '@solana/web3.js';

interface CameraData {
  owner: string;
  isActive: boolean;
  activityCounter?: number;
  lastActivityType?: number;
  metadata: {
    name: string;
    model: string;
    registrationDate: number;
    location?: [number, number] | null;
  };
  lastActivityAt?: number;
  publicKey: string;
}

export function NFCUrlGenerator() {
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  
  const [loading, setLoading] = useState(false);
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [generatedUrl, setGeneratedUrl] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [selectedCameraDetails, setSelectedCameraDetails] = useState<CameraData | null>(null);

  // Fetch registered cameras
  const fetchCameras = async () => {
    if (!primaryWallet?.address || !program) {
      setError('Wallet or program not available');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      // Get program accounts of type CameraAccount
      const cameraAccounts = await (program.account as any).cameraAccount.all();
      console.log(`Found ${cameraAccounts.length} camera accounts`);

      // Format the camera data
      const formattedCameras = cameraAccounts.map((account: any) => {
        const data = account.account;
        return {
          owner: data.owner.toString(),
          isActive: data.isActive,
          metadata: {
            name: data.metadata.name,
            model: data.metadata.model,
            registrationDate: data.metadata.registrationDate.toNumber(),
            location: data.metadata.location ? 
              [data.metadata.location[0].toNumber(), data.metadata.location[1].toNumber()] as [number, number] : 
              null
          },
          lastActivityAt: data.lastActivityAt ? data.lastActivityAt.toNumber() : 0,
          publicKey: account.publicKey.toString()
        };
      });

      console.log('Loaded cameras with public keys:', formattedCameras.map((c: any) => ({
        name: c.metadata.name,
        publicKey: c.publicKey
      })));

      setCameras(formattedCameras);
      console.log('Cameras loaded:', formattedCameras);
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch cameras');
    } finally {
      setLoading(false);
    }
  };

  // Update selected camera details when selection changes
  useEffect(() => {
    if (selectedCamera) {
      const camera = cameras.find(c => c.publicKey === selectedCamera);
      setSelectedCameraDetails(camera || null);
    } else {
      setSelectedCameraDetails(null);
    }
  }, [selectedCamera, cameras]);

  // Generate a unique URL for the selected camera
  const handleGenerateUrl = () => {
    if (!selectedCamera) {
      setError('Please select a camera first');
      return;
    }
    
    // Get the selected camera details
    const camera = cameras.find(c => c.publicKey === selectedCamera);
    if (!camera) {
      setError('Selected camera not found');
      return;
    }
    
    try {
      // Validate the public key format
      const publicKey = new PublicKey(camera.publicKey);
      console.log('Valid public key format:', publicKey.toString());
      
      // NOTE: We need to use encodeURIComponent for the public key as it contains special characters
      const encodedPublicKey = encodeURIComponent(publicKey.toString());
      console.log('Original public key:', publicKey.toString());
      console.log('Encoded public key for URL:', encodedPublicKey);
      
      // Add a timestamp parameter valid for 24 hours
      const expirationTime = Date.now() + (24 * 60 * 60 * 1000); // 24 hours from now
      
      // Generate a URL with both the camera ID and expiration timestamp
      const url = `${window.location.origin}/app/camera/${encodedPublicKey}?expires=${expirationTime}`;
      console.log('Generated URL:', url);
      
      setGeneratedUrl(url);
      console.log('URL generated for camera:', {
        name: camera.metadata.name,
        publicKey: camera.publicKey,
        owner: camera.owner
      });
      setError("");
    } catch (err) {
      console.error('Invalid public key format:', err);
      setError('Invalid camera public key');
    }
  };

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
  }, [primaryWallet?.address, program]);

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6">NFC URL Generator</h2>
      <p className="mb-4 text-gray-600">
        This tool generates URLs for NFC tags to provide physical-based access control to camera devices.
        Only users with physical access to the tag will be able to control the camera.
      </p>
      
      {/* Error message */}
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {/* Camera selection */}
      <div className="mb-6">
        <label className="block text-gray-700 font-medium mb-2">
          Select Camera
        </label>
        <select
          value={selectedCamera}
          onChange={(e) => setSelectedCamera(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
          disabled={loading || cameras.length === 0}
        >
          <option value="">-- Select a camera --</option>
          {cameras.map((camera) => (
            <option key={camera.publicKey} value={camera.publicKey}>
              {camera.metadata.name} - {camera.publicKey.slice(0, 8)}...
            </option>
          ))}
        </select>
      </div>
      
      {/* Selected camera details */}
      {selectedCameraDetails && (
        <div className="mb-6 p-4 bg-gray-50 rounded">
          <h3 className="font-bold mb-2">Selected Camera Details:</h3>
          <p><span className="font-medium">Name:</span> {selectedCameraDetails.metadata.name}</p>
          <p><span className="font-medium">Model:</span> {selectedCameraDetails.metadata.model}</p>
          <p><span className="font-medium">ID:</span> {selectedCameraDetails.publicKey}</p>
          <p>
            <span className="font-medium">Status:</span> 
            <span className={selectedCameraDetails.isActive ? 'text-green-600' : 'text-red-600'}>
              {selectedCameraDetails.isActive ? ' Active' : ' Inactive'}
            </span>
          </p>
          <p><span className="font-medium">Owner:</span> {selectedCameraDetails.owner}</p>
        </div>
      )}
      
      {/* Generate button */}
      <div className="mb-6">
        <button
          onClick={handleGenerateUrl}
          disabled={!selectedCamera || loading}
          className="px-4 py-2 bg-primary text-white rounded hover:bg-primary disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Generate NFC URL
        </button>
        
        <button
          onClick={fetchCameras}
          className="ml-2 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Refresh Cameras
        </button>
      </div>
      
      {/* Generated URL */}
      {generatedUrl && (
        <div className="p-4 bg-gray-100 rounded">
          <h3 className="font-bold mb-2">Generated URL:</h3>
          <div className="break-all mb-2">{generatedUrl}</div>
          <div className="flex space-x-2">
            <button
              onClick={() => {
                navigator.clipboard.writeText(generatedUrl);
                alert('URL copied to clipboard!');
              }}
              className="px-3 py-1 bg-green-500 text-white rounded text-sm"
            >
              Copy URL
            </button>
            <a
              href={generatedUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1 bg-purple-500 text-white rounded text-sm"
            >
              Test Camera Access
            </a>
            {selectedCameraDetails && (
              <a
                href={generatedUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1 bg-primary text-white rounded text-sm"
              >
                Open Camera
              </a>
            )}
          </div>
          <div className="mt-4 p-3 bg-yellow-50 text-yellow-800 rounded text-sm">
            <p className="font-bold">Security Note:</p>
            <p>This URL provides direct access to the camera control interface.</p>
            <p className="mt-2">In a production environment, this URL should include additional security measures such as:</p>
            <ul className="list-disc pl-5 mt-1">
              <li>Time-based expiration</li>
              <li>IP or network restrictions</li>
              <li>Location verification</li>
              <li>One-time use tokens</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
} 