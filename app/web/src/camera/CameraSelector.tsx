import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProgram } from '../anchor/setup';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { useCamera } from './CameraProvider';

interface CameraData {
  owner: string;
  publicKey: string;
  isActive: boolean;
  activityCounter?: number;
  lastActivityType?: number;
  metadata: {
    name: string;
    model: string;
    registrationDate: number;
    lastActivity: number;
    location?: [number, number] | null;
  };
}

export function CameraSelector({ onSelect }: { onSelect: (camera: CameraData) => void }) {
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const navigate = useNavigate();
  const { onCameraListRefresh } = useCamera();
  
  const [loading, setLoading] = useState(false);
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

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
      const cameraAccounts = await program.account.cameraAccount.all();

      // Format the camera data
      const formattedCameras = cameraAccounts.map(account => {
        const data = account.account;
        return {
          owner: data.owner.toString(),
          publicKey: account.publicKey.toString(),
          isActive: data.isActive,
          activityCounter: data.activityCounter ? data.activityCounter.toNumber() : undefined,
          lastActivityType: data.lastActivityType || 0,
          metadata: {
            name: data.metadata.name,
            model: data.metadata.model,
            registrationDate: data.metadata.registrationDate.toNumber(),
            lastActivity: data.lastActivityAt ? data.lastActivityAt.toNumber() : 0,
            location: data.metadata.location ? 
              [data.metadata.location[0].toNumber(), data.metadata.location[1].toNumber()] as [number, number] : 
              null
          }
        };
      });

      setCameras(formattedCameras);
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch cameras');
    } finally {
      setLoading(false);
    }
  };

  // Filter cameras based on search term
  const filteredCameras = cameras.filter(camera => 
    camera.metadata.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    camera.publicKey.toLowerCase().includes(searchTerm.toLowerCase()) ||
    camera.metadata.model.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Handle camera selection
  const handleSelectCamera = (camera: CameraData) => {
    onSelect(camera);
    // Navigate to the camera-specific URL using publicKey
    navigate(`/app/camera/${camera.publicKey}`);
  };

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
  }, [primaryWallet?.address, program]);

  // Subscribe to global camera list refresh events
  useEffect(() => {
    const unsubscribe = onCameraListRefresh(() => {
      console.log('[CameraSelector] Received camera list refresh event');
      fetchCameras();
    });
    return unsubscribe;
  }, [onCameraListRefresh]);

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Select a Camera</h2>
      
      {/* Search input */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search by name, ID, or model..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full p-2 border border-gray-300 rounded"
        />
      </div>
      
      {/* Error message */}
      {error && (
        <div className="mb-4 p-2 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
      
      {/* Loading state */}
      {loading ? (
        <div className="text-center py-4">Loading cameras...</div>
      ) : (
        <>
          {/* Camera list */}
          {filteredCameras.length > 0 ? (
            <div className="max-h-96 overflow-y-auto">
              {filteredCameras.map((camera) => (
                <div
                  key={camera.publicKey}
                  onClick={() => handleSelectCamera(camera)}
                  className="p-3 border-b border-gray-200 hover:bg-gray-50 cursor-pointer"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">{camera.metadata.name}</h3>
                      <p className="text-sm text-gray-500">ID: {camera.publicKey}</p>
                      <p className="text-sm text-gray-500">Model: {camera.metadata.model}</p>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${camera.isActive ? 'bg-green-500' : 'bg-red-500'}`} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              {searchTerm ? 'No cameras match your search' : 'No cameras found'}
            </div>
          )}
          
          {/* Refresh button */}
          <div className="mt-4 flex justify-center">
            <button
              onClick={fetchCameras}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Refresh Cameras
            </button>
          </div>
        </>
      )}
    </div>
  );
} 