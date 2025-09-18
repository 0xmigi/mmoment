// src/components/QuickStartView.tsx

import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useConnection } from '@solana/wallet-adapter-react';
import { motion } from 'framer-motion';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { IDL } from '../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { useCamera, CameraData, fetchCameraByPublicKey } from '../camera/CameraProvider';

interface CameraState {
  preview?: string;
  activeUsers: number;
  recentActivity: number;
}

export default function QuickStartView() {
  const { cameraId, timestamp } = useParams<{ cameraId: string; timestamp: string }>();
  const { connection } = useConnection();
  const { primaryWallet } = useDynamicContext();
  const { setSelectedCamera } = useCamera();
  const navigate = useNavigate();
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [camera, setCamera] = useState<CameraData | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>({
    activeUsers: 0,
    recentActivity: 0
  });
  const [, setProgram] = useState<Program<any> | null>(null);
  
  // Initialize program when wallet is connected
  useEffect(() => {
    if (!primaryWallet?.address || !connection) return;

    try {
      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      console.log('Setting up program with wallet address:', primaryWallet.address);
      console.log('Using program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());

      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: Transaction) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            const signer = await primaryWallet.getSigner();
            return await signer.signTransaction(tx);
          },
          signAllTransactions: async (txs: Transaction[]) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            const signer = await primaryWallet.getSigner();
            return await Promise.all(txs.map(tx => signer.signTransaction(tx)));
          },
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program
      const prog = new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      setProgram(prog);
      console.log('Program initialized with ID:', prog.programId.toString());
    } catch (err) {
      console.error('Failed to initialize program:', err);
      setError(`Program initialization error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [primaryWallet?.address, connection]);
  
  // Validate the session by checking the timestamp
  const isValidSession = () => {
    if (!timestamp) return false;
    
    const sessionTime = parseInt(timestamp, 10);
    const now = Date.now();
    const sessionAge = now - sessionTime;
    
    // Session is valid for 5 minutes (300000 ms)
    return sessionAge < 300000;
  };
  
  // Fetch camera data
  useEffect(() => {
    const fetchCameraData = async () => {
      if (!cameraId) {
        setError('No camera ID provided');
        setLoading(false);
        return;
      }

      try {
        // First check if the session is valid when there's a timestamp
        if (timestamp && !isValidSession()) {
          setError('This session has expired. Please scan the NFC tag again.');
          setLoading(false);
          return;
        }
        
        // Decode the camera public key from the URL
        const decodedPublicKey = decodeURIComponent(cameraId);
        console.log(`Attempting to fetch camera with public key: "${decodedPublicKey}"`);
        
        if (!connection) {
          setError('Solana connection not available');
          setLoading(false);
          return;
        }
        
        // Try to fetch the camera using its public key
        try {
          const cameraData = await fetchCameraByPublicKey(decodedPublicKey, connection);
          if (cameraData) {
            console.log(`Successfully loaded camera: ${cameraData.metadata.name} (${cameraData.publicKey})`);
            // Store the camera in the provider for later use
            setSelectedCamera(cameraData);
            setCamera(cameraData);
            
            // Store camera ID in localStorage for persistence
            localStorage.setItem('directCameraId', decodedPublicKey);
            
            // Simulate camera state with mock data
            setCameraState({
              preview: undefined,
              activeUsers: Math.floor(Math.random() * 5),
              recentActivity: Math.floor(Math.random() * 20)
            });
          } else {
            console.log(`Camera with public key ${decodedPublicKey} not found, but we'll show the ID anyway`);
            // Even if we can't fetch full camera data, still set the ID for navigation
            localStorage.setItem('directCameraId', decodedPublicKey);
          }
        } catch (fetchError) {
          console.error('Error fetching camera:', fetchError);
          setError(`Error fetching camera: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch camera data:', err);
        setError(`Failed to fetch camera information: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setLoading(false);
      }
    };
    
    fetchCameraData();
  }, [cameraId, connection, timestamp, setSelectedCamera]);

  const handleQuickStart = () => {
    if (!primaryWallet) {
      // User needs to authenticate first
      return;
    }
    
    // Navigate to the camera-specific view
    if (cameraId) {
      const decodedCameraId = decodeURIComponent(cameraId);
      navigate(`/app/camera/${decodedCameraId}`);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black/80">
        <div className="text-white flex flex-col items-center">
          <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-white mb-4"></div>
          <div>Loading camera...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black/80">
        <div className="max-w-md w-full mx-auto p-6 bg-white rounded-lg shadow-lg">
          <h2 className="text-xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-700 mb-6">{error}</p>
          <button
            onClick={() => navigate('/app')}
            className="w-full bg-blue-600 text-white rounded-lg py-3 font-bold"
          >
            Go to App
          </button>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      className="fixed inset-0 bg-black/80 flex flex-col"
    >
      <div className="flex-1 p-4">
        <div className="max-w-lg mx-auto bg-white rounded-lg overflow-hidden">
          <div className="aspect-video bg-gray-900 relative">
            {cameraState.preview ? (
              <img 
                src={cameraState.preview} 
                alt="Camera Preview" 
                className="absolute inset-0 w-full h-full object-cover"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-white">
                Camera Preview Not Available
              </div>
            )}
          </div>
          
          <div className="p-4">
            <h2 className="text-lg font-bold">Camera Ready</h2>
            {camera ? (
              <>
                <p className="text-sm text-gray-600 mb-2">
                  Name: {camera.metadata.name}
                </p>
                <p className="text-sm text-gray-600 mb-2">
                  Model: {camera.metadata.model}
                </p>
                <p className="text-sm text-gray-600 mb-2">
                  ID: {camera.publicKey}
                </p>
                <p className="text-sm text-gray-600 mb-4">
                  Active Users: {cameraState.activeUsers}
                </p>
              </>
            ) : (
              <>
                <p className="text-sm text-gray-600 mb-2">
                  ID: {cameraId ? decodeURIComponent(cameraId) : 'Unknown'}
                </p>
                <p className="text-sm text-gray-600 mb-4">
                  No additional camera information available
                </p>
              </>
            )}
            
            <p className="text-sm font-medium mb-6">
              This camera is ready to use. Click below to access the camera.
            </p>
            
            <button
              onClick={handleQuickStart}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium transition-colors"
            >
              Access Camera
            </button>
          </div>
        </div>
      </div>
      
      <div className="p-4 bg-white">
        <button
          onClick={() => navigate('/app')}
          className="w-full py-3 border border-gray-300 rounded-lg font-medium text-gray-800"
        >
          Go to App Home
        </button>
      </div>
    </motion.div>
  );
}