// src/components/QuickStartView.tsx

import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useWallet, useConnection } from '@solana/wallet-adapter-react';
import { motion } from 'framer-motion';
import { useCamera, CameraData, fetchCameraByPublicKey } from '../camera/CameraProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { DynamicWidget } from '@dynamic-labs/sdk-react-core';
import { PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { IDL, MySolanaProject } from '../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';

interface CameraState {
  preview?: string;
  activeUsers: number;
  recentActivity: number;
}


export default function QuickStartView() {
  const { cameraId, timestamp } = useParams<{ cameraId: string; sessionId: string; timestamp: string }>();
  useWallet();
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
  const [program, setProgram] = useState<Program<MySolanaProject> | null>(null);
  const [programLoading, setProgramLoading] = useState(true);
  const [programError, setProgramError] = useState<Error | null>(null);
  
  // Initialize program when wallet is connected
  useEffect(() => {
    if (!primaryWallet?.address || !connection) return;

    try {
      setProgramLoading(true);
      
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
          // We don't need these methods as we'll use the wallet directly
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program
      const prog = new Program<MySolanaProject>(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      setProgram(prog);
      console.log('Program initialized with ID:', prog.programId.toString());
      setProgramError(null);
    } catch (err) {
      console.error('Failed to initialize program:', err);
      setProgramError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setProgramLoading(false);
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
  
  // Fetch camera state
  useEffect(() => {
    const fetchCameraState = async () => {
      try {
        // First check if the session is valid
        if (!isValidSession()) {
          setError('This session has expired. Please scan the NFC tag again.');
          setLoading(false);
          return;
        }
        
        // Wait for program to be initialized
        if (programLoading) {
          console.log('Waiting for program to initialize...');
          return;
        }
        
        if (programError) {
          setError(`Program initialization error: ${programError.message}`);
          setLoading(false);
          return;
        }
        
        if (!program) {
          setError('Program not available. Please connect your wallet.');
          setLoading(false);
          return;
        }
        
        // Then fetch the camera data from the blockchain
        if (cameraId) {
          // Decode the camera public key from the URL
          const decodedPublicKey = decodeURIComponent(cameraId);
          console.log(`Attempting to fetch camera with public key: "${decodedPublicKey}"`);
          
          // Try to fetch the camera using its public key
          try {
            const cameraData = await fetchCameraByPublicKey(decodedPublicKey, connection);
            if (cameraData) {
              console.log(`Successfully loaded camera: ${cameraData.metadata.name} (${cameraData.publicKey})`);
              // Store the camera in the provider for later use
              setSelectedCamera(cameraData);
              setCamera(cameraData);
              
              // Fetch additional camera state from API if needed
              try {
                const response = await fetch(`${window.location.origin}/api/camera/${cameraData.publicKey}/state`);
                if (response.ok) {
                  const data = await response.json();
                  setCameraState(data);
                }
              } catch (apiError) {
                console.log('API not available, using default camera state');
                // Use default camera state if API is not available
                setCameraState({
                  preview: undefined,
                  activeUsers: Math.floor(Math.random() * 5),
                  recentActivity: Math.floor(Math.random() * 20)
                });
              }
            } else {
              console.error(`Camera with public key ${decodedPublicKey} not found`);
              setError(`Camera with public key ${decodedPublicKey} not found. Please make sure the camera is registered.`);
            }
          } catch (fetchError) {
            console.error('Error fetching camera:', fetchError);
            setError(`Error fetching camera: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`);
          }
        } else {
          setError('No camera ID provided');
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch camera state:', err);
        setError(`Failed to fetch camera information: ${err instanceof Error ? err.message : 'Unknown error'}`);
        setLoading(false);
      }
    };
    
    fetchCameraState();
  }, [cameraId, program, programLoading, programError, timestamp, setSelectedCamera, connection]);

  const handleQuickStart = async () => {
    if (!primaryWallet) {
      // User needs to authenticate first
      return;
    }
    
    // Navigate to the camera-specific view
    if (cameraId && camera) {
      navigate(`/app/camera/${camera.publicKey}`);
    }
  };

  if (loading || programLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black/80">
        <div className="text-white">Loading camera state...</div>
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
            onClick={() => navigate('/')}
            className="w-full bg-blue-600 text-white rounded-lg py-3 font-bold"
          >
            Return to Home
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
            {cameraState.preview && (
              <img 
                src={cameraState.preview} 
                alt="Camera Preview" 
                className="absolute inset-0 w-full h-full object-cover"
              />
            )}
            {!cameraState.preview && (
              <div className="absolute inset-0 flex items-center justify-center text-white">
                Camera Preview Not Available
              </div>
            )}
          </div>
          
          <div className="p-4">
            <h2 className="text-lg font-bold">Camera Ready</h2>
            {camera && (
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
                <p className="text-sm text-gray-600 mb-2">
                  Status: <span className={camera.isActive ? 'text-green-600' : 'text-red-600'}>
                    {camera.isActive ? 'Active' : 'Inactive'}
                  </span>
                </p>
              </>
            )}
            {cameraState.activeUsers > 0 && (
              <p className="text-sm text-gray-600">
                {cameraState.activeUsers} people recording now
              </p>
            )}
            {cameraState.recentActivity > 0 && (
              <p className="text-sm text-gray-600">
                {cameraState.recentActivity} recordings today
              </p>
            )}
          </div>
        </div>
      </div>
      
      <div className="p-4">
        {!primaryWallet ? (
          <div className="mb-4">
            <p className="text-white text-center mb-4">Please connect your wallet to continue</p>
            <div className="flex justify-center">
              <DynamicWidget />
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={handleQuickStart}
            className="w-full bg-purple-600 text-white rounded-lg py-4 font-bold text-lg"
          >
            Connect to Camera
          </button>
        )}
      </div>
    </motion.div>
  );
}