// src/components/QuickStartView.tsx

import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useWallet } from '@solana/wallet-adapter-react';
import { motion } from 'framer-motion';
import type { WalletName } from '@solana/wallet-adapter-base';

interface CameraState {
  preview?: string;
  activeUsers: number;
  recentActivity: number;
}

export default function QuickStartView() {
  const { cameraId } = useParams<{ cameraId: string }>();
  const { connected, select } = useWallet();
  const [loading, setLoading] = useState(true);
  const [cameraState, setCameraState] = useState<CameraState>({
    activeUsers: 0,
    recentActivity: 0
  });
  
  useEffect(() => {
    const fetchCameraState = async () => {
      try {
        const response = await fetch(`/api/camera/${cameraId}/state`);
        const data = await response.json();
        setCameraState(data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch camera state:', err);
      }
    };
    
    fetchCameraState();
    const interval = setInterval(fetchCameraState, 3000);
    return () => clearInterval(interval);
  }, [cameraId]);

  const handleQuickStart = async () => {
    if (!connected) {
      await select('Phantom' as WalletName);  // Type assertion
      return;
    }
    
    try {
      // Your existing camera activation code here
    // biome-ignore lint/correctness/noUnreachable: <explanation>
          } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-black/80">
        <div className="text-white">Loading camera state...</div>
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
          </div>
          
          <div className="p-4">
            <h2 className="text-lg font-bold">Camera Ready</h2>
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
        <button
          type="button"
          onClick={handleQuickStart}
          className="w-full bg-purple-600 text-white rounded-lg py-4 font-bold text-lg"
        >
          {connected ? 'Start Recording' : 'Connect Wallet to Start'}
        </button>
      </div>
    </motion.div>
  );
}