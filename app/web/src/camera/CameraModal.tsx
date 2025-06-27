/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User, Wifi } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, Transaction, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { timelineService } from '../timeline/timeline-service';
import { unifiedCameraService } from './unified-camera-service';
import { CONFIG } from '../core/config';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCheckStatusChange?: (isCheckedIn: boolean) => void;
  camera: {
    id: string;
    owner: string;
    ownerDisplayName?: string;
    ownerPfpUrl?: string;
    isLive: boolean;
    isStreaming: boolean;
    status: 'ok' | 'error' | 'offline';
    lastSeen?: number;
    activityCounter?: number;
    model?: string;
    // New properties for development info
    showDevInfo?: boolean;
    defaultDevCamera?: string;
  };
}

export function CameraModal({ isOpen, onClose, onCheckStatusChange, camera }: CameraModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [isCheckedIn, setIsCheckedIn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionTest, setConnectionTest] = useState<{
    testing: boolean;
    result: string | null;
    success: boolean;
  }>({ testing: false, result: null, success: false });

  // Configuration states for Jetson camera features
  const [gestureVisualization, setGestureVisualization] = useState(false);
  const [faceVisualization, setFaceVisualization] = useState(false);
  const [gestureControls, setGestureControls] = useState(false);
  const [configLoading, setConfigLoading] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<{ gesture: string; confidence: number } | null>(null);

  // Check if current user is the owner
  const isOwner = primaryWallet?.address === camera.owner;

  // Check if this is a Jetson camera (has advanced features)
  const isJetsonCamera = camera.id === CONFIG.JETSON_CAMERA_PDA || camera.model === 'jetson';

  // Add more frequent status updates to the parent component
  useEffect(() => {
    if (onCheckStatusChange) {
      console.log("[CameraModal] Notifying parent of check-in status:", isCheckedIn);
      onCheckStatusChange(isCheckedIn);
    }
  }, [isCheckedIn, onCheckStatusChange]);

  // Add a more frequent check for session status
  useEffect(() => {
    if (!isOpen || !camera.id || !primaryWallet?.address) return;
    
    console.log("[CameraModal] Checking session status on open");
    checkSessionStatus();
    
    // Also set up a periodic check while the modal is open
    const intervalId = setInterval(() => {
      console.log("[CameraModal] Periodic session status check");
      checkSessionStatus();
    }, 2000);
    
    return () => {
      console.log("[CameraModal] Cleaning up session status check");
      clearInterval(intervalId);
    };
  }, [isOpen, camera.id, primaryWallet]);

  // Clear errors and load configuration when modal opens
  useEffect(() => {
    const loadConfiguration = async () => {
      if (isOpen) {
        setError(null);
        
        // Load current configuration for Jetson cameras
        if (isJetsonCamera && camera.id) {
          try {
            console.log('[CameraModal] Loading current computer vision state...');
            
            // Load gesture controls state from unified service
            const gestureControlsEnabled = await unifiedCameraService.getGestureControlsStatus(camera.id);
            setGestureControls(gestureControlsEnabled);
            console.log('[CameraModal] Gesture controls state:', gestureControlsEnabled);
            
            // Load visualization states from localStorage (persist across modal opens)
            const storedGestureViz = localStorage.getItem(`jetson_gesture_viz_${camera.id}`) === 'true';
            const storedFaceViz = localStorage.getItem(`jetson_face_viz_${camera.id}`) === 'true';
            
            setGestureVisualization(storedGestureViz);
            setFaceVisualization(storedFaceViz);
            
            console.log('[CameraModal] Loaded visualization states - Gesture:', storedGestureViz, 'Face:', storedFaceViz);
            
            console.log('[CameraModal] Computer vision configuration loaded successfully');
          } catch (error) {
            console.error('Error loading computer vision configuration:', error);
            // Set defaults on error
            setGestureVisualization(false);
            setFaceVisualization(false);
            setGestureControls(false);
          }
        } else {
          // Reset states for non-Jetson cameras
          setGestureVisualization(false);
          setFaceVisualization(false);
          setGestureControls(false);
        }
      }
    };

    loadConfiguration();
  }, [isOpen, isJetsonCamera, camera.id]);

  // Expose test function to window for debugging
  useEffect(() => {
    if (isJetsonCamera && camera.id) {
      (window as any).testVisualizationEndpoints = testVisualizationEndpoints;
    }
    
    return () => {
      delete (window as any).testVisualizationEndpoints;
    };
  }, [isJetsonCamera, camera.id]);

  // Poll current gesture when modal is open and gesture visualization is enabled
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (isOpen && isJetsonCamera && camera.id && (gestureVisualization || gestureControls)) {
      const pollGesture = async () => {
        try {
          const result = await unifiedCameraService.getCurrentGesture(camera.id);
          if (result.success && result.data) {
            setCurrentGesture({
              gesture: result.data.gesture || 'none',
              confidence: result.data.confidence || 0
            });
          } else {
            setCurrentGesture(null);
          }
        } catch (error) {
          console.error('Error polling gesture:', error);
          setCurrentGesture(null);
        }
      };

      // Poll immediately and then every 2 seconds
      pollGesture();
      intervalId = setInterval(pollGesture, 2000);
    } else {
      setCurrentGesture(null);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isOpen, isJetsonCamera, camera.id, gestureVisualization, gestureControls]);

  const checkSessionStatus = async () => {
    if (!camera.id || !primaryWallet?.address || !connection) return;
    
    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Try to fetch the session account
      try {
        await program.account.userSession.fetch(sessionPda);
        console.log("[CameraModal] Session exists, setting checked-in: true");
        setIsCheckedIn(true);
      } catch (err) {
        console.log("[CameraModal] No session found, setting checked-in: false");
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error('[CameraModal] Error checking session status:', err);
      setIsCheckedIn(false);
    }
  };

  // Initialize program when needed
  const initializeProgram = async () => {
    if (!primaryWallet?.address || !connection) {
      throw new Error('Wallet or connection not available');
    }

    // Check if it's a Solana wallet
    if (!isSolanaWallet(primaryWallet)) {
      throw new Error('This is not a Solana wallet');
    }

    try {
      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          // We'll handle signing separately
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program
      return new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
    } catch (error) {
      console.error('Error initializing program:', error);
      throw new Error('Failed to initialize Solana program');
    }
  };

  const handleCameraExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.id}?cluster=devnet`, '_blank');
  };

  const handleOwnerExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.owner}?cluster=devnet`, '_blank');
  };

  // Format address for display
  const formatAddress = (address: string, start = 6, end = 6) => {
    if (!address) return '';
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  };

  const handleDevCameraClick = () => {
    if (camera.defaultDevCamera) {
      // Redirect to the correct camera page URL
      const baseUrl = window.location.origin;
      window.location.href = `${baseUrl}/app/camera/${camera.defaultDevCamera}`;
    }
  };

  // Add check-in event to timeline
  const addCheckInEvent = (transactionId: string) => {
    if (primaryWallet?.address) {
      timelineService.emitEvent({
        type: 'check_in',
        user: {
          address: primaryWallet.address
        },
        timestamp: Date.now(),
        transactionId: transactionId,
        cameraId: camera.id
      });
    }
  };

  // Add check-out event to timeline
  const addCheckOutEvent = (transactionId: string) => {
    if (primaryWallet?.address) {
      timelineService.emitEvent({
        type: 'check_out',
        user: {
          address: primaryWallet.address
        },
        timestamp: Date.now(),
        transactionId: transactionId,
        cameraId: camera.id
      });
    }
  };

  // Test connection to Jetson camera
  const testJetsonConnection = async () => {
    setConnectionTest({ testing: true, result: null, success: false });
    
    try {
      const result = await unifiedCameraService.testConnection(camera.id);
      setConnectionTest({
        testing: false,
        result: result.success ? result.data?.message || 'Connection successful' : result.error || 'Connection failed',
        success: result.success
      });
    } catch (error) {
      setConnectionTest({
        testing: false,
        result: error instanceof Error ? error.message : 'Connection test failed',
        success: false
      });
    }
  };

  // Test visualization endpoints directly
  const testVisualizationEndpoints = async () => {
    if (!isJetsonCamera || !camera.id) return;
    
    console.log('[CameraModal] Testing visualization endpoints...');
    
    try {
      // Test both visualization endpoints
      const baseUrl = 'https://jetson.mmoment.xyz';
      
      console.log('Testing face visualization endpoint...');
      const faceResponse = await fetch(`${baseUrl}/api/visualization/face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true })
      });
      
      console.log('Face viz response status:', faceResponse.status);
      const faceData = await faceResponse.json();
      console.log('Face viz response data:', faceData);
      
      console.log('Testing gesture visualization endpoint...');
      const gestureResponse = await fetch(`${baseUrl}/api/visualization/gesture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true })
      });
      
      console.log('Gesture viz response status:', gestureResponse.status);
      const gestureData = await gestureResponse.json();
      console.log('Gesture viz response data:', gestureData);
      

      
    } catch (error) {
      console.error('Error testing visualization endpoints:', error);
    }
  };

  // Handle gesture visualization toggle
  const handleGestureVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !gestureVisualization;
      console.log('[CameraModal] Toggling gesture visualization to:', newState);
      
      const result = await unifiedCameraService.toggleGestureVisualization(camera.id, newState);
      
      if (result.success) {
        setGestureVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_gesture_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Gesture visualization toggled successfully to:', newState);
        
        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle gesture visualization:', result.error);
        setError(result.error || 'Failed to toggle gesture visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling gesture visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle gesture visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle face visualization toggle
  const handleFaceVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !faceVisualization;
      console.log('[CameraModal] Toggling face visualization to:', newState);
      
      const result = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      
      if (result.success) {
        setFaceVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_face_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Face visualization toggled successfully to:', newState);
        
        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle face visualization:', result.error);
        setError(result.error || 'Failed to toggle face visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling face visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle face visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle gesture controls toggle
  const handleGestureControlsToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !gestureControls;
      const result = await unifiedCameraService.toggleGestureControls(camera.id, newState);
      
      if (result.success) {
        setGestureControls(newState);
      } else {
        setError(result.error || 'Failed to toggle gesture controls');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to toggle gesture controls');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle Livepeer stream start
  const handleStartLivepeerStream = async () => {
    setConfigLoading(true);
    try {
      // TODO: Add startLivepeerStream method to unified service
      const result = await unifiedCameraService.startStream(camera.id);
      
      if (result.success) {
        setError(null);
      } else {
        setError(result.error || 'Failed to start Livepeer stream');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start Livepeer stream');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle Livepeer stream stop
  const handleStopLivepeerStream = async () => {
    setConfigLoading(true);
    try {
      // TODO: Add stopLivepeerStream method to unified service  
      const result = await unifiedCameraService.stopStream(camera.id);
      
      if (result.success) {
        setError(null);
      } else {
        setError(result.error || 'Failed to stop Livepeer stream');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to stop Livepeer stream');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle check-in
  const handleCheckIn = async () => {
    if (!camera.id || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Create the accounts object for check-in
      const accounts: any = {
        user: userPublicKey,
        camera: cameraPublicKey,
        session: sessionPda,
        systemProgram: SystemProgram.programId
      };

      // Create check-in instruction
      const ix = await program.methods
        .checkIn(false)
        .accounts(accounts)
        .instruction();

      // Create and sign the transaction using Dynamic
      const tx = new Transaction().add(ix);
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Sign and send transaction
      try {
        // Get the signer object, accounting for different wallet interfaces
        let signedTx;
        
        // Check if wallet has a getSigner method (Dynamic wallet)
        if (typeof (primaryWallet as any).getSigner === 'function') {
          const signer = await (primaryWallet as any).getSigner();
          
          // Check if signer has signTransaction
          if (typeof signer.signTransaction === 'function') {
            console.log('Using signer.signTransaction for check-in');
            signedTx = await signer.signTransaction(tx);
          } else {
            console.log('Signer does not have signTransaction, using fallback for check-in');
            // Fallback to direct wallet signTransaction if available
            signedTx = await (primaryWallet as any).signTransaction(tx);
          }
        } else {
          // Standard wallet adapter interface
          console.log('Using primaryWallet.signTransaction directly for check-in');
          signedTx = await (primaryWallet as any).signTransaction(tx);
        }
        
        if (!signedTx) {
          throw new Error('Transaction signing failed - no signed transaction returned');
        }
        
        console.log('Transaction signed successfully, sending to network for check-in');
        const signature = await connection.sendRawTransaction(signedTx.serialize());
        
        console.log('Check-in transaction sent, confirming:', signature);
        await connection.confirmTransaction(signature, 'confirmed');
        console.log('Check-in transaction confirmed successfully');
        
        setIsCheckedIn(true);
        
        // Add to timeline
        addCheckInEvent(signature);
        
        // Refresh the timeline
        timelineService.refreshEvents();
        
        // Notify parent component
        if (onCheckStatusChange) {
          onCheckStatusChange(true);
        }
      } catch (signError) {
        console.error('Transaction signing error:', signError);
        if (signError instanceof Error) {
          setError(`Failed to sign check-in transaction: ${signError.message}`);
        } else {
          setError('Failed to sign check-in transaction. Please try again.');
        }
      }
      
    } catch (error) {
      console.error('Check-in error:', error);
      
      if (error instanceof Error) {
        let errorMsg = error.message;
        
        // Check for common error messages and provide more user-friendly versions
        if (errorMsg.includes('custom program error: 0x64')) {
          errorMsg = 'Program error: The camera may not be configured correctly.';
        } else if (errorMsg.includes('insufficient funds')) {
          errorMsg = 'Insufficient SOL in your wallet. Please add more SOL and try again.';
        } else if (errorMsg.includes('already in use')) {
          errorMsg = 'You are already checked in to this camera.';
          setIsCheckedIn(true);
          return;
        } else if (errorMsg.length > 150) {
          // If the error message is too long, provide a shorter, more general message
          errorMsg = 'An error occurred during check-in. Please check the console for details.';
        }
        
        setError(errorMsg);
      } else {
        setError('Unknown error during check-in');
      }
    } finally {
      setLoading(false);
    }
  };

  // Handle check-out
  const handleCheckOut = async () => {
    if (!camera.id || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Create check-out instruction
      const ix = await program.methods
        .checkOut()
        .accounts({
          user: userPublicKey,
          camera: cameraPublicKey,
          session: sessionPda,
        })
        .instruction();

      // Create and sign the transaction using Dynamic
      const tx = new Transaction().add(ix);
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Sign and send transaction
      try {
        // Get the signer object, accounting for different wallet interfaces
        let signedTx;
        
        // Check if wallet has a getSigner method (Dynamic wallet)
        if (typeof (primaryWallet as any).getSigner === 'function') {
          const signer = await (primaryWallet as any).getSigner();
          
          // Check if signer has signTransaction
          if (typeof signer.signTransaction === 'function') {
            console.log('Using signer.signTransaction');
            signedTx = await signer.signTransaction(tx);
          } else {
            console.log('Signer does not have signTransaction, using fallback');
            // Fallback to direct wallet signTransaction if available
            signedTx = await (primaryWallet as any).signTransaction(tx);
          }
        } else {
          // Standard wallet adapter interface
          console.log('Using primaryWallet.signTransaction directly');
          signedTx = await (primaryWallet as any).signTransaction(tx);
        }
        
        if (!signedTx) {
          throw new Error('Transaction signing failed - no signed transaction returned');
        }
        
        console.log('Transaction signed successfully, sending to network');
        const signature = await connection.sendRawTransaction(signedTx.serialize());
        
        console.log('Transaction sent, confirming:', signature);
        await connection.confirmTransaction(signature, 'confirmed');
        console.log('Transaction confirmed successfully');
        
        setIsCheckedIn(false);
        
        // Add to timeline
        addCheckOutEvent(signature);
        
        // Refresh the timeline
        timelineService.refreshEvents();
        
        // Notify parent component
        if (onCheckStatusChange) {
          onCheckStatusChange(false);
        }
      } catch (signError) {
        console.error('Transaction signing error:', signError);
        if (signError instanceof Error) {
          setError(`Failed to sign check-out transaction: ${signError.message}`);
        } else {
          setError('Failed to sign check-out transaction. Please try again.');
        }
      }
      
    } catch (error) {
      console.error('Check-out error:', error);
      
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('Unknown error during check-out');
      }
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Camera Details
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Camera Content */}
          <div className="p-4">
            {camera.showDevInfo ? (
              // Development section when no camera is connected
              <div className="space-y-4">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
                  <h3 className="text-sm font-medium text-yellow-800 mb-1">Development Mode</h3>
                  <p className="text-xs text-yellow-700 mb-3">
                    No camera is currently connected. Connect to a physical camera below:
                  </p>
                  <div className="space-y-2">
                    <button
                      onClick={handleDevCameraClick}
                      className="text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Pi5 <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                    <button
                      onClick={() => {
                        const baseUrl = window.location.origin;
                        window.location.href = `${baseUrl}/app/camera/WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD`;
                      }}
                      className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Orin Nano <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                    <button
                      onClick={testJetsonConnection}
                      disabled={connectionTest.testing}
                      className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      {connectionTest.testing ? 'Testing...' : 'Test Jetson Connection'} <Wifi className="h-3 w-3 ml-1" />
                    </button>
                  </div>
                </div>
                
                {/* Connection Test Results */}
                {connectionTest.result && (
                  <div className={`border rounded-lg p-3 mb-3 ${
                    connectionTest.success 
                      ? 'bg-green-50 border-green-200' 
                      : 'bg-red-50 border-red-200'
                  }`}>
                    <h3 className={`text-sm font-medium mb-1 ${
                      connectionTest.success ? 'text-green-800' : 'text-red-800'
                    }`}>
                      Connection Test {connectionTest.success ? 'Passed' : 'Failed'}
                    </h3>
                    <p className={`text-xs ${
                      connectionTest.success ? 'text-green-700' : 'text-red-700'
                    }`}>
                      {connectionTest.result}
                    </p>
                    <p className="text-xs text-gray-600 mt-1">
                      URL: {CONFIG.JETSON_CAMERA_URL}
                    </p>
                  </div>
                )}
                
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Camera PDA</div>
                  <div className="flex items-center">
                    <div className="text-sm font-medium">
                      {formatAddress(camera.defaultDevCamera || '')}
                    </div>
                    <button
                      onClick={() => window.open(`https://solscan.io/account/${camera.defaultDevCamera}?cluster=devnet`, '_blank')}
                      className="ml-1 text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                    >
                      <ExternalLink className="w-3 h-3 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              // Original camera details layout
              <>
                {/* Camera PDA */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-50 flex-shrink-0 flex items-center justify-center overflow-hidden">
                    <Camera className="w-5 h-5 text-blue-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">{formatAddress(camera.id)}</div>
                  </div>
                  <button
                    onClick={handleCameraExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Owner */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-gray-100 flex-shrink-0 flex items-center justify-center overflow-hidden ml-0">
                    <User className="w-5 h-5 text-gray-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700 flex items-center">
                      Owner
                      {isOwner && <span className="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">you</span>}
                    </div>
                    <div className="text-sm font-medium">
                      {formatAddress(camera.owner, 9, 5)}
                    </div>
                  </div>
                  <button
                    onClick={handleOwnerExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Camera Name */}
                {camera.ownerDisplayName && (
                  <div className="mb-4">
                    <div className="text-sm text-gray-700">Camera Name</div>
                    <div className="text-sm">{camera.ownerDisplayName}</div>
                  </div>
                )}

                {/* Type */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Type</div>
                  <div className="text-sm">{camera.model || "pi5"}</div>
                </div>

                {/* Activity Counter */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Activity</div>
                  <div className="text-sm font-medium">{camera.activityCounter || 0} total interactions</div>
                </div>

                {/* Computer Vision Controls - Only for Jetson cameras */}
                {isJetsonCamera && (
                  <div className="mb-4 pt-4 border-t border-gray-200">
                    <div className="space-y-2">
                      {/* Face Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Face Detection Overlay</div>
                          <div className="text-xs text-gray-500">Shows face detection boxes</div>
                        </div>
                        <button
                          onClick={handleFaceVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            faceVisualization 
                              ? 'bg-blue-600 hover:bg-blue-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              faceVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Gesture Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Gesture Detection Overlay</div>
                          <div className="text-xs text-gray-500">Shows hand gesture tracking</div>
                        </div>
                        <button
                          onClick={handleGestureVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            gestureVisualization 
                              ? 'bg-blue-600 hover:bg-blue-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              gestureVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Gesture Controls Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Gesture Photo/Video Capture</div>
                          <div className="text-xs text-gray-500">Peace sign = photo, thumbs up = video</div>
                        </div>
                        <button
                          onClick={handleGestureControlsToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 ${
                            gestureControls 
                              ? 'bg-green-600 hover:bg-green-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              gestureControls ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Current Gesture Status - Only when gesture features are enabled */}
                      {(gestureVisualization || gestureControls) && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <div className="text-xs text-gray-500 mb-1">Current Gesture Detected</div>
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-medium">
                              {currentGesture ? (
                                <span className="capitalize">
                                  {currentGesture.gesture === 'none' ? 'No gesture detected' : currentGesture.gesture.replace('_', ' ')}
                                </span>
                              ) : (
                                <span className="text-gray-400">Loading...</span>
                              )}
                            </div>
                            {currentGesture && currentGesture.gesture !== 'none' && (
                              <div className="text-xs text-gray-500">
                                {Math.round(currentGesture.confidence * 100)}% confidence
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <p className="text-xs text-red-700 mt-1">{error}</p>
                  </div>
                )}

                {/* Check In/Out Button */}
                {isCheckedIn ? (
                  <button
                    onClick={handleCheckOut}
                    disabled={loading}
                    className="w-full bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {loading ? 'Processing...' : 'Check Out'}
                  </button>
                ) : (
                  <button
                    onClick={handleCheckIn}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {loading ? 'Processing...' : 'Check In'}
                  </button>
                )}
              </>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}