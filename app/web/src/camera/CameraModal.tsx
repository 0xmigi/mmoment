/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import React, { useState, useEffect } from 'react';
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
    if (isOpen) {
      setError(null);
      
      // Load current configuration for Jetson cameras
      if (isJetsonCamera) {
        // Note: We could add API calls here to get current state
        // For now, we'll start with default states
        setGestureVisualization(false);
        setFaceVisualization(false);
        
        // Load gesture controls state from localStorage
        // TODO: Fix this to use unified service properly
        // const gestureControlsEnabled = await unifiedCameraService.getGestureControlsStatus(camera.id);
        // setGestureControls(gestureControlsEnabled);
        setGestureControls(false); // Default for now
      }
    }
  }, [isOpen, isJetsonCamera]);

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

  // Handle gesture visualization toggle
  const handleGestureVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !gestureVisualization;
      const result = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      
      if (result.success) {
        setGestureVisualization(newState);
      } else {
        setError(result.error || 'Failed to toggle gesture visualization');
      }
    } catch (error) {
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
      const result = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      
      if (result.success) {
        setFaceVisualization(newState);
      } else {
        setError(result.error || 'Failed to toggle face visualization');
      }
    } catch (error) {
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