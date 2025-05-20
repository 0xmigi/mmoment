/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useState, useEffect } from 'react';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, Transaction, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { timelineService } from '../timeline/timeline-service';

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

  // Check if current user is the owner
  const isOwner = primaryWallet?.address === camera.owner;

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
                  <p className="text-xs text-yellow-700 mb-2">
                    No camera is currently connected. Connect to the physical camera below:
                  </p>
                  <button
                    onClick={handleDevCameraClick}
                    className="text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                  >
                    Connect to Camera <ExternalLink className="h-3 w-3 ml-1" />
                  </button>
                </div>
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
          <div className="text-xs text-gray-500 mt-3 text-center px-4">
            {isCheckedIn ? 
              "You're currently checked in to this camera. Click the button above to check out." :
              "Check in to interact with this camera and record events on the timeline."}
            {error && <div className="mt-1 text-xs text-red-600">Try using a different wallet if you continue to see errors.</div>}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
} 