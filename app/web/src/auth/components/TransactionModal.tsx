import { Dialog } from '@headlessui/react';
import { X, CheckCircle } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from '../../anchor/setup';
import { SystemProgram, PublicKey, ComputeBudgetProgram, Transaction } from '@solana/web3.js';
import { useState, useEffect } from 'react';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../../anchor/idl';
import { timelineService } from '../../timeline/timeline-service';
import { unifiedCameraService } from '../../camera/unified-camera-service';
import { useSocialProfile } from '../social/useSocialProfile';

interface TransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactionData?: {
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  };
  onSuccess?: (data: { transactionId: string; cameraId: string }) => void;
}

export const TransactionModal: React.FC<TransactionModalProps> = ({
  isOpen,
  onClose,
  transactionData,
  onSuccess
}) => {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const { program } = useProgram();
  const { primaryProfile } = useSocialProfile();
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fee] = useState<number>(100); // Default fee in lamports
  const [isCheckedIn, setIsCheckedIn] = useState<boolean>(false);
  const [isCheckingIn, setIsCheckingIn] = useState<boolean>(false);
  const [checkInSuccess, setCheckInSuccess] = useState<boolean>(false);
  
  // Check if user is already checked in
  useEffect(() => {
    if (isOpen && transactionData?.cameraAccount && primaryWallet?.address && connection) {
      checkSessionStatus();
    }
  }, [isOpen, transactionData, primaryWallet, connection]);
  
  // Function to check if user is already checked in
  const checkSessionStatus = async () => {
    if (!transactionData?.cameraAccount || !primaryWallet?.address || !connection) return;
    
    try {
      // Initialize program
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );
      
      const program = new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      
      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          new PublicKey(primaryWallet.address).toBuffer(),
          new PublicKey(transactionData.cameraAccount).toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      
      // Try to fetch the session account
      try {
        await program.account.userSession.fetch(sessionPda);
        setIsCheckedIn(true);
      } catch (err) {
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error('Error checking session status:', err);
      setIsCheckedIn(false);
    }
  };
  
  // Function to handle check-in
  const handleCheckIn = async () => {
    if (!transactionData?.cameraAccount || !primaryWallet?.address || !connection) {
      setError('Wallet not connected or missing data');
      return;
    }
    
    setIsCheckingIn(true);
    setError(null);
    
    try {
      // Initialize program
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );
      
      const program = new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(transactionData.cameraAccount);
      
      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Derive recognition token PDA and check if it exists
      const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('recognition-token'), userPublicKey.toBuffer()],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Check if recognition token account exists
      let hasRecognitionToken = false;
      try {
        await (program.account as any).recognitionToken.fetch(recognitionTokenPda);
        hasRecognitionToken = true;
      } catch (err) {
        // No token - that's fine
      }

      // Build accounts object
      const checkInAccounts: any = {
        user: userPublicKey,
        camera: cameraPublicKey,
        session: sessionPda,
        systemProgram: SystemProgram.programId
      };

      // Only include recognition token if it exists
      if (hasRecognitionToken) {
        checkInAccounts.recognitionToken = recognitionTokenPda;
      }

      // Create check-in instruction (without face recognition for simplicity)
      const ix = await program.methods
        .checkIn(false) // No face recognition for simplicity in this flow
        .accounts(checkInAccounts)
        .instruction();
      
      // Create transaction
      const tx = new Transaction().add(ix);
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;
      
      // Use wallet to sign and send
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('Not a Solana wallet');
      }
      
      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');
      
      // Send user profile to camera for display name labeling
      if (primaryProfile?.displayName || primaryProfile?.username) {
        try {
          await unifiedCameraService.sendUserProfile(transactionData.cameraAccount, {
            wallet_address: primaryWallet.address,
            display_name: primaryProfile.displayName,
            username: primaryProfile.username
          });
        } catch (err) {
          console.warn('Failed to send display name to camera:', err);
          // Don't fail the check-in if this fails
        }
      }
      
      // NOTE: Check-in timeline event is now created by Jetson via buffer_checkin_activity()
      // for proper encryption and privacy-preserving timeline architecture

      setIsCheckedIn(true);
      setCheckInSuccess(true);
      timelineService.refreshEvents();
      
      // Automatically execute the camera action after successful check-in
      if (transactionData.type) {
        setTimeout(() => {
          handleCameraAction(signature);
        }, 500);
      }
      
    } catch (error) {
      console.error('Check-in error:', error);
      
      if (error instanceof Error) {
        let errorMsg = error.message;
        
        // Check for common error messages
        if (errorMsg.includes('custom program error: 0x64')) {
          errorMsg = 'Program error: The camera may not be configured correctly.';
        } else if (errorMsg.includes('insufficient funds')) {
          errorMsg = 'Insufficient SOL in your wallet. Please add more SOL and try again.';
        } else if (errorMsg.includes('already in use')) {
          errorMsg = 'You are already checked in to this camera.';
          setIsCheckedIn(true);
          return;
        } else if (errorMsg.length > 150) {
          errorMsg = 'An error occurred during check-in. Please check the console for details.';
        }
        
        setError(errorMsg);
      } else {
        setError('Unknown error during check-in');
      }
    } finally {
      setIsCheckingIn(false);
    }
  };

  // New function to handle camera actions after check-in
  const handleCameraAction = async (checkInSignature: string) => {
    if (!transactionData?.type || !primaryWallet?.address) return;
    
    setStatus(`Performing ${transactionData.type} action...`);
    setLoading(true);
    
    try {
      let response;
      
      // Use the existing signature for the action
      switch (transactionData.type) {
        case 'photo':
          response = await unifiedCameraService.takePhoto(transactionData.cameraAccount);
          break;
        case 'video':
          response = await unifiedCameraService.startVideoRecording(transactionData.cameraAccount);
          break;
        case 'stream':
          response = await unifiedCameraService.startStream(transactionData.cameraAccount);
          break;
        default:
          throw new Error(`Unknown action type: ${transactionData.type}`);
      }
      
      if (response.success) {
        setStatus(`${transactionData.type} action completed successfully`);
        
        // Pass back the transaction signature and camera ID
        onSuccess?.({
          transactionId: checkInSignature,
          cameraId: transactionData.cameraAccount
        });
        
        // Close modal after success
        setTimeout(() => onClose(), 1000);
      } else {
        setError(`Failed to perform ${transactionData.type} action: ${response.error || 'Unknown error'}`);
      }
    } catch (err) {
      console.error(`Error performing ${transactionData.type} action:`, err);
      setError(err instanceof Error ? err.message : 'Action failed');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmTransaction = async () => {
    if (!primaryWallet?.address || !program || !transactionData) {
      setError('Wallet not connected or missing data');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const cameraPublicKey = new PublicKey(transactionData.cameraAccount);
      
      const userPublicKey = new PublicKey(primaryWallet.address);

      // Find the session PDA for the transaction
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        program.programId
      );

      // Derive recognition token PDA and check if it exists
      const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('recognition-token'), userPublicKey.toBuffer()],
        program.programId
      );

      // Check if recognition token account exists
      let hasRecognitionToken = false;
      try {
        await (program.account as any).recognitionToken.fetch(recognitionTokenPda);
        hasRecognitionToken = true;
      } catch (err) {
        // No token - that's fine
      }

      setStatus('Preparing transaction...');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      // Build accounts object
      const checkInAccounts: any = {
        user: userPublicKey,
        camera: cameraPublicKey,
        session: sessionPda,
        systemProgram: SystemProgram.programId,
      };

      // Only include recognition token if it exists
      if (hasRecognitionToken) {
        checkInAccounts.recognitionToken = recognitionTokenPda;
      }

      // Instead of recordActivity (which doesn't exist), use checkIn with useFaceRecognition=false
      const ix = await program.methods
        .checkIn(false)
        .accounts(checkInAccounts)
        .instruction();
      
      // Add priority fee
      const addPriorityFee = ComputeBudgetProgram.setComputeUnitPrice({
        microLamports: 375000, // 0.375 lamports per compute unit
      });
      
      // Set compute unit limit
      const modifyComputeUnits = ComputeBudgetProgram.setComputeUnitLimit({
        units: 200000,
      });
      
      // Create new transaction and add instructions
      const transaction = new Transaction();
      transaction.add(addPriorityFee, modifyComputeUnits, ix);
      
      setStatus('Signing transaction...');
      
      // Get the signer
      const signer = await primaryWallet.getSigner();
      
      // Set recent blockhash and fee payer
      const walletConnection = await primaryWallet.getConnection();
      transaction.recentBlockhash = (await walletConnection.getLatestBlockhash()).blockhash;
      transaction.feePayer = new PublicKey(primaryWallet.address);
      
      // Sign and send the transaction
      const result = await signer.signAndSendTransaction(transaction);
      const signature = result.signature;
      
      setStatus('Transaction confirmed');
      
      // Automatically execute the camera action
      if (transactionData.type) {
        setTimeout(() => {
          handleCameraAction(signature);
        }, 500);
      } else {
        // If no camera action needed, just pass success
        onSuccess?.({
          transactionId: signature,
          cameraId: transactionData.cameraAccount
        });
        onClose();
      }
    } catch (err) {
      console.error('Transaction error:', err);
      setError(err instanceof Error ? err.message : 'Transaction failed');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const formatActionType = (type: string) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  };

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
              {!isCheckedIn 
                ? 'Check In Required' 
                : `Confirm ${transactionData?.type || ''} Action`}
            </Dialog.Title>
            {!loading && !isCheckingIn && (
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            )}
          </div>

          {/* Transaction Content */}
          <div className="p-3 space-y-4">
            {!isCheckedIn ? (
              // Show check-in UI if not checked in
              <>
                <div className="bg-yellow-50 border border-yellow-100 rounded-lg p-3">
                  <div className="flex items-start">
                    <div className="flex-1">
                      <p className="text-sm text-yellow-800 mb-1 font-medium">Camera Check-in Required</p>
                      <p className="text-xs text-yellow-700">
                        Please check in to continue with your {transactionData?.type || 'camera'} action.
                      </p>
                    </div>
                  </div>
                </div>
                
                {checkInSuccess && (
                  <div className="bg-green-50 text-green-700 px-3 py-2 rounded-lg text-sm flex items-center">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Starting {transactionData?.type || 'camera'} action...
                  </div>
                )}
                
                {error && (
                  <div className="bg-red-50 text-red-700 px-2 py-1.5 rounded-lg text-xs">
                    {error}
                  </div>
                )}
                
                <div className="flex gap-2 pt-2">
                  {checkInSuccess ? (
                    <button
                      disabled={true}
                      className="flex-1 bg-blue-400 text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors"
                    >
                      Processing...
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={handleCheckIn}
                        disabled={isCheckingIn}
                        className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isCheckingIn ? 'Checking in...' : 'Check In & Continue'}
                      </button>
                      <button
                        onClick={onClose}
                        className="flex-1 bg-gray-100 text-gray-700 py-2 px-3 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                      >
                        Cancel
                      </button>
                    </>
                  )}
                </div>
              </>
            ) : (
              // Show normal transaction UI if already checked in
              <>
                {/* Transaction Details */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                    <div>
                      <div className="text-xs font-medium text-gray-500">Action</div>
                      <div className="text-sm text-gray-900">
                        {formatActionType(transactionData?.type || '')}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                    <div>
                      <div className="text-xs font-medium text-gray-500">Network</div>
                      <div className="text-sm text-gray-900">Solana Devnet</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-1.5 bg-gray-50 px-2 rounded-lg">
                    <div>
                      <div className="text-xs font-medium text-gray-500">Fee</div>
                      <div className="text-sm text-gray-900">{fee} lamports</div>
                    </div>
                  </div>
                </div>

                {/* Status Messages */}
                {status && (
                  <div className="bg-blue-50 text-blue-700 px-2 py-1.5 rounded-lg text-xs flex items-center">
                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-current mr-2" />
                    {status}
                  </div>
                )}

                {error && (
                  <div className="bg-red-50 text-red-700 px-2 py-1.5 rounded-lg text-xs">
                    {error}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={handleConfirmTransaction}
                    disabled={loading}
                    className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Processing...' : 'Confirm'}
                  </button>
                  {!loading && (
                    <button
                      onClick={onClose}
                      className="flex-1 bg-gray-100 text-gray-700 py-2 px-3 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}; 