import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, CheckCircle2, Camera, UserRound, Info, Loader, AlertTriangle } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, Transaction, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';

interface CameraActionModalProps {
  isOpen: boolean;
  onClose: () => void;
  cameraId: string | null;
  onSuccess?: (signature: string) => void;
  onError?: (error: any) => void;
}

export const CameraActionModal = ({ 
  isOpen, 
  onClose, 
  cameraId,
  onSuccess,
  onError
}: CameraActionModalProps) => {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [useFaceRecognition, setUseFaceRecognition] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [walletWarning, setWalletWarning] = useState<string | null>(null);
  const [checkInStatus, setCheckInStatus] = useState<'idle' | 'checking' | 'success' | 'error'>('idle');
  const [isCheckedIn, setIsCheckedIn] = useState(false);
  const [showAdvancedInfo, setShowAdvancedInfo] = useState(false);

  // Validate camera address when the modal opens
  useEffect(() => {
    if (isOpen && cameraId) {
      try {
        // Validate that the camera address is a valid PublicKey
        new PublicKey(cameraId);
        
        // Reset states when modal opens
        setError(null);
        setWalletWarning(null);
        setCheckInStatus('idle');
        
        // Check wallet balance when modal opens
        if (primaryWallet?.address && connection) {
          connection.getBalance(new PublicKey(primaryWallet.address))
            .then((balance) => {
              const solBalance = balance / 1e9;
              if (solBalance < 0.05) {
                setWalletWarning(`Low SOL balance (${solBalance.toFixed(4)} SOL). You may need to add more SOL to your wallet.`);
              }
            })
            .catch(err => {
              console.error('Error checking wallet balance:', err);
            });
        }

        // Check if already checked in
        checkIsCheckedIn(cameraId);
        
      } catch (err) {
        setError(`Invalid camera address: ${cameraId}`);
        console.error('Invalid camera address:', err);
      }
    }
  }, [isOpen, cameraId, primaryWallet?.address, connection]);

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

  // Check if user is already checked in
  const checkIsCheckedIn = async (cameraId: string) => {
    if (!primaryWallet?.address || !connection) return;

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(cameraId);

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
        setIsCheckedIn(true);
      } catch (err) {
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error('Error checking session status:', err);
      setIsCheckedIn(false);
    }
  };

  // Handle check-in action
  const handleCheckIn = async () => {
    if (!cameraId || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setCheckInStatus('checking');
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(cameraId);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Check if user already has an active session - if so, check out first
      try {
        const existingSession = await program.account.userSession.fetch(sessionPda);
        console.log('[CameraActionModal] Existing session found, checking out first:', existingSession);

        // Check out existing session
        const checkOutIx = await program.methods
          .checkOut()
          .accounts({
            closer: userPublicKey,
            camera: cameraPublicKey,
            session: sessionPda,
            sessionUser: userPublicKey, // Rent goes back to user
          })
          .instruction();

        const checkOutTx = new Transaction().add(checkOutIx);
        const { blockhash: checkOutBlockhash } = await connection.getLatestBlockhash();
        checkOutTx.recentBlockhash = checkOutBlockhash;
        checkOutTx.feePayer = userPublicKey;

        // Use the Dynamic wallet to sign and send
        if (!isSolanaWallet(primaryWallet)) {
          throw new Error('Not a Solana wallet');
        }

        const signer = await primaryWallet.getSigner();
        const signedCheckOutTx = await signer.signTransaction(checkOutTx);
        const checkOutSig = await connection.sendRawTransaction(signedCheckOutTx.serialize());
        await connection.confirmTransaction(checkOutSig, 'confirmed');
        console.log('[CameraActionModal] Checked out existing session:', checkOutSig);
      } catch (err) {
        // No existing session - this is fine, continue with check-in
        console.log('[CameraActionModal] No existing session found, proceeding with check-in');
      }

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
        console.log('[CameraActionModal] Recognition token found for user');
      } catch (err) {
        console.log('[CameraActionModal] No recognition token found for user');
      }

      // Create the accounts object for check-in
      const accounts: any = {
        user: userPublicKey,
        camera: cameraPublicKey,
        session: sessionPda,
        systemProgram: SystemProgram.programId
      };

      // Only include recognition token if it exists
      if (hasRecognitionToken) {
        accounts.recognitionToken = recognitionTokenPda;
      }

      // Create check-in instruction
      const ix = await program.methods
        .checkIn(useFaceRecognition)
        .accounts(accounts)
        .instruction();

      // Create and sign the transaction using Dynamic
      const tx = new Transaction().add(ix);
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Use the Dynamic wallet to sign and send
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('Not a Solana wallet');
      }
      
      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');
      
      setIsCheckedIn(true);
      setCheckInStatus('success');
      
      // Wait a moment before closing modal
      setTimeout(() => {
        if (onSuccess) {
          onSuccess(signature);
        }
      }, 1500);
    } catch (error) {
      console.error('Check-in error:', error);
      setCheckInStatus('error');
      
      // Provide more friendly error messages
      if (error instanceof Error) {
        let errorMsg = error.message;
        
        // Check for common error messages and provide more user-friendly versions
        if (errorMsg.includes('custom program error: 0x64')) {
          errorMsg = 'Program error: The camera may not be configured correctly (error code 100).';
        } else if (errorMsg.includes('insufficient funds')) {
          errorMsg = 'Insufficient SOL in your wallet. Please add more SOL and try again.';
        } else if (errorMsg.includes('already in use')) {
          errorMsg = 'You are already checked in to this camera.';
          // This should be handled as a success case
          setCheckInStatus('success');
          setTimeout(() => {
            if (onSuccess) {
              onSuccess("already_checked_in");
            }
          }, 1500);
          setLoading(false);
          return;
        } else if (errorMsg.length > 150) {
          // If the error message is too long, provide a shorter, more general message
          errorMsg = 'An error occurred during check-in. Please check the browser console for details.';
        }
        
        setError(errorMsg);
      } else {
        setError('Unknown error during check-in');
      }
      
      if (onError) {
        onError(error);
      }
    } finally {
      setLoading(false);
    }
  };

  // Handle check-out action
  const handleCheckOut = async () => {
    if (!cameraId || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setCheckInStatus('checking');
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(cameraId);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Fetch session to get the original user (for rent reclamation)
      const sessionAccount = await program.account.userSession.fetch(sessionPda) as any;

      // Create check-out instruction
      const ix = await program.methods
        .checkOut()
        .accounts({
          closer: userPublicKey,  // Changed from 'user' to 'closer'
          camera: cameraPublicKey,
          session: sessionPda,
          sessionUser: sessionAccount.user as PublicKey,  // Original session creator
        })
        .instruction();

      // Create and sign the transaction using Dynamic
      const tx = new Transaction().add(ix);
      const { blockhash } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Use the Dynamic wallet to sign and send
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('Not a Solana wallet');
      }
      
      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');
      
      setIsCheckedIn(false);
      setCheckInStatus('success');
      
      // Wait a moment before closing modal
      setTimeout(() => {
        if (onSuccess) {
          onSuccess(signature);
        }
      }, 1500);
    } catch (error) {
      console.error('Check-out error:', error);
      setCheckInStatus('error');
      
      // Provide more friendly error messages
      if (error instanceof Error) {
        let errorMsg = error.message;
        
        // Check for common error messages and provide more user-friendly versions
        if (errorMsg.includes('custom program error')) {
          errorMsg = 'Program error: There was an issue with the check-out transaction.';
        } else if (errorMsg.includes('insufficient funds')) {
          errorMsg = 'Insufficient SOL in your wallet. Please add more SOL and try again.';
        } else if (errorMsg.length > 150) {
          // If the error message is too long, provide a shorter, more general message
          errorMsg = 'An error occurred during check-out. Please check the browser console for details.';
        }
        
        setError(errorMsg);
      } else {
        setError('Unknown error during check-out');
      }
      
      if (onError) {
        onError(error);
      }
    } finally {
      setLoading(false);
    }
  };

  const renderStatusMessage = () => {
    switch (checkInStatus) {
      case 'checking':
        return (
          <div className="bg-blue-50 text-blue-700 p-2 rounded-lg text-sm flex items-center">
            <Loader className="w-4 h-4 mr-2 animate-spin" />
            Processing your request... This may take a few moments.
          </div>
        );
      case 'success':
        return (
          <div className="bg-green-50 text-green-700 p-2 rounded-lg text-sm flex items-center">
            <CheckCircle2 className="w-4 h-4 mr-2" />
            {isCheckedIn ? 'Check-in successful!' : 'Check-out successful!'}
          </div>
        );
      case 'error':
        if (!error) return null;
        return (
          <div className="bg-red-50 text-red-700 p-2 rounded-lg text-sm flex items-center">
            <AlertTriangle className="w-4 h-4 mr-2" />
            {error}
          </div>
        );
      default:
        return null;
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog 
      open={isOpen} 
      onClose={() => {
        if (!loading) onClose();
      }}
      className="relative z-50"
    >
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />
      
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="mx-auto max-w-sm rounded-lg bg-white shadow-xl">
          <div className="flex items-center justify-between border-b border-gray-200 p-4">
            <Dialog.Title className="text-lg font-medium">
              Camera Actions
            </Dialog.Title>
            
            <button
              onClick={onClose}
              disabled={loading}
              className="text-gray-400 hover:text-gray-500"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          
          <div className="p-4 space-y-4">
            <div className="flex items-center space-x-3 p-2 bg-blue-50 rounded-lg">
              <Camera className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm text-blue-900">
                  {isCheckedIn 
                    ? "You are currently checked in to this camera" 
                    : "You need to check in to this camera to perform actions"}
                </p>
              </div>
            </div>

            {/* Camera address display */}
            <div className="text-xs bg-gray-50 p-2 rounded-lg overflow-hidden text-ellipsis">
              <span className="font-medium">Camera:</span> {cameraId || 'Not selected'}
            </div>

            {/* Status message */}
            {renderStatusMessage()}

            {/* Wallet warning if applicable */}
            {walletWarning && (
              <div className="bg-yellow-50 text-yellow-700 p-2 rounded-lg text-xs flex items-center">
                <AlertTriangle className="w-3.5 h-3.5 mr-1.5 flex-shrink-0" />
                {walletWarning}
              </div>
            )}

            {/* Face Recognition option */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-50 cursor-pointer">
                  <input
                    type="checkbox"
                    id="use-face-recognition"
                    checked={useFaceRecognition}
                    onChange={(e) => setUseFaceRecognition(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500"
                    disabled={loading}
                  />
                  <span className="text-sm flex items-center">
                    <UserRound className="w-4 h-4 mr-1.5 text-blue-600" />
                    Use Face Recognition
                  </span>
                </label>
                <button 
                  onClick={() => setShowAdvancedInfo(!showAdvancedInfo)}
                  className="text-gray-500 hover:text-blue-600 transition-colors"
                >
                  <Info className="w-4 h-4" />
                </button>
              </div>

              {showAdvancedInfo && (
                <div className="text-xs bg-gray-50 p-2 rounded-lg ml-7">
                  <p>Face recognition enables the camera to identify you automatically.</p>
                  <p className="mt-1">When enabled, your facial features will be stored securely in the blockchain.</p>
                </div>
              )}
            </div>

            {/* Action buttons */}
            <div className="pt-2">
              {isCheckedIn ? (
                <button
                  onClick={handleCheckOut}
                  disabled={loading}
                  className={`w-full py-2.5 rounded-lg font-medium ${
                    loading ? 'bg-red-300 text-red-800 cursor-not-allowed' : 'bg-red-600 text-white hover:bg-red-700'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <Loader className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </span>
                  ) : 'Check Out'}
                </button>
              ) : (
                <button
                  onClick={handleCheckIn}
                  disabled={loading}
                  className={`w-full py-2.5 rounded-lg font-medium ${
                    loading ? 'bg-blue-300 text-blue-800 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <Loader className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </span>
                  ) : 'Check In'}
                </button>
              )}
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default CameraActionModal; 