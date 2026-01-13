import { Dialog } from '@headlessui/react';
import { X, CheckCircle } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useState, useEffect } from 'react';
import { timelineService } from '../../timeline/timeline-service';
import { unifiedCameraService } from '../../camera/unified-camera-service';
import { createSignedRequest } from '../../camera/request-signer';
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
    if (isOpen && transactionData?.cameraAccount && primaryWallet?.address) {
      checkSessionStatus();
    }
  }, [isOpen, transactionData, primaryWallet]);

  // NEW PRIVACY ARCHITECTURE: Check session via localStorage (off-chain)
  const checkSessionStatus = async () => {
    if (!transactionData?.cameraAccount || !primaryWallet?.address) return;

    try {
      const sessionKey = `mmoment_session_${primaryWallet.address}_${transactionData.cameraAccount}`;
      const storedSession = localStorage.getItem(sessionKey);

      if (storedSession) {
        const session = JSON.parse(storedSession);
        const sessionAge = Date.now() - session.timestamp;
        const maxAge = 24 * 60 * 60 * 1000; // 24 hours

        if (sessionAge < maxAge) {
          setIsCheckedIn(true);
          return;
        } else {
          localStorage.removeItem(sessionKey);
        }
      }

      setIsCheckedIn(false);
    } catch (err) {
      console.error('Error checking session status:', err);
      setIsCheckedIn(false);
    }
  };
  
  // NEW PRIVACY ARCHITECTURE: Check-in is now off-chain via Jetson
  // PHASE 3: Requires Ed25519 signature for cryptographic handshake
  const handleCheckIn = async () => {
    if (!transactionData?.cameraAccount || !primaryWallet?.address) {
      setError('Wallet not connected or missing data');
      return;
    }

    setIsCheckingIn(true);
    setError(null);

    try {
      console.log('[TransactionModal] Starting off-chain check-in...');

      // PHASE 3: Create signed request for cryptographic handshake
      const signedParams = await createSignedRequest(primaryWallet);
      if (!signedParams) {
        setError('Failed to sign check-in request');
        setIsCheckingIn(false);
        return;
      }

      // Call Jetson check-in endpoint with Ed25519 signature
      const checkinResult = await unifiedCameraService.checkin(transactionData.cameraAccount, {
        ...signedParams,
        display_name: primaryProfile?.displayName,
        username: primaryProfile?.username
      });

      if (checkinResult.success) {
        console.log('[TransactionModal] Off-chain check-in successful!', checkinResult.data);

        // Store session locally
        const sessionKey = `mmoment_session_${primaryWallet.address}_${transactionData.cameraAccount}`;
        localStorage.setItem(sessionKey, JSON.stringify({
          sessionId: checkinResult.data?.session_id,
          timestamp: Date.now(),
          cameraId: transactionData.cameraAccount,
          walletAddress: primaryWallet.address
        }));

        setIsCheckedIn(true);
        setCheckInSuccess(true);
        timelineService.refreshEvents();

        // Automatically execute the camera action after successful check-in
        if (transactionData.type) {
          setTimeout(() => {
            handleCameraAction(checkinResult.data?.session_id || 'off_chain_session');
          }, 500);
        }
      } else {
        throw new Error(checkinResult.error || 'Check-in failed');
      }

    } catch (error) {
      console.error('Check-in error:', error);

      if (error instanceof Error) {
        let errorMsg = error.message;

        if (errorMsg.includes('already checked in')) {
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

  // NEW PRIVACY ARCHITECTURE: No blockchain transaction needed for camera actions
  // User is already checked in off-chain, just execute the camera action
  const handleConfirmTransaction = async () => {
    if (!primaryWallet?.address || !transactionData) {
      setError('Wallet not connected or missing data');
      return;
    }

    setLoading(true);
    setError('');

    try {
      setStatus('Executing camera action...');

      // Get session ID from localStorage
      const sessionKey = `mmoment_session_${primaryWallet.address}_${transactionData.cameraAccount}`;
      const storedSession = localStorage.getItem(sessionKey);
      const sessionId = storedSession ? JSON.parse(storedSession).sessionId : 'session';

      // Execute the camera action directly
      if (transactionData.type) {
        await handleCameraAction(sessionId);
      } else {
        // If no camera action needed, just pass success
        onSuccess?.({
          transactionId: sessionId,
          cameraId: transactionData.cameraAccount
        });
        onClose();
      }
    } catch (err) {
      console.error('Action error:', err);
      setError(err instanceof Error ? err.message : 'Action failed');
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
                      className="flex-1 bg-primary-muted text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors"
                    >
                      Processing...
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={handleCheckIn}
                        disabled={isCheckingIn}
                        className="flex-1 bg-primary text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
                  <div className="bg-primary-light text-primary px-2 py-1.5 rounded-lg text-xs flex items-center">
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
                    className="flex-1 bg-primary text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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