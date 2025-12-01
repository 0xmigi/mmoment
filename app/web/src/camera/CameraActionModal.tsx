import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, CheckCircle2, Camera, UserRound, Info, Loader, AlertTriangle } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey } from '@solana/web3.js';
import { unifiedCameraService } from './unified-camera-service';
import { createSignedRequest } from './request-signer';
import { useSocialProfile } from '../auth/social/useSocialProfile';

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
  const { primaryProfile } = useSocialProfile();
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

        // Check if already checked in (via local storage)
        checkIsCheckedIn(cameraId);

      } catch (err) {
        setError(`Invalid camera address: ${cameraId}`);
        console.error('Invalid camera address:', err);
      }
    }
  }, [isOpen, cameraId, primaryWallet?.address, connection]);

  // NEW PRIVACY ARCHITECTURE: Check session status via localStorage
  const checkIsCheckedIn = async (cameraId: string) => {
    if (!primaryWallet?.address) return;

    try {
      const sessionKey = `mmoment_session_${primaryWallet.address}_${cameraId}`;
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
    if (!cameraId || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setCheckInStatus('checking');
    setError(null);

    try {
      console.log('[CameraActionModal] Starting off-chain check-in via Jetson...');

      // PHASE 3: Create signed request for cryptographic handshake
      const signedParams = await createSignedRequest(primaryWallet);
      if (!signedParams) {
        setError('Failed to sign check-in request');
        setLoading(false);
        setCheckInStatus('error');
        return;
      }

      // Call Jetson check-in endpoint with Ed25519 signature
      const checkinResult = await unifiedCameraService.checkin(cameraId, {
        ...signedParams,
        display_name: primaryProfile?.displayName,
        username: primaryProfile?.username
      });

      if (checkinResult.success) {
        console.log('[CameraActionModal] Off-chain check-in successful!', checkinResult.data);

        // Store session locally
        const sessionKey = `mmoment_session_${primaryWallet.address}_${cameraId}`;
        localStorage.setItem(sessionKey, JSON.stringify({
          sessionId: checkinResult.data?.session_id,
          timestamp: Date.now(),
          cameraId,
          walletAddress: primaryWallet.address
        }));

        setIsCheckedIn(true);
        setCheckInStatus('success');

        // Wait a moment before notifying success
        setTimeout(() => {
          if (onSuccess) {
            onSuccess(checkinResult.data?.session_id || 'off_chain_session');
          }
        }, 1500);
      } else {
        throw new Error(checkinResult.error || 'Check-in failed');
      }
    } catch (error) {
      console.error('Check-in error:', error);
      setCheckInStatus('error');

      if (error instanceof Error) {
        let errorMsg = error.message;

        if (errorMsg.includes('already checked in')) {
          errorMsg = 'You are already checked in to this camera.';
          setCheckInStatus('success');
          setIsCheckedIn(true);
          setTimeout(() => {
            if (onSuccess) {
              onSuccess("already_checked_in");
            }
          }, 1500);
          setLoading(false);
          return;
        } else if (errorMsg.length > 150) {
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

  // NEW PRIVACY ARCHITECTURE: Check-out is now off-chain via Jetson
  const handleCheckOut = async () => {
    if (!cameraId || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setCheckInStatus('checking');
    setError(null);

    try {
      console.log('[CameraActionModal] Starting off-chain check-out via Jetson...');

      // Call Jetson checkout endpoint
      const checkoutResult = await unifiedCameraService.checkout(cameraId, {
        wallet_address: primaryWallet.address,
        transaction_signature: '' // No transaction in new architecture
      });

      if (checkoutResult.success) {
        console.log('[CameraActionModal] Off-chain check-out successful!', checkoutResult.data);

        // Clear local session state
        const sessionKey = `mmoment_session_${primaryWallet.address}_${cameraId}`;
        localStorage.removeItem(sessionKey);

        setIsCheckedIn(false);
        setCheckInStatus('success');

        // Wait a moment before notifying success
        setTimeout(() => {
          if (onSuccess) {
            onSuccess(checkoutResult.data?.session_id || 'off_chain_checkout');
          }
        }, 1500);
      } else {
        throw new Error(checkoutResult.error || 'Check-out failed');
      }
    } catch (error) {
      console.error('Check-out error:', error);
      setCheckInStatus('error');

      if (error instanceof Error) {
        let errorMsg = error.message;

        if (errorMsg.length > 150) {
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