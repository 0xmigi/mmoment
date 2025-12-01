import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, CheckCircle2, Camera, UserRound, Info, Loader, AlertTriangle } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey } from '@solana/web3.js';
import { useCamera } from './CameraProvider';

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
  // Use unified check-in state from CameraProvider context
  const { isCheckedIn, isCheckingIn, checkInError, checkIn, checkOut, refreshCheckInStatus } = useCamera();
  const [useFaceRecognition, setUseFaceRecognition] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [walletWarning, setWalletWarning] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [showAdvancedInfo, setShowAdvancedInfo] = useState(false);

  // Validate camera address and refresh check-in status when modal opens
  useEffect(() => {
    if (isOpen && cameraId) {
      try {
        // Validate that the camera address is a valid PublicKey
        new PublicKey(cameraId);

        // Reset local UI states when modal opens
        setError(null);
        setWalletWarning(null);
        setActionStatus('idle');

        // Refresh check-in status from context (queries Jetson)
        refreshCheckInStatus();

      } catch (err) {
        setError(`Invalid camera address: ${cameraId}`);
        console.error('Invalid camera address:', err);
      }
    }
  }, [isOpen, cameraId, primaryWallet?.address, refreshCheckInStatus]);

  // Handle check-in using unified context
  const handleCheckIn = async () => {
    setError(null);
    setActionStatus('idle');

    const success = await checkIn();

    if (success) {
      setActionStatus('success');
      // Wait a moment before notifying success
      setTimeout(() => {
        if (onSuccess) {
          onSuccess('off_chain_session');
        }
      }, 1500);
    } else {
      setActionStatus('error');
      setError(checkInError || 'Check-in failed');
      if (onError) {
        onError(new Error(checkInError || 'Check-in failed'));
      }
    }
  };

  // Handle check-out using unified context
  const handleCheckOut = async () => {
    setError(null);
    setActionStatus('idle');

    const success = await checkOut();

    if (success) {
      setActionStatus('success');
      // Wait a moment before notifying success
      setTimeout(() => {
        if (onSuccess) {
          onSuccess('off_chain_checkout');
        }
      }, 1500);
    } else {
      setActionStatus('error');
      setError(checkInError || 'Check-out failed');
      if (onError) {
        onError(new Error(checkInError || 'Check-out failed'));
      }
    }
  };

  const renderStatusMessage = () => {
    // Show loading state from context
    if (isCheckingIn) {
      return (
        <div className="bg-blue-50 text-blue-700 p-2 rounded-lg text-sm flex items-center">
          <Loader className="w-4 h-4 mr-2 animate-spin" />
          Processing your request... This may take a few moments.
        </div>
      );
    }

    // Show success state
    if (actionStatus === 'success') {
      return (
        <div className="bg-green-50 text-green-700 p-2 rounded-lg text-sm flex items-center">
          <CheckCircle2 className="w-4 h-4 mr-2" />
          {isCheckedIn ? 'Check-in successful!' : 'Check-out successful!'}
        </div>
      );
    }

    // Show error state
    if (actionStatus === 'error' && error) {
      return (
        <div className="bg-red-50 text-red-700 p-2 rounded-lg text-sm flex items-center">
          <AlertTriangle className="w-4 h-4 mr-2" />
          {error}
        </div>
      );
    }

    return null;
  };

  if (!isOpen) return null;

  return (
    <Dialog
      open={isOpen}
      onClose={() => {
        if (!isCheckingIn) onClose();
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
              disabled={isCheckingIn}
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
                    disabled={isCheckingIn}
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
                  disabled={isCheckingIn}
                  className={`w-full py-2.5 rounded-lg font-medium ${
                    isCheckingIn ? 'bg-red-300 text-red-800 cursor-not-allowed' : 'bg-red-600 text-white hover:bg-red-700'
                  }`}
                >
                  {isCheckingIn ? (
                    <span className="flex items-center justify-center">
                      <Loader className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </span>
                  ) : 'Check Out'}
                </button>
              ) : (
                <button
                  onClick={handleCheckIn}
                  disabled={isCheckingIn}
                  className={`w-full py-2.5 rounded-lg font-medium ${
                    isCheckingIn ? 'bg-blue-300 text-blue-800 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isCheckingIn ? (
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