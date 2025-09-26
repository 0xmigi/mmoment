import { Dialog } from '@headlessui/react';
import { X, ScanFace, CheckCircle, AlertCircle, Loader2, Shield, Clock } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { FacialEmbeddingManager } from '../../camera/FacialEmbeddingManager';
import { FacialEmbeddingStatus } from '../../hooks/useFacialEmbeddingStatus';

interface RecognitionTokenModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: FacialEmbeddingStatus;
  onStatusUpdate?: () => void;
}

export function RecognitionTokenModal({ isOpen, onClose, status, onStatusUpdate }: RecognitionTokenModalProps) {
  const { primaryWallet } = useDynamicContext();

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
            <Dialog.Title className="text-base font-medium flex items-center">
              <ScanFace className="w-4 h-4 mr-2" />
              Recognition Token
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="p-4">
            {/* Status Header */}
            <div className={`rounded-lg p-4 mb-4 border-2 ${
              status.hasEmbedding
                ? 'bg-green-50 border-green-200'
                : 'bg-orange-50 border-orange-200'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  {status.isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin text-blue-500 mr-2" />
                  ) : status.hasEmbedding ? (
                    <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-orange-500 mr-2" />
                  )}
                  <span className="font-medium">
                    {status.hasEmbedding ? 'Active' : 'Not Enrolled'}
                  </span>
                </div>
                <div className={`text-xs px-2 py-1 rounded-full font-medium ${
                  status.hasEmbedding
                    ? 'bg-green-100 text-green-700'
                    : 'bg-orange-100 text-orange-700'
                }`}>
                  {status.hasEmbedding ? 'SECURED' : 'PENDING'}
                </div>
              </div>

              <p className={`text-sm ${
                status.hasEmbedding ? 'text-green-700' : 'text-orange-700'
              }`}>
                {status.hasEmbedding
                  ? 'Your encrypted facial embedding is stored on-chain and ready for use'
                  : 'Enhanced security features require face enrollment at any camera'
                }
              </p>
            </div>

            {/* Detailed Status Information */}
            <div className="space-y-4 mb-4">
              {/* Security Level */}
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Shield className="w-4 h-4 text-gray-500 mr-3" />
                  <div>
                    <div className="text-sm font-medium">Security Level</div>
                    <div className="text-xs text-gray-500">Encryption & Privacy</div>
                  </div>
                </div>
                <div className={`text-sm font-medium ${
                  status.hasEmbedding ? 'text-green-600' : 'text-orange-600'
                }`}>
                  {status.hasEmbedding ? 'AES-256 Encrypted' : 'Not Protected'}
                </div>
              </div>

              {/* Storage Location */}
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="w-4 h-4 rounded bg-purple-500 mr-3 flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <div>
                    <div className="text-sm font-medium">Storage</div>
                    <div className="text-xs text-gray-500">Blockchain location</div>
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  {status.hasEmbedding ? 'Solana DevNet' : 'Not stored'}
                </div>
              </div>

              {/* Last Status Check */}
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Clock className="w-4 h-4 text-gray-500 mr-3" />
                  <div>
                    <div className="text-sm font-medium">Last Checked</div>
                    <div className="text-xs text-gray-500">Status verification</div>
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  {status.lastChecked
                    ? status.lastChecked.toLocaleTimeString()
                    : 'Never'
                  }
                </div>
              </div>

              {/* Error Display */}
              {status.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <div className="text-sm font-medium text-red-800">Error</div>
                  <div className="text-xs text-red-700 mt-1">{status.error}</div>
                </div>
              )}
            </div>

            {/* Action Section */}
            <div className="border-t border-gray-200 pt-4">
              {status.hasEmbedding ? (
                <div className="text-center">
                  <div className="text-sm text-gray-600 mb-3">
                    Your facial recognition is active and working across all cameras in the network.
                  </div>
                  <div className="text-xs text-gray-500">
                    This enables seamless authentication and enhanced security features.
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-sm text-gray-700 mb-3">
                    To enable face recognition, check in to any camera and complete the enrollment process.
                  </div>

                  {primaryWallet?.address && (
                    <div className="bg-gray-50 rounded-lg p-3">
                      <FacialEmbeddingManager
                        walletAddress={primaryWallet.address}
                        onComplete={() => {
                          if (onStatusUpdate) {
                            onStatusUpdate();
                          }
                          onClose();
                        }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}