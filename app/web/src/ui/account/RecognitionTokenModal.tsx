import { Dialog } from '@headlessui/react';
import { X, ScanFace, CheckCircle, AlertCircle, Loader2, Trash2, Smartphone } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PhoneSelfieEnrollment } from '../../camera/PhoneSelfieEnrollment';
import { FacialEmbeddingStatus } from '../../hooks/useFacialEmbeddingStatus';
import { useProgram } from '../../anchor/setup';
import { useState } from 'react';
import { useParams } from 'react-router-dom';

interface RecognitionTokenModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: FacialEmbeddingStatus;
  onStatusUpdate?: () => void;
}

export function RecognitionTokenModal({ isOpen, onClose, status, onStatusUpdate }: RecognitionTokenModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const { cameraId } = useParams<{ cameraId: string }>();
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showEnrollment, setShowEnrollment] = useState(false);

  const handleDeleteRecognitionToken = async () => {
    if (!primaryWallet?.address || !program) {
      setDeleteError('Wallet or program not available');
      return;
    }

    if (!confirm('Are you sure you want to permanently delete your Recognition Token? This action cannot be undone.')) {
      return;
    }

    setIsDeleting(true);
    setDeleteError(null);

    try {
      // Note: This will fail until we add a delete_face instruction to the Solana program
      // For now, we'll show a message explaining the limitation
      throw new Error('Delete functionality requires a program update. Contact support to remove your Recognition Token.');

      // Future implementation would be:
      // const tx = await program.methods
      //   .deleteFace()
      //   .accounts({
      //     user: userPublicKey,
      //     faceData: faceDataPda,
      //     systemProgram: SystemProgram.programId,
      //   })
      //   .rpc();

    } catch (error) {
      console.error('Delete error:', error);
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete Recognition Token');
    } finally {
      setIsDeleting(false);
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
            {/* Main Status Display */}
            <div className={`text-center py-6 mb-4 rounded-lg ${
              status.hasEmbedding
                ? 'bg-green-50'
                : 'bg-orange-50'
            }`}>
              {status.isLoading ? (
                <Loader2 className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-3" />
              ) : status.hasEmbedding ? (
                <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-3" />
              ) : (
                <AlertCircle className="w-8 h-8 text-orange-500 mx-auto mb-3" />
              )}

              <div className="font-medium text-lg mb-2">
                {status.hasEmbedding ? 'Active' : 'Not Enrolled'}
              </div>

              {status.hasEmbedding && (
                <div className="inline-block bg-green-100 text-green-800 text-xs font-medium px-2 py-1 rounded-full mb-2">
                  SECURED
                </div>
              )}

              <div className={`text-sm ${
                status.hasEmbedding ? 'text-green-700' : 'text-orange-700'
              }`}>
                {status.hasEmbedding
                  ? 'Your encrypted facial embedding is stored on-chain and ready for use'
                  : 'Check in to any camera to enroll'
                }
              </div>
            </div>

            {/* Detailed Information - Only show when active */}
            {status.hasEmbedding && (
              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center">
                    <div className="w-6 h-6 bg-blue-100 rounded p-1 mr-3">
                      <div className="w-full h-full bg-blue-500 rounded-sm"></div>
                    </div>
                    <div>
                      <div className="font-medium text-sm">Security Level</div>
                      <div className="text-xs text-gray-500">Encryption & Privacy</div>
                    </div>
                  </div>
                  <div className="text-sm font-medium text-green-600">AES-256 Encrypted</div>
                </div>

                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center">
                    <div className="w-6 h-6 bg-purple-100 rounded p-1 mr-3">
                      <div className="w-full h-full bg-purple-500 rounded-sm"></div>
                    </div>
                    <div>
                      <div className="font-medium text-sm">Storage</div>
                      <div className="text-xs text-gray-500">Blockchain location</div>
                    </div>
                  </div>
                  <div className="text-sm font-medium text-gray-700">Solana DevNet</div>
                </div>

                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center">
                    <div className="w-6 h-6 bg-gray-100 rounded p-1 mr-3">
                      <div className="w-2 h-2 bg-gray-400 rounded-full mx-auto mt-1"></div>
                    </div>
                    <div>
                      <div className="font-medium text-sm">Last Checked</div>
                      <div className="text-xs text-gray-500">Status verification</div>
                    </div>
                  </div>
                  <div className="text-sm font-medium text-gray-700">
                    {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            )}

            {/* Bottom description */}
            {status.hasEmbedding && (
              <div className="text-center text-sm text-gray-600 mb-4">
                <p className="mb-2">Your facial recognition is active and working across all cameras in the network.</p>
                <p className="text-xs">This enables seamless authentication and enhanced security features.</p>
              </div>
            )}

            {/* Error Display */}
            {status.error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                <div className="text-sm text-red-700">{status.error}</div>
              </div>
            )}

            {/* Actions */}
            {status.hasEmbedding ? (
              <div className="space-y-3">
                {deleteError && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <div className="text-sm text-red-700">{deleteError}</div>
                  </div>
                )}

                <button
                  onClick={handleDeleteRecognitionToken}
                  disabled={isDeleting}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                >
                  {isDeleting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                  {isDeleting ? 'Deleting...' : 'Delete Token'}
                </button>
              </div>
            ) : showEnrollment ? (
              <PhoneSelfieEnrollment
                cameraId={cameraId || ""}
                onEnrollmentComplete={(result) => {
                  if (result.success) {
                    setShowEnrollment(false);
                    // Wait 3 seconds for blockchain to finalize before refreshing status
                    setTimeout(() => {
                      console.log('[RecognitionTokenModal] Refreshing status after enrollment...');
                      if (onStatusUpdate) {
                        onStatusUpdate();
                      }
                    }, 3000);
                  }
                }}
                onCancel={() => setShowEnrollment(false)}
              />
            ) : (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="w-6 h-6 bg-blue-500 rounded p-1">
                      <div className="w-full h-full bg-white rounded-sm"></div>
                    </div>
                    <h4 className="text-base font-semibold text-blue-800">Create Facial Embedding</h4>
                  </div>
                  <p className="text-sm text-blue-700">
                    Create a secure facial embedding to use CV apps on mmoment cameras. This only needs to be done once.
                  </p>
                </div>

                <button
                  onClick={() => setShowEnrollment(true)}
                  className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 font-medium flex items-center justify-center gap-2"
                >
                  <Smartphone className="h-5 w-5" />
                  <span>Create Facial Embedding</span>
                </button>
              </div>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}