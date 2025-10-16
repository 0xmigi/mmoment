import { Dialog } from '@headlessui/react';
import { X, ScanFace, CheckCircle, AlertCircle, Loader2, Trash2, ExternalLink } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { FacialEmbeddingStatus } from '../../hooks/useFacialEmbeddingStatus';
import { useProgram } from '../../anchor/setup';
import { useState, useMemo } from 'react';
import { PublicKey } from '@solana/web3.js';

interface RecognitionTokenModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: FacialEmbeddingStatus;
  onStatusUpdate?: () => void;
}

export function RecognitionTokenModal({ isOpen, onClose, status }: RecognitionTokenModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Calculate the PDA for this wallet's face data
  const faceDataPda = useMemo(() => {
    if (!primaryWallet?.address || !program) return null;

    try {
      const [pda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('recognition-token'),
          new PublicKey(primaryWallet.address).toBuffer()
        ],
        program.programId
      );
      return pda.toString();
    } catch (error) {
      console.error('Error calculating PDA:', error);
      return null;
    }
  }, [primaryWallet?.address, program]);

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

              <div className={`text-sm px-4 ${
                status.hasEmbedding ? 'text-green-700' : 'text-orange-700'
              }`}>
                {status.hasEmbedding
                  ? "Encrypted facial hash stored on-chain"
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

                {/* On-chain PDA Link */}
                {faceDataPda && (
                  <a
                    href={`https://explorer.solana.com/address/${faceDataPda}?cluster=devnet`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between py-2 hover:bg-gray-50 rounded-lg px-2 -mx-2 transition-colors"
                  >
                    <div className="flex items-center">
                      <div className="w-6 h-6 bg-indigo-100 rounded p-1 mr-3">
                        <ExternalLink className="w-full h-full text-indigo-600" />
                      </div>
                      <div>
                        <div className="font-medium text-sm">View On-Chain</div>
                        <div className="text-xs text-gray-500">Solana Explorer</div>
                      </div>
                    </div>
                    <ExternalLink className="w-4 h-4 text-gray-400" />
                  </a>
                )}
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
            ) : (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="w-6 h-6 bg-blue-500 rounded p-1">
                      <div className="w-full h-full bg-white rounded-sm"></div>
                    </div>
                    <h4 className="text-base font-semibold text-blue-800">How to Create Recognition Token</h4>
                  </div>
                  <p className="text-sm text-blue-700 mb-2">
                    To create a Recognition Token, you need to check in to a camera and visit the Apps drawer.
                  </p>
                  <p className="text-sm text-blue-600">
                    Recognition Tokens enable CV apps to recognize you across the mmoment camera network.
                  </p>
                </div>
              </div>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}