import { Dialog } from '@headlessui/react';
import { X, ScanFace, CheckCircle, AlertCircle, Loader2, Trash2, ExternalLink } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { FacialEmbeddingStatus } from '../../hooks/useFacialEmbeddingStatus';
import { useProgram } from '../../anchor/setup';
import { useState, useMemo } from 'react';
import { PublicKey, Transaction } from '@solana/web3.js';
import { useConnection } from '@solana/wallet-adapter-react';

interface RecognitionTokenModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: FacialEmbeddingStatus;
  onStatusUpdate?: () => void;
}

export function RecognitionTokenModal({ isOpen, onClose, status, onStatusUpdate }: RecognitionTokenModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { program } = useProgram();
  const { connection } = useConnection();
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

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

  const confirmDelete = () => {
    setShowConfirmDelete(true);
  };

  const handleDeleteRecognitionToken = async () => {
    if (!primaryWallet?.address || !program || !connection) {
      setDeleteError('Wallet or program not available');
      return;
    }

    setShowConfirmDelete(false);
    setIsDeleting(true);
    setDeleteError(null);

    try {
      const userPublicKey = new PublicKey(primaryWallet.address);

      // Calculate the PDA for the recognition token
      const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('recognition-token'),
          userPublicKey.toBuffer()
        ],
        program.programId
      );

      console.log('[DeleteToken] Deleting recognition token...');
      console.log('[DeleteToken] User:', userPublicKey.toString());
      console.log('[DeleteToken] Recognition Token PDA:', recognitionTokenPda.toString());

      // Build the transaction instruction - DO NOT use .rpc()
      const instruction = await program.methods
        .deleteRecognitionToken()
        .accounts({
          user: userPublicKey,
          recognitionToken: recognitionTokenPda,
        })
        .instruction();

      // Create transaction with recent blockhash
      const { blockhash } = await connection.getLatestBlockhash();
      const transaction = new Transaction({
        feePayer: userPublicKey,
        blockhash,
        lastValidBlockHeight: (await connection.getLatestBlockhash()).lastValidBlockHeight,
      }).add(instruction);

      console.log('[DeleteToken] Transaction built, signing with Dynamic wallet...');

      // Sign transaction using Dynamic's getSigner() - EXACT pattern from working enrollment code
      const signer = await (primaryWallet as any).getSigner();
      const signedTx = await signer.signTransaction(transaction);

      console.log('[DeleteToken] Transaction signed successfully');

      // Submit signed transaction to Solana
      console.log('[DeleteToken] Submitting transaction to Solana...');
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      console.log('[DeleteToken] Transaction sent! Signature:', signature);

      // Wait for confirmation
      console.log('[DeleteToken] Waiting for confirmation...');
      await connection.confirmTransaction(signature, 'confirmed');
      console.log('[DeleteToken] ✅ Recognition token deleted successfully!');

      // Close the modal and refresh status
      onClose();

      // Trigger a status update if callback is provided
      if (onStatusUpdate) {
        onStatusUpdate();
      }

    } catch (error) {
      console.error('[DeleteToken] Delete error:', error);
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

                {showConfirmDelete ? (
                  <>
                    <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-3">
                      <div className="text-sm text-orange-900 font-medium mb-1">
                        Are you sure?
                      </div>
                      <div className="text-sm text-orange-700">
                        This will permanently delete your Recognition Token. You will reclaim ~0.009 SOL rent.
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setShowConfirmDelete(false)}
                        disabled={isDeleting}
                        className="flex-1 px-4 py-3 bg-gray-100 text-gray-700 font-medium rounded-lg hover:bg-gray-200 disabled:opacity-50 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleDeleteRecognitionToken}
                        disabled={isDeleting}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                      >
                        {isDeleting ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Trash2 className="w-4 h-4" />
                        )}
                        {isDeleting ? 'Deleting...' : 'Confirm Delete'}
                      </button>
                    </div>
                  </>
                ) : (
                  <button
                    onClick={confirmDelete}
                    disabled={isDeleting}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                    Delete Token
                  </button>
                )}
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