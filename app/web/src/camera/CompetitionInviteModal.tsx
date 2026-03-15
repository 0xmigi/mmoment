/**
 * Competition Invite Modal
 *
 * Shows pending competition invites and allows users to accept (join) or decline.
 * When accepted, the user's stake is deposited into the escrow.
 */

import { Dialog } from '@headlessui/react';
import { X, Users, DollarSign, Check, XCircle, Loader2, Clock } from 'lucide-react';
import { useState, useEffect, useCallback } from 'react';
import { useCompetitionEscrow, type ActiveCompetition } from '../hooks/useCompetitionEscrow';

interface CompetitionInviteModalProps {
  cameraDevicePubkey: string;
  walletAddress: string;
  isOpen: boolean;
  onClose: () => void;
  onJoined?: (escrowPda: string) => void;
}

export function CompetitionInviteModal({
  cameraDevicePubkey,
  walletAddress,
  isOpen,
  onClose,
  onJoined,
}: CompetitionInviteModalProps) {
  const [pendingInvites, setPendingInvites] = useState<ActiveCompetition[]>([]);
  const [isLoadingInvites, setIsLoadingInvites] = useState(false);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  const {
    joinCompetition,
    declineCompetition,
    fetchPendingInvites,
    loading,
    error,
    clearError,
  } = useCompetitionEscrow();

  // Fetch pending invites when modal opens
  const loadPendingInvites = useCallback(async () => {
    if (!cameraDevicePubkey || !walletAddress) return;

    setIsLoadingInvites(true);
    try {
      const invites = await fetchPendingInvites(walletAddress, cameraDevicePubkey);
      setPendingInvites(invites);
    } catch (err) {
      console.error('[CompetitionInviteModal] Error fetching invites:', err);
    } finally {
      setIsLoadingInvites(false);
    }
  }, [cameraDevicePubkey, walletAddress, fetchPendingInvites]);

  useEffect(() => {
    if (isOpen) {
      loadPendingInvites();
      clearError();
    }
  }, [isOpen, loadPendingInvites, clearError]);

  // Calculate remaining time for invite
  const getRemainingTime = (competition: ActiveCompetition): number => {
    const expiresAt = competition.createdAt + competition.inviteTimeoutSecs;
    const now = Math.floor(Date.now() / 1000);
    return Math.max(0, expiresAt - now);
  };

  const formatTime = (seconds: number): string => {
    if (seconds <= 0) return 'Expired';
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  const handleAccept = async (competition: ActiveCompetition) => {
    setActionInProgress(competition.escrowPda);
    clearError();

    try {
      const success = await joinCompetition(competition.escrowPda);
      if (success) {
        // Store escrow info in session storage for use during competition
        sessionStorage.setItem('competition_escrow_pda', competition.escrowPda);
        sessionStorage.setItem('competition_stake_sol', competition.stakePerPerson.toString());

        // Remove from pending invites
        setPendingInvites(prev => prev.filter(i => i.escrowPda !== competition.escrowPda));

        // Notify parent
        onJoined?.(competition.escrowPda);

        // Close modal if no more invites
        if (pendingInvites.length <= 1) {
          onClose();
        }
      }
    } finally {
      setActionInProgress(null);
    }
  };

  const handleDecline = async (competition: ActiveCompetition) => {
    setActionInProgress(competition.escrowPda);
    clearError();

    try {
      const success = await declineCompetition(competition.escrowPda);
      if (success) {
        // Remove from pending invites
        setPendingInvites(prev => prev.filter(i => i.escrowPda !== competition.escrowPda));

        // Close modal if no more invites
        if (pendingInvites.length <= 1) {
          onClose();
        }
      }
    } finally {
      setActionInProgress(null);
    }
  };

  // Truncate wallet address for display
  const truncateAddress = (address: string): string => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-[100]">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="mx-auto w-full max-w-md rounded-xl bg-white shadow-xl max-h-[80vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-100">
            <Dialog.Title className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
              <Users className="w-5 h-5 text-primary" />
              <span>Competition Invites</span>
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {error && (
              <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {isLoadingInvites ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="w-8 h-8 animate-spin text-primary mb-2" />
                <p className="text-sm text-gray-500">Loading invites...</p>
              </div>
            ) : pendingInvites.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Users className="w-12 h-12 text-gray-300 mb-3" />
                <p className="text-sm text-gray-500">No pending competition invites</p>
              </div>
            ) : (
              <div className="space-y-4">
                {pendingInvites.map((competition) => {
                  const remainingTime = getRemainingTime(competition);
                  const isExpired = remainingTime <= 0;
                  const isProcessing = actionInProgress === competition.escrowPda;

                  return (
                    <div
                      key={competition.escrowPda}
                      className={`border rounded-lg p-4 transition-colors ${
                        isExpired ? 'border-gray-200 bg-gray-50 opacity-60' : 'border-primary bg-primary-light'
                      }`}
                    >
                      {/* Invite Header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          <DollarSign className="w-5 h-5 text-primary" />
                          <span className="font-semibold text-gray-900">
                            {competition.stakePerPerson.toFixed(2)} SOL Stake
                          </span>
                        </div>
                        <div className={`flex items-center space-x-1 text-xs ${
                          isExpired ? 'text-red-500' : remainingTime < 30 ? 'text-orange-500' : 'text-gray-500'
                        }`}>
                          <Clock className="w-3 h-3" />
                          <span>{formatTime(remainingTime)}</span>
                        </div>
                      </div>

                      {/* Initiator Info */}
                      <div className="mb-3">
                        <p className="text-xs text-gray-500">Invited by</p>
                        <p className="text-sm font-medium text-gray-700">
                          {truncateAddress(competition.initiator)}
                        </p>
                      </div>

                      {/* Participants */}
                      <div className="mb-3">
                        <p className="text-xs text-gray-500 mb-1">
                          Participants ({competition.participants.length} joined)
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {competition.participants.map((p) => (
                            <span
                              key={p}
                              className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-green-100 text-green-700"
                            >
                              {truncateAddress(p)}
                            </span>
                          ))}
                          {competition.pendingInvites.map((p) => (
                            <span
                              key={p}
                              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${
                                p.toLowerCase() === walletAddress.toLowerCase()
                                  ? 'bg-primary text-white'
                                  : 'bg-gray-100 text-gray-500'
                              }`}
                            >
                              {p.toLowerCase() === walletAddress.toLowerCase() ? 'You' : truncateAddress(p)}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Betting Pool Info */}
                      <div className="mb-4 p-2 bg-white rounded-lg">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Total Pot</span>
                          <span className="font-semibold text-gray-900">
                            {competition.totalPool.toFixed(2)} SOL
                          </span>
                        </div>
                        <div className="flex justify-between text-xs mt-1">
                          <span className="text-gray-500">Your Bet</span>
                          <span className="font-semibold text-primary">
                            {competition.stakePerPerson.toFixed(2)} SOL
                          </span>
                        </div>
                      </div>

                      {/* Action Buttons */}
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleDecline(competition)}
                          disabled={isExpired || isProcessing || loading}
                          className="flex-1 flex items-center justify-center space-x-1 py-2 px-3 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                          {isProcessing ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <>
                              <XCircle className="w-4 h-4" />
                              <span className="text-sm">Decline</span>
                            </>
                          )}
                        </button>
                        <button
                          onClick={() => handleAccept(competition)}
                          disabled={isExpired || isProcessing || loading}
                          className="flex-1 flex items-center justify-center space-x-1 py-2 px-3 bg-primary text-white rounded-lg hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                          {isProcessing ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <>
                              <Check className="w-4 h-4" />
                              <span className="text-sm">Accept & Stake</span>
                            </>
                          )}
                        </button>
                      </div>

                      {isExpired && (
                        <p className="mt-2 text-xs text-center text-red-500">
                          This invite has expired
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100">
            <p className="text-xs text-gray-500 text-center">
              Accepting places your bet. Winner takes all!
            </p>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}

export default CompetitionInviteModal;
