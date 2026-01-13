import { Dialog } from '@headlessui/react';
import { X, Users, Play, Loader2, AlertCircle, Trophy, Swords, Settings2 } from 'lucide-react';
import { useState, useEffect, useMemo } from 'react';
import { unifiedCameraService } from './unified-camera-service';
import { useCompetitionEscrow } from '../hooks/useCompetitionEscrow';
import { useCamera } from './CameraProvider';
import { useResolveDisplayNames } from '../hooks/useResolveDisplayNames';

type CompetitionMode = 'none' | 'bet' | 'prize';

interface Competitor {
  wallet_address: string;
  display_name: string;
  isCurrentUser?: boolean;
}

interface PushupConfigModalProps {
  cameraId: string;
  walletAddress?: string;
  isOpen: boolean;
  onClose: () => void;
  onStartCompetition: () => void;
}

export function PushupConfigModal({
  cameraId,
  walletAddress,
  isOpen,
  onClose,
  onStartCompetition
}: PushupConfigModalProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [recognizedUsers, setRecognizedUsers] = useState<Competitor[]>([]);
  const [selectedCompetitors, setSelectedCompetitors] = useState<Set<string>>(new Set());
  const [duration, setDuration] = useState<number>(300); // Default 5 minutes
  const [competitionMode, setCompetitionMode] = useState<CompetitionMode>('none');
  const [stakeAmount, setStakeAmount] = useState<number>(0.01);
  const [prizeAmount, setPrizeAmount] = useState<number>(0.01);
  const [targetPushups, setTargetPushups] = useState<number>(10);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  // Extract wallet addresses for display name resolution from backend
  const userWallets = useMemo(() =>
    recognizedUsers.map(u => u.wallet_address),
    [recognizedUsers]
  );

  // Resolve display names from backend when Jetson only provides wallet addresses
  const { getDisplayName } = useResolveDisplayNames(userWallets);

  // Competition escrow integration
  const { selectedCamera } = useCamera();
  const {
    createCompetition,
    loading: escrowLoading,
    error: escrowError,
    clearError: clearEscrowError
  } = useCompetitionEscrow();

  // Fetch currently recognized users (those with recognition tokens who are checked in)
  useEffect(() => {
    if (isOpen) {
      fetchRecognizedUsers();
    }
  }, [isOpen, cameraId]);

  // Auto-select current user
  useEffect(() => {
    if (walletAddress && recognizedUsers.length > 0) {
      setSelectedCompetitors(new Set([walletAddress]));
    }
  }, [walletAddress, recognizedUsers]);

  const fetchRecognizedUsers = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Get currently recognized faces from the camera
      const result = await unifiedCameraService.recognizeFaces(cameraId);

      if (result.success && result.data?.recognized_data) {
        const users: Competitor[] = Object.entries(result.data.recognized_data).map(([wallet, data]: [string, any]) => ({
          wallet_address: wallet,
          display_name: data.display_name || `${wallet.substring(0, 8)}...${wallet.substring(wallet.length - 4)}`,
          isCurrentUser: wallet === walletAddress
        }));

        setRecognizedUsers(users);
      } else {
        // If no recognized users, at least add the current user
        if (walletAddress) {
          setRecognizedUsers([{
            wallet_address: walletAddress,
            display_name: `${walletAddress.substring(0, 8)}...${walletAddress.substring(walletAddress.length - 4)}`,
            isCurrentUser: true
          }]);
        }
      }
    } catch (err) {
      console.error('Error fetching recognized users:', err);
      setError('Failed to fetch recognized users');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCompetitor = (wallet: string) => {
    const newSelected = new Set(selectedCompetitors);
    if (newSelected.has(wallet)) {
      newSelected.delete(wallet);
    } else {
      newSelected.add(wallet);
    }
    setSelectedCompetitors(newSelected);
  };

  const handleLoadApp = async () => {
    if (selectedCompetitors.size === 0) {
      setError('Please select at least one competitor');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      clearEscrowError();

      // Get selected competitor details with resolved display names - store for later use
      const competitors = recognizedUsers
        .filter(user => selectedCompetitors.has(user.wallet_address))
        .map(user => ({
          wallet_address: user.wallet_address,
          display_name: getDisplayName(user.wallet_address, user.display_name)
        }));

      // Get invitees (competitors excluding current user)
      const invitees = competitors
        .filter(c => c.wallet_address !== walletAddress)
        .map(c => c.wallet_address);

      // If competition mode is enabled, create on-chain escrow
      let escrowPda: string | null = null;
      let createdAt: number | null = null;

      if (competitionMode !== 'none' && selectedCamera?.devicePubkey) {
        console.log('[PushupConfigModal] Creating on-chain competition escrow...', { mode: competitionMode });

        if (competitionMode === 'bet') {
          // Bet mode: everyone stakes, winner takes all
          const result = await createCompetition({
            cameraDevicePubkey: selectedCamera.devicePubkey,
            invitees,
            stakePerPersonSol: stakeAmount,
            payoutRule: 'winnerTakesAll',
            initiatorParticipates: true,
            inviteTimeoutSecs: 60,
          });

          if (!result) {
            throw new Error(escrowError || 'Failed to create competition escrow');
          }

          escrowPda = result.escrowPda;
          createdAt = result.createdAt;
        } else if (competitionMode === 'prize') {
          // Prize mode: initiator deposits prize, threshold split for winners
          const result = await createCompetition({
            cameraDevicePubkey: selectedCamera.devicePubkey,
            invitees,
            stakePerPersonSol: prizeAmount, // Initiator deposits this as the prize
            payoutRule: 'thresholdSplit',
            thresholdMinReps: targetPushups,
            initiatorParticipates: true, // Initiator can also compete
            inviteTimeoutSecs: 60,
          });

          if (!result) {
            throw new Error(escrowError || 'Failed to create prize escrow');
          }

          escrowPda = result.escrowPda;
          createdAt = result.createdAt;
        }

        console.log('[PushupConfigModal] Competition escrow created:', escrowPda);
      }

      // Store competitors and duration in sessionStorage for the start/stop controls
      sessionStorage.setItem('competition_competitors', JSON.stringify(competitors));
      sessionStorage.setItem('competition_duration', duration.toString());
      sessionStorage.setItem('competition_app', 'pushup');
      sessionStorage.setItem('competition_mode', competitionMode);

      // Store escrow info if created
      if (escrowPda) {
        sessionStorage.setItem('competition_escrow_pda', escrowPda);
        sessionStorage.setItem('competition_escrow_created_at', createdAt!.toString());
        const amount = competitionMode === 'prize' ? prizeAmount : stakeAmount;
        sessionStorage.setItem('competition_stake_sol', amount.toString());
        if (competitionMode === 'prize') {
          sessionStorage.setItem('competition_target_pushups', targetPushups.toString());
        }
      } else {
        // Clear any existing escrow data for non-staked competitions
        sessionStorage.removeItem('competition_escrow_pda');
        sessionStorage.removeItem('competition_escrow_created_at');
        sessionStorage.removeItem('competition_stake_sol');
        sessionStorage.removeItem('competition_target_pushups');
      }

      // Load the pushup app
      const loadResult = await unifiedCameraService.loadApp(cameraId, 'pushup');
      if (!loadResult.success) {
        throw new Error(loadResult.error || 'Failed to load pushup app');
      }

      // Activate the pushup app
      const activateResult = await unifiedCameraService.activateApp(cameraId, 'pushup');
      if (!activateResult.success) {
        throw new Error(activateResult.error || 'Failed to activate pushup app');
      }

      // Success! Close modal and return to camera view
      // Competition will be started by the user with start/stop controls
      onStartCompetition();
    } catch (err) {
      console.error('Error loading app:', err);
      setError(err instanceof Error ? err.message : 'Failed to load app');
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-[100]">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2">
        <Dialog.Panel className="mx-auto w-full sm:max-w-md rounded-xl bg-white shadow-xl max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-100">
            <Dialog.Title className="text-lg font-semibold text-gray-900">
              Pushup Competition Setup
            </Dialog.Title>
            <div className="flex items-center space-x-1">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-1.5 rounded-lg transition-colors ${showSettings ? 'bg-gray-200' : 'hover:bg-gray-100'}`}
                title="Settings"
              >
                <Settings2 className="w-5 h-5 text-gray-500" />
              </button>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {/* Competitor Selection */}
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <Users className="w-5 h-5 text-gray-700" />
                <h3 className="text-sm font-medium text-gray-900">Select Competitors</h3>
              </div>

              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              ) : recognizedUsers.length === 0 ? (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                  <p className="text-sm text-orange-800">
                    No recognized users found. Make sure you and your competitors are checked in with recognition tokens.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {recognizedUsers.map((user) => (
                    <button
                      key={user.wallet_address}
                      onClick={() => toggleCompetitor(user.wallet_address)}
                      className={`w-full flex items-center justify-between p-3 rounded-lg border-2 transition-all ${
                        selectedCompetitors.has(user.wallet_address)
                          ? 'border-primary bg-primary-light'
                          : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                          selectedCompetitors.has(user.wallet_address)
                            ? 'border-primary bg-primary'
                            : 'border-gray-300'
                        }`}>
                          {selectedCompetitors.has(user.wallet_address) && (
                            <div className="w-2 h-2 bg-white rounded-full" />
                          )}
                        </div>
                        <div className="text-left">
                          <div className="text-sm font-medium text-gray-900">
                            {getDisplayName(user.wallet_address, user.display_name)}
                            {user.isCurrentUser && (
                              <span className="ml-2 text-xs text-primary">(You)</span>
                            )}
                          </div>
                          <div className="text-xs text-gray-500">
                            {user.wallet_address.substring(0, 8)}...{user.wallet_address.substring(user.wallet_address.length - 4)}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Duration Selection - Hidden by default */}
            {showSettings && (
              <div className="bg-gray-50 rounded-lg p-3">
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Competition Duration
                </label>
                <select
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary bg-white"
                >
                  <option value={60}>1 minute</option>
                  <option value={180}>3 minutes</option>
                  <option value={300}>5 minutes</option>
                  <option value={600}>10 minutes</option>
                  <option value={0}>No limit</option>
                </select>
              </div>
            )}

            {/* Competition Mode Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-3">
                Competition Mode
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => setCompetitionMode('none')}
                  className={`p-3 rounded-lg border-2 transition-all text-center ${
                    competitionMode === 'none'
                      ? 'border-primary bg-primary-light'
                      : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  <Play className={`w-5 h-5 mx-auto mb-1 ${competitionMode === 'none' ? 'text-primary' : 'text-gray-400'}`} />
                  <span className={`text-xs font-medium ${competitionMode === 'none' ? 'text-gray-900' : 'text-gray-500'}`}>
                    Just Play
                  </span>
                </button>
                <button
                  onClick={() => setCompetitionMode('prize')}
                  className={`p-3 rounded-lg border-2 transition-all text-center ${
                    competitionMode === 'prize'
                      ? 'border-yellow-500 bg-yellow-50'
                      : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  <Trophy className={`w-5 h-5 mx-auto mb-1 ${competitionMode === 'prize' ? 'text-yellow-500' : 'text-gray-400'}`} />
                  <span className={`text-xs font-medium ${competitionMode === 'prize' ? 'text-gray-900' : 'text-gray-500'}`}>
                    Prize
                  </span>
                </button>
                <button
                  onClick={() => setCompetitionMode('bet')}
                  className={`p-3 rounded-lg border-2 transition-all text-center ${
                    competitionMode === 'bet'
                      ? 'border-primary bg-primary-light'
                      : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                  }`}
                >
                  <Swords className={`w-5 h-5 mx-auto mb-1 ${competitionMode === 'bet' ? 'text-primary' : 'text-gray-400'}`} />
                  <span className={`text-xs font-medium ${competitionMode === 'bet' ? 'text-gray-900' : 'text-gray-500'}`}>
                    Bet
                  </span>
                </button>
              </div>
            </div>

            {/* Prize Mode Config */}
            {competitionMode === 'prize' && (
              <div className="border-2 border-yellow-500 bg-yellow-50 rounded-lg p-3 space-y-3">
                {!selectedCamera?.devicePubkey && (
                  <div className="flex items-center space-x-2 p-2 bg-orange-50 border border-orange-200 rounded-lg">
                    <AlertCircle className="w-4 h-4 text-orange-500 flex-shrink-0" />
                    <p className="text-xs text-orange-700">
                      Camera device key not found. Prizes require a registered Jetson device.
                    </p>
                  </div>
                )}
                <div className="flex items-center space-x-3">
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-gray-700 mb-1">Prize</label>
                    <div className="flex items-center space-x-2">
                      <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        value={prizeAmount}
                        onChange={(e) => setPrizeAmount(Number(e.target.value))}
                        disabled={!selectedCamera?.devicePubkey}
                        className="w-full px-2 py-1.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-50 text-sm"
                      />
                      <span className="text-sm font-medium text-gray-700">SOL</span>
                    </div>
                  </div>
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-gray-700 mb-1">Target Reps</label>
                    <input
                      type="number"
                      min="1"
                      value={targetPushups}
                      onChange={(e) => setTargetPushups(Number(e.target.value))}
                      disabled={!selectedCamera?.devicePubkey}
                      className="w-full px-2 py-1.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 disabled:opacity-50 text-sm"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Bet Mode Config */}
            {competitionMode === 'bet' && (
              <div className="border-2 border-primary bg-primary-light rounded-lg p-3 space-y-2">
                {!selectedCamera?.devicePubkey && (
                  <div className="flex items-center space-x-2 p-2 bg-orange-50 border border-orange-200 rounded-lg">
                    <AlertCircle className="w-4 h-4 text-orange-500 flex-shrink-0" />
                    <p className="text-xs text-orange-700">
                      Camera device key not found. Bets require a registered Jetson device.
                    </p>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-gray-700">Stake per person</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      value={stakeAmount}
                      onChange={(e) => setStakeAmount(Number(e.target.value))}
                      disabled={!selectedCamera?.devicePubkey}
                      className="w-20 px-2 py-1.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 text-sm text-right"
                    />
                    <span className="text-sm font-medium text-gray-700">SOL</span>
                  </div>
                </div>
                {stakeAmount > 0 && selectedCompetitors.size > 1 && (
                  <div className="text-xs text-gray-600 text-right">
                    Pot: <span className="font-semibold">{(stakeAmount * selectedCompetitors.size).toFixed(2)} SOL</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100">
            <button
              onClick={handleLoadApp}
              disabled={isLoading || escrowLoading || selectedCompetitors.size === 0}
              className="w-full bg-primary text-white py-3 px-4 rounded-lg hover:bg-primary-hover transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {isLoading || escrowLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>{escrowLoading ? (competitionMode === 'prize' ? 'Depositing Prize...' : 'Placing Bet...') : 'Loading App...'}</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>
                    {competitionMode === 'prize' && prizeAmount > 0
                      ? `Deposit ${prizeAmount} SOL Prize & Load App`
                      : competitionMode === 'bet' && stakeAmount > 0
                      ? `Bet ${stakeAmount} SOL & Load App`
                      : `Load App (${selectedCompetitors.size} ${selectedCompetitors.size === 1 ? 'competitor' : 'competitors'})`
                    }
                  </span>
                </>
              )}
            </button>
            <p className="text-xs text-gray-500 text-center mt-2">
              {competitionMode === 'prize' && prizeAmount > 0
                ? `Hit ${targetPushups} reps to win the prize`
                : competitionMode === 'bet' && stakeAmount > 0
                ? 'Winner takes all. Good luck!'
                : 'Get in frame, press Start to begin'
              }
            </p>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}
