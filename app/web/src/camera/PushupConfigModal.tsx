import { Dialog } from '@headlessui/react';
import { X, Users, DollarSign, Play, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { unifiedCameraService } from './unified-camera-service';

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
  const [betAmount, setBetAmount] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

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

      // Get selected competitor details - store for later use
      const competitors = recognizedUsers
        .filter(user => selectedCompetitors.has(user.wallet_address))
        .map(user => ({
          wallet_address: user.wallet_address,
          display_name: user.display_name
        }));

      // Store competitors and duration in sessionStorage for the start/stop controls
      sessionStorage.setItem('competition_competitors', JSON.stringify(competitors));
      sessionStorage.setItem('competition_duration', duration.toString());
      sessionStorage.setItem('competition_app', 'pushup');

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
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
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
                            {user.display_name}
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

            {/* Duration Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Competition Duration
              </label>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value={60}>1 minute</option>
                <option value={180}>3 minutes</option>
                <option value={300}>5 minutes</option>
                <option value={600}>10 minutes</option>
                <option value={0}>No limit</option>
              </select>
            </div>

            {/* Betting Section (Placeholder for future implementation) */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <DollarSign className="w-5 h-5 text-gray-400" />
                <h3 className="text-sm font-medium text-gray-500">Put Your Money Where Your Muscles Are</h3>
              </div>
              <p className="text-xs text-gray-500">
                Bet on yourself or challenge friends - coming soon!
              </p>
              <div className="mt-3">
                <input
                  type="number"
                  value={betAmount}
                  onChange={(e) => setBetAmount(Number(e.target.value))}
                  placeholder="0.00 SOL"
                  disabled
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white opacity-50 cursor-not-allowed"
                />
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100">
            <button
              onClick={handleLoadApp}
              disabled={isLoading || selectedCompetitors.size === 0}
              className="w-full bg-primary text-white py-3 px-4 rounded-lg hover:bg-primary-hover transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Loading App...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Load App ({selectedCompetitors.size} {selectedCompetitors.size === 1 ? 'competitor' : 'competitors'})</span>
                </>
              )}
            </button>
            <p className="text-xs text-gray-500 text-center mt-2">
              Position yourself in frame, then use Start/Stop controls to begin
            </p>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}
