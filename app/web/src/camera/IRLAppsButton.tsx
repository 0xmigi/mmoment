import { useState } from 'react';
import { Lock, Zap, X, Play, ActivitySquare, UserPlus, TrendingUp, Dumbbell } from 'lucide-react';
import { useFacialEmbeddingStatus } from '../hooks/useFacialEmbeddingStatus';
import { PhoneSelfieEnrollment } from './PhoneSelfieEnrollment';
import { PushupConfigModal } from './PushupConfigModal';

interface IRLAppsButtonProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: () => void;
  devMode?: boolean;  // When true, bypass wallet and embedding requirements
}

// Dev mode fake wallet for testing
const DEV_WALLET_ADDRESS = 'DevWallet1111111111111111111111111111111111';

export function IRLAppsButton({ cameraId, walletAddress, onEnrollmentComplete, devMode = false }: IRLAppsButtonProps) {
  const [showAppsModal, setShowAppsModal] = useState(false);
  const [showEnrollment, setShowEnrollment] = useState(false);
  const [showPushupConfig, setShowPushupConfig] = useState(false);
  const facialEmbeddingStatus = useFacialEmbeddingStatus();

  // In dev mode, use fake wallet and assume embedding exists
  const effectiveWalletAddress = devMode ? DEV_WALLET_ADDRESS : walletAddress;
  const effectiveHasEmbedding = devMode ? true : facialEmbeddingStatus.hasEmbedding;


  const handleIRLAppsClick = () => {
    // Always show the app store first - users can see what's available
    setShowAppsModal(true);
  };

  // Define available CV apps for this camera
  const availableApps = [
    {
      id: 'pushup_competition',
      name: 'Pushup Competition',
      description: 'Compete in pushup challenges with friends',
      icon: <Dumbbell className="w-5 h-5" />,
      enabled: true,
      comingSoon: false
    },
    {
      id: 'basketball_tracker',
      name: 'Basketball Score Tracker',
      description: 'Track shots and scores automatically',
      icon: <ActivitySquare className="w-5 h-5" />,
      enabled: false,
      comingSoon: true
    },
    {
      id: 'bouldering_scoreboard',
      name: 'Bouldering Scoreboard',
      description: 'Track climbs and log progress',
      icon: <TrendingUp className="w-5 h-5" />,
      enabled: false,
      comingSoon: true
    },
    {
      id: 'contact_memory',
      name: 'Contact Memory',
      description: 'Remember everyone you meet',
      icon: <UserPlus className="w-5 h-5" />,
      enabled: false,
      comingSoon: true
    }
  ];

  // Only show the button if we have a wallet address (or in dev mode)
  if (!effectiveWalletAddress) {
    return null;
  }

  return (
    <>
      <button
        onClick={handleIRLAppsClick}
        className="flex items-center space-x-2 bg-primary hover:bg-primary-hover text-white px-1.5 py-0.5 rounded shadow-lg transition-colors text-xs"
        title={effectiveHasEmbedding ? "Access Apps" : "Create Recognition Token for Apps"}
      >
        <Zap className="w-3.5 h-3.5" />
        <span className="font-medium">Apps</span>
        {!effectiveHasEmbedding && (
          <Lock className="w-2.5 h-2.5 opacity-70" />
        )}
      </button>

      {/* Apps Full Page */}
      {showAppsModal && (
        <div className="fixed inset-0 bg-white z-50">
          <div className="min-h-screen bg-white">
            <div className="max-w-2xl mx-auto pt-8 px-4">
              <div className="bg-white mb-6 flex items-center justify-between">
                <h1 className="text-xl font-semibold">Apps</h1>
                <button
                  onClick={() => setShowAppsModal(false)}
                  className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>
            <div className="space-y-3 mb-4">
              {availableApps.map((app) => {
                const isAccessible = effectiveHasEmbedding && app.enabled && !app.comingSoon;
                const needsToken = !effectiveHasEmbedding && app.enabled && !app.comingSoon;
                const isComingSoon = app.comingSoon;

                return (
                  <div
                    key={app.id}
                    onClick={() => {
                      if (isAccessible) {
                        if (app.id === 'pushup_competition') {
                          setShowAppsModal(false);
                          setShowPushupConfig(true);
                        } else {
                          console.log(`Launching app: ${app.id}`);
                        }
                      }
                    }}
                    className={`flex items-center mb-4 rounded-lg p-3 ${
                      isComingSoon ? 'bg-gray-100 opacity-60' :
                      isAccessible ? 'bg-gray-50 hover:bg-gray-100 cursor-pointer' :
                      'bg-gray-50'
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center overflow-hidden ${
                      isComingSoon ? 'bg-gray-200 text-gray-400' :
                      isAccessible ? 'bg-green-50 text-green-600' :
                      needsToken ? 'bg-orange-50 text-orange-600' : 'bg-gray-100 text-gray-400'
                    }`}>
                      {app.icon}
                    </div>
                    <div className="ml-3 flex-1">
                      <div className={`text-sm font-medium ${isComingSoon ? 'text-gray-500' : 'text-gray-900'}`}>
                        {app.name}
                      </div>
                      <div className={`text-xs ${isComingSoon ? 'text-gray-400' : 'text-gray-500'}`}>
                        {app.description}
                      </div>
                    </div>
                    {isComingSoon ? (
                      <span className="text-[10px] font-medium bg-gray-300 text-gray-600 px-2 py-1 rounded whitespace-nowrap">
                        COMING SOON
                      </span>
                    ) : needsToken ? (
                      <div className="p-2 rounded-full bg-gray-200 text-gray-400">
                        <Lock className="w-4 h-4" />
                      </div>
                    ) : (
                      <div className="p-2 rounded-full bg-primary text-white">
                        <Play className="w-4 h-4" fill="currentColor" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {effectiveHasEmbedding && (
              <div className={`border rounded-lg p-3 ${devMode ? 'bg-yellow-50 border-yellow-200' : 'bg-green-50 border-green-200'}`}>
                <h3 className={`text-sm font-medium mb-1 ${devMode ? 'text-yellow-800' : 'text-green-800'}`}>
                  {devMode ? 'Dev Mode Active' : 'Recognition Token Active'}
                </h3>
                <p className={`text-xs ${devMode ? 'text-yellow-700' : 'text-green-700'}`}>
                  {devMode ? 'Apps unlocked for development testing' : 'All apps unlocked across the network'}
                </p>
              </div>
            )}

            {!effectiveHasEmbedding && (
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                <h3 className="text-sm font-medium text-orange-800 mb-1">Apps Locked</h3>
                <p className="text-xs text-orange-700">Create a recognition token to unlock</p>
              </div>
            )}

              {!effectiveHasEmbedding && (
                <div className="mt-6">
                  <button
                    onClick={() => {
                      setShowAppsModal(false);
                      setShowEnrollment(true);
                    }}
                    className="w-full bg-primary text-white py-3 px-4 rounded-lg hover:bg-primary-hover transition-colors text-sm font-medium"
                  >
                    Create Recognition Token
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Enrollment Drawer - Full screen camera drawer */}
      {showEnrollment && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/60 z-[99998]"
            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.6)', zIndex: 99998 }}
            onClick={() => setShowEnrollment(false)}
          />

          {/* Drawer - slides up from bottom, almost full screen */}
          <div className="fixed inset-x-0 bottom-0 top-3 bg-white rounded-t-2xl shadow-2xl z-[99999]" style={{ zIndex: 99999 }}>
            {/* Header - Mobile sized like CameraModal */}
            <div className="flex items-center justify-between p-3 border-b border-gray-100">
              <h3 className="text-base font-medium">Create Recognition Token</h3>
              <button
                onClick={() => setShowEnrollment(false)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>

            {/* Camera Content - Fill remaining drawer space with padding and rounded corners */}
            <div className="p-2 flex-1">
              <div className="h-full rounded-lg overflow-hidden">
                <PhoneSelfieEnrollment
                  cameraId={cameraId}
                  onEnrollmentComplete={(result) => {
                    if (result.success) {
                      setShowEnrollment(false);
                      if (onEnrollmentComplete) {
                        onEnrollmentComplete();
                      }
                      // Automatically show apps after successful enrollment
                      setTimeout(() => setShowAppsModal(true), 500);
                    }
                  }}
                  onCancel={undefined} // No duplicate X button
                />
              </div>
            </div>
          </div>
        </>
      )}

      {/* Pushup Competition Config Modal */}
      {showPushupConfig && effectiveWalletAddress && (
        <PushupConfigModal
          cameraId={cameraId}
          walletAddress={effectiveWalletAddress}
          isOpen={showPushupConfig}
          onClose={() => setShowPushupConfig(false)}
          onStartCompetition={() => {
            setShowPushupConfig(false);
            // Return to camera view - app will be loaded
          }}
        />
      )}
    </>
  );
}