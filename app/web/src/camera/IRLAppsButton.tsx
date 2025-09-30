import { useState } from 'react';
import { Lock, Zap, X, Play, ActivitySquare, UserPlus, TrendingUp } from 'lucide-react';
import { useFacialEmbeddingStatus } from '../hooks/useFacialEmbeddingStatus';
import { PhoneSelfieEnrollment } from './PhoneSelfieEnrollment';

interface IRLAppsButtonProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: () => void;
}

export function IRLAppsButton({ cameraId, walletAddress, onEnrollmentComplete }: IRLAppsButtonProps) {
  const [showAppsModal, setShowAppsModal] = useState(false);
  const [showEnrollment, setShowEnrollment] = useState(false);
  const facialEmbeddingStatus = useFacialEmbeddingStatus();


  const handleIRLAppsClick = () => {
    // Always show the app store first - users can see what's available
    setShowAppsModal(true);
  };

  // Define available CV apps for this camera
  const availableApps = [
    {
      id: 'basketball_tracker',
      name: 'Basketball Score Tracker',
      description: 'Track shots and scores automatically',
      icon: <ActivitySquare className="w-5 h-5" />,
      enabled: true
    },
    {
      id: 'bouldering_scoreboard',
      name: 'Bouldering Scoreboard',
      description: 'Track climbs and log progress',
      icon: <TrendingUp className="w-5 h-5" />,
      enabled: true
    },
    {
      id: 'contact_memory',
      name: 'Contact Memory',
      description: 'Remember everyone you meet',
      icon: <UserPlus className="w-5 h-5" />,
      enabled: false
    }
  ];

  // Only show the button if we have a wallet address
  if (!walletAddress) {
    return null;
  }

  return (
    <>
      <button
        onClick={handleIRLAppsClick}
        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg shadow-lg transition-colors"
        title={facialEmbeddingStatus.hasEmbedding ? "Access Apps" : "Create Recognition Token for Apps"}
      >
        <Zap className="w-4 h-4" />
        <span className="text-sm">Apps</span>
        {!facialEmbeddingStatus.hasEmbedding && (
          <Lock className="w-3 h-3 opacity-70" />
        )}
      </button>

      {/* Apps Drawer - Full screen with proper mobile sizing */}
      {showAppsModal && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/60 z-[99998]"
            style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.6)', zIndex: 99998 }}
            onClick={() => setShowAppsModal(false)}
          />

          {/* Drawer - slides up from bottom, almost full screen */}
          <div className="fixed inset-x-0 bottom-0 top-3 bg-white rounded-t-2xl shadow-2xl z-[99999]" style={{ zIndex: 99999 }}>
            {/* Header - Mobile sized like CameraModal */}
            <div className="flex items-center justify-between p-3 border-b border-gray-100">
              <h3 className="text-base font-medium">Apps</h3>
              <button
                onClick={() => setShowAppsModal(false)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>

            {/* Content - with padding for button */}
            <div className="p-4 overflow-y-auto" style={{ paddingBottom: '100px' }}>
              <div className="space-y-3 mb-4">
                {availableApps.map((app) => {
                  const isAccessible = facialEmbeddingStatus.hasEmbedding && app.enabled;
                  const needsToken = !facialEmbeddingStatus.hasEmbedding;

                  return (
                    <div key={app.id} className="flex items-center mb-4 bg-gray-50 rounded-lg p-3">
                      <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center overflow-hidden ${
                        isAccessible ? 'bg-green-50 text-green-600' : needsToken ? 'bg-orange-50 text-orange-600' : 'bg-gray-100 text-gray-400'
                      }`}>
                        {app.icon}
                      </div>
                      <div className="ml-3 flex-1">
                        <div className="text-sm font-medium text-gray-900">{app.name}</div>
                        <div className="text-xs text-gray-500">{app.description}</div>
                      </div>
                      <button
                        onClick={() => {
                          if (isAccessible) {
                            // TODO: Launch app
                            console.log(`Launching app: ${app.id}`);
                          }
                        }}
                        disabled={!isAccessible}
                        className={`p-2 rounded-full transition-colors ${
                          isAccessible
                            ? 'bg-blue-600 hover:bg-blue-700 text-white'
                            : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                        }`}
                        title={isAccessible ? 'Start app' : needsToken ? 'Locked - need token' : 'Not available'}
                      >
                        {isAccessible ? (
                          <Play className="w-4 h-4" fill="currentColor" />
                        ) : needsToken ? (
                          <Lock className="w-4 h-4" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  );
                })}
              </div>

              {facialEmbeddingStatus.hasEmbedding && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <h3 className="text-sm font-medium text-green-800 mb-1">Recognition Token Active</h3>
                  <p className="text-xs text-green-700">All apps unlocked across the network</p>
                </div>
              )}

              {!facialEmbeddingStatus.hasEmbedding && (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                  <h3 className="text-sm font-medium text-orange-800 mb-1">Apps Locked</h3>
                  <p className="text-xs text-orange-700">Create a recognition token to unlock</p>
                </div>
              )}
            </div>

            {/* Bottom Button - FIXED POSITION */}
            {!facialEmbeddingStatus.hasEmbedding && (
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-white border-t border-gray-100">
                <button
                  onClick={() => {
                    setShowAppsModal(false);
                    setShowEnrollment(true);
                  }}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                  Create Recognition Token
                </button>
              </div>
            )}
          </div>
        </>
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
    </>
  );
}