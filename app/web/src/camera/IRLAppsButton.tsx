import { useState } from 'react';
import { Lock, Zap, X, ActivitySquare, UserPlus, TrendingUp, Dumbbell, ListOrdered } from 'lucide-react';
import { useFacialEmbeddingStatus } from '../hooks/useFacialEmbeddingStatus';
import { useQueue, QueueEntry } from '../hooks/useQueue';
import { PhoneSelfieEnrollment } from './PhoneSelfieEnrollment';
import { PushupConfigModal } from './PushupConfigModal';
import { QueuePanel } from './QueuePanel';

interface IRLAppsButtonProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: () => void;
  devMode?: boolean;
}

const DEV_WALLET_ADDRESS = 'DevWallet1111111111111111111111111111111111';

// Avatar stack: show first 3 profile pics, then "+N" if more
function AvatarStack({ entries }: { entries: QueueEntry[] }) {
  if (entries.length === 0) return null;

  const shown = entries.slice(0, 3);
  const extra = entries.length - shown.length;

  return (
    <div className="flex items-center">
      {shown.map((entry, i) => (
        <div
          key={entry.id}
          className="w-6 h-6 rounded-full border-2 border-white flex-shrink-0 overflow-hidden bg-[#E8E8E3]"
          style={{ marginLeft: i === 0 ? 0 : -8, zIndex: shown.length - i }}
        >
          {entry.profileImage ? (
            <img src={entry.profileImage} alt="" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <span className="text-[9px] font-medium text-[#5C5C56]">
                {(entry.displayName?.[0] || entry.walletAddress[0]).toUpperCase()}
              </span>
            </div>
          )}
        </div>
      ))}
      {extra > 0 && (
        <div
          className="w-6 h-6 rounded-full border-2 border-white bg-[#F3F3EF] flex items-center justify-center flex-shrink-0"
          style={{ marginLeft: -8, zIndex: 0 }}
        >
          <span className="text-[9px] font-medium text-[#5C5C56]">+{extra}</span>
        </div>
      )}
    </div>
  );
}

export function IRLAppsButton({ cameraId, walletAddress, onEnrollmentComplete, devMode = false }: IRLAppsButtonProps) {
  const [showAppsModal, setShowAppsModal] = useState(false);
  const [showEnrollment, setShowEnrollment] = useState(false);
  const [showPushupConfig, setShowPushupConfig] = useState(false);
  const [showQueueModal, setShowQueueModal] = useState(false);
  const facialEmbeddingStatus = useFacialEmbeddingStatus();
  const { queueState } = useQueue(cameraId);

  const effectiveWalletAddress = devMode ? DEV_WALLET_ADDRESS : walletAddress;
  const effectiveHasEmbedding = devMode ? true : facialEmbeddingStatus.hasEmbedding;

  // Collect all queue entries for avatar stack
  const allQueueEntries: QueueEntry[] = [
    ...(queueState?.active ? [queueState.active] : []),
    ...(queueState?.queue || []),
  ];

  const availableApps = [
    {
      id: 'queue',
      name: 'Queue',
      description: 'Sign up for your turn',
      icon: <ListOrdered className="w-5 h-5" />,
      enabled: true,
      comingSoon: false,
      noTokenRequired: true,
    },
    {
      id: 'pushup_competition',
      name: 'Pushup Competition',
      description: 'Compete with friends',
      icon: <Dumbbell className="w-5 h-5" />,
      enabled: true,
      comingSoon: false,
      noTokenRequired: false,
    },
    {
      id: 'basketball_tracker',
      name: 'Basketball Score Tracker',
      description: 'Track shots and scores',
      icon: <ActivitySquare className="w-5 h-5" />,
      enabled: false,
      comingSoon: true,
      noTokenRequired: false,
    },
    {
      id: 'bouldering_scoreboard',
      name: 'Bouldering Scoreboard',
      description: 'Track climbs and progress',
      icon: <TrendingUp className="w-5 h-5" />,
      enabled: false,
      comingSoon: true,
      noTokenRequired: false,
    },
    {
      id: 'contact_memory',
      name: 'Contact Memory',
      description: 'Remember everyone you meet',
      icon: <UserPlus className="w-5 h-5" />,
      enabled: false,
      comingSoon: true,
      noTokenRequired: false,
    }
  ];

  if (!effectiveWalletAddress) {
    return null;
  }

  const handleAppClick = (appId: string) => {
    if (appId === 'queue') {
      setShowAppsModal(false);
      setShowQueueModal(true);
    } else if (appId === 'pushup_competition') {
      setShowAppsModal(false);
      setShowPushupConfig(true);
    }
  };

  return (
    <>
      <button
        onClick={() => setShowAppsModal(true)}
        className="flex items-center space-x-2 bg-gray-800/70 hover:bg-gray-800/90 text-white px-1.5 py-0.5 rounded shadow-lg backdrop-blur-sm transition-colors text-xs"
        title="Apps"
      >
        <Zap className="w-3.5 h-3.5" />
        <span className="font-medium">Apps</span>
      </button>

      {/* Apps Page */}
      {showAppsModal && (
        <div className="fixed inset-0 bg-[#FAFAF8] z-50">
          <div className="max-w-md mx-auto pt-8 px-5">
            <div className="flex items-center justify-between mb-6">
              <h1 className="text-xl font-semibold text-[#1A1A18]">Apps</h1>
              <button
                onClick={() => setShowAppsModal(false)}
                className="p-2 rounded-lg hover:bg-[#F3F3EF] transition-colors"
              >
                <X className="w-5 h-5 text-[#8A8A82]" />
              </button>
            </div>

            <div className="space-y-1.5">
              {availableApps.map((app) => {
                const isAccessible = (app.noTokenRequired || effectiveHasEmbedding) && app.enabled && !app.comingSoon;
                const needsToken = !app.noTokenRequired && !effectiveHasEmbedding && app.enabled && !app.comingSoon;
                const isLocked = needsToken || app.comingSoon;

                return (
                  <div
                    key={app.id}
                    onClick={() => isAccessible && handleAppClick(app.id)}
                    className={`flex items-center rounded-xl p-3.5 transition-colors ${
                      isAccessible
                        ? 'hover:bg-[#F3F3EF] cursor-pointer'
                        : ''
                    }`}
                  >
                    {/* Icon */}
                    <div className={`w-10 h-10 rounded-xl flex-shrink-0 flex items-center justify-center relative ${
                      isLocked ? 'bg-[#F3F3EF] text-[#E8E8E3]' : 'bg-[#F3F3EF] text-[#5C5C56]'
                    }`}>
                      {app.icon}
                      {isLocked && (
                        <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-[#E8E8E3] flex items-center justify-center">
                          <Lock className="w-2.5 h-2.5 text-[#8A8A82]" />
                        </div>
                      )}
                    </div>

                    {/* Name + description */}
                    <div className="ml-3 flex-1 min-w-0">
                      <div className={`text-sm font-medium ${isLocked ? 'text-[#8A8A82]' : 'text-[#1A1A18]'}`}>
                        {app.name}
                      </div>
                      <div className={`text-xs ${isLocked ? 'text-[#E8E8E3]' : 'text-[#8A8A82]'}`}>
                        {app.description}
                      </div>
                    </div>

                    {/* Right side: avatar stack for queue, status for others */}
                    <div className="flex-shrink-0 ml-3">
                      {app.id === 'queue' && allQueueEntries.length > 0 ? (
                        <AvatarStack entries={allQueueEntries} />
                      ) : app.comingSoon ? (
                        <span className="text-[10px] font-medium tracking-wide uppercase text-[#E8E8E3]">
                          Soon
                        </span>
                      ) : needsToken ? (
                        <span className="text-[11px] text-[#E8E8E3]">Locked</span>
                      ) : app.id === 'queue' ? (
                        <span className="text-xs text-[#8A8A82]">Empty</span>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Recognition token section */}
            {!effectiveHasEmbedding && (
              <div className="mt-8">
                <div className="border border-[#E8E8E3] rounded-xl p-4 bg-white">
                  <div className="text-sm font-medium text-[#1A1A18] mb-1">Recognition Token</div>
                  <p className="text-xs text-[#8A8A82] mb-3">
                    Create a token to unlock all apps and enable hands-free check-in
                  </p>
                  <button
                    onClick={() => {
                      setShowAppsModal(false);
                      setShowEnrollment(true);
                    }}
                    className="w-full py-2.5 bg-[#1A1A18] text-[#FAFAF8] text-sm font-medium rounded-xl hover:bg-[#5C5C56] transition-colors"
                  >
                    Create Recognition Token
                  </button>
                </div>
              </div>
            )}

            {effectiveHasEmbedding && !devMode && (
              <div className="mt-8 flex items-center gap-2 px-1">
                <div className="w-2 h-2 rounded-full bg-[#2F7D3E]" />
                <span className="text-xs text-[#8A8A82]">Recognition token active</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enrollment Drawer */}
      {showEnrollment && (
        <>
          <div
            className="fixed inset-0 bg-black/50 z-[99998]"
            onClick={() => setShowEnrollment(false)}
          />
          <div className="fixed inset-x-0 bottom-0 top-3 bg-white rounded-t-2xl shadow-2xl z-[99999]">
            <div className="flex items-center justify-between p-3 border-b border-[#E8E8E3]">
              <h3 className="text-base font-medium text-[#1A1A18]">Create Recognition Token</h3>
              <button
                onClick={() => setShowEnrollment(false)}
                className="p-1 rounded-lg hover:bg-[#F3F3EF] transition-colors"
              >
                <X className="w-4 h-4 text-[#8A8A82]" />
              </button>
            </div>
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
                      setTimeout(() => setShowAppsModal(true), 500);
                    }
                  }}
                  onCancel={undefined}
                />
              </div>
            </div>
          </div>
        </>
      )}

      {/* Queue Modal */}
      {showQueueModal && (
        <div className="fixed inset-0 bg-[#FAFAF8] z-50">
          <div className="max-w-md mx-auto pt-8 px-5">
            <div className="flex items-center justify-between mb-6">
              <h1 className="text-xl font-semibold text-[#1A1A18]">Queue</h1>
              <button
                onClick={() => setShowQueueModal(false)}
                className="p-2 rounded-lg hover:bg-[#F3F3EF] transition-colors"
              >
                <X className="w-5 h-5 text-[#8A8A82]" />
              </button>
            </div>
            <QueuePanel cameraId={cameraId} />
          </div>
        </div>
      )}

      {/* Pushup Competition Config */}
      {showPushupConfig && effectiveWalletAddress && (
        <PushupConfigModal
          cameraId={cameraId}
          walletAddress={effectiveWalletAddress}
          isOpen={showPushupConfig}
          onClose={() => setShowPushupConfig(false)}
          onStartCompetition={() => setShowPushupConfig(false)}
        />
      )}
    </>
  );
}
