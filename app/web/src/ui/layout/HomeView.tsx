import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Dialog } from '@headlessui/react';
import { X, Smartphone, Loader } from 'lucide-react';
import { useUserSessions, formatSessionDuration } from '../../hooks/useUserSessions';
import { useCamera } from '../../camera/CameraProvider';
import { CONFIG } from '../../core/config';

export function HomeView() {
  const { sessions, isLoading, fetchSessions } = useUserSessions();
  const { isCheckedIn } = useCamera();
  const navigate = useNavigate();
  const [showScanModal, setShowScanModal] = useState(false);
  const [devPda, setDevPda] = useState('');

  const isDev = !CONFIG.isProduction;

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const totalIrlTime = useMemo(() => {
    if (sessions.length === 0) return null;

    const totalMs = sessions.reduce((sum, s) => {
      const duration = s.endTime - s.startTime;
      return duration > 0 ? sum + duration : sum;
    }, 0);

    return {
      formatted: formatSessionDuration(0, totalMs),
      sessionCount: sessions.length,
    };
  }, [sessions]);

  const handleDevConnect = (pda: string) => {
    if (pda.trim()) {
      setShowScanModal(false);
      navigate(`/app/camera/${pda.trim()}`);
    }
  };

  return (
    <div className="relative flex flex-col min-h-[calc(100vh-4rem)]">
      {/* Centered content */}
      <div className="flex-1 flex items-center justify-center px-4">
        <div className="w-full max-w-sm space-y-8 text-center">
          {/* IRL Time Display */}
          <div className="space-y-2">
            <p className="text-sm text-gray-400 uppercase tracking-widest">
              Time in the moment
            </p>

            {isLoading ? (
              <Loader className="w-6 h-6 mx-auto animate-spin text-gray-300" />
            ) : totalIrlTime ? (
              <>
                <p className="text-5xl font-bold text-black tabular-nums">
                  {totalIrlTime.formatted}
                </p>
                <p className="text-sm text-gray-400">
                  across {totalIrlTime.sessionCount} session{totalIrlTime.sessionCount !== 1 ? 's' : ''}
                </p>
              </>
            ) : (
              <>
                <p className="text-5xl font-bold text-gray-200 tabular-nums">
                  0m
                </p>
                <p className="text-sm text-gray-400">
                  no sessions yet
                </p>
              </>
            )}
          </div>

          {/* Live session indicator */}
          {isCheckedIn && (
            <div className="flex items-center justify-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
              </span>
              <span className="text-sm text-green-600 font-medium">Live session</span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom-pinned button — matches TapLandingPage bottom sheet style */}
      <div className="px-4 pb-8">
        <button
          onClick={() => setShowScanModal(true)}
          className="w-full py-3 rounded-xl bg-white text-black font-semibold text-sm border border-gray-200 hover:bg-gray-50 transition-colors"
        >
          Check in to a camera
        </button>
      </div>

      {/* Check-in Modal */}
      <Dialog
        open={showScanModal}
        onClose={() => setShowScanModal(false)}
        className="relative z-[100]"
      >
        <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

        <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
          <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between p-3 border-b border-gray-100">
              <Dialog.Title className="text-base font-medium">
                Check in
              </Dialog.Title>
              <button
                onClick={() => setShowScanModal(false)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4">
              {/* Centered phone icon + tap prompt */}
              <div className="flex flex-col items-center py-6 space-y-4">
                <div className="w-20 h-20 rounded-full border-2 border-gray-200 flex items-center justify-center">
                  <Smartphone className="w-9 h-9 text-gray-500" />
                </div>
                <div className="text-center space-y-1">
                  <p className="text-base font-medium text-gray-900">Tap a Moment tag</p>
                  <p className="text-sm text-gray-500">Hold your phone near an NFC tag to check in</p>
                </div>
              </div>

              {/* Dev shortcut */}
              {isDev && (
                <div className="border-t border-gray-100 pt-4 space-y-3">
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                    <h3 className="text-sm font-medium text-yellow-800 mb-1">Development Mode</h3>
                    <p className="text-xs text-yellow-700 mb-3">
                      Skip NFC tap and connect directly:
                    </p>
                    <div className="space-y-2">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={devPda}
                          onChange={(e) => setDevPda(e.target.value)}
                          placeholder="Camera PDA"
                          className="flex-1 px-2 py-1.5 text-xs border border-yellow-300 rounded bg-white focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <button
                          onClick={() => handleDevConnect(devPda)}
                          className="text-xs bg-primary-light hover:bg-primary-muted text-primary px-3 py-1.5 rounded transition-colors"
                        >
                          Go
                        </button>
                      </div>
                      <button
                        onClick={() => handleDevConnect(CONFIG.JETSON_CAMERA_PDA)}
                        className="text-xs bg-primary-light hover:bg-primary-muted text-primary px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                      >
                        Connect to Jetson (ArQx...QqKv)
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Dialog.Panel>
        </div>
      </Dialog>
    </div>
  );
}
