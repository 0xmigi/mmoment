import { useState } from 'react';
import { Clock, LogOut, Settings, SkipForward, UserPlus } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useQueue, QueueEntry } from '../hooks/useQueue';
import { QueueConfigModal } from './QueueConfigModal';

interface QueuePanelProps {
  cameraId: string;
  isOwner?: boolean;
  displayOnly?: boolean;  // TV/desktop: no join form, just status
}

const DURATION_OPTIONS = [
  { label: '30 min', value: 1800 },
  { label: '1 hour', value: 3600 },
  { label: '1.5 hours', value: 5400 },
  { label: '2 hours', value: 7200 },
];

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatDurationLabel(seconds: number): string {
  if (seconds >= 3600) {
    const h = seconds / 3600;
    return h === 1 ? '1 hour' : `${h} hours`;
  }
  return `${seconds / 60} min`;
}

function truncateAddress(address: string): string {
  return `${address.slice(0, 4)}...${address.slice(-4)}`;
}

function entryName(entry: QueueEntry): string {
  return entry.displayName || truncateAddress(entry.walletAddress);
}

function entryTitle(entry: QueueEntry): string {
  return entry.title || `${entryName(entry)}'s session`;
}

function Avatar({ entry, size = 'sm' }: { entry: QueueEntry; size?: 'sm' | 'md' }) {
  const dim = size === 'md' ? 'w-8 h-8' : 'w-6 h-6';
  const textSize = size === 'md' ? 'text-xs' : 'text-[10px]';

  if (entry.profileImage) {
    return (
      <img
        src={entry.profileImage}
        alt=""
        className={`${dim} rounded-full object-cover flex-shrink-0`}
      />
    );
  }

  // Fallback: initial circle
  const initial = (entry.displayName?.[0] || entry.walletAddress[0]).toUpperCase();
  return (
    <div className={`${dim} rounded-full bg-[#E8E8E3] flex items-center justify-center flex-shrink-0`}>
      <span className={`${textSize} font-medium text-[#5C5C56]`}>{initial}</span>
    </div>
  );
}

export function QueuePanel({ cameraId, isOwner = false, displayOnly = false }: QueuePanelProps) {
  const { primaryWallet } = useDynamicContext();
  const walletAddress = primaryWallet?.address || null;

  const {
    queueState,
    isInQueue,
    isActive,
    remainingSeconds,
    joinQueue,
    leaveQueue,
    updateConfig,
    skipCurrent,
    loading,
    error,
  } = useQueue(cameraId);

  const [selectedDuration, setSelectedDuration] = useState(3600);
  const [sessionTitle, setSessionTitle] = useState('');
  const [joining, setJoining] = useState(false);
  const [showConfig, setShowConfig] = useState(false);

  if (loading) {
    return (
      <div className="py-4">
        <div className="text-sm text-[#8A8A82]">Loading queue...</div>
      </div>
    );
  }

  if (!queueState?.config.enabled && !isOwner) {
    return null;
  }

  const { active, queue, config } = queueState || {
    active: null,
    queue: [],
    config: { enabled: true, maxSlotDuration: 1800, minSlotDuration: 60 },
  };

  const availableDurations = DURATION_OPTIONS.filter(
    d => d.value >= config.minSlotDuration && d.value <= config.maxSlotDuration
  );
  if (availableDurations.length === 0) {
    availableDurations.push({ label: formatDurationLabel(config.maxSlotDuration), value: config.maxSlotDuration });
  }

  const handleJoin = async () => {
    setJoining(true);
    try {
      await joinQueue(selectedDuration, sessionTitle.trim() || undefined);
      setSessionTitle('');
    } catch {
      // error set by hook
    } finally {
      setJoining(false);
    }
  };

  const isEmpty = !active && queue.length === 0;

  // ── Display-only mode (TV / desktop) ──────────────────────────────────
  if (displayOnly) {
    return (
      <div className="py-2">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold tracking-[0.15em] uppercase text-[#8A8A82]">Queue</h3>
          {isOwner && (
            <button
              onClick={() => setShowConfig(true)}
              className="p-1 hover:bg-[#F3F3EF] rounded transition-colors"
              title="Queue settings"
            >
              <Settings className="w-3.5 h-3.5 text-[#8A8A82]" />
            </button>
          )}
        </div>

        {!config.enabled && isOwner && (
          <div className="text-xs text-[#8A8A82] mb-3">
            Queue is disabled.{' '}
            <button onClick={() => setShowConfig(true)} className="underline hover:text-[#5C5C56]">Enable it</button>
          </div>
        )}

        {config.enabled && (
          <>
            {active && (
              <div className="bg-[#F3F3EF] rounded-lg p-3 mb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <Avatar entry={active} size="md" />
                    <div>
                      <div className="text-sm font-medium text-[#1A1A18]">
                        {entryTitle(active)}
                      </div>
                      <div className="text-xs text-[#8A8A82]">
                        {entryName(active)} · {formatDurationLabel(active.requestedDuration)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-mono font-bold text-[#1A1A18] tabular-nums">
                      {remainingSeconds != null ? formatCountdown(remainingSeconds) : '--:--'}
                    </span>
                    {isOwner && (
                      <button
                        onClick={skipCurrent}
                        className="p-1 hover:bg-[#E8E8E3] rounded transition-colors"
                        title="Skip to next"
                      >
                        <SkipForward className="w-3.5 h-3.5 text-[#8A8A82]" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {queue.length > 0 && (
              <div className="space-y-1 mb-2">
                {queue.map((entry, idx) => (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between py-1.5 px-2"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs text-[#8A8A82] w-4 text-right tabular-nums">{idx + 1}</span>
                      <Avatar entry={entry} />
                      <span className="text-sm text-[#5C5C56]">
                        {entryName(entry)}
                      </span>
                    </div>
                    <span className="text-xs text-[#8A8A82]">
                      {formatDurationLabel(entry.requestedDuration)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {isEmpty && (
              <div className="text-sm text-[#8A8A82]">
                No one in line
              </div>
            )}
          </>
        )}

        {showConfig && queueState && (
          <QueueConfigModal
            isOpen={showConfig}
            onClose={() => setShowConfig(false)}
            config={queueState.config}
            onSave={updateConfig}
          />
        )}
      </div>
    );
  }

  // ── Interactive mode (mobile) ─────────────────────────────────────────
  return (
    <div>
      {/* Disabled notice for owner */}
      {!config.enabled && isOwner && (
        <div className="text-sm text-[#8A8A82] mb-4">
          Queue is disabled.{' '}
          <button onClick={() => setShowConfig(true)} className="underline hover:text-[#5C5C56]">Enable it</button>
        </div>
      )}

      {config.enabled && (
        <>
          {/* Active slot */}
          {active && (
            <div className="bg-[#F3F3EF] rounded-xl p-4 mb-4">
              <div className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-3">Now</div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Avatar entry={active} size="md" />
                  <div>
                    <div className="text-base font-medium text-[#1A1A18]">
                      {entryTitle(active)}
                    </div>
                    <div className="text-xs text-[#8A8A82]">
                      {entryName(active)} · {formatDurationLabel(active.requestedDuration)}
                    </div>
                  </div>
                </div>
                <span className="text-2xl font-mono font-bold text-[#1A1A18] tabular-nums">
                  {remainingSeconds != null ? formatCountdown(remainingSeconds) : '--:--'}
                </span>
              </div>
              {isActive && (
                <button
                  onClick={leaveQueue}
                  className="mt-3 w-full py-2 text-sm text-[#5C5C56] border border-[#E8E8E3] rounded-lg hover:bg-[#FAFAF8] transition-colors"
                >
                  End my turn early
                </button>
              )}
              {isOwner && !isActive && (
                <button
                  onClick={skipCurrent}
                  className="mt-3 flex items-center justify-center gap-1.5 w-full py-2 text-sm text-[#5C5C56] border border-[#E8E8E3] rounded-lg hover:bg-[#FAFAF8] transition-colors"
                >
                  <SkipForward className="w-3.5 h-3.5" />
                  Skip
                </button>
              )}
            </div>
          )}

          {/* Queue list */}
          {queue.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-2">Up next</div>
              <div className="space-y-1">
                {queue.map((entry, idx) => (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between py-2.5 px-3 rounded-lg hover:bg-[#FAFAF8] transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-[#8A8A82] w-5 text-right tabular-nums">{idx + 1}</span>
                      <Avatar entry={entry} />
                      <span className="text-sm font-medium text-[#1A1A18]">
                        {entryName(entry)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs text-[#8A8A82]">
                        {formatDurationLabel(entry.requestedDuration)}
                      </span>
                      {walletAddress && entry.walletAddress === walletAddress && (
                        <button
                          onClick={leaveQueue}
                          className="p-1 hover:bg-[#F3F3EF] rounded transition-colors"
                          title="Leave queue"
                        >
                          <LogOut className="w-3.5 h-3.5 text-[#8A8A82]" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {isEmpty && !isInQueue && !isActive && (
            <div className="text-center py-8 mb-4">
              <div className="text-sm text-[#8A8A82]">No one in line — you'd be first</div>
            </div>
          )}

          {/* Already in queue indicator */}
          {isInQueue && !isActive && (
            <div className="flex items-center justify-between py-3 px-4 bg-[#EEFBF0] rounded-xl mb-4">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-[#2F7D3E]" />
                <span className="text-sm font-medium text-[#2F7D3E]">You're in line</span>
              </div>
              <button
                onClick={leaveQueue}
                className="text-sm text-[#5C5C56] hover:text-[#1A1A18] transition-colors"
              >
                Leave
              </button>
            </div>
          )}

          {/* Join form */}
          {!isInQueue && !isActive && walletAddress && (
            <div>
              <input
                type="text"
                value={sessionTitle}
                onChange={(e) => setSessionTitle(e.target.value)}
                placeholder="Name your session (optional)"
                className="w-full px-3.5 py-2.5 text-sm rounded-xl border border-[#E8E8E3] bg-white text-[#1A1A18] placeholder-[#8A8A82] outline-none focus:border-[#8A8A82] transition-colors mb-3"
              />
              <div className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-2">How long?</div>
              <div className="flex gap-2 flex-wrap mb-3">
                {availableDurations.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setSelectedDuration(opt.value)}
                    className={`px-3.5 py-2 text-sm rounded-lg border transition-colors ${
                      selectedDuration === opt.value
                        ? 'bg-[#1A1A18] text-[#FAFAF8] border-[#1A1A18]'
                        : 'bg-white text-[#5C5C56] border-[#E8E8E3] hover:border-[#8A8A82]'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <button
                onClick={handleJoin}
                disabled={joining}
                className="w-full flex items-center justify-center gap-2 py-3 bg-[#1A1A18] text-[#FAFAF8] text-sm font-medium rounded-xl hover:bg-[#5C5C56] disabled:opacity-40 transition-colors"
              >
                <UserPlus className="w-4 h-4" />
                {joining ? 'Joining...' : 'Join Queue'}
              </button>
            </div>
          )}

          {/* Not signed in */}
          {!walletAddress && !isInQueue && (
            <div className="text-center py-6">
              <div className="text-sm text-[#8A8A82]">Sign in to join the queue</div>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="text-xs text-[#C73A3A] mt-3">{error}</div>
      )}

      {isOwner && (
        <button
          onClick={() => setShowConfig(true)}
          className="mt-4 flex items-center gap-1.5 text-xs text-[#8A8A82] hover:text-[#5C5C56] transition-colors"
        >
          <Settings className="w-3.5 h-3.5" />
          Queue settings
        </button>
      )}

      {showConfig && queueState && (
        <QueueConfigModal
          isOpen={showConfig}
          onClose={() => setShowConfig(false)}
          config={queueState.config}
          onSave={updateConfig}
        />
      )}
    </div>
  );
}
