/**
 * SessionCard Component
 *
 * Collapsible card showing a session summary with full timeline.
 * When expanded, shows ALL events at the camera during the user's session.
 * Used in the Activities view to show historical sessions.
 */

import { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, Clock, MapPin } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { SessionSummary, SessionParticipant, formatSessionDuration } from '../hooks/useUserSessions';
import { Timeline } from './Timeline';
import { TimelineEvent } from './timeline-types';
import { CONFIG } from '../core/config';

interface SessionCardProps {
  session: SessionSummary;
  defaultExpanded?: boolean;
}

// Backend response types
interface SessionTimelineEventFromAPI {
  id: string;
  type: string;
  userAddress: string;
  timestamp: number;
  cameraId: string;
  transactionId?: string;  // Solana transaction ID for on-chain writes
  encrypted?: {
    sessionId: string;
    activityType: number;
    encryptedContent: string;
    nonce: string;
    accessGrants: string;
  };
  // CV activity metadata (for pushups, squats, etc.)
  cvActivity?: {
    app_name: string;
    duration_seconds?: number;
    participant_count: number;
    results: Array<{
      wallet_address: string;
      display_name?: string;
      rank: number;
      stats: { reps?: number; [key: string]: unknown };
    }>;
    user_stats: { reps?: number; [key: string]: unknown };
  };
}

/**
 * ParticipantStack - Shows overlapping participant profile pictures
 * Excludes the current user (implied they were there)
 */
interface ParticipantStackProps {
  participants: SessionParticipant[];
  currentUserAddress: string;
  maxShow?: number;
}

function ParticipantStack({ participants, currentUserAddress, maxShow = 3 }: ParticipantStackProps) {
  // Filter out current user
  const otherParticipants = participants.filter(p => p.address !== currentUserAddress);

  if (otherParticipants.length === 0) {
    // Solo session - return empty space
    return <div className="w-8 h-8" />;
  }

  const displayParticipants = otherParticipants.slice(0, maxShow);
  const overflowCount = otherParticipants.length - maxShow;

  return (
    <div className="flex items-center -space-x-2">
      {displayParticipants.map((participant, index) => (
        <div
          key={participant.address}
          className="w-8 h-8 rounded-full bg-gray-200 border-2 border-white flex items-center justify-center overflow-hidden"
          style={{ zIndex: maxShow - index }}
          title={participant.displayName || participant.username || participant.address.slice(0, 8)}
        >
          {participant.pfpUrl ? (
            <img
              src={participant.pfpUrl}
              alt={participant.displayName || participant.username || 'User'}
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-xs font-medium text-gray-600">
              {(participant.displayName || participant.username || participant.address).charAt(0).toUpperCase()}
            </span>
          )}
        </div>
      ))}
      {overflowCount > 0 && (
        <div
          className="w-8 h-8 rounded-full bg-gray-300 border-2 border-white flex items-center justify-center"
          style={{ zIndex: 0 }}
          title={`+${overflowCount} more`}
        >
          <span className="text-xs font-medium text-gray-700">+{overflowCount}</span>
        </div>
      )}
    </div>
  );
}

/**
 * Format timestamp to readable date/time
 */
function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  const isYesterday = new Date(now.getTime() - 86400000).toDateString() === date.toDateString();

  const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (isToday) {
    return `Today at ${timeStr}`;
  } else if (isYesterday) {
    return `Yesterday at ${timeStr}`;
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ` at ${timeStr}`;
  }
}

/**
 * Convert API event to Timeline event format
 */
function apiEventToTimelineEvent(event: SessionTimelineEventFromAPI): TimelineEvent {
  return {
    id: event.id,
    type: event.type as TimelineEvent['type'],
    user: {
      address: event.userAddress,
    },
    timestamp: event.timestamp,
    cameraId: event.cameraId,
    // Include transaction ID if present (for on-chain actions like check-outs)
    ...(event.transactionId && { transactionId: event.transactionId }),
    // Include CV activity metadata if present
    ...(event.cvActivity && { cvActivity: event.cvActivity }),
  };
}

/**
 * Main SessionCard component
 */
export function SessionCard({ session, defaultExpanded = false }: SessionCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { primaryWallet } = useDynamicContext();

  // Fetch timeline events when expanded
  const fetchTimeline = useCallback(async () => {
    if (!primaryWallet?.address) {
      setError('Wallet not connected');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${CONFIG.BACKEND_URL}/api/session/${session.sessionId}/timeline?walletAddress=${primaryWallet.address}`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch timeline');
      }

      // Convert API events to Timeline format
      const events: TimelineEvent[] = (data.events || []).map(apiEventToTimelineEvent);

      console.log(`[SessionCard] Loaded ${events.length} timeline events for session ${session.sessionId.slice(0, 8)}...`);

      setTimelineEvents(events);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch timeline';
      console.error('[SessionCard] Error:', message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [primaryWallet?.address, session.sessionId]);

  // Fetch timeline when expanded
  useEffect(() => {
    if (isExpanded && timelineEvents.length === 0 && !isLoading && !error) {
      fetchTimeline();
    }
  }, [isExpanded, timelineEvents.length, isLoading, error, fetchTimeline]);

  // Format camera ID for display
  const shortCameraId = session.cameraId.slice(0, 8) + '...' + session.cameraId.slice(-4);

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      {/* Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {/* Participant profile pictures */}
          <ParticipantStack
            participants={session.participants || []}
            currentUserAddress={primaryWallet?.address || ''}
            maxShow={3}
          />

          {/* Session info */}
          <div className="text-left">
            <p className="text-sm font-medium text-gray-900">
              {formatTimestamp(session.startTime)}
            </p>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatSessionDuration(session.startTime, session.endTime)}
              </span>
              <span className="flex items-center gap-1">
                <MapPin className="w-3 h-3" />
                {shortCameraId}
              </span>
            </div>
          </div>
        </div>

        {/* Expand icon */}
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Expanded content - shows full Timeline */}
      {isExpanded && (
        <div className="border-t border-gray-100 px-4 py-3">
          {isLoading ? (
            <div className="py-4 text-center text-sm text-gray-500">
              <div className="animate-spin w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full mx-auto mb-2" />
              Loading timeline...
            </div>
          ) : error ? (
            <div className="py-4 text-center text-sm text-red-500">
              {error}
              <button
                onClick={fetchTimeline}
                className="block mx-auto mt-2 text-primary hover:text-primary-hover"
              >
                Retry
              </button>
            </div>
          ) : timelineEvents.length === 0 ? (
            <div className="py-4 text-center text-sm text-gray-500">
              No activity during this session
            </div>
          ) : (
            <Timeline
              initialEvents={timelineEvents}
              variant="full"
              showProfileStack={false}
              showAbsoluteTime={true}
            />
          )}
        </div>
      )}
    </div>
  );
}
