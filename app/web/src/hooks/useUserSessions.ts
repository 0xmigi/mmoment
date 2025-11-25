/**
 * useUserSessions Hook
 *
 * Fetches session summaries for the current user from the backend.
 * Sessions are grouped by check-in/check-out periods.
 */

import { useState, useCallback } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { CONFIG } from '../core/config';
import { ACTIVITY_TYPE } from '../utils/activity-crypto';

// Session summary from backend
export interface SessionSummary {
  sessionId: string;
  cameraId: string;
  startTime: number;
  endTime: number;
  activityCount: number;
  activityTypes: number[];
}

interface UseUserSessionsReturn {
  // State
  sessions: SessionSummary[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchSessions: () => Promise<void>;
  refreshSessions: () => Promise<void>;

  // Info
  userPubkey: string | null;
}

/**
 * Get human-readable summary of activity types in a session
 */
export function getSessionActivitySummary(activityTypes: number[]): string {
  const parts: string[] = [];
  const uniqueTypes = new Set(activityTypes);

  if (uniqueTypes.has(ACTIVITY_TYPE.PHOTO)) parts.push('photos');
  if (uniqueTypes.has(ACTIVITY_TYPE.VIDEO)) parts.push('videos');
  if (uniqueTypes.has(ACTIVITY_TYPE.STREAM_START)) parts.push('streams');

  return parts.join(', ') || 'activities';
}

/**
 * Format duration between two timestamps
 */
export function formatSessionDuration(startTime: number, endTime: number): string {
  const durationMs = endTime - startTime;
  const minutes = Math.floor(durationMs / 60000);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }
  return `${minutes}m`;
}

/**
 * Hook for fetching user session summaries.
 */
export function useUserSessions(): UseUserSessionsReturn {
  const { primaryWallet } = useDynamicContext();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const userPubkey = primaryWallet?.address || null;

  /**
   * Fetch all sessions for the current user
   */
  const fetchSessions = useCallback(async () => {
    if (!userPubkey) {
      setError('Wallet not connected');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${CONFIG.BACKEND_URL}/api/user/${userPubkey}/sessions?limit=50`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch sessions');
      }

      const fetchedSessions: SessionSummary[] = data.sessions || [];

      console.log(`[useUserSessions] Fetched ${fetchedSessions.length} sessions for user ${userPubkey.slice(0, 8)}...`);

      setSessions(fetchedSessions);

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch sessions';
      console.error('[useUserSessions] Error:', message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [userPubkey]);

  /**
   * Refresh sessions (alias for fetchSessions)
   */
  const refreshSessions = useCallback(async () => {
    await fetchSessions();
  }, [fetchSessions]);

  return {
    sessions,
    isLoading,
    error,
    fetchSessions,
    refreshSessions,
    userPubkey,
  };
}
