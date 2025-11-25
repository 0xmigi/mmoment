/**
 * useEncryptedActivities Hook
 *
 * Fetches and decrypts timeline activities for the current user.
 * Activities are encrypted by the Jetson and stored in the backend buffer.
 */

import { useState, useCallback, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { CONFIG } from '../core/config';
import {
  EncryptedActivity,
  DecryptedActivityContent,
  decryptActivity,
  decryptActivities,
  hasAccessToActivity,
  getActivityTypeName,
  ACTIVITY_TYPE,
} from '../utils/activity-crypto';

// Session summary from backend
export interface SessionSummary {
  sessionId: string;
  cameraId: string;
  startTime: number;
  endTime: number;
  activityCount: number;
  activityTypes: number[];
}

export interface DecryptedActivityWithMeta {
  // Original encrypted activity metadata
  sessionId: string;
  cameraId: string;
  userPubkey: string;
  timestamp: number;
  activityType: number;
  createdAt?: string;

  // Decrypted content
  content: DecryptedActivityContent;
}

interface UseEncryptedActivitiesReturn {
  // State
  activities: DecryptedActivityWithMeta[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchActivities: (sessionId: string) => Promise<void>;
  fetchActivitiesForCamera: (cameraId: string) => Promise<void>;
  clearActivities: () => void;
  refreshActivities: () => Promise<void>;

  // Info
  hasAccess: (activity: EncryptedActivity) => boolean;
  userPubkey: string | null;
}

/**
 * Hook for fetching and decrypting timeline activities.
 */
export function useEncryptedActivities(): UseEncryptedActivitiesReturn {
  const { primaryWallet } = useDynamicContext();
  const [activities, setActivities] = useState<DecryptedActivityWithMeta[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track last fetch params for refresh
  const lastFetchRef = useRef<{ type: 'session' | 'camera'; id: string } | null>(null);

  // Get user's wallet address
  const userPubkey = primaryWallet?.address || null;

  /**
   * Fetch encrypted activities from backend by session ID
   */
  const fetchActivities = useCallback(async (sessionId: string) => {
    if (!userPubkey) {
      setError('Wallet not connected');
      return;
    }

    setIsLoading(true);
    setError(null);
    lastFetchRef.current = { type: 'session', id: sessionId };

    try {
      const response = await fetch(
        `${CONFIG.BACKEND_URL}/api/session/activities/${sessionId}`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch activities');
      }

      const encryptedActivities: EncryptedActivity[] = data.activities || [];

      console.log(`[useEncryptedActivities] Fetched ${encryptedActivities.length} activities for session ${sessionId.slice(0, 8)}...`);

      // Decrypt all activities the user has access to
      const decryptedResults = await decryptActivities(encryptedActivities, userPubkey);

      // Map to our output format
      const decryptedActivities: DecryptedActivityWithMeta[] = decryptedResults.map(
        ({ activity, content }) => ({
          sessionId: activity.sessionId,
          cameraId: activity.cameraId,
          userPubkey: activity.userPubkey,
          timestamp: activity.timestamp,
          activityType: activity.activityType,
          createdAt: activity.createdAt,
          content,
        })
      );

      // Sort by timestamp (newest first)
      decryptedActivities.sort((a, b) => b.timestamp - a.timestamp);

      setActivities(decryptedActivities);
      console.log(`[useEncryptedActivities] Decrypted ${decryptedActivities.length}/${encryptedActivities.length} activities`);

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch activities';
      console.error('[useEncryptedActivities] Error:', message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [userPubkey]);

  /**
   * Fetch activities for a camera (all sessions)
   * This queries the backend for activities by camera ID
   */
  const fetchActivitiesForCamera = useCallback(async (cameraId: string) => {
    if (!userPubkey) {
      setError('Wallet not connected');
      return;
    }

    setIsLoading(true);
    setError(null);
    lastFetchRef.current = { type: 'camera', id: cameraId };

    try {
      const response = await fetch(
        `${CONFIG.BACKEND_URL}/api/camera/activities/${cameraId}`
      );

      if (!response.ok) {
        // If endpoint doesn't exist yet, fall back to empty
        if (response.status === 404) {
          console.log(`[useEncryptedActivities] Camera activities endpoint not found, using empty list`);
          setActivities([]);
          return;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch activities');
      }

      const encryptedActivities: EncryptedActivity[] = data.activities || [];

      console.log(`[useEncryptedActivities] Fetched ${encryptedActivities.length} activities for camera ${cameraId.slice(0, 8)}...`);

      // Decrypt all activities the user has access to
      const decryptedResults = await decryptActivities(encryptedActivities, userPubkey);

      // Map to our output format
      const decryptedActivities: DecryptedActivityWithMeta[] = decryptedResults.map(
        ({ activity, content }) => ({
          sessionId: activity.sessionId,
          cameraId: activity.cameraId,
          userPubkey: activity.userPubkey,
          timestamp: activity.timestamp,
          activityType: activity.activityType,
          createdAt: activity.createdAt,
          content,
        })
      );

      // Sort by timestamp (newest first)
      decryptedActivities.sort((a, b) => b.timestamp - a.timestamp);

      setActivities(decryptedActivities);
      console.log(`[useEncryptedActivities] Decrypted ${decryptedActivities.length}/${encryptedActivities.length} activities`);

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch activities';
      console.error('[useEncryptedActivities] Error:', message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [userPubkey]);

  /**
   * Clear all activities
   */
  const clearActivities = useCallback(() => {
    setActivities([]);
    setError(null);
    lastFetchRef.current = null;
  }, []);

  /**
   * Refresh activities using last fetch params
   */
  const refreshActivities = useCallback(async () => {
    if (!lastFetchRef.current) return;

    if (lastFetchRef.current.type === 'session') {
      await fetchActivities(lastFetchRef.current.id);
    } else {
      await fetchActivitiesForCamera(lastFetchRef.current.id);
    }
  }, [fetchActivities, fetchActivitiesForCamera]);

  /**
   * Check if user has access to a specific activity
   */
  const hasAccess = useCallback((activity: EncryptedActivity): boolean => {
    if (!userPubkey) return false;
    return hasAccessToActivity(activity, userPubkey);
  }, [userPubkey]);

  return {
    activities,
    isLoading,
    error,
    fetchActivities,
    fetchActivitiesForCamera,
    clearActivities,
    refreshActivities,
    hasAccess,
    userPubkey,
  };
}

// Re-export utils for convenience
export { ACTIVITY_TYPE, getActivityTypeName } from '../utils/activity-crypto';
export type { EncryptedActivity, DecryptedActivityContent } from '../utils/activity-crypto';
