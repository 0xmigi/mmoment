import { useState, useEffect, useCallback, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { CONFIG } from '../core/config';
import { timelineService } from '../timeline/timeline-service';

export interface QueueEntry {
  id: string;
  cameraId: string;
  walletAddress: string;
  displayName: string | null;
  profileImage: string | null;
  requestedDuration: number;
  position: number;
  status: 'waiting' | 'active' | 'completed' | 'left';
  joinedAt: number;
  startedAt: number | null;
  expiresAt: number | null;
  completedAt: number | null;
  remainingSeconds?: number;
}

export interface QueueConfig {
  enabled: boolean;
  maxSlotDuration: number;
  minSlotDuration: number;
}

export interface QueueState {
  config: QueueConfig;
  active: QueueEntry | null;
  queue: QueueEntry[];
  totalWaiting: number;
}

export function useQueue(cameraId: string | null) {
  const [queueState, setQueueState] = useState<QueueState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { primaryWallet } = useDynamicContext();
  const countdownRef = useRef<NodeJS.Timeout | null>(null);

  const walletAddress = primaryWallet?.address || null;

  // Fetch queue state from backend
  const fetchQueueState = useCallback(async () => {
    if (!cameraId) return;
    try {
      const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/queue`);
      const data = await res.json();
      if (data.success) {
        setQueueState({
          config: data.config,
          active: data.active || null,
          queue: data.queue || [],
          totalWaiting: data.totalWaiting || 0,
        });
        setError(null);
      }
    } catch (err) {
      console.error('[useQueue] Failed to fetch queue state:', err);
      setError('Failed to load queue');
    } finally {
      setLoading(false);
    }
  }, [cameraId]);

  // Initial fetch
  useEffect(() => {
    fetchQueueState();
  }, [fetchQueueState]);

  // Listen for real-time queue updates via socket
  useEffect(() => {
    if (!cameraId) return;

    const unsubscribe = timelineService.onSocketEvent('queueUpdate', (state: QueueState) => {
      setQueueState(state);
      setError(null);
    });

    return unsubscribe;
  }, [cameraId]);

  // Local countdown timer for active slot
  useEffect(() => {
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }

    if (!queueState?.active?.expiresAt) return;

    countdownRef.current = setInterval(() => {
      setQueueState(prev => {
        if (!prev?.active?.expiresAt) return prev;
        const remaining = Math.max(0, Math.round((prev.active.expiresAt! - Date.now()) / 1000));
        if (remaining === prev.active.remainingSeconds) return prev;
        return {
          ...prev,
          active: { ...prev.active, remainingSeconds: remaining },
        };
      });
    }, 1000);

    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, [queueState?.active?.id, queueState?.active?.expiresAt]);

  // Derived state
  const myEntry = queueState?.active?.walletAddress === walletAddress
    ? queueState.active
    : queueState?.queue.find(e => e.walletAddress === walletAddress) || null;
  const isInQueue = !!myEntry;
  const isActive = queueState?.active?.walletAddress === walletAddress;
  const remainingSeconds = queueState?.active?.remainingSeconds ?? null;

  // Actions
  const joinQueue = useCallback(async (duration: number, displayName?: string) => {
    if (!cameraId || !walletAddress) return;
    try {
      const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/queue/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet_address: walletAddress, duration, display_name: displayName }),
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error || 'Failed to join queue');
        throw new Error(data.error);
      }
      // Socket broadcast will update state
    } catch (err: any) {
      if (!err.message?.includes('Already in queue')) {
        console.error('[useQueue] Failed to join:', err);
      }
      throw err;
    }
  }, [cameraId, walletAddress]);

  const leaveQueue = useCallback(async () => {
    if (!cameraId || !walletAddress) return;
    try {
      const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/queue/leave?wallet_address=${walletAddress}`, {
        method: 'DELETE',
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error || 'Failed to leave queue');
      }
    } catch (err) {
      console.error('[useQueue] Failed to leave:', err);
    }
  }, [cameraId, walletAddress]);

  const updateConfig = useCallback(async (config: Partial<{ enabled: boolean; maxSlotDuration: number; minSlotDuration: number }>) => {
    if (!cameraId) return;
    try {
      const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/queue/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: config.enabled,
          max_slot_duration: config.maxSlotDuration,
          min_slot_duration: config.minSlotDuration,
        }),
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error || 'Failed to update config');
      }
    } catch (err) {
      console.error('[useQueue] Failed to update config:', err);
    }
  }, [cameraId]);

  const skipCurrent = useCallback(async () => {
    if (!cameraId) return;
    try {
      const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/queue/skip`, {
        method: 'POST',
      });
      const data = await res.json();
      if (!data.success) {
        setError(data.error || 'Failed to skip');
      }
    } catch (err) {
      console.error('[useQueue] Failed to skip:', err);
    }
  }, [cameraId]);

  return {
    queueState,
    isInQueue,
    isActive,
    myEntry,
    remainingSeconds,
    joinQueue,
    leaveQueue,
    updateConfig,
    skipCurrent,
    loading,
    error,
    refetch: fetchQueueState,
  };
}
