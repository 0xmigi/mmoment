/**
 * Unified Camera Polling Service
 *
 * Single source of truth for all camera status polling across the entire app.
 * Prevents request storms by consolidating all polling into one interval per camera.
 */

import { unifiedCameraService } from './unified-camera-service';

export interface CameraStatusData {
  owner: string;
  isLive: boolean;
  isStreaming: boolean;
  status: 'ok' | 'error' | 'offline';
  lastSeen?: number;
  hardwareState?: {
    temperature?: number;
    cpuUsage?: number;
    memoryUsage?: number;
    diskUsage?: number;
  };
}

type StatusCallback = (status: CameraStatusData) => void;

class UnifiedCameraPollingService {
  private static instance: UnifiedCameraPollingService;

  // Map of cameraId -> callbacks
  private subscribers = new Map<string, Set<StatusCallback>>();

  // Map of cameraId -> current status
  private statusCache = new Map<string, CameraStatusData>();

  // Map of cameraId -> polling interval
  private pollIntervals = new Map<string, NodeJS.Timeout>();

  // Polling frequency
  private readonly POLL_INTERVAL = 10000; // 10 seconds - balance between responsiveness and load

  // Pending requests to prevent duplicate concurrent fetches
  private pendingRequests = new Map<string, Promise<CameraStatusData>>();

  // Stagger initial subscriptions to prevent thundering herd
  private initialFetchDelay = 0;

  private constructor() {
    console.log('[UnifiedCameraPolling] Service initialized');
  }

  static getInstance(): UnifiedCameraPollingService {
    if (!UnifiedCameraPollingService.instance) {
      UnifiedCameraPollingService.instance = new UnifiedCameraPollingService();
    }
    return UnifiedCameraPollingService.instance;
  }

  /**
   * Subscribe to status updates for a specific camera.
   * Returns unsubscribe function.
   */
  subscribe(cameraId: string, callback: StatusCallback): () => void {
    console.log(`[UnifiedCameraPolling] New subscription for camera ${cameraId}`);

    // Get or create subscriber set for this camera
    if (!this.subscribers.has(cameraId)) {
      this.subscribers.set(cameraId, new Set());
    }

    const callbacks = this.subscribers.get(cameraId)!;
    callbacks.add(callback);

    // If this is the first subscriber, start polling
    if (callbacks.size === 1) {
      this.startPolling(cameraId);
    } else {
      // Immediately call with cached status if available
      const cached = this.statusCache.get(cameraId);
      if (cached) {
        callback(cached);
      }
    }

    // Return unsubscribe function
    return () => {
      console.log(`[UnifiedCameraPolling] Unsubscribe from camera ${cameraId}`);
      callbacks.delete(callback);

      // If no more subscribers, stop polling
      if (callbacks.size === 0) {
        this.stopPolling(cameraId);
      }
    };
  }

  /**
   * Get current cached status without triggering a fetch
   */
  getCachedStatus(cameraId: string): CameraStatusData | undefined {
    return this.statusCache.get(cameraId);
  }

  /**
   * Force an immediate status check for a camera
   */
  async forceCheck(cameraId: string): Promise<CameraStatusData> {
    console.log(`[UnifiedCameraPolling] Force check for camera ${cameraId}`);
    return this.fetchStatus(cameraId);
  }

  private startPolling(cameraId: string): void {
    console.log(`[UnifiedCameraPolling] Starting polling for camera ${cameraId} (interval: ${this.POLL_INTERVAL}ms)`);

    // Stagger initial fetches to prevent thundering herd on page load
    const delay = this.initialFetchDelay;
    this.initialFetchDelay += 500; // Each new subscription waits 500ms longer

    // Reset stagger after 5 seconds
    setTimeout(() => {
      this.initialFetchDelay = Math.max(0, this.initialFetchDelay - 500);
    }, 5000);

    // Do delayed initial fetch to prevent overwhelming the server
    setTimeout(() => {
      this.fetchStatus(cameraId).catch(err =>
        console.error(`[UnifiedCameraPolling] Initial fetch failed for ${cameraId}:`, err)
      );
    }, delay);

    // Start interval
    const interval = setInterval(() => {
      this.fetchStatus(cameraId).catch(err =>
        console.error(`[UnifiedCameraPolling] Polling fetch failed for ${cameraId}:`, err)
      );
    }, this.POLL_INTERVAL);

    this.pollIntervals.set(cameraId, interval);
  }

  private stopPolling(cameraId: string): void {
    console.log(`[UnifiedCameraPolling] Stopping polling for camera ${cameraId}`);

    const interval = this.pollIntervals.get(cameraId);
    if (interval) {
      clearInterval(interval);
      this.pollIntervals.delete(cameraId);
    }

    // Clean up cache and subscribers
    this.subscribers.delete(cameraId);
    this.statusCache.delete(cameraId);
  }

  private async fetchStatus(cameraId: string): Promise<CameraStatusData> {
    // Check if there's already a pending request for this camera
    const pending = this.pendingRequests.get(cameraId);
    if (pending) {
      console.log(`[UnifiedCameraPolling] Reusing pending request for ${cameraId}`);
      return pending;
    }

    // Create new request
    const request = this.performFetch(cameraId);
    this.pendingRequests.set(cameraId, request);

    try {
      const status = await request;
      this.statusCache.set(cameraId, status);
      this.notifySubscribers(cameraId, status);
      return status;
    } finally {
      this.pendingRequests.delete(cameraId);
    }
  }

  private async performFetch(cameraId: string): Promise<CameraStatusData> {
    try {
      // Use unified service to get comprehensive state
      const comprehensiveState = await unifiedCameraService.getComprehensiveState(cameraId);

      if (comprehensiveState.status.success && comprehensiveState.status.data) {
        const statusData = comprehensiveState.status.data;
        const streamData = comprehensiveState.streamInfo.data;

        return {
          owner: statusData.owner || '',
          isLive: statusData.isOnline,
          isStreaming: streamData?.isActive || false,
          status: statusData.isOnline ? 'ok' : 'offline',
          lastSeen: statusData.lastSeen
        };
      } else {
        // Camera is offline
        return {
          owner: '',
          isLive: false,
          isStreaming: false,
          status: 'offline',
          lastSeen: Date.now()
        };
      }
    } catch (error) {
      console.error(`[UnifiedCameraPolling] Error fetching status for ${cameraId}:`, error);
      return {
        owner: '',
        isLive: false,
        isStreaming: false,
        status: 'error',
        lastSeen: Date.now()
      };
    }
  }

  private notifySubscribers(cameraId: string, status: CameraStatusData): void {
    const callbacks = this.subscribers.get(cameraId);
    if (!callbacks) return;

    callbacks.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error(`[UnifiedCameraPolling] Error in subscriber callback for ${cameraId}:`, error);
      }
    });
  }

  /**
   * Get statistics about current polling state (for debugging)
   */
  getDebugStats() {
    return {
      activeCameras: this.pollIntervals.size,
      totalSubscribers: Array.from(this.subscribers.values()).reduce((sum, set) => sum + set.size, 0),
      cachedStatuses: this.statusCache.size,
      pendingRequests: this.pendingRequests.size
    };
  }
}

export const unifiedCameraPolling = UnifiedCameraPollingService.getInstance();
