import { CONFIG } from '../config';

type StatusCallback = (status: { isLive: boolean; isStreaming: boolean }) => void;

class CameraStatusService {
  private static instance: CameraStatusService;
  private callbacks: Set<StatusCallback> = new Set();
  private pollInterval: NodeJS.Timeout | null = null;
  private retryCount = 0;
  private currentStatus = { isLive: false, isStreaming: false };
  private lastCheckTime = 0;
  private readonly MIN_CHECK_INTERVAL = 2000;

  private constructor() {
    this.startPolling();
  }

  static getInstance(): CameraStatusService {
    if (!CameraStatusService.instance) {
      CameraStatusService.instance = new CameraStatusService();
    }
    return CameraStatusService.instance;
  }

  private async checkStatus(force = false): Promise<void> {
    const now = Date.now();
    if (!force && now - this.lastCheckTime < this.MIN_CHECK_INTERVAL) {
      return;
    }
    this.lastCheckTime = now;

    try {
      const healthResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/health`, {
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });

      if (!healthResponse.ok) {
        throw new Error('Health check failed');
      }

      const streamResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`, {
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });

      if (!streamResponse.ok) {
        throw new Error('Stream info check failed');
      }

      const streamData = await streamResponse.json();
      const newStatus = {
        isLive: true,
        isStreaming: streamData.isActive
      };

      if (this.hasStatusChanged(newStatus)) {
        this.currentStatus = newStatus;
        this.notifyCallbacks();
      }

      this.retryCount = 0;
    } catch (error) {
      console.error('Status check failed:', error);
      if (this.retryCount < 3) {
        this.retryCount++;
        await new Promise(resolve => setTimeout(resolve, Math.min(1000 * Math.pow(2, this.retryCount), 10000)));
        return this.checkStatus(true);
      } else {
        const newStatus = { isLive: false, isStreaming: false };
        if (this.hasStatusChanged(newStatus)) {
          this.currentStatus = newStatus;
          this.notifyCallbacks();
        }
      }
    }
  }

  private hasStatusChanged(newStatus: typeof this.currentStatus): boolean {
    return this.currentStatus.isLive !== newStatus.isLive || 
           this.currentStatus.isStreaming !== newStatus.isStreaming;
  }

  private startPolling() {
    this.checkStatus(true);
    this.pollInterval = setInterval(() => this.checkStatus(), 5000);
  }

  subscribe(callback: StatusCallback): () => void {
    this.callbacks.add(callback);
    callback(this.currentStatus);
    return () => {
      this.callbacks.delete(callback);
    };
  }

  private notifyCallbacks() {
    this.callbacks.forEach(callback => callback(this.currentStatus));
  }

  getCurrentStatus() {
    return { ...this.currentStatus };
  }

  cleanup() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    this.callbacks.clear();
  }
}

export const cameraStatus = CameraStatusService.getInstance(); 