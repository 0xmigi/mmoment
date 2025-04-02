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
  private debugMode = true; // Enable debug logs

  private constructor() {
    this.logDebug('CameraStatusService initialized');
    this.startPolling();
  }

  static getInstance(): CameraStatusService {
    if (!CameraStatusService.instance) {
      CameraStatusService.instance = new CameraStatusService();
    }
    return CameraStatusService.instance;
  }

  private logDebug(message: string, ...args: any[]) {
    if (this.debugMode) {
      console.log(`[CameraStatus] ${message}`, ...args);
    }
  }

  private async checkStatus(force = false): Promise<void> {
    const now = Date.now();
    if (!force && now - this.lastCheckTime < this.MIN_CHECK_INTERVAL) {
      return;
    }
    this.lastCheckTime = now;

    this.logDebug(`Checking camera status using API URL: ${CONFIG.CAMERA_API_URL}`);

    try {
      // Create a simple fetch without cache-control headers which might cause CORS issues
      const healthResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/health`);
      this.logDebug('Health response status:', healthResponse.status);

      if (!healthResponse.ok) {
        throw new Error(`Health check failed with status: ${healthResponse.status}`);
      }

      let healthData;
      try {
        healthData = await healthResponse.json();
        this.logDebug('Health data:', healthData);
      } catch (e) {
        this.logDebug('Failed to parse health response as JSON:', e);
      }

      // Try to fetch stream info - if it fails, we'll still consider the camera live
      // but not streaming
      let isStreaming = false;
      try {
        const streamResponse = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`);
        this.logDebug('Stream info response status:', streamResponse.status);

        if (streamResponse.ok) {
          const streamData = await streamResponse.json();
          this.logDebug('Stream data:', streamData);
          isStreaming = streamData.isActive;
        } else {
          this.logDebug('Stream info request failed, assuming not streaming');
        }
      } catch (streamError) {
        this.logDebug('Error fetching stream info:', streamError);
        // Don't throw here, we still want to update the isLive status
      }

      const newStatus = {
        isLive: true,
        isStreaming
      };

      this.logDebug('New status:', newStatus, 'Current status:', this.currentStatus);

      if (this.hasStatusChanged(newStatus)) {
        this.logDebug('Status changed, updating');
        this.currentStatus = newStatus;
        this.notifyCallbacks();
      }

      this.retryCount = 0;
    } catch (error) {
      this.logDebug('Status check failed:', error);
      
      if (this.retryCount < 3) {
        this.retryCount++;
        const retryDelay = Math.min(1000 * Math.pow(2, this.retryCount), 10000);
        this.logDebug(`Retrying in ${retryDelay}ms (attempt ${this.retryCount})`);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
        return this.checkStatus(true);
      } else {
        this.logDebug('Max retries reached, setting camera to offline');
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
    this.logDebug('Starting status polling');
    this.checkStatus(true);
    this.pollInterval = setInterval(() => this.checkStatus(), 5000);
  }

  subscribe(callback: StatusCallback): () => void {
    this.logDebug('New subscriber added');
    this.callbacks.add(callback);
    callback(this.currentStatus);
    return () => {
      this.logDebug('Subscriber removed');
      this.callbacks.delete(callback);
    };
  }

  private notifyCallbacks() {
    this.logDebug(`Notifying ${this.callbacks.size} subscribers of status:`, this.currentStatus);
    this.callbacks.forEach(callback => callback(this.currentStatus));
  }

  getCurrentStatus() {
    return { ...this.currentStatus };
  }

  // Force check and reset - useful for debugging
  public forceCheck() {
    this.logDebug('Force checking status');
    this.retryCount = 0;
    return this.checkStatus(true);
  }

  cleanup() {
    this.logDebug('Cleaning up CameraStatusService');
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    this.callbacks.clear();
  }
}

export const cameraStatus = CameraStatusService.getInstance(); 