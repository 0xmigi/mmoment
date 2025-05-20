import { CONFIG } from '../core/config';

type StatusCallback = (status: { isLive: boolean; isStreaming: boolean; owner: string }) => void;

class CameraStatusService {
  private static instance: CameraStatusService;
  private callbacks: Set<StatusCallback> = new Set();
  private pollInterval: NodeJS.Timeout | null = null;
  private retryCount = 0;
  private currentStatus = { isLive: false, isStreaming: false, owner: '' };
  private lastCheckTime = 0;
  private readonly MIN_CHECK_INTERVAL = 3000; // Increased to reduce load
  private readonly MOBILE_CHECK_INTERVAL = 10000; // Longer interval for mobile
  private readonly FETCH_TIMEOUT = 8000; // 8 second timeout for fetch operations
  private debugMode = true; // Enable debug logs
  private forceOnlineMode = false; // New flag to force online status when we know the camera is working

  private constructor() {
    this.logDebug('CameraStatusService initialized');
    this.startPolling();
    
    // Check localStorage for previous status
    try {
      const savedStatus = localStorage.getItem('camera_status');
      if (savedStatus) {
        const parsed = JSON.parse(savedStatus);
        // Apply saved status if it exists
        if (parsed && typeof parsed === 'object') {
          this.logDebug('Restored camera status from localStorage:', parsed);
          this.currentStatus = {...this.currentStatus, ...parsed};
          this.notifyCallbacks();
        }
      }
    } catch (e) {
      this.logDebug('Error loading saved camera status:', e);
    }
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

  // Helper function to safely fetch with timeout
  private async fetchWithTimeout(url: string, options: RequestInit = {}, timeout = this.FETCH_TIMEOUT): Promise<Response> {
    const controller = new AbortController();
    const { signal } = controller;
    
    // Create timeout that aborts the fetch
    const timeoutId = setTimeout(() => {
      this.logDebug(`Fetch timeout reached for ${url}`);
      controller.abort();
    }, timeout);
    
    try {
      const response = await fetch(url, { ...options, signal });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private async checkStatus(force = false): Promise<void> {
    // If we're in forced online mode, don't do normal status checks
    if (this.forceOnlineMode) {
      this.logDebug('In forced online mode, skipping normal status check');
      return;
    }

    const now = Date.now();
    const isMobile = CONFIG.isMobileBrowser;
    const minInterval = isMobile ? this.MOBILE_CHECK_INTERVAL : this.MIN_CHECK_INTERVAL;
    
    if (!force && now - this.lastCheckTime < minInterval) {
      return;
    }
    this.lastCheckTime = now;

    this.logDebug(`Checking camera status using API URL: ${CONFIG.CAMERA_API_URL}, mobile: ${isMobile}`);

    // Use a timeout promise to prevent the check from hanging
    const timeoutPromise = new Promise<void>((_, reject) => {
      setTimeout(() => {
        reject(new Error('Status check timed out'));
      }, this.FETCH_TIMEOUT * 1.5);
    });

    try {
      // Race the status check with a timeout to prevent UI freezes
      await Promise.race([this._performStatusCheck(), timeoutPromise]);
    } catch (error) {
      this.logDebug('Status check failed or timed out:', error);
      
      // Don't set camera offline if we've recently had successful operations
      if (this.hasRecentSuccessfulOperation()) {
        this.logDebug('Ignoring status check failure due to recent successful operation');
        return;
      }
      
      // Mobile gets more retries with longer delays
      const maxRetries = CONFIG.isMobileBrowser ? 5 : 3;
      
      if (this.retryCount < maxRetries) {
        this.retryCount++;
        const retryDelay = Math.min(1000 * Math.pow(2, this.retryCount), 20000);
        this.logDebug(`Retrying in ${retryDelay}ms (attempt ${this.retryCount}/${maxRetries})`);
        
        // Schedule retry instead of recursively calling to avoid stack issues
        setTimeout(() => {
          this.checkStatus(true).catch(e => 
            this.logDebug('Retry check status failed:', e)
          );
        }, retryDelay);
      } else {
        this.logDebug('Max retries reached, setting camera to offline');
        
        // Don't set offline if we know it was working recently
        if (!this.hasRecentSuccessfulOperation()) {
          const newStatus = { isLive: false, isStreaming: false, owner: this.currentStatus.owner };
          if (this.hasStatusChanged(newStatus)) {
            this.currentStatus = newStatus;
            this.notifyCallbacks();
            this.saveStatus();
          }
        }
        
        // Reset retry count after a while to allow future attempts
        setTimeout(() => {
          this.retryCount = 0;
        }, 60000); // Wait 1 minute before allowing retries again
      }
    }
  }

  // Check if we've had a successful camera operation recently (in the last 5 minutes)
  private hasRecentSuccessfulOperation(): boolean {
    try {
      const lastSuccessTimeStr = localStorage.getItem('last_successful_camera_operation');
      if (!lastSuccessTimeStr) return false;
      
      const lastSuccessTime = parseInt(lastSuccessTimeStr, 10);
      const now = Date.now();
      const fiveMinutes = 5 * 60 * 1000;
      
      return !isNaN(lastSuccessTime) && (now - lastSuccessTime < fiveMinutes);
    } catch (e) {
      return false;
    }
  }

  // Record a successful camera operation
  public recordSuccessfulOperation(): void {
    try {
      localStorage.setItem('last_successful_camera_operation', Date.now().toString());
      this.logDebug('Recorded successful camera operation');
    } catch (e) {
      this.logDebug('Error recording successful operation:', e);
    }
  }

  // New method: set streaming mode
  public setStreaming(isStreaming: boolean): void {
    this.logDebug(`Setting streaming mode to: ${isStreaming}`);
    this.forceOnlineMode = true;
    this.recordSuccessfulOperation();
    
    const newStatus = { 
      ...this.currentStatus,
      isLive: true, 
      isStreaming 
    };
    
    if (this.hasStatusChanged(newStatus)) {
      this.currentStatus = newStatus;
      this.notifyCallbacks();
      this.saveStatus();
    }
  }

  // Save current status to localStorage
  private saveStatus(): void {
    try {
      localStorage.setItem('camera_status', JSON.stringify(this.currentStatus));
    } catch (e) {
      this.logDebug('Error saving camera status:', e);
    }
  }

  // Separate the actual status check logic for better error handling
  private async _performStatusCheck(): Promise<void> {
    const isMobile = CONFIG.isMobileBrowser;
    
    try {
      console.log(`[CameraStatus] Checking health at ${new Date().toISOString()} - ${CONFIG.CAMERA_API_URL}/api/health`);
      
      // Try a simple health check first
      const healthResponse = await this.fetchWithTimeout(
        `${CONFIG.CAMERA_API_URL}/api/health`,
        {
          // Add cache-busting for mobile
          headers: isMobile ? {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          } : undefined
        },
        isMobile ? this.FETCH_TIMEOUT * 1.5 : this.FETCH_TIMEOUT // Longer timeout for mobile
      );
      
      this.logDebug('Health response status:', healthResponse.status);

      if (!healthResponse.ok) {
        console.error(`[CameraStatus] Health check failed with status: ${healthResponse.status}, trying direct camera API`);
        throw new Error(`Health check failed with status: ${healthResponse.status}`);
      }

      let healthData;
      try {
        healthData = await healthResponse.json();
        this.logDebug('Health data:', healthData);
        console.log(`[CameraStatus] Health data:`, healthData);
        
        // If the health check returns camera_api field and it contains status: connected
        // This is a more reliable indicator than just a 200 OK
        if (healthData.camera_api && healthData.camera_api.status === 'connected') {
          this.logDebug('Camera API is explicitly connected');
          this.recordSuccessfulOperation();
        }
        
        // If healthData contains a camera object with an "available" state
        if (healthData.camera_api?.camera?.state === 'available' || 
            healthData.camera?.state === 'available' || 
            healthData.camera_api?.state === 'available') {
          this.logDebug('Camera is marked as available');
          this.recordSuccessfulOperation();
        }
      } catch (e) {
        this.logDebug('Failed to parse health response as JSON:', e);
      }

      // Try to fetch stream info with a shorter timeout
      let isStreaming = false;
      try {
        const streamResponse = await this.fetchWithTimeout(
          `${CONFIG.CAMERA_API_URL}/api/stream/info`,
          {
            // Add cache-busting for mobile
            headers: isMobile ? {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            } : undefined
          },
          isMobile ? 5000 : 3000 // Shorter timeout for stream info
        );
        
        this.logDebug('Stream info response status:', streamResponse.status);

        if (streamResponse.ok) {
          const streamData = await streamResponse.json();
          this.logDebug('Stream data:', streamData);
          isStreaming = streamData.isActive;
          
          // If streaming is active, record a successful operation
          if (isStreaming) {
            this.recordSuccessfulOperation();
          }
        } else {
          this.logDebug('Stream info request failed, assuming not streaming');
        }
      } catch (streamError) {
        this.logDebug('Error fetching stream info:', streamError);
        // Don't throw here, we still want to update the isLive status
      }

      // Extract owner from the health data if available
      const owner = healthData?.owner || healthData?.camera?.owner || '';

      const newStatus = {
        isLive: true,
        isStreaming,
        owner
      };

      this.logDebug('New status:', newStatus, 'Current status:', this.currentStatus);

      if (this.hasStatusChanged(newStatus)) {
        this.logDebug('Status changed, updating');
        this.currentStatus = newStatus;
        this.notifyCallbacks();
        this.saveStatus();
      }

      this.retryCount = 0;
      this.recordSuccessfulOperation();
    } catch (error) {
      this.logDebug('Status check operation failed:', error);
      
      // Try direct camera API as fallback
      console.log(`[CameraStatus] Trying direct camera API at ${CONFIG.CAMERA_API_URL.replace('middleware', 'camera')}/api/health`);
      try {
        const directCameraResponse = await this.fetchWithTimeout(
          `${CONFIG.CAMERA_API_URL.replace('middleware', 'camera')}/api/health`,
          {
            headers: isMobile ? {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            } : undefined
          },
          this.FETCH_TIMEOUT
        );
        
        if (directCameraResponse.ok) {
          console.log(`[CameraStatus] Direct camera API check succeeded`);
          let cameraData;
          try {
            cameraData = await directCameraResponse.json();
            console.log(`[CameraStatus] Direct camera data:`, cameraData);
            
            // If we can directly connect to the camera, record it as a successful operation
            this.recordSuccessfulOperation();
          } catch (e) {
            console.log(`[CameraStatus] Failed to parse direct camera response as JSON`);
          }
          
          const newStatus = {
            isLive: true,
            isStreaming: false, // We don't know for sure, default to false
            owner: cameraData?.owner || ''
          };
          
          if (this.hasStatusChanged(newStatus)) {
            this.logDebug('Status changed from direct camera check, updating');
            this.currentStatus = newStatus;
            this.notifyCallbacks();
            this.saveStatus();
          }
          
          this.retryCount = 0;
          return; // Successfully used direct camera API
        } else {
          console.log(`[CameraStatus] Direct camera API check failed with status: ${directCameraResponse.status}`);
        }
      } catch (directError) {
        console.log(`[CameraStatus] Direct camera API check failed:`, directError);
      }
      
      // If we get here, both middleware and direct camera checks failed
      throw error; // Re-throw to let the main checkStatus method handle retries
    }
  }

  private hasStatusChanged(newStatus: Partial<typeof this.currentStatus>): boolean {
    return (newStatus.isLive !== undefined && this.currentStatus.isLive !== newStatus.isLive) || 
           (newStatus.isStreaming !== undefined && this.currentStatus.isStreaming !== newStatus.isStreaming) ||
           (newStatus.owner !== undefined && this.currentStatus.owner !== newStatus.owner);
  }

  private startPolling() {
    // Stop existing interval if any
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    
    // Start a new polling interval
    const pollingInterval = CONFIG.isMobileBrowser ? 15000 : 10000; // Longer on mobile
    this.pollInterval = setInterval(() => {
      this.checkStatus().catch(e => this.logDebug('Poll check status failed:', e));
    }, pollingInterval);
    
    this.logDebug(`Status polling started with interval: ${pollingInterval}ms`);
    
    // Do an initial check
    this.checkStatus(true).catch(e => this.logDebug('Initial check status failed:', e));
  }

  subscribe(callback: StatusCallback): () => void {
    this.callbacks.add(callback);
    callback(this.currentStatus); // immediately call with current status
    
    // Return unsubscribe function
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

  public forceCheck() {
    // Reset forced online mode for new check
    this.forceOnlineMode = false;
    return this.checkStatus(true);
  }

  cleanup() {
    this.logDebug('Cleaning up CameraStatusService');
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  public setOnline(isStreaming: boolean = false) {
    this.logDebug(`Setting camera status to online, streaming: ${isStreaming}`);
    this.forceOnlineMode = true;
    this.recordSuccessfulOperation();
    
    const newStatus = {
      isLive: true,
      isStreaming: isStreaming || this.currentStatus.isStreaming,
      owner: this.currentStatus.owner
    };
    
    if (this.hasStatusChanged(newStatus)) {
      this.currentStatus = newStatus;
      this.notifyCallbacks();
      this.saveStatus();
    }
  }
}

export const cameraStatus = CameraStatusService.getInstance(); 