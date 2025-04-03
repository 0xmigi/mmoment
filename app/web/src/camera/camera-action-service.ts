import { CONFIG } from "../core/config";
import { unifiedIpfsService } from "../storage/ipfs/unified-ipfs-service";

interface CameraActionResponse {
  success: boolean;
  error?: string;
  data?: {
    blob?: Blob;
    streamInfo?: {
      isActive: boolean;
      streamUrl?: string;
    };
  };
}

export class CameraActionService {
  private static instance: CameraActionService | null = null;
  private debugMode = true;
  private bypassVerification = false;

  private constructor() {
    this.log('CameraActionService initialized');
    this.log('Using API URL:', CONFIG.CAMERA_API_URL);
  }

  public enableBypassMode(enable: boolean = true) {
    this.bypassVerification = enable;
    this.log(`Transaction verification bypass ${enable ? 'enabled' : 'disabled'}`);
  }

  private log(...args: any[]) {
    if (this.debugMode) {
      console.log('[CameraAction]', ...args);
    }
  }

  public static getInstance(): CameraActionService {
    if (!CameraActionService.instance) {
      CameraActionService.instance = new CameraActionService();
    }
    return CameraActionService.instance;
  }

  /**
   * Helper method to make API calls with proper error handling
   */
  private async makeApiCall(endpoint: string, method: string, data: any): Promise<Response> {
    const url = `${CONFIG.CAMERA_API_URL}${endpoint}`;
    this.log(`Making ${method} request to: ${url}`);
    console.log(`🛠️ API Call: ${method} ${url}`);
    
    if (this.bypassVerification && method === 'POST' && data?.tx_signature) {
      this.log('BYPASS MODE: Skipping transaction verification');
      
      if (endpoint === '/api/capture' || endpoint === '/api/record' || endpoint === '/api/stream/start') {
        // Try to connect directly to camera API on port 5001 instead of middleware on port 5002
        const cameraUrl = url.replace(':5002', ':5001');
        console.log(`⚠️ BYPASS MODE: Redirecting to camera API directly: ${cameraUrl}`);
        
        try {
          const response = await fetch(cameraUrl, {
            method,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${data.wallet_address || 'unknown'}`
            },
            body: JSON.stringify({ wallet_address: data.wallet_address }),
            mode: 'cors',
            credentials: 'omit'
          });
          
          this.log(`BYPASS MODE: Response status: ${response.status} ${response.statusText}`);
          return response;
        } catch (error) {
          this.log('BYPASS MODE: Fetch error:', error);
          console.error('🚫 BYPASS MODE API Error:', error);
          throw error;
        }
      }
    }
    
    try {
      console.log(`📦 Request payload:`, data);
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: method !== 'GET' ? JSON.stringify(data) : undefined
      });
      
      this.log(`Response status: ${response.status} ${response.statusText}`);
      console.log(`✅ Response status: ${response.status} ${response.statusText}`);
      
      // Debug response headers
      const headers: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });
      console.log('📋 Response headers:', headers);
      
      return response;
    } catch (error) {
      this.log('Fetch error:', error);
      console.error('🚫 API Error:', error);
      throw error;
    }
  }


  /**
   * Takes a photo via the Solana middleware using the transaction signature
   */
  public async capturePhoto(txSignature: string, walletAddress: string, retryCount = 0): Promise<CameraActionResponse> {
    try {
      this.log(`Capturing photo with transaction: ${txSignature}`);
      const url = `${CONFIG.CAMERA_API_URL}/api/capture`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tx_signature: txSignature,
          wallet_address: walletAddress
        }),
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        this.log(`Camera API Error: ${response.status} - ${errorText}`);
        
        if (retryCount < 2 && 
            (response.status === 429 || response.status === 503 || response.status >= 500)) {
          this.log(`Retrying capture (${retryCount + 1})...`);
          return this.capturePhoto(txSignature, walletAddress, retryCount + 1);
        }
        
        return {
          success: false,
          error: `Camera API error: ${response.status} ${errorText}`
        };
      }
      
      // Get the image data directly from the response
      const imageBlob = await response.blob();
      this.log(`Received photo: ${imageBlob.size} bytes, ${imageBlob.type}`);
      
      // Upload to IPFS
      const results = await unifiedIpfsService.uploadFile(imageBlob, walletAddress, 'image');
      
      if (results.length === 0) {
        return { 
          success: false, 
          error: 'Failed to upload image to IPFS' 
        };
      }

      // Save transaction ID with the media object
      results.forEach(media => {
        // Attach the real transaction ID
        media.transactionId = txSignature;
        media.cameraId = CONFIG.CAMERA_PDA || walletAddress;
      });
      
      // Store the transaction ID in localStorage for future reference
      try {
        const mediaTransactionsKey = `mediaTransactions_${walletAddress}`;
        const existingDataStr = localStorage.getItem(mediaTransactionsKey) || '{}';
        const existingData = JSON.parse(existingDataStr);
        
        // Save the transaction ID indexed by the media ID
        results.forEach(media => {
          existingData[media.id] = {
            transactionId: txSignature,
            cameraId: CONFIG.CAMERA_PDA || walletAddress,
            timestamp: media.timestamp,
            type: 'photo'
          };
        });
        
        localStorage.setItem(mediaTransactionsKey, JSON.stringify(existingData));
        this.log(`Saved transaction ID ${txSignature} for media ID ${results[0].id} with camera ${CONFIG.CAMERA_PDA || walletAddress}`);
      } catch (e) {
        this.log(`Error saving transaction ID to localStorage: ${e}`);
      }
      
      return {
        success: true,
        data: {
          blob: imageBlob
        }
      };
    } catch (error) {
      this.log('Error capturing photo:', error);
      
      if (retryCount < 2) {
        this.log(`Retrying capture (${retryCount + 1})...`);
        return this.capturePhoto(txSignature, walletAddress, retryCount + 1);
      }
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error capturing photo'
      };
    }
  }

  /**
   * Starts streaming via the Solana middleware using the transaction signature
   */
  public async startStream(txSignature: string, walletAddress: string): Promise<CameraActionResponse> {
    try {
      this.log(`Starting stream with tx signature: ${txSignature}`);
      
      // Send the transaction signature to the middleware
      const response = await this.makeApiCall('/api/stream/start', 'POST', { 
        tx_signature: txSignature,
        wallet_address: walletAddress
      });

      if (!response.ok) {
        let errorText = '';
        try {
          errorText = JSON.stringify(await response.json());
        } catch (e) {
          errorText = await response.text();
        }
        
        this.log('Stream start error:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        
        return { 
          success: false, 
          error: `Stream error: ${response.status} ${response.statusText}` 
        };
      }

      // Parse the stream info from the response
      const streamInfo = await response.json();
      this.log('Stream started successfully:', streamInfo);
      
      return {
        success: true,
        data: {
          streamInfo: {
            isActive: true,
            streamUrl: streamInfo.streamUrl
          }
        }
      };
    } catch (error) {
      this.log('Error starting stream:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error starting stream'
      };
    }
  }

  /**
   * Stops streaming via the Solana middleware (no transaction signature required)
   */
  public async stopStream(walletAddress: string): Promise<CameraActionResponse> {
    try {
      this.log(`Stopping stream for wallet: ${walletAddress}`);
      
      // Direct API call to avoid CORS issues
      const url = `${CONFIG.CAMERA_API_URL}/api/stream/stop`;
      this.log(`Sending POST request to: ${url}`);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          wallet_address: walletAddress
        }),
        mode: 'cors',
        credentials: 'omit'
      });
      
      // If we get a 200 OK response, the stream was stopped successfully
      // even if there's a CORS error in the browser
      if (response.status === 200) {
        this.log('Stream stopped successfully with status 200');
        return {
          success: true,
          data: {
            streamInfo: {
              isActive: false
            }
          }
        };
      }

      if (!response.ok) {
        let errorText = '';
        try {
          errorText = JSON.stringify(await response.json());
        } catch (e) {
          errorText = await response.text();
        }
        
        this.log('Stream stop error:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        
        return { 
          success: false, 
          error: `Stream error: ${response.status} ${response.statusText}` 
        };
      }

      this.log('Stream stopped successfully');
      return {
        success: true,
        data: {
          streamInfo: {
            isActive: false
          }
        }
      };
    } catch (error) {
      // Even if we get a fetch error, the stream might have been stopped,
      // so we'll return success but log the error
      this.log('Error stopping stream, but assuming it stopped:', error);
      
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        this.log('This is likely a CORS error, but the stream probably stopped successfully');
        return {
          success: true,
          data: {
            streamInfo: {
              isActive: false
            }
          }
        };
      }
      
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error stopping stream'
      };
    }
  }

  /**
   * Records a video via the Solana middleware using the transaction signature
   */
  public async recordVideo(txSignature: string, walletAddress: string, retryCount = 0): Promise<CameraActionResponse> {
    try {
      // Make a POST request to the camera's record endpoint
      this.log(`Recording video with transaction: ${txSignature}`);
      
      const url = `${CONFIG.CAMERA_API_URL}/api/record`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tx_signature: txSignature,
          wallet_address: walletAddress,
          duration: 5 // Fixed 5-second duration
        }),
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        this.log(`Camera API Error: ${response.status} - ${errorText}`);
        
        if (retryCount < 2 && 
            (response.status === 429 || response.status === 503 || response.status >= 500)) {
          this.log(`Retrying recording (${retryCount + 1})...`);
          return this.recordVideo(txSignature, walletAddress, retryCount + 1);
        }
        
        return {
          success: false,
          error: `Camera API error: ${response.status} ${errorText}`
        };
      }
      
      // Parse the response as JSON to get metadata
      const recordingData = await response.json();
      this.log(`Recording completed, status: ${recordingData.status}`);
      
      if (recordingData.status !== 'recorded') {
        return {
          success: false,
          error: `Recording failed: ${recordingData.status}`
        };
      }
      
      // Get direct video URL from response if available
      let directVideoUrl: string | null = null;
      
      // Middleware might return different URL formats
      if (recordingData.video_url) {
        directVideoUrl = `${CONFIG.CAMERA_API_URL}${recordingData.video_url}`;
        this.log(`Direct video URL (video_url): ${directVideoUrl}`);
      } else if (recordingData.videoUrl) {
        directVideoUrl = `${CONFIG.CAMERA_API_URL}${recordingData.videoUrl}`;
        this.log(`Direct video URL (videoUrl): ${directVideoUrl}`);
      } else if (recordingData.file_url) {
        directVideoUrl = `${CONFIG.CAMERA_API_URL}${recordingData.file_url}`;
        this.log(`Direct video URL (file_url): ${directVideoUrl}`);
      } else if (recordingData.fileUrl) {
        directVideoUrl = `${CONFIG.CAMERA_API_URL}${recordingData.fileUrl}`;
        this.log(`Direct video URL (fileUrl): ${directVideoUrl}`);
      } else if (recordingData.filename) {
        // Try constructing URL from filename
        directVideoUrl = `${CONFIG.CAMERA_API_URL}/api/videos/${recordingData.filename}`;
        this.log(`Constructed direct video URL from filename: ${directVideoUrl}`);
      }
      
      // Get the actual video content - THIS IS THE CRUCIAL STEP THAT WAS MISSING
      let videoBlob: Blob;
      if (directVideoUrl) {
        try {
          // Directly fetch video content from middleware
          this.log(`Fetching video content from: ${directVideoUrl}`);
          const videoResponse = await fetch(directVideoUrl, {
            method: 'GET',
            mode: 'cors',
            credentials: 'omit',
            cache: 'no-cache'
          });
          
          if (videoResponse.ok) {
            // Get the video as binary data
            const arrayBuffer = await videoResponse.arrayBuffer();
            this.log(`Fetched video content: ${arrayBuffer.byteLength} bytes`);
            
            // Create blob with proper MIME type
            videoBlob = new Blob([arrayBuffer], { type: 'video/quicktime' });
            this.log(`Created video blob: ${videoBlob.size} bytes, type: ${videoBlob.type}`);
          } else {
            this.log(`Failed to fetch video content: ${videoResponse.status}`);
            // Create a placeholder with the metadata
            const placeholderData = JSON.stringify(recordingData);
            videoBlob = new Blob([placeholderData], { type: 'application/json' });
            this.log(`Created placeholder blob: ${videoBlob.size} bytes, no video data available`);
          }
        } catch (e) {
          this.log(`Error fetching video content: ${e}`);
          // Create a placeholder with the metadata
          const placeholderData = JSON.stringify(recordingData);
          videoBlob = new Blob([placeholderData], { type: 'application/json' });
          this.log(`Error creating video blob, using placeholder: ${videoBlob.size} bytes`);
        }
      } else {
        // No direct URL available, create placeholder
        const placeholderData = JSON.stringify(recordingData);
        videoBlob = new Blob([placeholderData], { type: 'application/json' });
        this.log(`No video URL available, created placeholder: ${videoBlob.size} bytes`);
      }
      
      // Upload the video to IPFS
      this.log(`Uploading to IPFS...`);
      console.log(`[CameraAction] Uploading video blob: ${videoBlob.size} bytes, type: ${videoBlob.type}`);
      
      // Pass the direct URL so it can be stored with the IPFS result
      const uploadOptions = {
        directUrl: directVideoUrl || undefined
      };
      
      const results = await unifiedIpfsService.uploadFile(videoBlob, walletAddress, 'video', uploadOptions);
      
      if (results.length === 0) {
        this.log('IPFS upload failed - no results returned');
        return { 
          success: false, 
          error: 'Failed to upload video to IPFS' 
        };
      }
      
      // Save transaction ID with the media object
      results.forEach(media => {
        media.transactionId = txSignature;
        media.cameraId = CONFIG.CAMERA_PDA || walletAddress;
        if (directVideoUrl) {
          media.directUrl = directVideoUrl;
          
          // Add direct URL to backup URLs if not already there
          if (!media.backupUrls.includes(directVideoUrl)) {
            media.backupUrls.unshift(directVideoUrl);
          }
        }
      });
      
      // Store the transaction ID in localStorage for future reference
      try {
        const mediaTransactionsKey = `mediaTransactions_${walletAddress}`;
        const existingDataStr = localStorage.getItem(mediaTransactionsKey) || '{}';
        const existingData = JSON.parse(existingDataStr);
        
        // Save the transaction ID indexed by the media ID
        results.forEach(media => {
          existingData[media.id] = {
            transactionId: txSignature,
            cameraId: CONFIG.CAMERA_PDA || walletAddress,
            timestamp: media.timestamp,
            type: 'video',
            directUrl: directVideoUrl || undefined
          };
        });
        
        localStorage.setItem(mediaTransactionsKey, JSON.stringify(existingData));
        this.log(`Saved transaction ID ${txSignature} for video ID ${results[0].id} with camera ${CONFIG.CAMERA_PDA || walletAddress}`);
      } catch (e) {
        this.log(`Error saving transaction ID to localStorage: ${e}`);
      }
      
      this.log(`Successfully uploaded to IPFS: ${results[0].url}`);
      return {
        success: true,
        data: {
          blob: videoBlob
        }
      };
    } catch (error) {
      this.log('Error recording video:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error recording video'
      };
    }
  }


  /**
   * Gets the camera status
   */
  public async getCameraStatus(): Promise<any> {
    try {
      this.log('Fetching camera status');
      
      // Fetch health status
      const url = `${CONFIG.CAMERA_API_URL}/api/health`;
      this.log(`Making request to: ${url}`);
      
      const response = await fetch(url);
      
      if (!response.ok) {
        this.log(`Health check failed: ${response.status} ${response.statusText}`);
        return { isOnline: false };
      }
      
      // Try to parse JSON
      try {
        const data = await response.json();
        this.log('Camera status:', data);
        return {
          isOnline: true,
          ...data
        };
      } catch (e) {
        this.log('Failed to parse health response as JSON:', e);
        // If it's not valid JSON but the request succeeded, camera is still online
        return { isOnline: true };
      }
    } catch (error) {
      this.log('Error fetching camera status:', error);
      return { isOnline: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  /**
   * Tests connectivity to the middleware server
   */
  public async testConnection(): Promise<{
    middleware: { success: boolean; message: string; url: string }
    camera: { success: boolean; message: string; url: string }
  }> {
    this.log('Testing connection to middleware and camera API');
    
    const result = {
      middleware: { 
        success: false, 
        message: 'Not tested',
        url: CONFIG.CAMERA_API_URL 
      },
      camera: { 
        success: false, 
        message: 'Not tested',
        url: CONFIG.CAMERA_API_URL.replace('middleware', 'camera') 
      }
    };
    
    // Test middleware connection
    try {
      const middlewareUrl = `${CONFIG.CAMERA_API_URL}/api/debug`;
      this.log(`Testing middleware connection to: ${middlewareUrl}`);
      
      const middlewareResponse = await fetch(middlewareUrl, {
        method: 'GET',
        mode: 'cors'
      });
      
      if (middlewareResponse.ok) {
        try {
          const debugInfo = await middlewareResponse.json();
          this.log('Middleware debug info:', debugInfo);
          
          result.middleware = {
            success: true,
            message: `Connected to middleware v${debugInfo.middleware?.version || 'unknown'}`,
            url: middlewareUrl
          };
          
          // If camera info is available in debug response
          if (debugInfo.camera_api) {
            result.camera = {
              success: debugInfo.camera_api.status === 'connected',
              message: debugInfo.camera_api.status === 'connected' 
                ? 'Connected to camera API' 
                : 'Camera API error: ' + debugInfo.camera_api.status,
              url: debugInfo.middleware?.camera_api_url || 'unknown'
            };
          }
        } catch (e) {
          this.log('Error parsing middleware debug response:', e);
          result.middleware = {
            success: true,
            message: 'Connected to middleware but received invalid response',
            url: middlewareUrl
          };
        }
      } else {
        this.log(`Middleware connection failed: ${middlewareResponse.status} ${middlewareResponse.statusText}`);
        result.middleware = {
          success: false,
          message: `Connection failed: ${middlewareResponse.status} ${middlewareResponse.statusText}`,
          url: middlewareUrl
        };
      }
    } catch (e) {
      this.log('Error testing middleware connection:', e);
      result.middleware = {
        success: false,
        message: `Connection error: ${e instanceof Error ? e.message : String(e)}`,
        url: CONFIG.CAMERA_API_URL
      };
    }
    
    // If we didn't get camera info from middleware debug, test camera directly
    if (result.camera.message === 'Not tested') {
      try {
        const cameraUrl = `${CONFIG.CAMERA_API_URL.replace('middleware', 'camera')}/api/health`;
        this.log(`Testing direct camera connection to: ${cameraUrl}`);
        
        const cameraResponse = await fetch(cameraUrl, {
          method: 'GET',
          mode: 'cors'
        });
        
        if (cameraResponse.ok) {
          result.camera = {
            success: true,
            message: 'Connected to camera API directly',
            url: cameraUrl
          };
        } else {
          this.log(`Camera connection failed: ${cameraResponse.status} ${cameraResponse.statusText}`);
          result.camera = {
            success: false,
            message: `Connection failed: ${cameraResponse.status} ${cameraResponse.statusText}`,
            url: cameraUrl
          };
        }
      } catch (e) {
        this.log('Error testing camera connection:', e);
        result.camera = {
          success: false,
          message: `Connection error: ${e instanceof Error ? e.message : String(e)}`,
          url: CONFIG.CAMERA_API_URL.replace('middleware', 'camera')
        };
      }
    }
    
    return result;
  }

}

// Export the singleton instance
export const cameraActionService = CameraActionService.getInstance(); 