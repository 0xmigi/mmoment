/**
 * Unified Camera Service
 * 
 * This service provides a single interface for all camera operations,
 * making the frontend completely camera-agnostic. It automatically
 * detects camera types and routes operations to the appropriate implementation.
 */

import { 
  ICamera, 
  CameraActionResponse, 
  CameraMediaResponse, 
  CameraStreamInfo, 
  CameraStatus,
  CameraSession,
  CameraGestureResponse
} from './camera-interface';
import { cameraRegistry } from './camera-registry';

export class UnifiedCameraService {
  private static instance: UnifiedCameraService | null = null;
  private debugMode = true;

  private constructor() {
    this.log('UnifiedCameraService initialized');
  }

  public static getInstance(): UnifiedCameraService {
    if (!UnifiedCameraService.instance) {
      UnifiedCameraService.instance = new UnifiedCameraService();
    }
    return UnifiedCameraService.instance;
  }

  private log(...args: any[]) {
    if (this.debugMode) {
      console.log('[UnifiedCameraService]', ...args);
    }
  }

  /**
   * Get a camera instance by ID
   */
  private async getCamera(cameraId: string): Promise<ICamera | null> {
    if (!cameraId) {
      this.log('No camera ID provided');
      return null;
    }

    const camera = await cameraRegistry.getCamera(cameraId);
    if (!camera) {
      this.log(`Camera not found or failed to create: ${cameraId}`);
      return null;
    }

    return camera;
  }

  /**
   * Check if a camera exists and is registered
   */
  public hasCamera(cameraId: string): boolean {
    return cameraRegistry.hasCamera(cameraId);
  }

  /**
   * Get camera type by ID
   */
  public getCameraType(cameraId: string): string | null {
    return cameraRegistry.getCameraType(cameraId);
  }

  /**
   * Check if camera supports a specific capability
   */
  public cameraSupports(cameraId: string, capability: keyof import('./camera-interface').CameraCapabilities): boolean {
    return cameraRegistry.cameraSupports(cameraId, capability);
  }

  /**
   * Connect to a camera
   */
  public async connect(cameraId: string, walletAddress?: string): Promise<CameraActionResponse<CameraSession>> {
    try {
      this.log(`Connecting to camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.connect(walletAddress);
      this.log(`Connection result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Connection error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to connect'
      };
    }
  }

  /**
   * Disconnect from a camera
   */
  public async disconnect(cameraId: string): Promise<CameraActionResponse> {
    try {
      this.log(`Disconnecting from camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.disconnect();
      this.log(`Disconnect result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Disconnect error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to disconnect'
      };
    }
  }

  /**
   * Test connection to a camera
   */
  public async testConnection(cameraId: string): Promise<CameraActionResponse<{ message: string; url: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: { message: '', url: '' }
        };
      }

      return await camera.testConnection();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to test connection',
        data: { message: '', url: '' }
      };
    }
  }

  /**
   * Check if camera is connected
   */
  public async isConnected(cameraId: string): Promise<boolean> {
    try {
      const camera = await this.getCamera(cameraId);
      return camera ? camera.isConnected() : false;
    } catch (error) {
      this.log(`Error checking connection for ${cameraId}:`, error);
      return false;
    }
  }

  /**
   * Get camera status
   */
  public async getStatus(cameraId: string): Promise<CameraActionResponse<CameraStatus>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: {
            isOnline: false,
            isStreaming: false,
            isRecording: false,
            lastSeen: Date.now()
          }
        };
      }

      return await camera.getStatus();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get status',
        data: {
          isOnline: false,
          isStreaming: false,
          isRecording: false,
          lastSeen: Date.now()
        }
      };
    }
  }

  /**
   * Take a photo
   */
  public async takePhoto(cameraId: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log(`Taking photo with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`Camera not found for takePhoto: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      this.log(`Camera instance created successfully: ${camera.cameraType} at ${camera.apiUrl}`);
      
      const result = await camera.takePhoto();
      this.log(`Photo result for ${cameraId}:`, result.success ? 'SUCCESS' : `FAILED - ${result.error}`);
      
      return result;
    } catch (error) {
      this.log(`Photo error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to take photo'
      };
    }
  }

  /**
   * Start video recording
   */
  public async startVideoRecording(cameraId: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log(`Starting video recording with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.startVideoRecording();
      this.log(`Start recording result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Start recording error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start recording'
      };
    }
  }

  /**
   * Stop video recording
   */
  public async stopVideoRecording(cameraId: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log(`Stopping video recording with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.stopVideoRecording();
      this.log(`Stop recording result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Stop recording error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop recording'
      };
    }
  }

  /**
   * Get recorded video
   */
  public async getRecordedVideo(cameraId: string, filename: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      return await camera.getRecordedVideo(filename);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get video'
      };
    }
  }

  /**
   * List available videos
   */
  public async listVideos(cameraId: string): Promise<CameraActionResponse<{ videos: any[] }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: { videos: [] }
        };
      }

      return await camera.listVideos();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list videos',
        data: { videos: [] }
      };
    }
  }

  /**
   * Start streaming
   */
  public async startStream(cameraId: string): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      this.log(`Starting stream with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.startStream();
      this.log(`Start stream result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Start stream error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start stream'
      };
    }
  }

  /**
   * Stop streaming
   */
  public async stopStream(cameraId: string): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      this.log(`Stopping stream with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const result = await camera.stopStream();
      this.log(`Stop stream result for ${cameraId}:`, result.success ? 'SUCCESS' : 'FAILED');
      
      return result;
    } catch (error) {
      this.log(`Stop stream error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop stream'
      };
    }
  }

  /**
   * Get stream information
   */
  public async getStreamInfo(cameraId: string): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: {
            isActive: false,
            format: 'mjpeg'
          }
        };
      }

      return await camera.getStreamInfo();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get stream info',
        data: {
          isActive: false,
          format: 'mjpeg'
        }
      };
    }
  }

  /**
   * Get stream URL
   */
  public async getStreamUrl(cameraId: string): Promise<string | null> {
    try {
      const camera = await this.getCamera(cameraId);
      return camera ? camera.getStreamUrl() : null;
    } catch (error) {
      this.log(`Error getting stream URL for ${cameraId}:`, error);
      return null;
    }
  }

  /**
   * Get current gesture (if supported)
   */
  public async getCurrentGesture(cameraId: string): Promise<CameraActionResponse<CameraGestureResponse>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.getCurrentGesture) {
        return {
          success: false,
          error: 'Gesture detection not supported by this camera'
        };
      }

      return await camera.getCurrentGesture();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get gesture'
      };
    }
  }

  /**
   * Toggle gesture controls (if supported)
   */
  public async toggleGestureControls(cameraId: string, enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.toggleGestureControls) {
        return {
          success: false,
          error: 'Gesture controls not supported by this camera'
        };
      }

      return await camera.toggleGestureControls(enabled);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle gesture controls'
      };
    }
  }

  /**
   * Toggle face visualization (if supported)
   */
  public async toggleFaceVisualization(cameraId: string, enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.toggleFaceVisualization) {
        return {
          success: false,
          error: 'Face visualization not supported by this camera'
        };
      }

      return await camera.toggleFaceVisualization(enabled);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle face visualization'
      };
    }
  }

  /**
   * Get current session
   */
  public async getCurrentSession(cameraId: string): Promise<CameraSession | null> {
    try {
      const camera = await this.getCamera(cameraId);
      return camera ? camera.getCurrentSession() : null;
    } catch (error) {
      this.log(`Error getting session for ${cameraId}:`, error);
      return null;
    }
  }

  /**
   * Check if currently recording
   */
  public async isCurrentlyRecording(cameraId: string): Promise<boolean> {
    try {
      const camera = await this.getCamera(cameraId);
      return camera ? camera.isCurrentlyRecording() : false;
    } catch (error) {
      this.log(`Error checking recording status for ${cameraId}:`, error);
      return false;
    }
  }

  /**
   * Health check all cameras
   */
  public async healthCheckAll(): Promise<Map<string, boolean>> {
    return await cameraRegistry.healthCheckAll();
  }

  /**
   * Get all registered cameras
   */
  public getAllCameras() {
    return cameraRegistry.getAllCameras();
  }

  /**
   * Get online cameras only
   */
  public getOnlineCameras() {
    return cameraRegistry.getOnlineCameras();
  }

  /**
   * Get most recent video
   */
  public async getMostRecentVideo(cameraId: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log(`Getting most recent video from camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.getMostRecentVideo) {
        return {
          success: false,
          error: 'Camera does not support getMostRecentVideo'
        };
      }
      
      const result = await camera.getMostRecentVideo();
      this.log(`Most recent video result for ${cameraId}:`, result.success ? 'SUCCESS' : `FAILED - ${result.error}`);
      
      return result;
    } catch (error) {
      this.log(`Most recent video error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get most recent video'
      };
    }
  }

  /**
   * Check for gesture trigger
   */
  public async checkForGestureTrigger(cameraId?: string): Promise<{
    shouldCapture: boolean;
    gestureType: 'photo' | 'video' | null;
    gesture?: string;
    confidence?: number;
  }> {
    try {
      // If no camera ID provided, try to get the first available camera
      if (!cameraId) {
        const availableCameras = this.getAllCameras();
        if (availableCameras.length === 0) {
          return { shouldCapture: false, gestureType: null };
        }
        cameraId = availableCameras[0].cameraId;
      }

      this.log(`Checking gesture trigger for camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return { shouldCapture: false, gestureType: null };
      }

      if (!camera.checkForGestureTrigger) {
        return { shouldCapture: false, gestureType: null };
      }
      
      const result = await camera.checkForGestureTrigger();
      this.log(`Gesture trigger result for ${cameraId}:`, result);
      
      return result;
    } catch (error) {
      this.log(`Gesture trigger error for ${cameraId}:`, error);
      return { shouldCapture: false, gestureType: null };
    }
  }

  /**
   * Get gesture controls status
   */
  public async getGestureControlsStatus(cameraId?: string): Promise<boolean> {
    try {
      // If no camera ID provided, try to get the first available camera
      if (!cameraId) {
        const availableCameras = this.getAllCameras();
        if (availableCameras.length === 0) {
          return false;
        }
        cameraId = availableCameras[0].cameraId;
      }

      this.log(`Getting gesture controls status for camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return false;
      }

      if (!camera.getGestureControlsStatus) {
        return false;
      }
      
      const result = camera.getGestureControlsStatus();
      this.log(`Gesture controls status for ${cameraId}:`, result);
      
      return result;
    } catch (error) {
      this.log(`Gesture controls status error for ${cameraId}:`, error);
      return false;
    }
  }

  /**
   * Start video recording with automatic timeout
   */
  public async startTimedVideoRecording(cameraId: string, durationSeconds: number = 30): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log(`Starting timed video recording (${durationSeconds}s) with camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      // Ensure camera is connected first
      this.log('Checking camera connection...');
      const isConnected = await this.isConnected(cameraId);
      if (!isConnected) {
        this.log('Camera not connected, attempting to connect...');
        const connectResult = await this.connect(cameraId, 'default-wallet'); // Use a default wallet for now
        if (!connectResult.success) {
          return {
            success: false,
            error: `Failed to connect to camera: ${connectResult.error}`
          };
        }
        this.log('✅ Camera connected successfully');
      } else {
        this.log('✅ Camera already connected');
      }
      
      // Start the recording
      const startResult = await this.startVideoRecording(cameraId);
      if (!startResult.success) {
        return startResult;
      }

      this.log(`Video recording started, will auto-stop in ${durationSeconds} seconds`);
      
      // Set up automatic stop after the specified duration
      setTimeout(async () => {
        try {
          this.log(`Auto-stopping video recording after ${durationSeconds} seconds`);
          const stopResult = await this.stopVideoRecording(cameraId);
          
          if (stopResult.success) {
            this.log('✅ Video recording auto-stopped successfully:', stopResult.data?.filename);
            
            // Try to get the video and trigger download
            if (stopResult.data?.filename) {
              this.log('📥 Attempting to download recorded video...');
              const videoResult = await this.getRecordedVideo(cameraId, stopResult.data.filename);
              
              if (videoResult.success && videoResult.data?.blob) {
                // Create download link
                const url = URL.createObjectURL(videoResult.data.blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = stopResult.data.filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.log('🎉 Video downloaded successfully!');
              } else {
                this.log('❌ Failed to download video:', videoResult.error);
              }
            }
          } else {
            this.log('❌ Failed to auto-stop recording:', stopResult.error);
          }
        } catch (error) {
          this.log('❌ Error during auto-stop:', error);
        }
      }, durationSeconds * 1000);

      return startResult;
    } catch (error) {
      this.log(`Timed recording error for ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start timed recording'
      };
    }
  }

  // Debug function to test stream display fix
  async debugStreamDisplayFix(cameraId?: string): Promise<void> {
    // Use a default camera ID if none provided
    const targetCameraId = cameraId || 'WT9o3rLT3oNLpBRCzwSLoWPpwdkCZZJmjEZZjMuvD';
    
    console.log('🔧 Testing stream display fix for camera:', targetCameraId);
    
    try {
      const camera = await cameraRegistry.getCamera(targetCameraId);
      if (!camera) {
        console.error('❌ Camera not found:', targetCameraId);
        return;
      }

      // Check if it's a Jetson camera with the debug method
      if (camera.cameraType === 'jetson' && 'debugStreamDisplayFix' in camera) {
        console.log('📡 Testing Jetson camera stream display fix...');
        await (camera as any).debugStreamDisplayFix();
      } else {
        console.log('📡 Testing generic stream info...');
        const streamInfo = await this.getStreamInfo(targetCameraId);
        console.log('Stream info result:', streamInfo);
        
        if (streamInfo.success && streamInfo.data) {
          console.log('✅ Stream info retrieved successfully');
          console.log('- Is Active:', streamInfo.data.isActive);
          console.log('- Format:', streamInfo.data.format);
          
          if (streamInfo.data.isActive) {
            console.log('🎉 Stream should display properly!');
          } else {
            console.log('⚠️ Stream is not active');
          }
        } else {
          console.log('❌ Failed to get stream info:', streamInfo.error);
        }
      }
      
    } catch (error) {
      console.error('❌ Stream display fix test failed:', error);
    }
  }

  // Debug function to test timed video recording
  async debugTimedVideoRecording(cameraId?: string, duration: number = 10): Promise<void> {
    const targetCameraId = cameraId || 'WT9o3rLT3oNLpBRCzwSLoWPpwdkCZZJmjEZZjMuvD';
    
    console.log(`🎬 Testing timed video recording (${duration}s) for camera:`, targetCameraId);
    
    try {
      const result = await this.startTimedVideoRecording(targetCameraId, duration);
      
      if (result.success) {
        console.log('✅ Timed video recording started successfully!');
        console.log(`⏱️ Recording will auto-stop in ${duration} seconds and download automatically`);
        console.log('📹 You should see the recording stop and download after the timeout');
      } else {
        console.log('❌ Failed to start timed recording:', result.error);
      }
      
    } catch (error) {
      console.error('❌ Timed recording test failed:', error);
    }
  }

  // Debug function to check camera registry status
  async debugCameraRegistry(): Promise<void> {
    console.log('🔍 Checking camera registry status...');
    
    try {
      // Check what cameras are available
      const allCameras = this.getAllCameras();
      console.log('📋 All registered cameras:', allCameras);
      
      const onlineCameras = this.getOnlineCameras();
      console.log('🟢 Online cameras:', onlineCameras);
      
      // Check specific camera
      const targetCameraId = 'WT9o3rLT3oNLpBRCzwSLoWPpwdkCZZJmjEZZjMuvD';
      const hasCamera = this.hasCamera(targetCameraId);
      console.log(`🎯 Has target camera (${targetCameraId}):`, hasCamera);
      
      if (hasCamera) {
        const cameraType = this.getCameraType(targetCameraId);
        console.log('📷 Camera type:', cameraType);
        
        const camera = await this.getCamera(targetCameraId);
        console.log('🔧 Camera instance:', camera ? 'Found' : 'Not found');
      } else {
        console.log('❌ Camera not found in registry');
        console.log('💡 Trying to register camera...');
        
        // Try to register the camera manually
        try {
          cameraRegistry.registerCamera({
            cameraId: targetCameraId,
            cameraType: 'jetson',
            apiUrl: 'https://jetson.mmoment.xyz',
            name: 'Jetson Camera',
            description: 'Jetson Nano camera with AI capabilities'
          });
          console.log('✅ Camera registered successfully');
          
          // Check again
          const hasNow = this.hasCamera(targetCameraId);
          console.log('🔄 Has camera after registration:', hasNow);
        } catch (regError) {
          console.log('❌ Failed to register camera:', regError);
        }
      }
      
    } catch (error) {
      console.error('❌ Camera registry debug failed:', error);
    }
  }

  // Debug function to test direct Jetson recording (bypass session complexity)
  async debugDirectJetsonRecording(): Promise<void> {
    console.log('🎬 Testing direct Jetson recording API...');
    
    try {
      const jetsonUrl = 'https://jetson.mmoment.xyz';
      
      // Test 1: Check if we can reach the API
      console.log('1️⃣ Testing API connectivity...');
      const healthResponse = await fetch(`${jetsonUrl}/api/health`);
      console.log('Health check status:', healthResponse.status);
      
      if (healthResponse.ok) {
        const healthData = await healthResponse.json();
        console.log('✅ API is reachable:', healthData);
      }
      
      // Test 2: Try to connect/create session
      console.log('2️⃣ Testing session creation...');
      const connectResponse = await fetch(`${jetsonUrl}/api/session/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: '9gERcKdpaTNLfFNNYANzs1P73iMHJpVqhvKMKLa6Xvo'
        })
      });
      
      console.log('Connect response status:', connectResponse.status);
      
      if (connectResponse.ok) {
        const connectData = await connectResponse.json();
        console.log('✅ Session created:', connectData);
        
        // Test 3: Try to start recording
        console.log('3️⃣ Testing recording start...');
        const recordResponse = await fetch(`${jetsonUrl}/api/record`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            wallet_address: '9gERcKdpaTNLfFNNYANzs1P73iMHJpVqhvKMKLa6Xvo',
            session_id: connectData.session_id,
            duration: 5  // 5 seconds for testing
          })
        });
        
        console.log('Recording response status:', recordResponse.status);
        
        if (recordResponse.ok) {
          const recordData = await recordResponse.json();
          console.log('✅ Recording started:', recordData);
          
          // Test 4: Auto-stop after 7 seconds
          setTimeout(async () => {
            console.log('4️⃣ Testing recording stop...');
            const stopResponse = await fetch(`${jetsonUrl}/api/record`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                action: 'stop',
                wallet_address: '9gERcKdpaTNLfFNNYANzs1P73iMHJpVqhvKMKLa6Xvo',
                session_id: connectData.session_id
              })
            });
            
            console.log('Stop response status:', stopResponse.status);
            
            if (stopResponse.ok) {
              const stopData = await stopResponse.json();
              console.log('✅ Recording stopped:', stopData);
              
              if (stopData.filename) {
                console.log('5️⃣ Testing video download...');
                const videoUrl = `${jetsonUrl}/api/videos/${stopData.filename}`;
                console.log('📥 Video URL:', videoUrl);
                
                // Try to download
                const videoResponse = await fetch(videoUrl);
                if (videoResponse.ok) {
                  const videoBlob = await videoResponse.blob();
                  console.log('✅ Video downloaded:', videoBlob.size, 'bytes');
                  
                  // Trigger browser download
                  const url = URL.createObjectURL(videoBlob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = stopData.filename;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                  
                  console.log('🎉 Direct Jetson recording test SUCCESSFUL!');
                } else {
                  console.log('❌ Failed to download video:', videoResponse.status);
                }
              }
            } else {
              const stopError = await stopResponse.text();
              console.log('❌ Failed to stop recording:', stopError);
            }
          }, 7000);
          
        } else {
          const recordError = await recordResponse.text();
          console.log('❌ Failed to start recording:', recordError);
        }
      } else {
        const connectError = await connectResponse.text();
        console.log('❌ Failed to connect:', connectError);
      }
      
    } catch (error) {
      console.error('❌ Direct Jetson test failed:', error);
    }
  }
}

// Export singleton instance
export const unifiedCameraService = UnifiedCameraService.getInstance();

// Add debug functions to global scope for testing
if (typeof window !== 'undefined') {
  (window as any).debugStreamDisplayFix = () => unifiedCameraService.debugStreamDisplayFix();
  (window as any).debugTimedVideoRecording = (duration?: number) => unifiedCameraService.debugTimedVideoRecording(undefined, duration);
  (window as any).debugCameraRegistry = () => unifiedCameraService.debugCameraRegistry();
  (window as any).debugDirectJetsonRecording = () => unifiedCameraService.debugDirectJetsonRecording();
  (window as any).unifiedCameraService = unifiedCameraService;
} 