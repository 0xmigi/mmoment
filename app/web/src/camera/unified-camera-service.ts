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
import { CONFIG } from '../core/config';

export class UnifiedCameraService {
  private static instance: UnifiedCameraService | null = null;
  private debugMode = true;

  // Cache for getComprehensiveState to prevent duplicate concurrent requests
  private comprehensiveStateCache = new Map<string, {
    promise: Promise<any>;
    timestamp: number;
  }>();
  private readonly COMPREHENSIVE_STATE_CACHE_TTL = 2000; // 2 seconds cache

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

  private processComprehensiveState(statusResponse: any, streamResponse: any, cameraId: string) {
    // Check consistency between status and stream info
    const statusStreaming = statusResponse.success ? statusResponse.data?.isStreaming || false : false;
    const streamActive = streamResponse.success ? streamResponse.data?.isActive || false : false;
    const isConsistent = statusStreaming === streamActive;

    if (!isConsistent) {
      this.log(`⚠️ State inconsistency detected for ${cameraId}:`, {
        statusStreaming,
        streamActive
      });
    }

    return {
      status: statusResponse,
      streamInfo: streamResponse,
      isConsistent
    };
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
   * Get camera API URL by ID (for direct API calls)
   */
  public getCameraApiUrl(cameraId: string): string | null {
    if (!cameraId) return null;
    try {
      return CONFIG.getCameraApiUrlByPda(cameraId);
    } catch {
      return null;
    }
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
   * Set session locally without making an API call.
   * Used when session was created via /api/checkin (Phase 3) to set the
   * camera's currentSession without calling /api/session/connect.
   */
  public async setSession(cameraId: string, session: CameraSession): Promise<boolean> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`Cannot set session - camera not found: ${cameraId}`);
        return false;
      }

      if (camera.setSession) {
        camera.setSession(session);
        this.log(`Session set for ${cameraId}:`, session.sessionId);
        return true;
      } else {
        this.log(`Camera ${cameraId} does not support setSession`);
        return false;
      }
    } catch (error) {
      this.log(`Error setting session for ${cameraId}:`, error);
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
   * Toggle gesture visualization (if supported)
   */
  public async toggleGestureVisualization(cameraId: string, enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.toggleGestureVisualization) {
        return {
          success: false,
          error: 'Gesture visualization not supported by this camera'
        };
      }

      return await camera.toggleGestureVisualization(enabled);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle gesture visualization'
      };
    }
  }

  /**
   * Toggle pose visualization (if supported)
   */
  public async togglePoseVisualization(cameraId: string, enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.togglePoseVisualization) {
        return {
          success: false,
          error: 'Pose visualization not supported by this camera'
        };
      }

      return await camera.togglePoseVisualization(enabled);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle pose visualization'
      };
    }
  }

  /**
   * Enroll face for recognition (if supported)
   */
  public async enrollFace(cameraId: string, walletAddress: string): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.enrollFace) {
        return {
          success: false,
          error: 'Face enrollment not supported by this camera'
        };
      }

      return await camera.enrollFace(walletAddress);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to enroll face'
      };
    }
  }

  /**
   * Prepare face enrollment transaction (Phase 1 of two-phase flow)
   */
  public async prepareFaceEnrollmentTransaction(cameraId: string, walletAddress: string): Promise<CameraActionResponse<{ transactionBuffer: string; faceId: string; metadata?: any }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.prepareFaceEnrollmentTransaction) {
        return {
          success: false,
          error: 'Face enrollment transaction preparation not supported by this camera'
        };
      }

      return await camera.prepareFaceEnrollmentTransaction(walletAddress);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to prepare face enrollment transaction'
      };
    }
  }

  /**
   * Confirm face enrollment transaction (Phase 2 of two-phase flow)
   */
  public async confirmFaceEnrollmentTransaction(
    cameraId: string, 
    walletAddress: string, 
    confirmationData: { signedTransaction: string; faceId: string; biometricSessionId?: string }
  ): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string; transactionId?: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.confirmFaceEnrollmentTransaction) {
        return {
          success: false,
          error: 'Face enrollment transaction confirmation not supported by this camera'
        };
      }

      return await camera.confirmFaceEnrollmentTransaction(walletAddress, confirmationData);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to confirm face enrollment transaction'
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
   * Send user profile to camera for display name labeling using new PDA-based API
   */
  public async sendUserProfile(cameraId: string, profile: {
    wallet_address: string;
    display_name?: string;
    username?: string;
  }): Promise<CameraActionResponse> {
    try {
      this.log(`[DEBUG] Sending user profile to camera: ${cameraId}`, profile);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[ERROR] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const apiUrl = (camera as any).apiUrl;
      this.log(`[DEBUG] Camera API URL: ${apiUrl}`);

      // Use the new /api/user/profile endpoint for sending profile data
      const url = `${apiUrl}/api/user/profile`;
      this.log(`[DEBUG] Making request to: ${url}`);
      
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify(profile)
      });

      this.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        this.log(`[ERROR] HTTP error response: ${errorText}`);
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      this.log(`[SUCCESS] Profile sent successfully to ${cameraId}`, result);
      
      return {
        success: true,
        data: result
      };
    } catch (error) {
      this.log(`[ERROR] Error sending profile to ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to send profile'
      };
    }
  }

  /**
   * Check-in to camera with Ed25519 cryptographic handshake
   *
   * PHASE 3 PRIVACY ARCHITECTURE:
   * Check-in requires a cryptographic signature to prove wallet ownership.
   * This prevents impersonation and ensures the camera knows FOR CERTAIN who is checked in.
   *
   * @param cameraId - Camera PDA
   * @param profile.wallet_address - User's wallet address (base58)
   * @param profile.request_signature - Ed25519 signature (base58 encoded)
   * @param profile.request_timestamp - Unix timestamp in ms when signed
   * @param profile.request_nonce - Random UUID for replay protection
   * @param profile.display_name - Optional display name for timeline
   * @param profile.username - Optional username for timeline
   */
  public async checkin(cameraId: string, profile: {
    wallet_address: string;
    request_signature: string;    // Required: Ed25519 signature (base58)
    request_timestamp: number;    // Required: Unix ms timestamp
    request_nonce: string;        // Required: UUID for replay protection
    display_name?: string;
    username?: string;
  }): Promise<CameraActionResponse<{
    wallet_address: string;
    display_name: string;
    session_id: string;
    camera_pda: string;
    camera_url: string;
    message: string;
  }>> {
    try {
      this.log(`[DEBUG] Unified check-in to camera: ${cameraId}`, profile);

      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[ERROR] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const apiUrl = (camera as any).apiUrl;
      const url = `${apiUrl}/api/checkin`;
      this.log(`[DEBUG] Making unified check-in request to: ${url}`);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify(profile)
      });

      this.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        this.log(`[ERROR] HTTP error response: ${errorText}`);
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      this.log(`[SUCCESS] Unified check-in successful for ${cameraId}`, result);

      return {
        success: true,
        data: result
      };
    } catch (error) {
      this.log(`[ERROR] Error during unified check-in to ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unified check-in failed'
      };
    }
  }

  /**
   * Notify Jetson about checkout - passes transaction signature for Solscan link
   *
   * @param cameraId - Camera PDA
   * @param data - Checkout data with wallet address and transaction signature
   */
  public async checkout(cameraId: string, data: {
    wallet_address: string;
    transaction_signature: string;
  }): Promise<CameraActionResponse<{
    wallet_address: string;
    session_id: string;
    message: string;
  }>> {
    try {
      this.log(`[DEBUG] Checkout notification to camera: ${cameraId}`, data);

      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[ERROR] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const apiUrl = (camera as any).apiUrl;
      const url = `${apiUrl}/api/checkout`;
      this.log(`[DEBUG] Making checkout notification request to: ${url}`);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify(data)
      });

      this.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        this.log(`[ERROR] HTTP error response: ${errorText}`);
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      this.log(`[SUCCESS] Checkout notification successful for ${cameraId}`, result);

      return {
        success: true,
        data: result
      };
    } catch (error) {
      this.log(`[ERROR] Error during checkout notification to ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Checkout notification failed'
      };
    }
  }

  /**
   * Check if a user is currently checked in at a camera
   *
   * PHASE 3 PRIVACY ARCHITECTURE:
   * Queries the Jetson directly (source of truth) to check session status.
   * This is consistent with how activeSessionCount already works.
   *
   * @param cameraId - Camera PDA
   * @param walletAddress - User's wallet address to check
   * @returns { isCheckedIn: boolean, activeSessionCount: number }
   */
  public async getSessionStatus(cameraId: string, walletAddress: string): Promise<CameraActionResponse<{
    isCheckedIn: boolean;
    activeSessionCount: number;
  }>> {
    try {
      this.log(`[SESSION-STATUS] Checking session for ${walletAddress.slice(0, 8)}... at camera ${cameraId.slice(0, 8)}...`);

      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[SESSION-STATUS] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: { isCheckedIn: false, activeSessionCount: 0 }
        };
      }

      const apiUrl = (camera as any).apiUrl;
      const url = `${apiUrl}/api/session/status/${walletAddress}`;
      this.log(`[SESSION-STATUS] Querying: ${url}`);

      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit'
      });

      if (!response.ok) {
        const errorText = await response.text();
        this.log(`[SESSION-STATUS] Error: ${errorText}`);
        return {
          success: false,
          error: `HTTP ${response.status}: ${errorText}`,
          data: { isCheckedIn: false, activeSessionCount: 0 }
        };
      }

      const result = await response.json();
      this.log(`[SESSION-STATUS] Result: isCheckedIn=${result.isCheckedIn}, activeCount=${result.activeSessionCount}`);

      return {
        success: true,
        data: {
          isCheckedIn: result.isCheckedIn,
          activeSessionCount: result.activeSessionCount
        }
      };
    } catch (error) {
      this.log(`[SESSION-STATUS] Error:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to check session status',
        data: { isCheckedIn: false, activeSessionCount: 0 }
      };
    }
  }

  /**
   * Get user profile from camera
   */
  public async getUserProfile(cameraId: string, walletAddress: string): Promise<CameraActionResponse<{
    wallet_address: string;
    display_name?: string;
    username?: string;
  }>> {
    try {
      this.log(`[DEBUG] Getting user profile from camera: ${cameraId} for wallet: ${walletAddress}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[ERROR] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const apiUrl = (camera as any).apiUrl;
      
      // Use the new /api/user/profile/<wallet_address> endpoint
      const url = `${apiUrl}/api/user/profile/${walletAddress}`;
      this.log(`[DEBUG] Making request to: ${url}`);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit'
      });

      this.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) {
        if (response.status === 404) {
          return {
            success: false,
            error: 'User profile not found'
          };
        }
        const errorText = await response.text();
        this.log(`[ERROR] HTTP error response: ${errorText}`);
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      this.log(`[SUCCESS] Profile retrieved successfully from ${cameraId}`, result);
      
      return {
        success: true,
        data: result
      };
    } catch (error) {
      this.log(`[ERROR] Error getting profile from ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get profile'
      };
    }
  }

  /**
   * Remove user profile from camera
   */
  public async removeUserProfile(cameraId: string, walletAddress: string): Promise<CameraActionResponse> {
    try {
      this.log(`[DEBUG] Removing user profile from camera: ${cameraId} for wallet: ${walletAddress}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        this.log(`[ERROR] Camera not found: ${cameraId}`);
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      const apiUrl = (camera as any).apiUrl;
      
      // Use the new /api/user/profile/<wallet_address> endpoint with DELETE method
      const url = `${apiUrl}/api/user/profile/${walletAddress}`;
      this.log(`[DEBUG] Making DELETE request to: ${url}`);
      
      const response = await fetch(url, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit'
      });

      this.log(`[DEBUG] Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        this.log(`[ERROR] HTTP error response: ${errorText}`);
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      this.log(`[SUCCESS] Profile removed successfully from ${cameraId}`, result);
      
      return {
        success: true,
        data: result
      };
    } catch (error) {
      this.log(`[ERROR] Error removing profile from ${cameraId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to remove profile'
      };
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

  /**
   * Get comprehensive camera state (status + stream info)
   * This is used for state reconciliation to ensure UI matches hardware reality
   */
  public async getComprehensiveState(cameraId: string): Promise<{
    status: CameraActionResponse<CameraStatus>;
    streamInfo: CameraActionResponse<CameraStreamInfo>;
    isConsistent: boolean;
  }> {
    try {
      this.log(`Getting comprehensive state for camera: ${cameraId}`);
      
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        const errorResponse = {
          success: false,
          error: `Camera not found: ${cameraId}`,
          data: {
            isOnline: false,
            isStreaming: false,
            isRecording: false,
            lastSeen: Date.now()
          }
        };
        
        return {
          status: errorResponse,
          streamInfo: {
            success: false,
            error: `Camera not found: ${cameraId}`,
            data: { isActive: false, streamUrl: '', playbackId: '', format: 'mjpeg' as const }
          },
          isConsistent: false
        };
      }

      // Check cache first to prevent duplicate concurrent requests
      const now = Date.now();
      const cached = this.comprehensiveStateCache.get(cameraId);

      if (cached && (now - cached.timestamp) < this.COMPREHENSIVE_STATE_CACHE_TTL) {
        this.log(`Using cached comprehensive state for ${cameraId}`);
        return cached.promise;
      }

      // Create new request and cache it
      const fetchPromise = (async () => {
        // Get both status and stream info in parallel
        const [statusResponse, streamResponse] = await Promise.all([
          camera.getStatus(),
          camera.getStreamInfo()
        ]);

        return { statusResponse, streamResponse };
      })();

      // Cache the promise
      this.comprehensiveStateCache.set(cameraId, {
        promise: fetchPromise.then(({ statusResponse, streamResponse }) => {
          // Process and return
          const result = this.processComprehensiveState(statusResponse, streamResponse, cameraId);

          // Clear cache after TTL
          setTimeout(() => {
            this.comprehensiveStateCache.delete(cameraId);
          }, this.COMPREHENSIVE_STATE_CACHE_TTL);

          return result;
        }),
        timestamp: now
      });

      const { statusResponse, streamResponse } = await fetchPromise;

      return this.processComprehensiveState(statusResponse, streamResponse, cameraId);
    } catch (error) {
      this.log(`Error getting comprehensive state for ${cameraId}:`, error);
      
      const errorResponse = {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get comprehensive state',
        data: {
          isOnline: false,
          isStreaming: false,
          isRecording: false,
          lastSeen: Date.now()
        }
      };

      return {
        status: errorResponse,
        streamInfo: {
          success: false,
          error: error instanceof Error ? error.message : 'Failed to get stream info',
          data: { isActive: false, streamUrl: '', playbackId: '', format: 'mjpeg' as const }
        },
        isConsistent: false
      };
    }
  }

  /**
   * Recognize faces in current frame (if supported)
   */
  public async recognizeFaces(cameraId: string): Promise<CameraActionResponse<{ recognized_data: Record<string, any> }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.recognizeFaces) {
        return {
          success: false,
          error: 'Face recognition not supported by this camera'
        };
      }

      return await camera.recognizeFaces();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to recognize faces'
      };
    }
  }

  /**
   * Load a CV app (if supported)
   */
  public async loadApp(cameraId: string, appName: string): Promise<CameraActionResponse<{ message: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.loadApp) {
        return {
          success: false,
          error: 'CV apps not supported by this camera'
        };
      }

      return await camera.loadApp(appName);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to load app'
      };
    }
  }

  /**
   * Activate a CV app (if supported)
   */
  public async activateApp(cameraId: string, appName: string): Promise<CameraActionResponse<{ active_app: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.activateApp) {
        return {
          success: false,
          error: 'CV apps not supported by this camera'
        };
      }

      return await camera.activateApp(appName);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to activate app'
      };
    }
  }

  /**
   * Deactivate current CV app (if supported)
   */
  public async deactivateApp(cameraId: string): Promise<CameraActionResponse> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.deactivateApp) {
        return {
          success: false,
          error: 'CV apps not supported by this camera'
        };
      }

      return await camera.deactivateApp();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to deactivate app'
      };
    }
  }

  /**
   * Get CV app status (if supported)
   */
  public async getAppStatus(cameraId: string): Promise<CameraActionResponse<{ active_app: string | null; loaded_apps: string[]; state: any }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.getAppStatus) {
        return {
          success: false,
          error: 'CV apps not supported by this camera'
        };
      }

      return await camera.getAppStatus();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get app status'
      };
    }
  }

  /**
   * Start a competition (if supported)
   */
  public async startCompetition(
    cameraId: string,
    competitors: Array<{ wallet_address: string; display_name: string }>,
    durationLimit?: number
  ): Promise<CameraActionResponse<{ message: string }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.startCompetition) {
        return {
          success: false,
          error: 'Competitions not supported by this camera'
        };
      }

      return await camera.startCompetition(competitors, durationLimit);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start competition'
      };
    }
  }

  /**
   * End a competition (if supported)
   */
  public async endCompetition(cameraId: string): Promise<CameraActionResponse<{ result: any }>> {
    try {
      const camera = await this.getCamera(cameraId);
      if (!camera) {
        return {
          success: false,
          error: `Camera not found: ${cameraId}`
        };
      }

      if (!camera.endCompetition) {
        return {
          success: false,
          error: 'Competitions not supported by this camera'
        };
      }

      return await camera.endCompetition();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to end competition'
      };
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
  (window as any).unifiedCameraService = unifiedCameraService;
} 