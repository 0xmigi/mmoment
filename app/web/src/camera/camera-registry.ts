/**
 * Camera Registry System
 * 
 * This system automatically detects, configures, and manages all cameras in the network.
 * It provides a single point of access for all camera operations, making the system
 * completely camera-agnostic.
 */

import { ICamera, CameraConfig, CameraRegistryEntry } from './camera-interface';
import { JetsonCamera } from './implementations/jetson-camera';
import { Pi5Camera } from './implementations/pi5-camera';
import { CONFIG } from '../core/config';

export class CameraRegistry {
  private static instance: CameraRegistry | null = null;
  private cameras: Map<string, CameraRegistryEntry> = new Map();
  private debugMode = true;

  private constructor() {
    this.log('CameraRegistry initialized');
    this.initializeKnownCameras();
  }

  public static getInstance(): CameraRegistry {
    if (!CameraRegistry.instance) {
      CameraRegistry.instance = new CameraRegistry();
    }
    return CameraRegistry.instance;
  }

  private log(...args: any[]) {
    if (this.debugMode) {
      console.log('[CameraRegistry]', ...args);
    }
  }

  /**
   * Initialize known cameras from configuration
   */
  private initializeKnownCameras() {
    // Register known cameras from config
    const knownCameras: CameraConfig[] = [
      {
        cameraId: CONFIG.JETSON_CAMERA_PDA,
        cameraType: 'jetson',
        apiUrl: CONFIG.JETSON_CAMERA_URL,
        name: 'Jetson Orin Nano Camera',
        description: 'NVIDIA Jetson Orin Nano with advanced computer vision',
        capabilities: {
          canTakePhotos: true,
          canRecordVideos: true,
          canStream: true,
          canDetectGestures: true,
          canRecognizeFaces: true,
          hasLivepeerStreaming: true,
          supportedStreamFormats: ['livepeer', 'mjpeg']
        }
      },
      {
        cameraId: CONFIG.CAMERA_PDA,
        cameraType: 'pi5',
        apiUrl: CONFIG.CAMERA_API_URL,
        name: 'Raspberry Pi 5 Camera',
        description: 'Raspberry Pi 5 with camera module',
        capabilities: {
          canTakePhotos: true,
          canRecordVideos: true,
          canStream: true,
          canDetectGestures: false,
          canRecognizeFaces: false,
          hasLivepeerStreaming: false,
          supportedStreamFormats: ['mjpeg']
        }
      }
    ];

    for (const config of knownCameras) {
      this.registerCamera(config);
    }

    this.log(`Registered ${knownCameras.length} known cameras`);
  }

  /**
   * Register a new camera in the registry
   */
  public registerCamera(config: CameraConfig): void {
    const entry: CameraRegistryEntry = {
      ...config,
      isOnline: false,
      lastSeen: 0,
      instance: undefined
    };

    this.cameras.set(config.cameraId, entry);
    this.log(`Registered camera: ${config.cameraId} (${config.cameraType})`);
  }

  /**
   * Get a camera instance by ID
   */
  public async getCamera(cameraId: string): Promise<ICamera | null> {
    const entry = this.cameras.get(cameraId);
    if (!entry) {
      this.log(`Camera not found: ${cameraId}`);
      return null;
    }

    // Create instance if it doesn't exist
    if (!entry.instance) {
      entry.instance = this.createCameraInstance(entry);
      if (!entry.instance) {
        this.log(`Failed to create camera instance: ${cameraId}`);
        return null;
      }
    }

    return entry.instance;
  }

  /**
   * Create a camera instance based on its type
   */
  private createCameraInstance(entry: CameraRegistryEntry): ICamera | undefined {
    try {
      switch (entry.cameraType.toLowerCase()) {
        case 'jetson':
          return new JetsonCamera(entry.cameraId, entry.apiUrl);
        case 'pi5':
          return new Pi5Camera(entry.cameraId, entry.apiUrl);
        default:
          this.log(`Unknown camera type: ${entry.cameraType}`);
          return undefined;
      }
    } catch (error) {
      this.log(`Error creating camera instance:`, error);
      return undefined;
    }
  }

  /**
   * Get all registered cameras
   */
  public getAllCameras(): CameraRegistryEntry[] {
    return Array.from(this.cameras.values());
  }

  /**
   * Get online cameras only
   */
  public getOnlineCameras(): CameraRegistryEntry[] {
    return this.getAllCameras().filter(camera => camera.isOnline);
  }

  /**
   * Check if a camera exists in the registry
   */
  public hasCamera(cameraId: string): boolean {
    return this.cameras.has(cameraId);
  }

  /**
   * Get camera configuration
   */
  public getCameraConfig(cameraId: string): CameraConfig | null {
    const entry = this.cameras.get(cameraId);
    if (!entry) return null;

    return {
      cameraId: entry.cameraId,
      cameraType: entry.cameraType,
      apiUrl: entry.apiUrl,
      name: entry.name,
      description: entry.description,
      capabilities: entry.capabilities,
      config: entry.config
    };
  }

  /**
   * Update camera online status
   */
  public updateCameraStatus(cameraId: string, isOnline: boolean): void {
    const entry = this.cameras.get(cameraId);
    if (entry) {
      entry.isOnline = isOnline;
      entry.lastSeen = Date.now();
      this.log(`Updated camera status: ${cameraId} -> ${isOnline ? 'online' : 'offline'}`);
    }
  }

  /**
   * Health check all registered cameras
   */
  public async healthCheckAll(): Promise<Map<string, boolean>> {
    const results = new Map<string, boolean>();
    const cameras = this.getAllCameras();

    this.log(`Running health check on ${cameras.length} cameras...`);

    const healthChecks = cameras.map(async (entry) => {
      try {
        const camera = await this.getCamera(entry.cameraId);
        if (!camera) {
          results.set(entry.cameraId, false);
          this.updateCameraStatus(entry.cameraId, false);
          return;
        }

        const response = await camera.testConnection();
        const isHealthy = response.success;
        
        results.set(entry.cameraId, isHealthy);
        this.updateCameraStatus(entry.cameraId, isHealthy);
        
        this.log(`Health check ${entry.cameraId}: ${isHealthy ? 'PASS' : 'FAIL'}`);
      } catch (error) {
        this.log(`Health check error for ${entry.cameraId}:`, error);
        results.set(entry.cameraId, false);
        this.updateCameraStatus(entry.cameraId, false);
      }
    });

    await Promise.all(healthChecks);
    return results;
  }

  /**
   * Auto-discover cameras on the network
   * This could be extended to scan for cameras automatically
   */
  public async discoverCameras(): Promise<CameraConfig[]> {
    this.log('Auto-discovery not yet implemented');
    // TODO: Implement network scanning for cameras
    // This could scan common ports, check for camera APIs, etc.
    return [];
  }

  /**
   * Remove a camera from the registry
   */
  public removeCamera(cameraId: string): boolean {
    const entry = this.cameras.get(cameraId);
    if (entry && entry.instance) {
      // Disconnect the camera if it's connected
      entry.instance.disconnect().catch(err => 
        this.log(`Error disconnecting camera during removal:`, err)
      );
    }

    const removed = this.cameras.delete(cameraId);
    if (removed) {
      this.log(`Removed camera: ${cameraId}`);
    }
    return removed;
  }

  /**
   * Get camera type by ID
   */
  public getCameraType(cameraId: string): string | null {
    const entry = this.cameras.get(cameraId);
    return entry ? entry.cameraType : null;
  }

  /**
   * Check if camera supports a specific capability
   */
  public cameraSupports(cameraId: string, capability: keyof import('./camera-interface').CameraCapabilities): boolean {
    const entry = this.cameras.get(cameraId);
    if (!entry || !entry.capabilities) return false;
    
    return entry.capabilities[capability] === true;
  }
}

// Export singleton instance
export const cameraRegistry = CameraRegistry.getInstance(); 