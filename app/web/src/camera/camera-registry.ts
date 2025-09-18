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
   * Initialize known cameras from configuration using PDA-based URLs
   */
  private initializeKnownCameras() {
    // Register known cameras using PDA-based URL system
    const knownCameras: CameraConfig[] = [
      {
        cameraId: CONFIG.JETSON_CAMERA_PDA,
        cameraType: 'jetson',
        apiUrl: CONFIG.getCameraApiUrlByPda(CONFIG.JETSON_CAMERA_PDA),
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
        },
        // Store legacy URL for fallback
        config: {
          legacyUrl: (CONFIG.KNOWN_CAMERAS as any)[CONFIG.JETSON_CAMERA_PDA]?.legacyUrl
        }
      },
      {
        cameraId: CONFIG.CAMERA_PDA,
        cameraType: 'pi5',
        apiUrl: CONFIG.getCameraApiUrlByPda(CONFIG.CAMERA_PDA),
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
        },
        config: {
          legacyUrl: (CONFIG.KNOWN_CAMERAS as any)[CONFIG.CAMERA_PDA]?.legacyUrl
        }
      }
    ];

    for (const config of knownCameras) {
      this.registerCamera(config);
    }

    this.log(`Registered ${knownCameras.length} known cameras with PDA-based URLs`);
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
    this.log(`Registered camera: ${config.cameraId} (${config.cameraType}) at ${config.apiUrl}`);
  }

  /**
   * Discover and register a camera by PDA
   * This method attempts to connect to a camera using its PDA-based URL
   */
  public async discoverCameraByPda(cameraPda: string): Promise<CameraConfig | null> {
    try {
      this.log(`Attempting to discover camera: ${cameraPda}`);
      
      // Generate PDA-based URL
      const apiUrl = CONFIG.getCameraApiUrlByPda(cameraPda);
      
      // Try to get camera info from the /api/camera/info endpoint
      const response = await fetch(`${apiUrl}/api/camera/info`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        credentials: 'omit'
      });

      if (!response.ok) {
        this.log(`Failed to discover camera ${cameraPda}: HTTP ${response.status}`);
        return null;
      }

      const cameraInfo = await response.json();
      
      // Determine camera type based on capabilities if not provided
      let cameraType = cameraInfo.type;
      if (!cameraType || cameraType === 'unknown') {
        // Auto-detect camera type based on capabilities
        const caps = cameraInfo.capabilities;
        if (caps && caps.face_recognition && caps.gesture_detection && caps.livepeer_streaming) {
          cameraType = 'jetson';
        } else if (caps && caps.media_capture && !caps.face_recognition) {
          cameraType = 'pi5';
        } else {
          // Default to jetson for PDA-based cameras with advanced capabilities
          cameraType = 'jetson';
        }
      }
      
      // Create camera configuration from discovered info
      const config: CameraConfig = {
        cameraId: cameraPda,
        cameraType: cameraType,
        apiUrl: apiUrl,
        name: cameraInfo.name || `Camera ${cameraPda.slice(0, 8)}`,
        description: cameraInfo.description || 'Discovered camera',
        capabilities: {
          canTakePhotos: cameraInfo.capabilities?.media_capture || true,
          canRecordVideos: cameraInfo.capabilities?.media_capture || true,
          canStream: true,
          canDetectGestures: cameraInfo.capabilities?.gesture_detection || false,
          canRecognizeFaces: cameraInfo.capabilities?.face_recognition || false,
          hasLivepeerStreaming: cameraInfo.capabilities?.livepeer_streaming || false,
          supportedStreamFormats: cameraInfo.capabilities?.livepeer_streaming ? ['livepeer', 'mjpeg'] : ['mjpeg']
        }
      };

      // Register the discovered camera
      this.registerCamera(config);
      
      this.log(`Successfully discovered and registered camera: ${cameraPda}`);
      return config;
      
    } catch (error) {
      this.log(`Error discovering camera ${cameraPda}:`, error);
      return null;
    }
  }

  /**
   * Get a camera instance by ID
   */
  public async getCamera(cameraId: string): Promise<ICamera | null> {
    let entry = this.cameras.get(cameraId);
    
    // If camera not found, try to discover it
    if (!entry) {
      this.log(`Camera ${cameraId} not found in registry, attempting discovery...`);
      const discoveredConfig = await this.discoverCameraByPda(cameraId);
      if (discoveredConfig) {
        entry = this.cameras.get(cameraId);
      }
    }
    
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
   * Create a camera instance based on its type with fallback support
   */
  private createCameraInstance(entry: CameraRegistryEntry): ICamera | undefined {
    try {
      // Use PDA-based URL by default, with legacy URL as fallback
      const apiUrl = entry.apiUrl;
      const legacyUrl = entry.config?.legacyUrl;
      
      switch (entry.cameraType.toLowerCase()) {
        case 'jetson':
          // For Jetson cameras, try PDA-based URL first, then legacy
          const jetsonCamera = new JetsonCamera(entry.cameraId, apiUrl);
          
          // Store legacy URL for potential fallback
          if (legacyUrl) {
            (jetsonCamera as any).legacyUrl = legacyUrl;
          }
          
          return jetsonCamera;
          
        case 'pi5':
          return new Pi5Camera(entry.cameraId, apiUrl);
          
        default:
          // For unknown camera types, try to determine based on PDA or capabilities
          this.log(`Unknown camera type: ${entry.cameraType}, attempting auto-detection`);
          
          // Default to Jetson for unknown types since they're likely PDA-based advanced cameras
          return new JetsonCamera(entry.cameraId, apiUrl);
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

    // Check all cameras in parallel
    const promises = cameras.map(async (entry) => {
      try {
        // Try multiple health check endpoints in order of preference
        const healthEndpoints = ['/api/health', '/api/status', '/api/camera/info'];
        let isHealthy = false;
        
        for (const endpoint of healthEndpoints) {
          try {
            const response = await fetch(`${entry.apiUrl}${endpoint}`, {
              method: 'GET',
              timeout: 5000,
              mode: 'cors',
              credentials: 'omit'
            } as any);
            
            if (response.ok) {
              isHealthy = true;
              break; // Stop trying once we get a successful response
            }
          } catch (endpointError) {
            // Continue to next endpoint
            continue;
          }
        }
        
        results.set(entry.cameraId, isHealthy);
        this.updateCameraStatus(entry.cameraId, isHealthy);
        
        // If PDA-based URL fails and we have a legacy URL, try that
        if (!isHealthy && entry.config?.legacyUrl) {
          for (const endpoint of healthEndpoints) {
            try {
              const legacyResponse = await fetch(`${entry.config.legacyUrl}${endpoint}`, {
                method: 'GET',
                timeout: 5000,
                mode: 'cors',
                credentials: 'omit'
              } as any);
              
              if (legacyResponse.ok) {
                this.log(`Camera ${entry.cameraId} responded to legacy URL, updating API URL`);
                entry.apiUrl = entry.config.legacyUrl;
                results.set(entry.cameraId, true);
                this.updateCameraStatus(entry.cameraId, true);
                isHealthy = true;
                break;
              }
            } catch (legacyError) {
              // Continue to next endpoint
              continue;
            }
          }
          
          if (!isHealthy) {
            this.log(`Both PDA and legacy URLs failed for ${entry.cameraId}`);
          }
        }
        
      } catch (error) {
        this.log(`Health check failed for ${entry.cameraId}:`, error);
        results.set(entry.cameraId, false);
        this.updateCameraStatus(entry.cameraId, false);
      }
    });

    await Promise.all(promises);
    
    const onlineCount = Array.from(results.values()).filter(Boolean).length;
    this.log(`Health check complete: ${onlineCount}/${cameras.length} cameras online`);
    
    return results;
  }

  /**
   * Discover cameras from the network
   * This method can be extended to discover cameras from various sources
   */
  public async discoverCameras(): Promise<CameraConfig[]> {
    const discovered: CameraConfig[] = [];
    
    // For now, we only discover known cameras, but this can be extended
    // to discover cameras from blockchain, DNS, or other sources
    this.log('Camera discovery not yet implemented for network scanning');
    
    return discovered;
  }

  /**
   * Remove a camera from the registry
   */
  public removeCamera(cameraId: string): boolean {
    const entry = this.cameras.get(cameraId);
    if (entry) {
      // Disconnect the camera instance if it exists
      if (entry.instance) {
        try {
          entry.instance.disconnect();
        } catch (error) {
          this.log(`Error disconnecting camera ${cameraId} during removal:`, error);
        }
      }
      
      this.cameras.delete(cameraId);
      this.log(`Removed camera: ${cameraId}`);
      return true;
    }
    return false;
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
    return entry && entry.capabilities ? !!entry.capabilities[capability] : false;
  }
}

// Export singleton instance
export const cameraRegistry = CameraRegistry.getInstance(); 