/**
 * Standardized Camera Interface
 * 
 * This interface defines the contract that all camera implementations must follow.
 * This ensures that any camera type (Jetson, Pi5, future cameras) can be used
 * interchangeably without changing the frontend code.
 */

export interface CameraCapabilities {
  canTakePhotos: boolean;
  canRecordVideos: boolean;
  canStream: boolean;
  canDetectGestures: boolean;
  canRecognizeFaces: boolean;
  hasLivepeerStreaming: boolean;
  supportedStreamFormats: string[];
}

export interface CameraStatus {
  isOnline: boolean;
  isStreaming: boolean;
  isRecording: boolean;
  lastSeen: number;
  owner?: string;
  error?: string;
}

export interface CameraStreamInfo {
  isActive: boolean;
  streamUrl?: string;
  playbackId?: string;
  streamKey?: string;
  hlsUrl?: string;
  format: 'mjpeg' | 'livepeer' | 'webrtc';
}

export interface CameraActionResponse<T = any> {
  success: boolean;
  error?: string;
  data?: T;
}

export interface CameraMediaResponse {
  blob?: Blob;
  filename?: string;
  path?: string;
  timestamp?: number;
  size?: number;
  width?: number;
  height?: number;
}

export interface CameraGestureResponse {
  gesture?: string;
  confidence?: number;
  timestamp?: number;
}

export interface CameraSession {
  sessionId: string;
  walletAddress: string;
  cameraPda: string;
  timestamp: number;
  isActive: boolean;
}

/**
 * Base Camera Interface
 * All camera implementations must implement this interface
 */
export interface ICamera {
  // Camera identification
  readonly cameraId: string;
  readonly cameraType: string;
  readonly apiUrl: string;
  
  // Camera capabilities
  getCapabilities(): CameraCapabilities;
  
  // Connection management
  connect(walletAddress?: string): Promise<CameraActionResponse<CameraSession>>;
  disconnect(): Promise<CameraActionResponse>;
  testConnection(): Promise<CameraActionResponse<{ message: string; url: string }>>;
  isConnected(): boolean;
  
  // Status monitoring
  getStatus(): Promise<CameraActionResponse<CameraStatus>>;
  
  // Media capture
  takePhoto(): Promise<CameraActionResponse<CameraMediaResponse>>;
  startVideoRecording(): Promise<CameraActionResponse<CameraMediaResponse>>;
  stopVideoRecording(): Promise<CameraActionResponse<CameraMediaResponse>>;
  getRecordedVideo(filename: string): Promise<CameraActionResponse<CameraMediaResponse>>;
  getMostRecentVideo?(): Promise<CameraActionResponse<CameraMediaResponse>>;
  listVideos(): Promise<CameraActionResponse<{ videos: any[] }>>;
  
  // Streaming
  startStream(): Promise<CameraActionResponse<CameraStreamInfo>>;
  stopStream(): Promise<CameraActionResponse<CameraStreamInfo>>;
  getStreamInfo(): Promise<CameraActionResponse<CameraStreamInfo>>;
  getStreamUrl(): string;
  
  // Advanced features (optional - cameras can return not supported)
  getCurrentGesture?(): Promise<CameraActionResponse<CameraGestureResponse>>;
  toggleGestureControls?(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>>;
  toggleFaceVisualization?(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>>;
  toggleGestureVisualization?(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>>;
  togglePoseVisualization?(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>>;
  checkForGestureTrigger?(): Promise<{
    shouldCapture: boolean;
    gestureType: 'photo' | 'video' | null;
    gesture?: string;
    confidence?: number;
  }>;
  getGestureControlsStatus?(): boolean;
  
  // Face enrollment (Jetson-specific)
  enrollFace?(walletAddress: string): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string }>>;
  
  // Face enrollment transaction flow (two-phase for user wallet payment)
  prepareFaceEnrollmentTransaction?(walletAddress: string): Promise<CameraActionResponse<{ transactionBuffer: string; faceId: string; metadata?: any }>>;
  confirmFaceEnrollmentTransaction?(walletAddress: string, confirmationData: { signedTransaction: string; faceId: string; biometricSessionId?: string }): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string; transactionId?: string }>>;

  // Face recognition
  recognizeFaces?(): Promise<CameraActionResponse<{ recognized_data: Record<string, any> }>>;

  // CV Apps (Jetson-specific)
  loadApp?(appName: string): Promise<CameraActionResponse<{ message: string }>>;
  activateApp?(appName: string): Promise<CameraActionResponse<{ active_app: string }>>;
  deactivateApp?(): Promise<CameraActionResponse>;
  getAppStatus?(): Promise<CameraActionResponse<{ active_app: string | null; loaded_apps: string[]; state: any }>>;

  // Competition support (for CompetitionApp types)
  startCompetition?(competitors: Array<{ wallet_address: string; display_name: string }>, durationLimit?: number): Promise<CameraActionResponse<{ message: string }>>;
  endCompetition?(): Promise<CameraActionResponse<{ result: any }>>;

  // Session management
  getCurrentSession(): CameraSession | null;

  // Recording state
  isCurrentlyRecording(): Promise<boolean> | boolean;
}

/**
 * Camera Configuration
 * Defines how to connect to and identify a camera
 */
export interface CameraConfig {
  cameraId: string;
  cameraType: string;
  apiUrl: string;
  name?: string;
  description?: string;
  capabilities?: Partial<CameraCapabilities>;
  // Additional camera-specific configuration
  config?: Record<string, any>;
}

/**
 * Camera Registry Entry
 * Used by the camera registry to track available cameras
 */
export interface CameraRegistryEntry extends CameraConfig {
  isOnline: boolean;
  lastSeen: number;
  instance?: ICamera;
} 