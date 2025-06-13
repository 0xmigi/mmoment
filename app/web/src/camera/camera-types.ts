// src/types/camera.ts

export interface CameraState {
    preview?: string;
    activeUsers: number;
    recentActivity: number;
    isActive: boolean;
  }
  
  export interface Event {
    timestamp: string;
    signature: string;
    user: string | undefined;
    cameraAccount: string | undefined;
  }

// Jetson camera API types
export interface JetsonCameraSession {
  success: boolean;
  session_id: string;
  wallet_address: string;
  camera_pda: string;
  timestamp: number;
}

export interface JetsonCaptureResponse {
  success: boolean;
  path?: string;
  filename?: string;
  timestamp?: number;
  size?: number;
  width?: number;
  height?: number;
  image_data?: string;
  error?: string;
}

export interface JetsonRecordingResponse {
  success: boolean;
  path?: string;
  filename?: string;
  size?: number;
  recording?: boolean;
  error?: string;
}