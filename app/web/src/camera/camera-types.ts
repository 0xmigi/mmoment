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

// Device signature types for DePIN authentication
export interface DeviceSignature {
  device_pubkey: string;
  signature: string;
  timestamp: number;
  version: string;
}

// Extended response types with device signatures
export interface DeviceSignedResponse {
  device_signature?: DeviceSignature;
}

// Jetson camera API types
export interface JetsonCameraSession extends DeviceSignedResponse {
  success: boolean;
  session_id: string;
  wallet_address: string;
  camera_pda: string;
  timestamp: number;
}

export interface JetsonCaptureResponse extends DeviceSignedResponse {
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

export interface JetsonRecordingResponse extends DeviceSignedResponse {
  success: boolean;
  path?: string;
  filename?: string;
  size?: number;
  recording?: boolean;
  error?: string;
}

export interface JetsonStatusResponse extends DeviceSignedResponse {
  status: string;
  camera_count: number;
  uptime: number;
  active_sessions: number;
}

export interface JetsonDeviceInfoResponse extends DeviceSignedResponse {
  device_pubkey: string;
  hardware_id: string;
  model: string;
  version: string;
}