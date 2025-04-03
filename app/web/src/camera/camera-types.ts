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