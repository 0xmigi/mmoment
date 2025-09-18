// app/web/src/types/timeline.ts
export interface TimelineEvent {
    id: string;
    type: 'initialization' | 'user_connected' | 'photo_captured' | 'video_recorded' | 'stream_started' | 'stream_ended' | 'face_enrolled';
    user: {
      address: string;
      username?: string;
    };
    timestamp: number;
    cameraId?: string;
  }
  
  export interface TimelineState {
    events: TimelineEvent[];
    isConnected: boolean;
  }