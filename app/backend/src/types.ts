// Define timeline event types
export interface TimelineEvent {
  id: string;
  type: 'initialization' | 'user_connected' | 'photo_captured' | 'video_recorded' | 'stream_started' | 'stream_ended';
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
  };
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
  message?: string;
  metadata?: Record<string, any>;
} 