// app/web/src/types/timeline.ts
export type TimelineEventType =
  | 'initialization'
  | 'user_connected'
  | 'photo_captured'
  | 'video_recorded'
  | 'stream_started'
  | 'stream_ended'
  | 'check_in'
  | 'check_out'
  | 'auto_check_out'
  | 'face_enrolled';

export interface TimelineUser {
  address: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
}

export interface TimelineEvent {
  id: string;
  type: TimelineEventType;
  user: TimelineUser;
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
}
  
export interface TimelineState {
  events: TimelineEvent[];
  isConnected: boolean;
}