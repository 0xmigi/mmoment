// app/web/src/types/timeline.ts

// Re-export encrypted activity types from crypto utils
export {
  ACTIVITY_TYPE,
  type ActivityType,
  type AccessGrant,
  type EncryptedActivity,
  type DecryptedPhotoActivity,
  type DecryptedVideoActivity,
  type DecryptedStreamActivity,
  type DecryptedActivityContent,
  type TimelineEventFromActivity,
  decryptActivity,
  decryptActivities,
  hasAccessToActivity,
  getActivityTypeName,
  activityToTimelineEvent,
  activitiesToTimelineEvents,
} from '../utils/activity-crypto';

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
  | 'face_enrolled'
  | 'cv_activity'
  | 'other';

export interface TimelineUser {
  address: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
  provider?: string;
}

/** CV activity result for a single participant */
export interface CVActivityResult {
  wallet_address: string;
  display_name?: string;
  rank: number;
  stats: {
    reps?: number;
    [key: string]: unknown;
  };
}

/** CV activity metadata (competition results, etc.) */
export interface CVActivityMetadata {
  app_name: string;
  duration_seconds?: number;
  participant_count: number;
  results: CVActivityResult[];
  user_stats: {
    reps?: number;
    [key: string]: unknown;
  };
}

export interface TimelineEvent {
  id: string;
  type: TimelineEventType;
  user: TimelineUser;
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
  /** CV activity metadata (only present for cv_activity events) */
  cvActivity?: CVActivityMetadata;
}
  
export interface TimelineState {
  events: TimelineEvent[];
  isConnected: boolean;
}