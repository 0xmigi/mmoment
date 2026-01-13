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

/** Competition/Prize metadata for financialized CV activities */
export interface CompetitionMetadata {
  /** Competition mode: 'none' | 'bet' | 'prize' */
  mode: 'none' | 'bet' | 'prize';
  /** Escrow PDA address (on-chain) */
  escrowPda?: string;
  /** Amount staked/deposited in SOL */
  stakeAmountSol: number;
  /** Target reps (for prize/threshold mode) */
  targetReps?: number;
  /** Did the user win? */
  won: boolean;
  /** Amount won in SOL (0 if lost) */
  amountWonSol: number;
  /** Amount lost in SOL (0 if won) */
  amountLostSol: number;
  /** Who received the lost funds (camera wallet for prize mode) */
  lostTo?: string;
  /** Settlement transaction signature */
  settlementTxId?: string;
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
  /** Competition/prize info (if financialized) */
  competition?: CompetitionMetadata;
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