/**
 * Activity Decryption Utilities
 *
 * Decrypts timeline activities encrypted by the Jetson camera service.
 *
 * Architecture:
 * 1. Activities are encrypted with AES-256-GCM (random key per activity)
 * 2. Access grants contain the AES key encrypted for each user
 * 3. Currently uses XOR fallback (PyNaCl not on Jetson yet)
 * 4. Future: Sealed box encryption with Ed25519→X25519 conversion
 */

// Activity types from Solana program (state.rs) - MUST match ActivityType enum exactly
export const ACTIVITY_TYPE = {
  // Core camera activities (0-49)
  CHECK_IN: 0,
  CHECK_OUT: 1,
  PHOTO: 2,             // PhotoCapture
  VIDEO: 3,             // VideoRecord
  STREAM: 4,            // LiveStream
  FACE_RECOGNITION: 5,

  // CV app activity results (50-99)
  CV_APP: 50,

  // Other
  OTHER: 255,
} as const;

export type ActivityType = typeof ACTIVITY_TYPE[keyof typeof ACTIVITY_TYPE];

// Access grant structure from Jetson
export interface AccessGrant {
  pubkey: string;           // User's wallet address (base58)
  encryptedKey: string;     // Base64 encrypted AES key
}

// Encrypted activity from backend
export interface EncryptedActivity {
  sessionId: string;
  cameraId: string;
  userPubkey: string;       // Who triggered the activity
  timestamp: number;        // Milliseconds
  activityType: ActivityType;
  encryptedContent: string; // Base64 AES-256-GCM ciphertext
  nonce: string;            // Base64 12-byte nonce
  accessGrants: AccessGrant[];
  createdAt?: string;
}

// Decrypted activity content (varies by type)
export interface DecryptedPhotoActivity {
  type: 'photo';
  pipe_file_name: string;
  pipe_file_id: string;
  camera_id: string;
  captured_by: string;
  timestamp: number;
  device_signature?: string;
  width?: number;
  height?: number;
  filename?: string;
}

export interface DecryptedVideoActivity {
  type: 'video';
  pipe_file_name: string;
  pipe_file_id: string;
  camera_id: string;
  recorded_by: string;
  timestamp: number;
  duration_seconds?: number;
  device_signature?: string;
  filename?: string;
  size?: number;
}

export interface DecryptedStreamActivity {
  type: 'stream_start' | 'stream_end';
  stream_id: string;
  playback_id: string;
  camera_id: string;
  started_by?: string;
  ended_by?: string;
  timestamp: number;
  duration_seconds?: number;
  resolution?: string;
  fps?: number;
}

export type DecryptedActivityContent =
  | DecryptedPhotoActivity
  | DecryptedVideoActivity
  | DecryptedStreamActivity;

/**
 * Derive the wrapping key from a user's pubkey using HKDF.
 * Must match Jetson's _fallback_encrypt_key implementation.
 */
async function deriveWrappingKey(userPubkey: string): Promise<Uint8Array> {
  const encoder = new TextEncoder();

  // Match Jetson's salt and info
  const salt = encoder.encode('mmoment-activity-encryption-v1');
  const info = encoder.encode('activity-key-wrap');
  const pubkeyBytes = encoder.encode(userPubkey);

  // Import pubkey as raw key material for HKDF
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    pubkeyBytes.buffer as ArrayBuffer,
    'HKDF',
    false,
    ['deriveBits']
  );

  // Derive 256 bits (32 bytes) using HKDF-SHA256
  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: 'HKDF',
      hash: 'SHA-256',
      salt: salt.buffer as ArrayBuffer,
      info: info.buffer as ArrayBuffer,
    },
    keyMaterial,
    256 // 32 bytes
  );

  return new Uint8Array(derivedBits);
}

/**
 * XOR two byte arrays (for fallback encryption/decryption)
 */
function xorBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.length !== b.length) {
    throw new Error(`XOR length mismatch: ${a.length} vs ${b.length}`);
  }
  const result = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] ^ b[i];
  }
  return result;
}

/**
 * Base64 decode helper
 */
function base64Decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

/**
 * Decrypt the AES key from an access grant using XOR fallback.
 *
 * This matches Jetson's _fallback_encrypt_key which uses HKDF + XOR.
 * Works because both sides can derive the same key from the pubkey.
 *
 * @param encryptedKey - Base64 encoded encrypted AES key
 * @param userPubkey - User's wallet address
 * @returns Decrypted 32-byte AES key
 */
async function decryptAccessGrantFallback(
  encryptedKey: string,
  userPubkey: string
): Promise<Uint8Array> {
  // Derive the same wrapping key Jetson used
  const wrappingKey = await deriveWrappingKey(userPubkey);

  // Decode the encrypted key
  const encryptedKeyBytes = base64Decode(encryptedKey);

  // XOR to decrypt (same operation reverses the encryption)
  const activityKey = xorBytes(encryptedKeyBytes, wrappingKey);

  return activityKey;
}

/**
 * Decrypt activity content using AES-256-GCM.
 *
 * @param encryptedContent - Base64 encoded ciphertext
 * @param nonce - Base64 encoded 12-byte nonce
 * @param activityKey - 32-byte AES key
 * @returns Decrypted content as JSON object
 */
async function decryptContent(
  encryptedContent: string,
  nonce: string,
  activityKey: Uint8Array
): Promise<DecryptedActivityContent> {
  // Decode base64
  const ciphertext = base64Decode(encryptedContent);
  const iv = base64Decode(nonce);

  // Import the AES key
  const cryptoKey = await crypto.subtle.importKey(
    'raw',
    activityKey.buffer as ArrayBuffer,
    { name: 'AES-GCM' },
    false,
    ['decrypt']
  );

  // Decrypt
  const decryptedBuffer = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: iv.buffer as ArrayBuffer },
    cryptoKey,
    ciphertext.buffer as ArrayBuffer
  );

  // Parse JSON
  const decoder = new TextDecoder();
  const jsonString = decoder.decode(decryptedBuffer);
  return JSON.parse(jsonString);
}

/**
 * Find the access grant for a specific user.
 */
function findAccessGrant(
  accessGrants: AccessGrant[],
  userPubkey: string
): AccessGrant | null {
  return accessGrants.find(grant => grant.pubkey === userPubkey) || null;
}

/**
 * Decrypt an encrypted activity for a specific user.
 *
 * @param activity - Encrypted activity from backend
 * @param userPubkey - User's wallet address (to find their access grant)
 * @returns Decrypted activity content, or null if user doesn't have access
 */
export async function decryptActivity(
  activity: EncryptedActivity,
  userPubkey: string
): Promise<DecryptedActivityContent | null> {
  try {
    // Find the user's access grant
    const grant = findAccessGrant(activity.accessGrants, userPubkey);

    if (!grant) {
      console.warn(`[ActivityCrypto] No access grant for ${userPubkey.slice(0, 8)}...`);
      return null;
    }

    // Decrypt the AES key from the access grant
    // Currently using XOR fallback - will upgrade to sealed box when Jetson has PyNaCl
    const activityKey = await decryptAccessGrantFallback(
      grant.encryptedKey,
      userPubkey
    );

    // Decrypt the content
    const content = await decryptContent(
      activity.encryptedContent,
      activity.nonce,
      activityKey
    );

    console.log(`[ActivityCrypto] Decrypted activity type=${activity.activityType} for ${userPubkey.slice(0, 8)}...`);

    return content;

  } catch (error) {
    console.error('[ActivityCrypto] Decryption failed:', error);
    return null;
  }
}

/**
 * Decrypt multiple activities for a user.
 * Returns only successfully decrypted activities.
 */
export async function decryptActivities(
  activities: EncryptedActivity[],
  userPubkey: string
): Promise<Array<{ activity: EncryptedActivity; content: DecryptedActivityContent }>> {
  const results: Array<{ activity: EncryptedActivity; content: DecryptedActivityContent }> = [];

  for (const activity of activities) {
    const content = await decryptActivity(activity, userPubkey);
    if (content) {
      results.push({ activity, content });
    }
  }

  return results;
}

/**
 * Check if a user has access to an activity.
 */
export function hasAccessToActivity(
  activity: EncryptedActivity,
  userPubkey: string
): boolean {
  return findAccessGrant(activity.accessGrants, userPubkey) !== null;
}

/**
 * Get activity type name for display.
 */
export function getActivityTypeName(activityType: ActivityType): string {
  switch (activityType) {
    case ACTIVITY_TYPE.CHECK_IN:
      return 'Check In';
    case ACTIVITY_TYPE.CHECK_OUT:
      return 'Check Out';
    case ACTIVITY_TYPE.PHOTO:
      return 'Photo';
    case ACTIVITY_TYPE.VIDEO:
      return 'Video';
    case ACTIVITY_TYPE.STREAM:
      return 'Stream';
    case ACTIVITY_TYPE.FACE_RECOGNITION:
      return 'Face Recognition';
    case ACTIVITY_TYPE.CV_APP:
      return 'CV Activity';
    case ACTIVITY_TYPE.OTHER:
      return 'Other';
    default:
      return 'Unknown';
  }
}

/**
 * Timeline event type for compatibility with existing Timeline component
 */
export interface TimelineEventFromActivity {
  id: string;
  type: 'check_in' | 'check_out' | 'photo_captured' | 'video_recorded' | 'stream_started' | 'face_enrolled' | 'cv_activity';
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
  };
  timestamp: number;
  cameraId: string;
  // Pipe-specific metadata for media retrieval
  pipeMedia?: {
    fileName: string;
    fileId: string;
  };
  // Legacy mediaUrl (empty for Pipe - use pipeMedia instead)
  mediaUrl?: string;
}

/**
 * Convert activity type to timeline event type
 */
function activityTypeToEventType(
  activityType: ActivityType
): TimelineEventFromActivity['type'] {
  switch (activityType) {
    case ACTIVITY_TYPE.CHECK_IN:
      return 'check_in';
    case ACTIVITY_TYPE.CHECK_OUT:
      return 'check_out';
    case ACTIVITY_TYPE.PHOTO:
      return 'photo_captured';
    case ACTIVITY_TYPE.VIDEO:
      return 'video_recorded';
    case ACTIVITY_TYPE.STREAM:
      return 'stream_started';
    case ACTIVITY_TYPE.FACE_RECOGNITION:
      return 'face_enrolled';
    case ACTIVITY_TYPE.CV_APP:
      return 'cv_activity';
    default:
      return 'photo_captured'; // Fallback
  }
}

/**
 * Convert a decrypted activity to a TimelineEvent format.
 * This allows encrypted activities to be displayed in the existing Timeline component.
 */
export function activityToTimelineEvent(
  activity: EncryptedActivity,
  content: DecryptedActivityContent
): TimelineEventFromActivity {
  // Get user address from content or activity metadata
  let userAddress = activity.userPubkey;
  if ('captured_by' in content && content.captured_by) {
    userAddress = content.captured_by;
  } else if ('recorded_by' in content && content.recorded_by) {
    userAddress = content.recorded_by;
  } else if ('started_by' in content && content.started_by) {
    userAddress = content.started_by;
  } else if ('ended_by' in content && content.ended_by) {
    userAddress = content.ended_by;
  }

  // Build Pipe media metadata if available
  let pipeMedia: TimelineEventFromActivity['pipeMedia'];
  if ('pipe_file_name' in content && 'pipe_file_id' in content) {
    pipeMedia = {
      fileName: content.pipe_file_name,
      fileId: content.pipe_file_id,
    };
  }

  return {
    id: `encrypted-${activity.sessionId}-${activity.timestamp}`,
    type: activityTypeToEventType(activity.activityType),
    user: {
      address: userAddress,
    },
    timestamp: activity.timestamp,
    cameraId: activity.cameraId,
    pipeMedia,
  };
}

/**
 * Convert multiple decrypted activities to TimelineEvent format.
 */
export function activitiesToTimelineEvents(
  decryptedActivities: Array<{ activity: EncryptedActivity; content: DecryptedActivityContent }>
): TimelineEventFromActivity[] {
  return decryptedActivities.map(({ activity, content }) =>
    activityToTimelineEvent(activity, content)
  );
}
