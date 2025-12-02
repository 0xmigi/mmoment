// backend/src/database.ts
import sqlite3 from 'sqlite3';

// Database instance
let db: sqlite3.Database | null = null;

// File mapping interface
export interface FileMapping {
  signature: string;
  signatureType: 'device' | 'blockchain';
  walletAddress: string;
  fileId: string;
  fileName: string;
  cameraId: string;
  uploadedAt: Date;
  fileType: 'photo' | 'video';
}

// Timeline event interface
export interface TimelineEvent {
  id: string;
  type: string;
  userAddress: string;
  userUsername?: string;
  timestamp: number;
  cameraId?: string;
}

// User profile interface
export interface UserProfile {
  walletAddress: string;
  displayName?: string;
  username?: string;
  profileImage?: string;
  provider?: string;
  lastUpdated: Date;
}

// Session activity buffer interface (for privacy-preserving bundling)
export interface SessionActivityBuffer {
  sessionId: string;
  cameraId: string;
  userPubkey: string;
  timestamp: number;
  activityType: number;
  encryptedContent: Buffer;  // AES-256-GCM encrypted by Jetson
  nonce: Buffer;             // 12-byte nonce
  accessGrants: Buffer;      // JSON array of encrypted keys, serialized
  createdAt: Date;
}

// Initialize database with tables
export async function initializeDatabase(dbPath: string = './mmoment.db'): Promise<void> {
  return new Promise((resolve, reject) => {
    db = new sqlite3.Database(dbPath, async (err) => {
      if (err) {
        console.error('❌ Failed to open database:', err);
        reject(err);
        return;
      }

      console.log(`📦 SQLite database connected: ${dbPath}`);

      try {
        // Enable foreign keys
        await runQuery('PRAGMA foreign_keys = ON');

        // Create file_mappings table
        await runQuery(`
          CREATE TABLE IF NOT EXISTS file_mappings (
            signature TEXT PRIMARY KEY,
            signature_type TEXT NOT NULL,
            wallet_address TEXT NOT NULL,
            file_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            uploaded_at INTEGER NOT NULL,
            file_type TEXT NOT NULL
          )
        `);

        // Create indexes for file_mappings
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_file_mappings_wallet ON file_mappings(wallet_address)`);
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_file_mappings_camera ON file_mappings(camera_id)`);
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_file_mappings_file_name ON file_mappings(file_name)`);

        // NOTE: timeline_events table has been removed. All events now go to session_activity_buffers.

        // Create user_profiles table
        await runQuery(`
          CREATE TABLE IF NOT EXISTS user_profiles (
            wallet_address TEXT PRIMARY KEY,
            display_name TEXT,
            username TEXT,
            profile_image TEXT,
            provider TEXT,
            last_updated INTEGER NOT NULL
          )
        `);

        // Create index for user_profiles
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_profiles_updated ON user_profiles(last_updated)`);

        // Create session_activity_buffers table (for privacy-preserving timeline)
        await runQuery(`
          CREATE TABLE IF NOT EXISTS session_activity_buffers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            user_pubkey TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            activity_type INTEGER NOT NULL,
            encrypted_content BLOB NOT NULL,
            nonce BLOB NOT NULL,
            access_grants BLOB NOT NULL,
            created_at INTEGER NOT NULL
          )
        `);

        // Create indexes for session_activity_buffers
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_session_buffers_session ON session_activity_buffers(session_id)`);
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_session_buffers_camera ON session_activity_buffers(camera_id, timestamp)`);
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_session_buffers_created ON session_activity_buffers(created_at)`);

        console.log('✅ Database tables initialized');
        resolve();
      } catch (error) {
        console.error('❌ Failed to create tables:', error);
        reject(error);
      }
    });
  });
}

// Helper function to run queries with promises
function runQuery(sql: string, params: any[] = []): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }
    db.run(sql, params, (err: Error | null) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}

// Close database connection
export function closeDatabase(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      resolve();
      return;
    }

    db.close((err) => {
      if (err) {
        reject(err);
      } else {
        db = null;
        resolve();
      }
    });
  });
}

// ============================================================================
// FILE MAPPINGS OPERATIONS
// ============================================================================

// Save file mapping to database
export async function saveFileMapping(mapping: FileMapping): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stmt = db.prepare(`
      INSERT OR REPLACE INTO file_mappings
      (signature, signature_type, wallet_address, file_id, file_name, camera_id, uploaded_at, file_type)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      mapping.signature,
      mapping.signatureType,
      mapping.walletAddress,
      mapping.fileId,
      mapping.fileName,
      mapping.cameraId,
      mapping.uploadedAt.getTime(),
      mapping.fileType,
      (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      }
    );

    stmt.finalize();
  });
}

// Get file mapping by signature
export async function getFileMappingBySignature(signature: string): Promise<FileMapping | null> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.get(
      'SELECT * FROM file_mappings WHERE signature = ?',
      [signature],
      (err, row: any) => {
        if (err) {
          reject(err);
        } else if (!row) {
          resolve(null);
        } else {
          resolve({
            signature: row.signature,
            signatureType: row.signature_type,
            walletAddress: row.wallet_address,
            fileId: row.file_id,
            fileName: row.file_name,
            cameraId: row.camera_id,
            uploadedAt: new Date(row.uploaded_at),
            fileType: row.file_type
          });
        }
      }
    );
  });
}

// Get all signatures for a wallet address
export async function getSignaturesForWallet(walletAddress: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.all(
      'SELECT signature FROM file_mappings WHERE wallet_address = ? ORDER BY uploaded_at DESC',
      [walletAddress],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map(row => row.signature));
        }
      }
    );
  });
}

// Get all file mappings for a wallet address
export async function getFileMappingsForWallet(walletAddress: string): Promise<FileMapping[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.all(
      'SELECT * FROM file_mappings WHERE wallet_address = ? ORDER BY uploaded_at DESC',
      [walletAddress],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map(row => ({
            signature: row.signature,
            signatureType: row.signature_type,
            walletAddress: row.wallet_address,
            fileId: row.file_id,
            fileName: row.file_name,
            cameraId: row.camera_id,
            uploadedAt: new Date(row.uploaded_at),
            fileType: row.file_type
          })));
        }
      }
    );
  });
}

// Delete file mapping by file name and wallet
export async function deleteFileMappingByFileName(fileName: string, walletAddress: string): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.run(
      'DELETE FROM file_mappings WHERE file_name = ? AND wallet_address = ?',
      [fileName, walletAddress],
      function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.changes);
        }
      }
    );
  });
}

// Load all file mappings into memory Maps (for backward compatibility)
export async function loadAllFileMappingsToMaps(): Promise<{
  signatureToFileMapping: Map<string, FileMapping>;
  walletToSignatures: Map<string, string[]>;
}> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const signatureToFileMapping = new Map<string, FileMapping>();
    const walletToSignatures = new Map<string, string[]>();

    db.all('SELECT * FROM file_mappings', [], (err, rows: any[]) => {
      if (err) {
        reject(err);
        return;
      }

      for (const row of rows) {
        const mapping: FileMapping = {
          signature: row.signature,
          signatureType: row.signature_type,
          walletAddress: row.wallet_address,
          fileId: row.file_id,
          fileName: row.file_name,
          cameraId: row.camera_id,
          uploadedAt: new Date(row.uploaded_at),
          fileType: row.file_type
        };

        signatureToFileMapping.set(mapping.signature, mapping);

        const walletSigs = walletToSignatures.get(mapping.walletAddress) || [];
        walletSigs.push(mapping.signature);
        walletToSignatures.set(mapping.walletAddress, walletSigs);
      }

      console.log(`✅ Loaded ${signatureToFileMapping.size} file mappings from database`);
      console.log(`✅ Loaded mappings for ${walletToSignatures.size} wallets`);

      resolve({ signatureToFileMapping, walletToSignatures });
    });
  });
}

// ============================================================================
// TIMELINE EVENTS - REMOVED
// ============================================================================
// NOTE: timeline_events table and functions have been removed.
// All events now use session_activity_buffers with activity_type:
//   0 = CHECK_IN, 1 = CHECK_OUT, 2 = PHOTO, 3 = VIDEO, 4 = STREAM, 5 = FACE_RECOGNITION, 50 = CV_APP, 255 = OTHER

// Get database statistics
export async function getDatabaseStats(): Promise<{
  fileMappings: number;
  uniqueWallets: number;
  uniqueCameras: number;
  userProfiles: number;
  sessionActivityBuffers: number;
}> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stats = {
      fileMappings: 0,
      uniqueWallets: 0,
      uniqueCameras: 0,
      userProfiles: 0,
      sessionActivityBuffers: 0
    };

    const dbInstance = db;
    if (!dbInstance) {
      reject(new Error('Database not initialized'));
      return;
    }

    dbInstance.get('SELECT COUNT(*) as count FROM file_mappings', [], (err, row: any) => {
      if (err) { reject(err); return; }
      stats.fileMappings = row.count;

      dbInstance.get('SELECT COUNT(DISTINCT wallet_address) as count FROM file_mappings', [], (err, row: any) => {
        if (err) { reject(err); return; }
        stats.uniqueWallets = row.count;

        dbInstance.get('SELECT COUNT(DISTINCT camera_id) as count FROM session_activity_buffers WHERE camera_id IS NOT NULL', [], (err, row: any) => {
          if (err) { reject(err); return; }
          stats.uniqueCameras = row.count;

          dbInstance.get('SELECT COUNT(*) as count FROM user_profiles', [], (err, row: any) => {
            if (err) { reject(err); return; }
            stats.userProfiles = row.count;

            dbInstance.get('SELECT COUNT(*) as count FROM session_activity_buffers', [], (err, row: any) => {
              if (err) { reject(err); return; }
              stats.sessionActivityBuffers = row.count;

              resolve(stats);
            });
          });
        });
      });
    });
  });
}

// ============================================================================
// USER PROFILES OPERATIONS
// ============================================================================

// Save or update user profile to database
export async function saveUserProfile(profile: UserProfile): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stmt = db.prepare(`
      INSERT OR REPLACE INTO user_profiles
      (wallet_address, display_name, username, profile_image, provider, last_updated)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      profile.walletAddress,
      profile.displayName || null,
      profile.username || null,
      profile.profileImage || null,
      profile.provider || null,
      profile.lastUpdated.getTime(),
      (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      }
    );

    stmt.finalize();
  });
}

// Get user profile by wallet address
export async function getUserProfile(walletAddress: string): Promise<UserProfile | null> {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.warn('Database not initialized, returning null for user profile');
      resolve(null);
      return;
    }

    db.get(
      'SELECT * FROM user_profiles WHERE wallet_address = ?',
      [walletAddress],
      (err, row: any) => {
        if (err) {
          reject(err);
        } else if (!row) {
          resolve(null);
        } else {
          resolve({
            walletAddress: row.wallet_address,
            displayName: row.display_name || undefined,
            username: row.username || undefined,
            profileImage: row.profile_image || undefined,
            provider: row.provider || undefined,
            lastUpdated: new Date(row.last_updated)
          });
        }
      }
    );
  });
}

// Get multiple user profiles by wallet addresses
export async function getUserProfiles(walletAddresses: string[]): Promise<Map<string, UserProfile>> {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.warn('Database not initialized, returning empty map for user profiles');
      resolve(new Map());
      return;
    }

    if (walletAddresses.length === 0) {
      resolve(new Map());
      return;
    }

    const placeholders = walletAddresses.map(() => '?').join(',');
    const query = `SELECT * FROM user_profiles WHERE wallet_address IN (${placeholders})`;

    db.all(query, walletAddresses, (err, rows: any[]) => {
      if (err) {
        reject(err);
      } else {
        const profilesMap = new Map<string, UserProfile>();
        for (const row of rows) {
          profilesMap.set(row.wallet_address, {
            walletAddress: row.wallet_address,
            displayName: row.display_name || undefined,
            username: row.username || undefined,
            profileImage: row.profile_image || undefined,
            provider: row.provider || undefined,
            lastUpdated: new Date(row.last_updated)
          });
        }
        resolve(profilesMap);
      }
    });
  });
}

// Load all user profiles into memory map (for backward compatibility)
export async function loadAllUserProfilesToMap(): Promise<Map<string, UserProfile>> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const profilesMap = new Map<string, UserProfile>();

    db.all('SELECT * FROM user_profiles', [], (err, rows: any[]) => {
      if (err) {
        reject(err);
        return;
      }

      for (const row of rows) {
        profilesMap.set(row.wallet_address, {
          walletAddress: row.wallet_address,
          displayName: row.display_name || undefined,
          username: row.username || undefined,
          profileImage: row.profile_image || undefined,
          provider: row.provider || undefined,
          lastUpdated: new Date(row.last_updated)
        });
      }

      console.log(`✅ Loaded ${profilesMap.size} user profiles from database`);
      resolve(profilesMap);
    });
  });
}

// Delete user profile by wallet address
export async function deleteUserProfile(walletAddress: string): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.run(
      'DELETE FROM user_profiles WHERE wallet_address = ?',
      [walletAddress],
      function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.changes);
        }
      }
    );
  });
}

// ============================================================================
// SESSION ACTIVITY BUFFER OPERATIONS (Privacy-Preserving Timeline)
// ============================================================================

// Save encrypted activity to session buffer (called by Jetson during session)
export async function saveSessionActivity(activity: SessionActivityBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stmt = db.prepare(`
      INSERT INTO session_activity_buffers
      (session_id, camera_id, user_pubkey, timestamp, activity_type, encrypted_content, nonce, access_grants, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      activity.sessionId,
      activity.cameraId,
      activity.userPubkey,
      activity.timestamp,
      activity.activityType,
      activity.encryptedContent,
      activity.nonce,
      activity.accessGrants,
      activity.createdAt.getTime(),
      (err: Error | null) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      }
    );

    stmt.finalize();
  });
}

// Get all buffered activities for a session (called by auto-checkout bot)
export async function getSessionActivities(sessionId: string): Promise<SessionActivityBuffer[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.all(
      'SELECT * FROM session_activity_buffers WHERE session_id = ? ORDER BY timestamp ASC',
      [sessionId],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map(row => ({
            sessionId: row.session_id,
            cameraId: row.camera_id,
            userPubkey: row.user_pubkey,
            timestamp: row.timestamp,
            activityType: row.activity_type,
            encryptedContent: row.encrypted_content,
            nonce: row.nonce,
            accessGrants: row.access_grants,
            createdAt: new Date(row.created_at)
          })));
        }
      }
    );
  });
}

// Clear buffered activities after successful checkout (called after blockchain commit)
export async function clearSessionActivities(sessionId: string): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.run(
      'DELETE FROM session_activity_buffers WHERE session_id = ?',
      [sessionId],
      function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.changes);
        }
      }
    );
  });
}

// Clean up old orphaned activities (older than 7 days, for crashed sessions)
export async function cleanupOldSessionActivities(daysOld: number = 7): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const cutoffTime = Date.now() - (daysOld * 24 * 60 * 60 * 1000);

    db.run(
      'DELETE FROM session_activity_buffers WHERE created_at < ?',
      [cutoffTime],
      function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.changes);
        }
      }
    );
  });
}

// Get buffer statistics for monitoring
export async function getSessionBufferStats(): Promise<{
  totalActivities: number;
  uniqueSessions: number;
  uniqueCameras: number;
  oldestActivity: Date | null;
  newestActivity: Date | null;
}> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stats = {
      totalActivities: 0,
      uniqueSessions: 0,
      uniqueCameras: 0,
      oldestActivity: null as Date | null,
      newestActivity: null as Date | null
    };

    const dbInstance = db;
    if (!dbInstance) {
      reject(new Error('Database not initialized'));
      return;
    }

    dbInstance.get('SELECT COUNT(*) as count FROM session_activity_buffers', [], (err, row: any) => {
      if (err) { reject(err); return; }
      stats.totalActivities = row.count;

      dbInstance.get('SELECT COUNT(DISTINCT session_id) as count FROM session_activity_buffers', [], (err, row: any) => {
        if (err) { reject(err); return; }
        stats.uniqueSessions = row.count;

        dbInstance.get('SELECT COUNT(DISTINCT camera_id) as count FROM session_activity_buffers', [], (err, row: any) => {
          if (err) { reject(err); return; }
          stats.uniqueCameras = row.count;

          dbInstance.get('SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM session_activity_buffers', [], (err, row: any) => {
            if (err) { reject(err); return; }
            if (row.oldest) stats.oldestActivity = new Date(row.oldest);
            if (row.newest) stats.newestActivity = new Date(row.newest);

            resolve(stats);
          });
        });
      });
    });
  });
}

// Session summary interface for listing
export interface SessionParticipant {
  address: string;
  displayName?: string;
  username?: string;
  pfpUrl?: string;
}

export interface SessionSummary {
  sessionId: string;
  cameraId: string;
  startTime: number;
  endTime: number;
  activityCount: number;
  activityTypes: number[];  // Unique activity types in this session
  participants: SessionParticipant[];  // All users who were at the camera during this session
}

// Activity type constants (must match Solana ActivityType enum in state.rs)
const ACTIVITY_TYPE = {
  CHECK_IN: 0,
  CHECK_OUT: 1,
  PHOTO: 2,
  VIDEO: 3,
  STREAM: 4,
  FACE_RECOGNITION: 5,
  CV_APP: 50,
  OTHER: 255,
} as const;

// Get all COMPLETED sessions for a user (check-in to check-out periods)
// Sessions are defined by CHECK_IN (0) and CHECK_OUT (1) activity types in session_activity_buffers
export async function getUserSessions(walletAddress: string, limit: number = 50): Promise<SessionSummary[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const dbInstance = db;

    // Step 1: Get all CHECK_IN activities for this user (activity_type = 0)
    dbInstance.all(
      `SELECT id, session_id, camera_id, timestamp
       FROM session_activity_buffers
       WHERE user_pubkey = ? AND activity_type = ?
       ORDER BY timestamp DESC
       LIMIT ?`,
      [walletAddress, ACTIVITY_TYPE.CHECK_IN, limit * 2],
      async (err, checkIns: any[]) => {
        if (err) {
          reject(err);
          return;
        }

        if (!checkIns || checkIns.length === 0) {
          console.log(`[getUserSessions] No CHECK_IN activities found for ${walletAddress.slice(0, 8)}...`);
          resolve([]);
          return;
        }

        console.log(`[getUserSessions] Found ${checkIns.length} CHECK_IN activities for ${walletAddress.slice(0, 8)}...`);

        const sessions: SessionSummary[] = [];
        const seenSessionIds = new Set<string>(); // Track already processed session IDs to prevent duplicates

        // Step 2: For each check-in, find the corresponding check-out
        for (const checkIn of checkIns) {
          if (sessions.length >= limit) break;

          // Skip if we've already processed this session_id (prevents duplicate sessions)
          const sessionId = checkIn.session_id || `${checkIn.id}`;
          if (seenSessionIds.has(sessionId)) {
            console.log(`[getUserSessions] Skipping duplicate session_id: ${sessionId.slice(0, 8)}...`);
            continue;
          }
          seenSessionIds.add(sessionId);

          try {
            // Find the next CHECK_OUT activity for this user at this camera after the check_in
            const checkOut = await new Promise<any>((res, rej) => {
              dbInstance.get(
                `SELECT id, timestamp
                 FROM session_activity_buffers
                 WHERE user_pubkey = ?
                   AND camera_id = ?
                   AND activity_type = ?
                   AND timestamp > ?
                 ORDER BY timestamp ASC
                 LIMIT 1`,
                [walletAddress, checkIn.camera_id, ACTIVITY_TYPE.CHECK_OUT, checkIn.timestamp],
                (err, row) => err ? rej(err) : res(row)
              );
            });

            // Only include COMPLETED sessions (with a check-out)
            if (!checkOut) {
              console.log(`[getUserSessions] Check-in at ${checkIn.camera_id.slice(0, 8)}... has no check-out yet (active session)`);
              continue;
            }

            // Step 3: Get activity stats for this time window (excluding check-in/check-out themselves)
            // IMPORTANT: Filter by user_pubkey to only count THIS user's activities, not all users at the camera
            const activityStats = await new Promise<any>((res, rej) => {
              dbInstance.get(
                `SELECT
                   COUNT(*) as activity_count,
                   GROUP_CONCAT(DISTINCT activity_type) as activity_types
                 FROM session_activity_buffers
                 WHERE camera_id = ?
                   AND user_pubkey = ?
                   AND timestamp >= ?
                   AND timestamp <= ?
                   AND activity_type NOT IN (?, ?)`,
                [checkIn.camera_id, walletAddress, checkIn.timestamp, checkOut.timestamp, ACTIVITY_TYPE.CHECK_IN, ACTIVITY_TYPE.CHECK_OUT],
                (err, row) => err ? rej(err) : res(row || { activity_count: 0, activity_types: '' })
              );
            });

            // Step 4: Get all unique participants (users) at this camera during the session
            const participantAddresses = await new Promise<string[]>((res, rej) => {
              dbInstance.all(
                `SELECT DISTINCT user_pubkey
                 FROM session_activity_buffers
                 WHERE camera_id = ?
                   AND timestamp >= ?
                   AND timestamp <= ?`,
                [checkIn.camera_id, checkIn.timestamp, checkOut.timestamp],
                (err, rows: any[]) => {
                  if (err) rej(err);
                  else res(rows.map(r => r.user_pubkey));
                }
              );
            });

            // Step 5: Fetch profiles for participants (batch)
            const profilesMap = await getUserProfiles(participantAddresses);

            // Step 6: Build participant list with profile enrichment
            const participants: SessionParticipant[] = participantAddresses.map(address => {
              const profile = profilesMap.get(address);
              return {
                address,
                displayName: profile?.displayName,
                username: profile?.username,
                pfpUrl: profile?.profileImage,
              };
            });

            // Create session with session_id from check-in as sessionId
            sessions.push({
              sessionId, // Use the already-derived sessionId
              cameraId: checkIn.camera_id,
              startTime: checkIn.timestamp,
              endTime: checkOut.timestamp,
              activityCount: activityStats.activity_count || 0,
              activityTypes: activityStats.activity_types
                ? activityStats.activity_types.split(',').map(Number).filter((n: number) => !isNaN(n))
                : [],
              participants
            });

            console.log(`[getUserSessions] Session: ${sessionId.slice(0, 8)}... at camera ${checkIn.camera_id.slice(0, 8)}... (${activityStats.activity_count} activities)`);

          } catch (error) {
            console.error('[getUserSessions] Error processing check-in:', error);
          }
        }

        console.log(`[getUserSessions] Returning ${sessions.length} completed sessions`);
        resolve(sessions);
      }
    );
  });
}

// Get all activities for a camera (most recent first)
export async function getCameraActivities(cameraId: string, limit: number = 100): Promise<SessionActivityBuffer[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.all(
      'SELECT * FROM session_activity_buffers WHERE camera_id = ? ORDER BY timestamp DESC LIMIT ?',
      [cameraId, limit],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map(row => ({
            sessionId: row.session_id,
            cameraId: row.camera_id,
            userPubkey: row.user_pubkey,
            timestamp: row.timestamp,
            activityType: row.activity_type,
            encryptedContent: row.encrypted_content,
            nonce: row.nonce,
            accessGrants: row.access_grants,
            createdAt: new Date(row.created_at)
          })));
        }
      }
    );
  });
}

// Get all events at a camera during a time window (for historical session timeline)
// All events now come from session_activity_buffers (unified storage)
export interface SessionTimelineEvent {
  id: string;
  type: string;  // TimelineEventType: 'check_in', 'check_out', 'photo_captured', etc.
  userAddress: string;
  timestamp: number;
  cameraId: string;
  // For encrypted activities (all activities have this now)
  encrypted?: {
    sessionId: string;
    activityType: number;
    encryptedContent: string;  // base64
    nonce: string;  // base64
    accessGrants: string;  // JSON string
  };
}

// Get all events at a camera during a session's time window
// Now queries ONLY session_activity_buffers (single source of truth)
export async function getSessionTimelineEvents(
  cameraId: string,
  startTime: number,
  endTime: number
): Promise<SessionTimelineEvent[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const dbInstance = db;

    // Get all activities from session_activity_buffers
    dbInstance.all(
      `SELECT * FROM session_activity_buffers
       WHERE camera_id = ? AND timestamp >= ? AND timestamp <= ?
       ORDER BY timestamp DESC`,
      [cameraId, startTime, endTime],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
          return;
        }

        // Map activity_type to event type string (matches Solana ActivityType enum)
        const activityTypeToEventType: Record<number, string> = {
          0: 'check_in',
          1: 'check_out',
          2: 'photo_captured',
          3: 'video_recorded',
          4: 'stream_started',
          5: 'face_enrolled',
          50: 'cv_activity',
          255: 'other'
        };

        // Convert all activities to timeline events
        const events: SessionTimelineEvent[] = rows.map((row: any) => ({
          id: `activity-${row.session_id}-${row.timestamp}`,
          type: activityTypeToEventType[row.activity_type] || 'unknown',
          userAddress: row.user_pubkey,
          timestamp: row.timestamp,
          cameraId: row.camera_id,
          encrypted: {
            sessionId: row.session_id,
            activityType: row.activity_type,
            encryptedContent: row.encrypted_content.toString('base64'),
            nonce: row.nonce.toString('base64'),
            accessGrants: row.access_grants.toString('utf-8')
          }
        }));

        resolve(events);
      }
    );
  });
}

// Get all activities a user has access to (across all sessions)
export async function getUserActivities(walletAddress: string, limit: number = 100): Promise<SessionActivityBuffer[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    // Query activities where user has an access grant
    db.all(
      `SELECT * FROM session_activity_buffers
       WHERE access_grants LIKE ?
       ORDER BY timestamp DESC
       LIMIT ?`,
      [`%"pubkey":"${walletAddress}"%`, limit],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map(row => ({
            sessionId: row.session_id,
            cameraId: row.camera_id,
            userPubkey: row.user_pubkey,
            timestamp: row.timestamp,
            activityType: row.activity_type,
            encryptedContent: row.encrypted_content,
            nonce: row.nonce,
            accessGrants: row.access_grants,
            createdAt: new Date(row.created_at)
          })));
        }
      }
    );
  });
}
