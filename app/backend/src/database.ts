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

        // Create timeline_events table
        await runQuery(`
          CREATE TABLE IF NOT EXISTS timeline_events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            user_address TEXT NOT NULL,
            user_username TEXT,
            timestamp INTEGER NOT NULL,
            camera_id TEXT
          )
        `);

        // Create indexes for timeline_events
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_timeline_camera ON timeline_events(camera_id, timestamp)`);
        await runQuery(`CREATE INDEX IF NOT EXISTS idx_timeline_timestamp ON timeline_events(timestamp)`);

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
// TIMELINE EVENTS OPERATIONS
// ============================================================================

// Save timeline event to database
export async function saveTimelineEvent(event: TimelineEvent): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!db) {
      console.warn('Database not initialized, skipping timeline event save');
      resolve();
      return;
    }

    const stmt = db.prepare(`
      INSERT OR REPLACE INTO timeline_events
      (id, type, user_address, user_username, timestamp, camera_id)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      event.id,
      event.type,
      event.userAddress,
      event.userUsername || null,
      event.timestamp,
      event.cameraId || null,
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

// Get recent timeline events for a camera (limit to most recent N)
export async function getRecentTimelineEvents(cameraId?: string, limit: number = 100): Promise<TimelineEvent[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    let query = 'SELECT * FROM timeline_events';
    let params: any[] = [];

    if (cameraId) {
      query += ' WHERE camera_id = ?';
      params.push(cameraId);
    }

    query += ' ORDER BY timestamp DESC LIMIT ?';
    params.push(limit);

    db.all(query, params, (err, rows: any[]) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows.map(row => ({
          id: row.id,
          type: row.type,
          userAddress: row.user_address,
          userUsername: row.user_username || undefined,
          timestamp: row.timestamp,
          cameraId: row.camera_id || undefined
        })));
      }
    });
  });
}

// Clean up old timeline events (keep only last N per camera)
export async function cleanupOldTimelineEvents(keepPerCamera: number = 100): Promise<number> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    // Delete old events, keeping only the most recent N per camera
    db.run(`
      DELETE FROM timeline_events
      WHERE id NOT IN (
        SELECT id FROM (
          SELECT id, ROW_NUMBER() OVER (PARTITION BY camera_id ORDER BY timestamp DESC) as rn
          FROM timeline_events
        ) WHERE rn <= ?
      )
    `, [keepPerCamera], function(err) {
      if (err) {
        reject(err);
      } else {
        resolve(this.changes);
      }
    });
  });
}

// Load all timeline events into memory array (for backward compatibility)
export async function loadAllTimelineEventsToArray(): Promise<TimelineEvent[]> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    db.all(
      'SELECT * FROM timeline_events ORDER BY timestamp DESC LIMIT 1000',
      [],
      (err, rows: any[]) => {
        if (err) {
          reject(err);
          return;
        }

        const events = rows.map(row => ({
          id: row.id,
          type: row.type,
          userAddress: row.user_address,
          userUsername: row.user_username || undefined,
          timestamp: row.timestamp,
          cameraId: row.camera_id || undefined
        }));

        console.log(`✅ Loaded ${events.length} timeline events from database`);

        resolve(events);
      }
    );
  });
}

// Get database statistics
export async function getDatabaseStats(): Promise<{
  fileMappings: number;
  timelineEvents: number;
  uniqueWallets: number;
  uniqueCameras: number;
  userProfiles: number;
}> {
  return new Promise((resolve, reject) => {
    if (!db) {
      reject(new Error('Database not initialized'));
      return;
    }

    const stats = {
      fileMappings: 0,
      timelineEvents: 0,
      uniqueWallets: 0,
      uniqueCameras: 0,
      userProfiles: 0
    };

    const dbInstance = db;
    if (!dbInstance) {
      reject(new Error('Database not initialized'));
      return;
    }

    dbInstance.get('SELECT COUNT(*) as count FROM file_mappings', [], (err, row: any) => {
      if (err) { reject(err); return; }
      stats.fileMappings = row.count;

      dbInstance.get('SELECT COUNT(*) as count FROM timeline_events', [], (err, row: any) => {
        if (err) { reject(err); return; }
        stats.timelineEvents = row.count;

        dbInstance.get('SELECT COUNT(DISTINCT wallet_address) as count FROM file_mappings', [], (err, row: any) => {
          if (err) { reject(err); return; }
          stats.uniqueWallets = row.count;

          dbInstance.get('SELECT COUNT(DISTINCT camera_id) as count FROM timeline_events WHERE camera_id IS NOT NULL', [], (err, row: any) => {
            if (err) { reject(err); return; }
            stats.uniqueCameras = row.count;

            dbInstance.get('SELECT COUNT(*) as count FROM user_profiles', [], (err, row: any) => {
              if (err) { reject(err); return; }
              stats.userProfiles = row.count;

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
