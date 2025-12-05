// Session Access Key Storage Service
// In the new privacy architecture, sessions are managed off-chain by Jetson.
// This service stores access keys for users when sessions end (fallback for when user doesn't store their own keys).
import { Connection, Keypair, PublicKey, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet, BN } from '@coral-xyz/anchor';
import { IDL } from './idl';
import { Server } from 'socket.io';
import { updateActivityTransactionId } from './database';

const PROGRAM_ID = new PublicKey('E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL');
const RETRY_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes for retrying pending keys

// Pending access keys to be stored (received from Jetson on session end)
interface PendingAccessKey {
  userPubkey: string;
  keyCiphertext: number[];
  nonce: number[];
  timestamp: number;
  retryCount: number;
}

// Type for timeline event without ID (kept for backwards compatibility)
type TimelineEventWithoutId = {
  type: string;
  user: {
    address: string;
  };
  timestamp: number;
  cameraId: string;
  transactionId?: string;
};

type AddTimelineEventFn = (event: TimelineEventWithoutId, socketServer: Server) => any;

let connection: Connection | null = null;
let cronBotKeypair: Keypair | null = null;
let program: Program | null = null;
let retryIntervalId: NodeJS.Timeout | null = null;
let io: Server | null = null;
let addTimelineEventFn: AddTimelineEventFn | null = null;

// Queue of pending access keys to store (keyed by user pubkey)
const pendingKeys: Map<string, PendingAccessKey[]> = new Map();

export function initializeSessionCleanupCron(
  rpcUrl: string,
  cronBotSecretKey: string,
  socketServer?: Server,
  addTimelineEvent?: AddTimelineEventFn
) {
  try {
    // Parse the secret key from JSON array format
    const secretKeyArray = JSON.parse(cronBotSecretKey);
    cronBotKeypair = Keypair.fromSecretKey(new Uint8Array(secretKeyArray));
    connection = new Connection(rpcUrl, 'confirmed');

    // Store Socket.IO server reference and timeline event handler
    if (socketServer) {
      io = socketServer;
      console.log('   Socket.IO server connected for session events');
    }
    if (addTimelineEvent) {
      addTimelineEventFn = addTimelineEvent;
      console.log('   Timeline event handler connected');
    }

    // Create Anchor provider and program
    const wallet = new Wallet(cronBotKeypair);
    const provider = new AnchorProvider(connection, wallet, { commitment: 'confirmed' });
    program = new Program(IDL as any, PROGRAM_ID, provider);

    console.log('✅ Session Access Key Service Initialized');
    console.log(`   Cron Bot (Authority): ${cronBotKeypair.publicKey.toString()}`);
    console.log('   NOTE: Sessions are now managed off-chain by Jetson');
    console.log('   This service stores access keys as fallback when users don\'t store their own');

    // Start retry interval for pending keys
    startRetryInterval();

    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Session Access Key Service:', error);
    return false;
  }
}

function startRetryInterval() {
  if (retryIntervalId) {
    console.log('⚠️  Retry interval already running');
    return;
  }

  // Retry pending keys periodically
  retryIntervalId = setInterval(() => {
    processAllPendingKeys();
  }, RETRY_INTERVAL_MS);

  console.log('🔄 Access key retry interval started');
}

export function stopCleanupCron() {
  if (retryIntervalId) {
    clearInterval(retryIntervalId);
    retryIntervalId = null;
    console.log('🛑 Session Access Key Service stopped');
  }
}

/**
 * Queue an access key to be stored for a user
 * Called by Jetson when a session ends
 */
export async function queueAccessKeyForUser(
  userPubkey: string,
  keyCiphertext: number[],
  nonce: number[],
  timestamp: number
): Promise<boolean> {
  const key: PendingAccessKey = {
    userPubkey,
    keyCiphertext,
    nonce,
    timestamp,
    retryCount: 0
  };

  if (!pendingKeys.has(userPubkey)) {
    pendingKeys.set(userPubkey, []);
  }
  pendingKeys.get(userPubkey)!.push(key);

  console.log(`📥 Queued access key for user ${userPubkey.slice(0, 8)}...`);

  // Try to store immediately
  return await processAccessKeyForUser(userPubkey);
}

/**
 * Process pending access keys for a user
 */
async function processAccessKeyForUser(userPubkey: string): Promise<boolean> {
  if (!connection || !cronBotKeypair || !program) {
    console.error('Session Access Key Service not initialized');
    return false;
  }

  const keys = pendingKeys.get(userPubkey);
  if (!keys || keys.length === 0) {
    return true; // Nothing to process
  }

  try {
    const userKey = new PublicKey(userPubkey);

    // Derive the UserSessionChain PDA
    const [userSessionChainPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user-session-chain'), userKey.toBuffer()],
      PROGRAM_ID
    );

    // Check if user's session chain exists
    const chainAccount = await connection.getAccountInfo(userSessionChainPda);

    if (!chainAccount) {
      console.log(`   ⚠️ User ${userPubkey.slice(0, 8)}... has no session chain. Keys held in queue.`);
      // User needs to create their session chain first
      // Keep keys in queue for retry
      return false;
    }

    // Convert keys to the format expected by the program
    const keysToStore = keys.map(k => ({
      keyCiphertext: Buffer.from(k.keyCiphertext),
      nonce: k.nonce,
      timestamp: new BN(k.timestamp),
    }));

    // Build store_session_access_keys transaction
    const tx = await program.methods
      .storeSessionAccessKeys(keysToStore)
      .accounts({
        signer: cronBotKeypair.publicKey,
        user: userKey,
        userSessionChain: userSessionChainPda,
        systemProgram: SystemProgram.programId,
      })
      .transaction();

    // Sign and send
    const { blockhash } = await connection.getLatestBlockhash();
    tx.recentBlockhash = blockhash;
    tx.feePayer = cronBotKeypair.publicKey;
    tx.sign(cronBotKeypair);

    const signature = await connection.sendRawTransaction(tx.serialize());
    await connection.confirmTransaction(signature, 'confirmed');

    console.log(`   ✅ Stored ${keys.length} access key(s) for user ${userPubkey.slice(0, 8)}...`);
    console.log(`      Tx: ${signature.slice(0, 8)}...`);

    // Update the check-out timeline event with the transaction ID
    // Use the most recent key's timestamp to find the matching event
    const mostRecentKey = keys.reduce((a, b) => a.timestamp > b.timestamp ? a : b);
    try {
      // Normalize timestamp: access key timestamps are in seconds, database uses milliseconds
      const normalizedTimestamp = mostRecentKey.timestamp < 10000000000
        ? mostRecentKey.timestamp * 1000
        : mostRecentKey.timestamp;

      // Directly update the database (more reliable than HTTP call on Railway)
      const dbUpdated = await updateActivityTransactionId(
        userPubkey,
        normalizedTimestamp,
        1, // activity_type 1 = check_out
        signature
      );

      if (dbUpdated) {
        console.log(`   �� Updated timeline event with transaction ID in database`);
      } else {
        console.log(`   ⚠️ No matching check_out event found in database (may have been cleared)`);
      }

      // Also emit socket event if io is available (for real-time updates)
      if (io) {
        // We don't have the cameraId here, so we can't emit to the right room
        // The database update is the important part for historical timeline
        console.log(`   📡 Socket.IO available for real-time updates`);
      }
    } catch (updateError) {
      console.log(`   ⚠️ Could not update timeline event (non-critical):`, updateError);
    }

    // Clear processed keys
    pendingKeys.delete(userPubkey);

    return true;

  } catch (error: any) {
    console.error(`   ❌ Failed to store access keys for user ${userPubkey.slice(0, 8)}...:`, error.message);

    // Increment retry count for all keys
    const updatedKeys = keys.map(k => ({ ...k, retryCount: k.retryCount + 1 }));

    // Remove keys that have failed too many times (5 retries)
    const retryableKeys = updatedKeys.filter(k => k.retryCount < 5);
    if (retryableKeys.length < keys.length) {
      console.log(`   🗑️ Dropped ${keys.length - retryableKeys.length} keys after max retries`);
    }

    if (retryableKeys.length > 0) {
      pendingKeys.set(userPubkey, retryableKeys);
    } else {
      pendingKeys.delete(userPubkey);
    }

    return false;
  }
}

/**
 * Process all pending access keys (called periodically or on demand)
 */
export async function processAllPendingKeys() {
  if (pendingKeys.size === 0) {
    return;
  }

  console.log('\n🔄 Processing all pending access keys...');

  const users = Array.from(pendingKeys.keys());
  console.log(`   ${users.length} users with pending keys`);

  let successCount = 0;
  let failCount = 0;

  for (const userPubkey of users) {
    const success = await processAccessKeyForUser(userPubkey);
    if (success) {
      successCount++;
    } else {
      failCount++;
    }
  }

  console.log(`   ✅ Processed: ${successCount} success, ${failCount} pending`);
}

// Get service status
export function getCleanupCronStatus() {
  return {
    running: cronBotKeypair !== null,
    cronBot: cronBotKeypair?.publicKey.toString() || null,
    pendingKeysCount: Array.from(pendingKeys.values()).reduce((sum, keys) => sum + keys.length, 0),
    usersWithPendingKeys: pendingKeys.size,
    retryIntervalMinutes: RETRY_INTERVAL_MS / 1000 / 60
  };
}

// For backwards compatibility and manual trigger
export async function triggerManualCleanup() {
  console.log('🔧 Manual key processing triggered');
  await processAllPendingKeys();
}

// Alias for clearer naming
export { triggerManualCleanup as runCleanup };
