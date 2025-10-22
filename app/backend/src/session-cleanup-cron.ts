// Session Cleanup Cron Job
// Automatically checks out expired sessions and collects rent as reward
import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import { IDL } from './idl';
import { Server } from 'socket.io';

const CLEANUP_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes
const PROGRAM_ID = new PublicKey('E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL');

let connection: Connection | null = null;
let cronBotKeypair: Keypair | null = null;
let program: Program | null = null;
let cleanupIntervalId: NodeJS.Timeout | null = null;
let io: Server | null = null;

export function initializeSessionCleanupCron(rpcUrl: string, cronBotSecretKey: string, socketServer?: Server) {
  try {
    // Parse the secret key from JSON array format
    const secretKeyArray = JSON.parse(cronBotSecretKey);
    cronBotKeypair = Keypair.fromSecretKey(new Uint8Array(secretKeyArray));
    connection = new Connection(rpcUrl, 'confirmed');

    // Store Socket.IO server reference for timeline events
    if (socketServer) {
      io = socketServer;
      console.log('   Socket.IO server connected for timeline events');
    }

    // Create Anchor provider and program
    const wallet = new Wallet(cronBotKeypair);
    const provider = new AnchorProvider(connection, wallet, { commitment: 'confirmed' });
    program = new Program(IDL as any, PROGRAM_ID, provider);

    console.log('✅ Session Cleanup Cron Initialized');
    console.log(`   Cron Bot: ${cronBotKeypair.publicKey.toString()}`);
    console.log(`   Cleanup Interval: ${CLEANUP_INTERVAL_MS / 1000 / 60} minutes`);

    // Start the cron job
    startCleanupCron();

    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Session Cleanup Cron:', error);
    return false;
  }
}

function startCleanupCron() {
  if (cleanupIntervalId) {
    console.log('⚠️  Cleanup cron already running');
    return;
  }

  // Run immediately on start
  runCleanup();

  // Then run every interval
  cleanupIntervalId = setInterval(() => {
    runCleanup();
  }, CLEANUP_INTERVAL_MS);

  console.log('🤖 Session cleanup cron started');
}

export function stopCleanupCron() {
  if (cleanupIntervalId) {
    clearInterval(cleanupIntervalId);
    cleanupIntervalId = null;
    console.log('🛑 Session cleanup cron stopped');
  }
}

async function runCleanup() {
  if (!connection || !cronBotKeypair || !program) {
    console.error('Cleanup cron not initialized');
    return;
  }

  try {
    console.log('\n🧹 [Cleanup Cron] Starting session cleanup scan...');
    const now = Math.floor(Date.now() / 1000);

    // Fetch all session accounts (102 bytes)
    const sessionAccounts = await connection.getProgramAccounts(PROGRAM_ID, {
      filters: [
        { dataSize: 102 } // UserSession size
      ]
    });

    console.log(`   Found ${sessionAccounts.length} total sessions`);
    console.log(`   Current time: ${now} (${new Date(now * 1000).toISOString()})`);

    let expiredCount = 0;
    let cleanedCount = 0;
    let errorCount = 0;

    // Check each session for expiration
    for (const accountInfo of sessionAccounts) {
      try {
        const session = program.coder.accounts.decode('userSession', accountInfo.account.data);
        const sessionPubkey = accountInfo.pubkey;

        // Debug: Log session details
        const expiresAt = session.autoCheckoutAt.toNumber();
        const expiresDate = new Date(expiresAt * 1000);
        const isExpired = now > expiresAt;

        console.log(`   Session ${sessionPubkey.toString().slice(0, 8)}...`);
        console.log(`      User: ${session.user.toString().slice(0, 8)}...`);
        console.log(`      Expires at: ${expiresDate.toISOString()} (${expiresAt})`);
        console.log(`      Now: ${new Date(now * 1000).toISOString()} (${now})`);
        console.log(`      Is expired: ${isExpired}`);

        // Check if expired
        if (isExpired) {
          expiredCount++;

          console.log(`   📤 Cleaning up expired session:`);
          console.log(`      Session: ${sessionPubkey.toString().slice(0, 8)}...`);
          console.log(`      User: ${session.user.toString().slice(0, 8)}...`);
          console.log(`      Expired at: ${new Date(session.autoCheckoutAt.toNumber() * 1000).toISOString()}`);

          try {
            // Build check-out transaction
            const checkOutTx = await program.methods
              .checkOut()
              .accounts({
                closer: cronBotKeypair.publicKey,
                camera: session.camera,
                session: sessionPubkey,
                sessionUser: session.user,
                rentDestination: cronBotKeypair.publicKey, // Cron bot collects rent!
              })
              .transaction();

            // Sign and send
            const { blockhash } = await connection.getLatestBlockhash();
            checkOutTx.recentBlockhash = blockhash;
            checkOutTx.feePayer = cronBotKeypair.publicKey;
            checkOutTx.sign(cronBotKeypair);

            const signature = await connection.sendRawTransaction(checkOutTx.serialize());
            await connection.confirmTransaction(signature, 'confirmed');

            cleanedCount++;
            console.log(`      ✅ Cleaned up! Tx: ${signature.slice(0, 8)}...`);
            console.log(`         Rent collected by cron bot`);

            // Emit timeline event for auto-checkout
            if (io) {
              const timelineEvent = {
                type: 'auto_check_out',
                user: {
                  address: session.user.toString(),
                },
                timestamp: Date.now(),
                cameraId: session.camera.toString(),
                transactionId: signature,
              };

              // Emit to the specific camera room (use 'timelineEvent' not 'newTimelineEvent')
              io.to(session.camera.toString()).emit('timelineEvent', timelineEvent);
              console.log(`      📡 Timeline event emitted for auto-checkout to room ${session.camera.toString().slice(0, 8)}...`);
            }

          } catch (cleanupError: any) {
            errorCount++;
            console.error(`      ❌ Failed to cleanup:`, cleanupError.message);
          }
        }
      } catch (decodeError: any) {
        // Skip sessions we can't decode (old format, etc.)
        console.log(`   ⚠️  Could not decode session:`, decodeError.message);
        continue;
      }
    }

    console.log(`\n   Summary:`);
    console.log(`   - Total sessions: ${sessionAccounts.length}`);
    console.log(`   - Expired: ${expiredCount}`);
    console.log(`   - Cleaned up: ${cleanedCount}`);
    console.log(`   - Errors: ${errorCount}`);
    console.log(`   🏁 Cleanup scan complete\n`);

  } catch (error) {
    console.error('Error during cleanup scan:', error);
  }
}

// Export for manual trigger (useful for testing)
export async function triggerManualCleanup() {
  console.log('🔧 Manual cleanup triggered');
  await runCleanup();
}

// Get cron status
export function getCleanupCronStatus() {
  return {
    running: cleanupIntervalId !== null,
    cronBot: cronBotKeypair?.publicKey.toString() || null,
    intervalMinutes: CLEANUP_INTERVAL_MS / 1000 / 60
  };
}
