// Gas Sponsorship Service
import { Connection, Keypair, PublicKey, Transaction, SystemProgram } from '@solana/web3.js';

// Constants
const FREE_TIER_LIMIT = 10;
const ALLOWED_PROGRAMS = [
  'E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL', // Camera Network Program
  SystemProgram.programId.toString(), // System Program (for account creation)
];

// In-memory storage for sponsored transaction counts
// In production, this should use a database
interface UserSponsorshipData {
  count: number;
  firstUsed: Date;
  lastUsed: Date;
  actions: Array<{ action: string; timestamp: Date }>;
}

const sponsoredTransactions = new Map<string, UserSponsorshipData>();

// Fee Payer Configuration
let feePayerKeypair: Keypair | null = null;
let connection: Connection | null = null;

export function initializeGasSponsorshipService(rpcUrl: string, feePayerSecretKey: string) {
  try {
    // Parse the secret key from JSON array format
    const secretKeyArray = JSON.parse(feePayerSecretKey);
    feePayerKeypair = Keypair.fromSecretKey(new Uint8Array(secretKeyArray));
    connection = new Connection(rpcUrl, 'confirmed');

    console.log('✅ Gas Sponsorship Service Initialized');
    console.log(`   Fee Payer: ${feePayerKeypair.publicKey.toString()}`);
    console.log(`   RPC URL: ${rpcUrl}`);

    // Check balance
    checkFeePayerBalance();

    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Gas Sponsorship Service:', error);
    return false;
  }
}

export async function checkFeePayerBalance(): Promise<number> {
  if (!connection || !feePayerKeypair) {
    console.error('Gas Sponsorship Service not initialized');
    return 0;
  }

  try {
    const balance = await connection.getBalance(feePayerKeypair.publicKey);
    const solBalance = balance / 1e9;
    console.log(`💰 Fee Payer Balance: ${solBalance} SOL`);

    if (solBalance < 0.1) {
      console.warn('⚠️  WARNING: Fee payer balance is low!');
    }

    return solBalance;
  } catch (error) {
    console.error('Failed to check fee payer balance:', error);
    return 0;
  }
}

export function getUserSponsorshipStatus(userWallet: string): {
  eligible: boolean;
  count: number;
  remaining: number;
  message: string;
} {
  const userData = sponsoredTransactions.get(userWallet);
  const count = userData?.count || 0;
  const remaining = Math.max(0, FREE_TIER_LIMIT - count);

  return {
    eligible: count < FREE_TIER_LIMIT,
    count,
    remaining,
    message: count >= FREE_TIER_LIMIT
      ? `You've used all ${FREE_TIER_LIMIT} free interactions. Please add SOL to your wallet to continue.`
      : `${remaining} free interactions remaining.`
  };
}

export async function sponsorTransaction(
  userWallet: string,
  serializedTransaction: string,
  action: string
): Promise<{
  success: boolean;
  transaction?: string;
  signature?: string;
  error?: string;
  remaining?: number;
}> {
  // Validate initialization
  if (!connection || !feePayerKeypair) {
    return {
      success: false,
      error: 'Gas sponsorship service not initialized'
    };
  }

  // Check user eligibility
  const status = getUserSponsorshipStatus(userWallet);
  if (!status.eligible) {
    return {
      success: false,
      error: status.message
    };
  }

  try {
    // Deserialize the transaction
    const txBuffer = Buffer.from(serializedTransaction, 'base64');
    const transaction = Transaction.from(txBuffer);

    // Security validations
    const userPubkey = new PublicKey(userWallet);

    // 1. Check that user is a required signer (in the transaction, not necessarily signed yet)
    // The transaction should have user in the signers list (feePayer will be there too after we set it)
    // We'll validate the user is actually a signer by checking the instruction accounts
    const userIsInInstructions = transaction.instructions.some(ix =>
      ix.keys.some(key =>
        key.pubkey.equals(userPubkey) && key.isSigner
      )
    );
    if (!userIsInInstructions) {
      return {
        success: false,
        error: 'User must be a signer in the transaction'
      };
    }

    // 2. Validate program IDs (only allow camera network and system program)
    for (const instruction of transaction.instructions) {
      const programId = instruction.programId.toString();
      if (!ALLOWED_PROGRAMS.includes(programId)) {
        console.error(`Unauthorized program: ${programId}`);
        return {
          success: false,
          error: `Unauthorized program in transaction: ${programId}`
        };
      }
    }

    // 3. Replace user as payer in instruction accounts with our fee payer
    // This allows gas sponsorship for account rent (not just transaction fees)
    for (const instruction of transaction.instructions) {
      // Find all instances where the user appears as a signer
      const userSignerIndices: number[] = [];
      instruction.keys.forEach((key, index) => {
        if (key.pubkey.equals(userPubkey) && key.isSigner) {
          userSignerIndices.push(index);
        }
      });

      // If user appears multiple times as signer, the second one is likely the payer
      // In check_in instruction: index 0 = user, index 1 = payer
      if (userSignerIndices.length >= 2) {
        const payerIndex = userSignerIndices[1]; // Second occurrence is the payer
        console.log(`🔄 Replacing payer account at index ${payerIndex} with fee payer`);
        console.log(`   User: ${userPubkey.toString().slice(0, 8)}...`);
        console.log(`   Fee Payer: ${feePayerKeypair!.publicKey.toString().slice(0, 8)}...`);

        instruction.keys[payerIndex] = {
          pubkey: feePayerKeypair!.publicKey,
          isSigner: true,
          isWritable: true
        };
      } else {
        console.log(`ℹ️  User appears ${userSignerIndices.length} time(s) - no payer replacement needed`);
      }
    }

    // Set fee payer and get recent blockhash
    transaction.feePayer = feePayerKeypair.publicKey;
    transaction.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;

    // Log transaction details before signing
    console.log('📋 Transaction details before signing:');
    console.log('   Fee payer:', transaction.feePayer.toString());
    console.log('   Instruction accounts:');
    transaction.instructions.forEach((ix, idx) => {
      console.log(`   Instruction ${idx}:`, ix.keys.map((k, i) => ({
        index: i,
        pubkey: k.pubkey.toString().slice(0, 8) + '...',
        isSigner: k.isSigner,
        isWritable: k.isWritable
      })));
    });

    // Sign with fee payer
    transaction.partialSign(feePayerKeypair);

    console.log('✍️  Fee payer signed. Signatures:', transaction.signatures.map(s =>
      `${s.publicKey.toString().slice(0,8)}... signed: ${s.signature !== null}`
    ));

    // Serialize the sponsored transaction to send back to frontend
    const sponsoredTx = transaction.serialize({
      requireAllSignatures: false, // User hasn't signed yet
      verifySignatures: false
    });

    // Update usage counter
    const userData = sponsoredTransactions.get(userWallet) || {
      count: 0,
      firstUsed: new Date(),
      lastUsed: new Date(),
      actions: []
    };

    userData.count += 1;
    userData.lastUsed = new Date();
    userData.actions.push({ action, timestamp: new Date() });
    sponsoredTransactions.set(userWallet, userData);

    console.log(`✅ Sponsored transaction for ${userWallet.slice(0, 8)}... (${userData.count}/${FREE_TIER_LIMIT})`);
    console.log(`   Action: ${action}`);

    return {
      success: true,
      transaction: sponsoredTx.toString('base64'),
      remaining: FREE_TIER_LIMIT - userData.count
    };

  } catch (error) {
    console.error('Error sponsoring transaction:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to sponsor transaction'
    };
  }
}

// Get sponsorship statistics (for debugging/monitoring)
export function getSponsorshipStats() {
  const totalUsers = sponsoredTransactions.size;
  const totalTransactions = Array.from(sponsoredTransactions.values())
    .reduce((sum, user) => sum + user.count, 0);
  const usersAtLimit = Array.from(sponsoredTransactions.values())
    .filter(user => user.count >= FREE_TIER_LIMIT).length;

  return {
    totalUsers,
    totalTransactions,
    usersAtLimit,
    activeUsers: totalUsers - usersAtLimit,
    averagePerUser: totalUsers > 0 ? (totalTransactions / totalUsers).toFixed(2) : 0
  };
}

// Reset user sponsorship count (for testing/admin purposes)
export function resetUserSponsorship(userWallet: string): boolean {
  const existed = sponsoredTransactions.has(userWallet);
  sponsoredTransactions.delete(userWallet);
  return existed;
}

// Clear all sponsorship data (for testing)
export function clearAllSponsorships(): number {
  const count = sponsoredTransactions.size;
  sponsoredTransactions.clear();
  return count;
}
