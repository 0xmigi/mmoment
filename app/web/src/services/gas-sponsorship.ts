// Gas Sponsorship Service - Frontend
import { Transaction, Connection, PublicKey } from '@solana/web3.js';
import { CONFIG } from '../core/config';

const BACKEND_URL = CONFIG.BACKEND_URL;

export interface SponsorshipStatus {
  eligible: boolean;
  count: number;
  remaining: number;
  message: string;
}

export interface SponsorshipResult {
  success: boolean;
  transaction?: Transaction;
  remaining?: number;
  message?: string;
  error?: string;
  requiresUserPayment?: boolean;
}

/**
 * Check if user is eligible for sponsored transactions
 */
export async function checkSponsorshipStatus(userWallet: string): Promise<SponsorshipStatus> {
  try {
    const response = await fetch(`${BACKEND_URL}/api/sponsorship-status/${userWallet}`);

    if (!response.ok) {
      console.error('Failed to check sponsorship status:', response.statusText);
      return {
        eligible: false,
        count: 0,
        remaining: 0,
        message: 'Unable to check sponsorship status'
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Error checking sponsorship status:', error);
    return {
      eligible: false,
      count: 0,
      remaining: 0,
      message: 'Network error'
    };
  }
}

/**
 * Request gas sponsorship for a transaction
 *
 * @param userWallet - User's wallet public key
 * @param transaction - The transaction to sponsor (must be signed by user first!)
 * @param action - Description of the action (e.g., 'check_in', 'enroll_face')
 * @param connection - Solana connection for sending the transaction
 * @returns Result with success status and either transaction signature or error
 */
export async function requestSponsoredTransaction(
  userWallet: string,
  transaction: Transaction,
  action: string,
  _connection: Connection
): Promise<SponsorshipResult> {
  try {
    // Serialize the user-signed transaction
    const serializedTx = transaction.serialize({
      requireAllSignatures: false,
      verifySignatures: false
    }).toString('base64');

    console.log(`🔄 Requesting sponsorship for action: ${action}`);

    // Send to backend for sponsorship
    const response = await fetch(`${BACKEND_URL}/api/sponsor-transaction`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userWallet,
        transaction: serializedTx,
        action
      })
    });

    const result = await response.json();

    if (!result.success) {
      console.error('❌ Sponsorship denied:', result.error);
      return {
        success: false,
        error: result.error,
        requiresUserPayment: result.requiresUserPayment || false
      };
    }

    // Deserialize the sponsored transaction
    const sponsoredTxBuffer = Buffer.from(result.transaction, 'base64');
    const sponsoredTx = Transaction.from(sponsoredTxBuffer);

    console.log(`✅ Transaction sponsored! ${result.remaining} free interactions remaining`);

    return {
      success: true,
      transaction: sponsoredTx,
      remaining: result.remaining,
      message: result.message
    };

  } catch (error) {
    console.error('Error requesting sponsored transaction:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to request sponsorship'
    };
  }
}

/**
 * Build, sign, and submit a sponsored transaction
 *
 * This is the main function to use for gas sponsorship.
 * It handles the full flow: build tx → sign with user → request sponsorship → submit
 */
export async function buildAndSponsorTransaction(
  userWallet: PublicKey,
  walletSigner: any, // Dynamic wallet signer
  buildTransaction: () => Promise<Transaction>,
  action: string,
  connection: Connection
): Promise<{
  success: boolean;
  signature?: string;
  error?: string;
  requiresUserPayment?: boolean;
}> {
  try {
    console.log(`🔨 Building transaction for ${action}...`);

    // Step 1: Build the transaction
    const transaction = await buildTransaction();

    // Step 2: Add temporary blockhash for serialization (backend will replace it)
    const { blockhash } = await connection.getLatestBlockhash();
    transaction.recentBlockhash = blockhash;
    transaction.feePayer = userWallet; // Temporary, backend will replace

    // Step 3: Serialize the unsigned transaction
    const serializedTx = transaction.serialize({
      requireAllSignatures: false,
      verifySignatures: false
    }).toString('base64');

    // Step 4: Request backend to sponsor it (backend adds fee payer + blockhash)
    console.log(`🔄 Requesting sponsorship for action: ${action}`);
    const response = await fetch(`${BACKEND_URL}/api/sponsor-transaction`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userWallet: userWallet.toString(),
        transaction: serializedTx,
        action
      })
    });

    const result = await response.json();

    if (!result.success) {
      console.error('❌ Sponsorship denied:', result.error);
      return {
        success: false,
        error: result.error,
        requiresUserPayment: result.requiresUserPayment || false
      };
    }

    console.log(`✅ Transaction sponsored! ${result.remaining} free interactions remaining`);

    // Step 4: Deserialize the sponsored transaction (has fee payer + blockhash)
    const sponsoredTxBuffer = Buffer.from(result.transaction, 'base64');
    const sponsoredTx = Transaction.from(sponsoredTxBuffer);

    // Step 5: NOW user signs the sponsored transaction
    console.log(`✍️  Requesting user signature...`);
    const userSignedTx = await walletSigner.signTransaction(sponsoredTx);

    // Step 6: Send the fully-signed transaction
    console.log(`📤 Sending sponsored transaction...`);
    const signature = await connection.sendRawTransaction(
      userSignedTx.serialize()
    );

    // Step 7: Confirm the transaction
    console.log(`⏳ Confirming transaction ${signature}...`);
    await connection.confirmTransaction(signature, 'confirmed');

    console.log(`✅ Transaction confirmed: ${signature}`);

    return {
      success: true,
      signature
    };

  } catch (error) {
    console.error('Error in sponsored transaction flow:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Transaction failed'
    };
  }
}

/**
 * Display sponsorship status to user
 */
export function formatSponsorshipMessage(status: SponsorshipStatus): string {
  if (!status.eligible) {
    return `❌ ${status.message}`;
  }

  if (status.remaining <= 10) {
    return `⚠️  ${status.remaining} free interactions remaining`;
  }

  return `✅ ${status.remaining} free interactions available`;
}
