// Kora Gas Sponsorship Client
// Talks to a local or deployed Kora JSON-RPC server for gasless transactions.
// Replaces the old gas-sponsorship.ts backend proxy approach.

import { Transaction, Connection, PublicKey } from '@solana/web3.js';
import { CONFIG } from '../core/config';

// Kora server URL — from centralized config
const KORA_URL = CONFIG.KORA_URL;

interface KoraRpcResponse<T> {
  jsonrpc: string;
  id: number;
  result?: T;
  error?: { code: number; message: string };
}

interface KoraConfig {
  fee_payer: string;
  allowed_programs: string[];
  allowed_tokens: string[];
}

interface SignTransactionResult {
  signed_transaction: string; // base64 encoded signed transaction
  signer_pubkey: string;
}

interface SignAndSendResult {
  signature: string;
}

// ─── Low-level JSON-RPC helper ───────────────────────────────────

async function koraRpc<T>(method: string, params?: unknown): Promise<T> {
  const body = JSON.stringify({
    id: 1,
    jsonrpc: '2.0',
    method,
    params: params ?? undefined,
  });

  const response = await fetch(KORA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });

  const json: KoraRpcResponse<T> = await response.json();

  if (json.error) {
    throw new Error(`Kora RPC Error ${json.error.code}: ${json.error.message}`);
  }

  return json.result as T;
}

// ─── Public API ──────────────────────────────────────────────────

/**
 * Get the Kora fee payer's public key.
 * Use this to set as feePayer on transactions before sending to Kora.
 */
export async function getKoraFeePayer(): Promise<PublicKey> {
  const result = await koraRpc<{ signer_address: string }>('getPayerSigner');
  return new PublicKey(result.signer_address);
}

/**
 * Get Kora server config (allowed programs, tokens, etc.)
 */
export async function getKoraConfig(): Promise<KoraConfig> {
  return koraRpc<KoraConfig>('getConfig');
}

/**
 * Check if Kora is reachable and healthy
 */
export async function isKoraAvailable(): Promise<boolean> {
  try {
    await getKoraConfig();
    return true;
  } catch {
    return false;
  }
}

/**
 * Send a transaction to Kora for fee payer signing only.
 * Returns the transaction with Kora's signature added (user still needs to sign).
 */
export async function koraSignTransaction(
  transaction: Transaction
): Promise<Transaction> {
  const serialized = transaction.serialize({
    requireAllSignatures: false,
    verifySignatures: false,
  }).toString('base64');

  const result = await koraRpc<SignTransactionResult>('signTransaction', {
    transaction: serialized,
  });

  return Transaction.from(Buffer.from(result.signed_transaction, 'base64'));
}

/**
 * Send a transaction to Kora for signing AND submission to the network.
 * The transaction must already be signed by the user.
 * Returns the transaction signature.
 */
export async function koraSignAndSend(
  transaction: Transaction
): Promise<string> {
  const serialized = transaction.serialize({
    requireAllSignatures: false,
    verifySignatures: false,
  }).toString('base64');

  const result = await koraRpc<SignAndSendResult>('signAndSendTransaction', {
    transaction: serialized,
  });

  return result.signature;
}

// ─── High-level sponsored transaction flow ───────────────────────

/**
 * Build, sponsor, sign, and submit a transaction via Kora.
 *
 * This is the main function components should use. It handles the full flow:
 *   1. Build the transaction (caller provides this)
 *   2. Set Kora as fee payer
 *   3. Send to Kora for fee payer signature
 *   4. User signs the sponsored transaction
 *   5. Send fully-signed transaction to the network
 *
 * @param userWallet - User's public key
 * @param walletSigner - Dynamic wallet signer (has .signTransaction)
 * @param buildTransaction - Async function that builds the unsigned transaction
 * @param connection - Solana connection
 * @param action - Description for logging (e.g. 'check_in', 'enroll_face')
 */
export async function buildAndSubmitSponsored(
  _userWallet: PublicKey,
  walletSigner: { signTransaction: (tx: Transaction) => Promise<Transaction> },
  buildTransaction: () => Promise<Transaction>,
  connection: Connection,
  action: string = 'unknown'
): Promise<{
  success: boolean;
  signature?: string;
  error?: string;
}> {
  try {
    // 1. Get Kora's fee payer pubkey
    const feePayer = await getKoraFeePayer();

    // 2. Build the transaction
    const transaction = await buildTransaction();

    // 3. Set fee payer to Kora and add blockhash
    transaction.feePayer = feePayer;
    const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
    transaction.recentBlockhash = blockhash;

    // 4. Send to Kora for fee payer signature
    const sponsoredTx = await koraSignTransaction(transaction);

    // 5. User signs the sponsored transaction
    const userSignedTx = await walletSigner.signTransaction(sponsoredTx);

    // 6. Submit to the network
    const signature = await connection.sendRawTransaction(
      userSignedTx.serialize(),
      { skipPreflight: false }
    );

    // 7. Confirm
    await connection.confirmTransaction(
      { signature, blockhash, lastValidBlockHeight },
      'confirmed'
    );

    console.log(`[Kora] ${action} confirmed: ${signature}`);

    return { success: true, signature };

  } catch (error) {
    console.error(`[Kora] ${action} failed:`, error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Transaction failed',
    };
  }
}

