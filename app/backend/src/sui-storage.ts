/**
 * Sui Storage Service
 *
 * Manages Sui Ed25519 keypairs for users to own Walrus blobs.
 * - Generates keypairs on-demand when user first uploads
 * - Encrypts private keys before storing in database
 * - mmoment backend controls keys for ownership transfer
 */

import * as crypto from 'crypto';
import { getUserSuiWallet, saveUserSuiWallet } from './database';

// Encryption key derived from environment secret
// In production, use a proper secret management system
const ENCRYPTION_SECRET = process.env.SUI_KEY_ENCRYPTION_SECRET || 'mmoment-sui-key-encryption-v1-secret';

/**
 * Generate a new Ed25519 keypair for Sui.
 * Returns raw bytes for the keypair.
 */
function generateSuiKeypair(): { publicKey: Buffer; privateKey: Buffer } {
  // Generate Ed25519 keypair
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519', {
    publicKeyEncoding: { type: 'spki', format: 'der' },
    privateKeyEncoding: { type: 'pkcs8', format: 'der' }
  });

  // Extract raw 32-byte keys from DER encoding
  // Ed25519 public key in DER SPKI: last 32 bytes
  // Ed25519 private key in PKCS8: bytes 16-48 (the seed)
  const rawPublicKey = publicKey.slice(-32);
  const rawPrivateKey = privateKey.slice(16, 48);

  return {
    publicKey: rawPublicKey,
    privateKey: rawPrivateKey
  };
}

/**
 * Convert Ed25519 public key to Sui address.
 * Sui uses Blake2b-256 hash with a scheme flag prefix.
 */
function publicKeyToSuiAddress(publicKey: Buffer): string {
  // Sui Ed25519 scheme flag is 0x00
  const schemeFlag = Buffer.from([0x00]);
  const prefixedKey = Buffer.concat([schemeFlag, publicKey]);

  // Blake2b-256 hash
  const hash = crypto.createHash('blake2b512').update(prefixedKey).digest();
  // Take first 32 bytes of Blake2b-512 to simulate Blake2b-256
  const addressBytes = hash.slice(0, 32);

  // Sui addresses are hex with 0x prefix
  return '0x' + addressBytes.toString('hex');
}

/**
 * Encrypt private key for database storage.
 * Uses AES-256-GCM with key derived from secret.
 */
function encryptPrivateKey(privateKey: Buffer, walletAddress: string): string {
  // Derive encryption key from secret + wallet address
  const derivedKey = crypto
    .createHash('sha256')
    .update(ENCRYPTION_SECRET + walletAddress)
    .digest();

  // Generate random IV
  const iv = crypto.randomBytes(12);

  // Encrypt with AES-256-GCM
  const cipher = crypto.createCipheriv('aes-256-gcm', derivedKey, iv);
  const encrypted = Buffer.concat([cipher.update(privateKey), cipher.final()]);
  const authTag = cipher.getAuthTag();

  // Combine: iv (12) + authTag (16) + encrypted (32)
  const combined = Buffer.concat([iv, authTag, encrypted]);
  return combined.toString('base64');
}

/**
 * Decrypt private key from database.
 */
function decryptPrivateKey(encryptedData: string, walletAddress: string): Buffer {
  // Derive decryption key from secret + wallet address
  const derivedKey = crypto
    .createHash('sha256')
    .update(ENCRYPTION_SECRET + walletAddress)
    .digest();

  // Parse combined data
  const combined = Buffer.from(encryptedData, 'base64');
  const iv = combined.slice(0, 12);
  const authTag = combined.slice(12, 28);
  const encrypted = combined.slice(28);

  // Decrypt with AES-256-GCM
  const decipher = crypto.createDecipheriv('aes-256-gcm', derivedKey, iv);
  decipher.setAuthTag(authTag);
  const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);

  return decrypted;
}

/**
 * Get or create a Sui wallet for a user.
 * If the user doesn't have a Sui wallet yet, generates one.
 *
 * @param walletAddress - User's Solana wallet address
 * @returns Sui address (0x prefixed hex)
 */
export async function getOrCreateSuiWallet(walletAddress: string): Promise<{
  suiAddress: string;
  isNew: boolean;
}> {
  // Check if user already has a Sui wallet
  const existing = await getUserSuiWallet(walletAddress);
  if (existing) {
    console.log(`📦 Found existing Sui wallet for ${walletAddress.slice(0, 8)}...`);
    return {
      suiAddress: existing.suiAddress,
      isNew: false
    };
  }

  // Generate new keypair
  console.log(`🔑 Generating new Sui wallet for ${walletAddress.slice(0, 8)}...`);
  const { publicKey, privateKey } = generateSuiKeypair();
  const suiAddress = publicKeyToSuiAddress(publicKey);

  // Encrypt private key for storage
  const encryptedKey = encryptPrivateKey(privateKey, walletAddress);

  // Save to database
  await saveUserSuiWallet({
    walletAddress,
    suiAddress,
    encryptedSuiKey: encryptedKey,
    createdAt: new Date()
  });

  console.log(`✅ Created Sui wallet: ${suiAddress.slice(0, 16)}...`);

  return {
    suiAddress,
    isNew: true
  };
}

/**
 * Get the private key for a user's Sui wallet.
 * Used for signing Sui transactions (e.g., blob ownership transfer).
 *
 * @param walletAddress - User's Solana wallet address
 * @returns Raw 32-byte Ed25519 private key or null if not found
 */
export async function getSuiPrivateKey(walletAddress: string): Promise<Buffer | null> {
  const suiWallet = await getUserSuiWallet(walletAddress);
  if (!suiWallet) {
    return null;
  }

  return decryptPrivateKey(suiWallet.encryptedSuiKey, walletAddress);
}

/**
 * Get Sui address for a user without creating if it doesn't exist.
 *
 * @param walletAddress - User's Solana wallet address
 * @returns Sui address or null if not found
 */
export async function getSuiAddress(walletAddress: string): Promise<string | null> {
  const suiWallet = await getUserSuiWallet(walletAddress);
  return suiWallet?.suiAddress || null;
}

/**
 * Export keypair in Sui CLI format (for manual operations).
 * Only use for debugging/admin purposes.
 */
export async function exportSuiKeypairForCLI(walletAddress: string): Promise<string | null> {
  const suiWallet = await getUserSuiWallet(walletAddress);
  if (!suiWallet) {
    return null;
  }

  const privateKey = decryptPrivateKey(suiWallet.encryptedSuiKey, walletAddress);

  // Sui CLI expects base64-encoded private key with scheme flag
  const schemeFlag = Buffer.from([0x00]); // Ed25519
  const suiFormat = Buffer.concat([schemeFlag, privateKey]);

  return suiFormat.toString('base64');
}
