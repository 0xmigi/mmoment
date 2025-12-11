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

/**
 * Convert Ed25519 private key to X25519 private key.
 * Used for NaCl sealed box decryption.
 *
 * X25519 private key is derived by SHA-512 hashing the Ed25519 seed,
 * taking the first 32 bytes, and clamping for Curve25519.
 */
export function ed25519ToX25519PrivateKey(ed25519PrivateKey: Buffer): Buffer {
  // SHA-512 hash of the 32-byte seed
  const hash = crypto.createHash('sha512').update(ed25519PrivateKey).digest();
  const x25519Private = Buffer.from(hash.slice(0, 32));

  // Clamp for Curve25519
  x25519Private[0] &= 248;
  x25519Private[31] &= 127;
  x25519Private[31] |= 64;

  return x25519Private;
}

/**
 * Convert Ed25519 public key to X25519 public key.
 * Uses the formula: x = (1 + y) / (1 - y) mod p
 * where y is the Ed25519 point's y-coordinate.
 *
 * This is a simplified implementation that works for key conversion.
 */
export function ed25519ToX25519PublicKey(ed25519PublicKey: Buffer): Buffer {
  // Ed25519 public key is a compressed point (y-coordinate with sign bit)
  // For X25519, we need to convert Edwards to Montgomery form

  // Use crypto's X25519 key derivation by generating a keypair
  // and using the public key derivation. This is a workaround since
  // Node.js crypto doesn't expose the conversion directly.

  // For now, we'll use a mathematical conversion
  // The Ed25519 public key encodes (x, y) in compressed form as y with the sign of x
  // Montgomery u = (1 + y) / (1 - y) mod p

  const p = BigInt('57896044618658097711785492504343953926634992332820282019728792003956564819949'); // 2^255 - 19

  // Extract y from the 32-byte compressed Ed25519 public key
  const yBytes = Buffer.from(ed25519PublicKey);
  const signBit = (yBytes[31] & 0x80) >> 7;
  yBytes[31] &= 0x7f; // Clear sign bit to get y

  // Convert to BigInt (little-endian)
  let y = BigInt(0);
  for (let i = 0; i < 32; i++) {
    y += BigInt(yBytes[i]) << BigInt(8 * i);
  }

  // Calculate u = (1 + y) / (1 - y) mod p
  const one = BigInt(1);
  const numerator = (one + y) % p;
  const denominator = (p + one - y) % p;

  // Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
  const modInverse = (a: bigint, m: bigint): bigint => {
    let result = BigInt(1);
    let base = a % m;
    let exp = m - BigInt(2);
    while (exp > BigInt(0)) {
      if (exp % BigInt(2) === BigInt(1)) {
        result = (result * base) % m;
      }
      exp = exp / BigInt(2);
      base = (base * base) % m;
    }
    return result;
  };

  const u = (numerator * modInverse(denominator, p)) % p;

  // Convert u to 32-byte little-endian buffer
  const uBytes = Buffer.alloc(32);
  let remaining = u;
  for (let i = 0; i < 32; i++) {
    uBytes[i] = Number(remaining & BigInt(0xff));
    remaining = remaining >> BigInt(8);
  }

  return uBytes;
}

/**
 * Get the X25519 public key for a user's Sui wallet.
 * Used by Jetson for encrypting content keys with NaCl sealed box.
 */
export async function getSuiX25519PublicKey(walletAddress: string): Promise<{
  suiAddress: string;
  x25519PublicKey: Buffer;
  ed25519PublicKey: Buffer;
} | null> {
  const suiWallet = await getUserSuiWallet(walletAddress);
  if (!suiWallet) {
    return null;
  }

  // Reconstruct Ed25519 public key from private key
  const privateKey = decryptPrivateKey(suiWallet.encryptedSuiKey, walletAddress);

  // Generate Ed25519 keypair from the seed to get public key
  const { publicKey } = crypto.generateKeyPairSync('ed25519', {
    privateKeyEncoding: { type: 'pkcs8', format: 'der' },
    publicKeyEncoding: { type: 'spki', format: 'der' }
  });

  // Actually, we need to derive from the seed. Let's use a different approach.
  // Create key object from the private key and derive public
  const keyObject = crypto.createPrivateKey({
    key: Buffer.concat([
      Buffer.from('302e020100300506032b6570042204', 'hex'), // PKCS8 prefix for Ed25519
      privateKey
    ]),
    format: 'der',
    type: 'pkcs8'
  });

  const publicKeyObj = crypto.createPublicKey(keyObject);
  const publicKeyDer = publicKeyObj.export({ type: 'spki', format: 'der' });
  const ed25519PublicKey = Buffer.from(publicKeyDer.slice(-32));

  // Convert to X25519
  const x25519PublicKey = ed25519ToX25519PublicKey(ed25519PublicKey);

  return {
    suiAddress: suiWallet.suiAddress,
    ed25519PublicKey,
    x25519PublicKey
  };
}

/**
 * Decrypt a content key that was encrypted with NaCl sealed box.
 *
 * Sealed box format: ephemeral_pubkey (32) + ciphertext + poly1305_tag (16)
 * The actual encryption uses XSalsa20-Poly1305 with a key derived from
 * ephemeral ECDH.
 */
export async function decryptSealedBox(
  encryptedKey: Buffer,
  walletAddress: string
): Promise<Buffer | null> {
  try {
    const suiWallet = await getUserSuiWallet(walletAddress);
    if (!suiWallet) {
      console.error('No Sui wallet found for decryption');
      return null;
    }

    // Get Ed25519 private key and convert to X25519
    const ed25519PrivateKey = decryptPrivateKey(suiWallet.encryptedSuiKey, walletAddress);
    const x25519PrivateKey = ed25519ToX25519PrivateKey(ed25519PrivateKey);

    // Get X25519 public key
    const keyInfo = await getSuiX25519PublicKey(walletAddress);
    if (!keyInfo) {
      return null;
    }
    const x25519PublicKey = keyInfo.x25519PublicKey;

    // Sealed box format: ephemeral_pubkey (32) + box_content
    // box_content = nonce (24, derived) + ciphertext + tag (16)
    // Actually NaCl sealed box nonce is derived from ephemeral_pubkey + recipient_pubkey

    // For this we need a proper NaCl implementation
    // Node.js crypto doesn't have native sealed box support
    // We'll use tweetnacl via a synchronous call

    // Since we're in Node.js backend, we can use the tweetnacl package directly
    const nacl = await import('tweetnacl');
    const sealedBox = await import('tweetnacl-sealedbox-js');

    const decrypted = sealedBox.default.open(
      new Uint8Array(encryptedKey),
      new Uint8Array(x25519PublicKey),
      new Uint8Array(x25519PrivateKey)
    );

    if (!decrypted) {
      console.error('Sealed box decryption failed');
      return null;
    }

    return Buffer.from(decrypted);
  } catch (error) {
    console.error('Error decrypting sealed box:', error);
    return null;
  }
}

/**
 * Decrypt AES-256-GCM encrypted content.
 * Format: nonce (12 bytes) + ciphertext + tag (16 bytes)
 */
export function decryptAesGcm(encryptedData: Buffer, contentKey: Buffer): Buffer {
  // Extract nonce (first 12 bytes)
  const nonce = encryptedData.slice(0, 12);
  // Remaining is ciphertext + auth tag (last 16 bytes are the tag)
  const ciphertext = encryptedData.slice(12);

  const decipher = crypto.createDecipheriv('aes-256-gcm', contentKey, nonce);
  // In Node.js, the auth tag is the last 16 bytes of the ciphertext
  const authTag = ciphertext.slice(-16);
  const actualCiphertext = ciphertext.slice(0, -16);

  decipher.setAuthTag(authTag);
  const decrypted = Buffer.concat([
    decipher.update(actualCiphertext),
    decipher.final()
  ]);

  return decrypted;
}
