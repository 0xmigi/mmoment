/**
 * Walrus Upload Service with Upload Relay
 *
 * Uses the official @mysten/walrus SDK with upload relay for fast uploads.
 * This service receives encrypted blobs from Jetson and uploads them to Walrus.
 */

import { SuiClient, getFullnodeUrl } from '@mysten/sui/client';
import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';
import { walrus } from '@mysten/walrus';
import { fromBase64 } from '@mysten/sui/utils';

// Walrus configuration
const WALRUS_UPLOAD_RELAY = process.env.WALRUS_UPLOAD_RELAY || 'https://upload-relay.mainnet.walrus.space';
const WALRUS_AGGREGATOR = process.env.WALRUS_AGGREGATOR || 'https://aggregator.walrus-mainnet.walrus.space';
const SUI_NETWORK = (process.env.SUI_NETWORK || 'mainnet') as 'mainnet' | 'testnet' | 'devnet';

// Backend's Sui wallet for paying storage costs
// This should be a funded wallet with SUI and WAL tokens
const BACKEND_SUI_PRIVATE_KEY = process.env.BACKEND_SUI_PRIVATE_KEY;

let walrusClient: ReturnType<typeof createWalrusClient> | null = null;
let backendKeypair: Ed25519Keypair | null = null;

interface WalrusUploadResult {
  blobId: string;
  downloadUrl: string;
  uploadDurationMs: number;
}

/**
 * Create a Sui client extended with Walrus capabilities
 */
function createWalrusClient() {
  const suiClient = new SuiClient({
    url: getFullnodeUrl(SUI_NETWORK),
  });

  // Extend with walrus and upload relay
  // @ts-ignore - $extend is dynamically added by the SDK
  return suiClient.$extend(
    walrus({
      network: SUI_NETWORK,
      aggregatorUrl: WALRUS_AGGREGATOR,
      uploadRelay: {
        host: WALRUS_UPLOAD_RELAY,
        sendTip: {
          max: 10_000_000, // 0.01 SUI max tip
        },
      },
    })
  );
}

/**
 * Initialize the Walrus upload service
 */
export function initializeWalrusUpload(): boolean {
  if (!BACKEND_SUI_PRIVATE_KEY) {
    console.warn('⚠️ BACKEND_SUI_PRIVATE_KEY not set - Walrus upload relay disabled');
    console.warn('   Jetson will continue using direct HTTP publisher (slow)');
    return false;
  }

  try {
    // Parse the private key (expects base64 with 0x00 prefix for Ed25519)
    const keyBytes = fromBase64(BACKEND_SUI_PRIVATE_KEY);

    // If the key has a scheme flag prefix (0x00 for Ed25519), remove it
    const privateKeyBytes = keyBytes.length === 33 && keyBytes[0] === 0x00
      ? keyBytes.slice(1)
      : keyBytes;

    backendKeypair = Ed25519Keypair.fromSecretKey(privateKeyBytes);
    walrusClient = createWalrusClient();

    const address = backendKeypair.getPublicKey().toSuiAddress();
    console.log(`✅ Walrus upload relay initialized`);
    console.log(`   Network: ${SUI_NETWORK}`);
    console.log(`   Relay: ${WALRUS_UPLOAD_RELAY}`);
    console.log(`   Backend wallet: ${address.slice(0, 16)}...`);

    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Walrus upload:', error);
    return false;
  }
}

/**
 * Check if Walrus upload relay is available
 */
export function isWalrusRelayEnabled(): boolean {
  return walrusClient !== null && backendKeypair !== null;
}

/**
 * Upload an encrypted blob to Walrus via the upload relay
 *
 * @param encryptedData - The encrypted file data
 * @param epochs - Number of epochs to store (default 5)
 * @returns Upload result with blob ID and download URL
 */
export async function uploadToWalrus(
  encryptedData: Buffer,
  epochs: number = 5
): Promise<WalrusUploadResult> {
  if (!walrusClient || !backendKeypair) {
    throw new Error('Walrus upload relay not initialized. Set BACKEND_SUI_PRIVATE_KEY.');
  }

  const startTime = Date.now();

  try {
    console.log(`🚀 Uploading ${encryptedData.length} bytes to Walrus via relay...`);

    // Upload using the SDK with relay
    const result = await walrusClient.walrus.writeBlob({
      blob: new Uint8Array(encryptedData),
      deletable: false,
      epochs,
      signer: backendKeypair,
    });

    const uploadDurationMs = Date.now() - startTime;
    const downloadUrl = `${WALRUS_AGGREGATOR}/v1/blobs/${result.blobId}`;

    console.log(`✅ Walrus upload complete in ${uploadDurationMs}ms`);
    console.log(`   Blob ID: ${result.blobId.slice(0, 20)}...`);

    return {
      blobId: result.blobId,
      downloadUrl,
      uploadDurationMs,
    };
  } catch (error) {
    const uploadDurationMs = Date.now() - startTime;
    console.error(`❌ Walrus upload failed after ${uploadDurationMs}ms:`, error);
    throw error;
  }
}

/**
 * Get the backend's Sui wallet address (for debugging/funding)
 */
export function getBackendSuiAddress(): string | null {
  if (!backendKeypair) return null;
  return backendKeypair.getPublicKey().toSuiAddress();
}
