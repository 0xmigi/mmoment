/**
 * Walrus Decryption Service
 *
 * Decrypts Walrus blobs by calling the backend decrypt endpoint.
 * The backend handles all crypto operations using the Sui private key.
 *
 * This approach:
 * - No client-side crypto needed
 * - No wallet signature prompts for decryption
 * - Backend verifies authorization before decrypting
 * - Sui-native: encryption uses Sui keys that match blob ownership
 */

import { CONFIG } from '../../core/config';

/**
 * Result of decryption attempt
 */
export interface DecryptionResult {
  success: boolean;
  blob?: Blob;
  objectUrl?: string;
  error?: string;
}

/**
 * Decrypt a Walrus blob by calling the backend decrypt endpoint.
 *
 * @param blobId - The Walrus blob ID
 * @param walletAddress - User's Solana wallet address (for authorization)
 * @returns Decrypted blob as an object URL
 */
export async function decryptWalrusBlob(
  blobId: string,
  walletAddress: string
): Promise<DecryptionResult> {
  try {
    console.log(`[WalrusDecrypt] Requesting decryption for blob ${blobId.slice(0, 12)}...`);

    const response = await fetch(
      `${CONFIG.BACKEND_URL}/api/walrus/decrypt/${blobId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ walletAddress }),
      }
    );

    if (!response.ok) {
      // Check if it's a redirect to direct URL (unencrypted file)
      if (response.redirected) {
        console.log(`[WalrusDecrypt] File not encrypted, using direct URL`);
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        return { success: true, blob, objectUrl };
      }

      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.error || `Decryption failed: ${response.status}`;
      console.error(`[WalrusDecrypt] ${errorMessage}`);
      return { success: false, error: errorMessage };
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);

    console.log(`[WalrusDecrypt] Success! Decrypted ${blob.size} bytes`);

    return {
      success: true,
      blob,
      objectUrl,
    };
  } catch (error) {
    console.error('[WalrusDecrypt] Decryption failed:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown decryption error',
    };
  }
}

/**
 * Decrypt a Walrus file using the gallery item data.
 * Convenience wrapper that extracts blobId from the file.
 */
export async function decryptWalrusFile(
  _downloadUrl: string,  // Unused - kept for API compatibility
  blobId: string,
  walletAddress: string
): Promise<DecryptionResult> {
  return decryptWalrusBlob(blobId, walletAddress);
}

/**
 * Revoke object URL to free memory.
 */
export function revokeDecryptedUrl(objectUrl: string): void {
  URL.revokeObjectURL(objectUrl);
}

/**
 * Check if a blob needs decryption by checking with the backend.
 * For now, we assume all Walrus files are encrypted.
 */
export function needsDecryption(encrypted: boolean | undefined): boolean {
  // Default to true - most Walrus files are encrypted
  return encrypted !== false;
}
