/**
 * Request Signing Utility for Secure Camera API Calls
 *
 * This module provides cryptographic signing of requests to prove the
 * requester actually owns the wallet private key.
 *
 * Without this, anyone who knows your wallet address could impersonate you.
 *
 * Flow:
 * 1. Frontend signs: "{wallet}|{timestamp}|{nonce}"
 * 2. Sends: { wallet_address, request_timestamp, request_nonce, request_signature }
 * 3. Jetson verifies ed25519 signature before processing
 */

import bs58 from 'bs58';

/**
 * Parameters required for a signed request
 */
export interface SignedRequestParams {
  wallet_address: string;
  request_timestamp: number;
  request_nonce: string;
  request_signature: string;
}

/**
 * Generate a cryptographic nonce for replay attack prevention
 */
export function generateNonce(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Create the message to be signed
 */
export function createSignatureMessage(
  walletAddress: string,
  timestamp: number,
  nonce: string
): string {
  return `${walletAddress}|${timestamp}|${nonce}`;
}

/**
 * Sign a request using the wallet's signMessage function
 */
export async function signRequest(
  walletAddress: string,
  signMessage: (message: Uint8Array) => Promise<Uint8Array>
): Promise<SignedRequestParams> {
  const timestamp = Date.now();
  const nonce = generateNonce();
  const message = createSignatureMessage(walletAddress, timestamp, nonce);

  console.log('[RequestSigner] Signing message:', message);

  const messageBytes = new TextEncoder().encode(message);
  const signatureBytes = await signMessage(messageBytes);
  const signature = bs58.encode(signatureBytes);

  console.log('[RequestSigner] Signature generated:', signature.slice(0, 20) + '...');

  return {
    wallet_address: walletAddress,
    request_timestamp: timestamp,
    request_nonce: nonce,
    request_signature: signature,
  };
}

/**
 * Helper to get signMessage function from Dynamic Labs wallet
 */
export async function getSignMessageFunction(
  primaryWallet: any
): Promise<((message: Uint8Array) => Promise<Uint8Array>) | null> {
  if (!primaryWallet) {
    console.error('[RequestSigner] No wallet provided');
    return null;
  }

  try {
    const signer = await primaryWallet.getSigner();

    if (signer && typeof signer.signMessage === 'function') {
      console.log('[RequestSigner] Using signer.signMessage');
      return async (message: Uint8Array) => {
        const sig = await signer.signMessage(message);
        console.log('[RequestSigner] Raw signature result:', sig);
        console.log('[RequestSigner] Signature type:', typeof sig);
        console.log('[RequestSigner] Is Uint8Array:', sig instanceof Uint8Array);
        if (sig && typeof sig === 'object') {
          console.log('[RequestSigner] Signature keys:', Object.keys(sig));
          // Handle case where signature might be wrapped in an object
          if ('signature' in sig) {
            console.log('[RequestSigner] Found signature property, using it');
            const innerSig = (sig as any).signature;
            return innerSig instanceof Uint8Array ? innerSig : new Uint8Array(innerSig);
          }
        }
        return sig instanceof Uint8Array ? sig : new Uint8Array(sig);
      };
    }

    if (typeof primaryWallet.signMessage === 'function') {
      console.log('[RequestSigner] Using primaryWallet.signMessage');
      return async (message: Uint8Array) => {
        const sig = await primaryWallet.signMessage(message);
        console.log('[RequestSigner] Raw signature result:', sig);
        console.log('[RequestSigner] Signature type:', typeof sig);
        if (sig && typeof sig === 'object' && 'signature' in sig) {
          console.log('[RequestSigner] Found signature property, using it');
          const innerSig = (sig as any).signature;
          return innerSig instanceof Uint8Array ? innerSig : new Uint8Array(innerSig);
        }
        return sig instanceof Uint8Array ? sig : new Uint8Array(sig);
      };
    }

    console.warn('[RequestSigner] Wallet does not support signMessage');
    return null;
  } catch (error) {
    console.error('[RequestSigner] Error getting signMessage function:', error);
    return null;
  }
}

/**
 * Create signed request parameters using a Dynamic Labs wallet
 */
export async function createSignedRequest(
  primaryWallet: any
): Promise<SignedRequestParams | null> {
  if (!primaryWallet?.address) {
    console.error('[RequestSigner] No wallet address');
    return null;
  }

  const signMessage = await getSignMessageFunction(primaryWallet);

  if (!signMessage) {
    console.warn('[RequestSigner] Wallet signing not available');
    return null;
  }

  try {
    return await signRequest(primaryWallet.address, signMessage);
  } catch (error) {
    console.error('[RequestSigner] Failed to sign request:', error);
    return null;
  }
}

/**
 * Merge signed parameters into existing request body
 */
export function addSignatureToBody<T extends object>(
  body: T,
  signedParams: SignedRequestParams | null
): T & Partial<SignedRequestParams> {
  if (!signedParams) {
    return body;
  }
  return { ...body, ...signedParams };
}
