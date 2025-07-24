import { DeviceSignature, DeviceSignedResponse } from './camera-types';

/**
 * Device signature verification utilities for DePIN authentication
 */

/**
 * Validates that a device signature exists and has basic integrity
 */
export function hasValidDeviceSignature(response: DeviceSignedResponse): boolean {
  const signature = response.device_signature;
  
  if (!signature) {
    console.warn('Device signature missing from response');
    return false;
  }

  // Validate required fields
  if (!signature.device_pubkey || !signature.signature || !signature.timestamp) {
    console.warn('Device signature missing required fields');
    return false;
  }

  // Validate timestamp is recent (within 5 minutes)
  const now = Date.now();
  const signatureAge = Math.abs(now - signature.timestamp);
  const maxAge = 5 * 60 * 1000; // 5 minutes in milliseconds

  if (signatureAge > maxAge) {
    console.warn(`Device signature too old: ${signatureAge}ms (max: ${maxAge}ms)`);
    return false;
  }

  return true;
}

/**
 * Extracts device public key from a signed response
 */
export function getDevicePublicKey(response: DeviceSignedResponse): string | null {
  return response.device_signature?.device_pubkey || null;
}

/**
 * Validates that the response came from a specific expected device
 */
export function validateDeviceIdentity(
  response: DeviceSignedResponse, 
  expectedDevicePubkey: string
): boolean {
  if (!hasValidDeviceSignature(response)) {
    return false;
  }

  const devicePubkey = getDevicePublicKey(response);
  if (devicePubkey !== expectedDevicePubkey) {
    console.warn(`Device identity mismatch. Expected: ${expectedDevicePubkey}, Got: ${devicePubkey}`);
    return false;
  }

  return true;
}

/**
 * Creates a signature verification info object for logging/debugging
 */
export function getSignatureInfo(response: DeviceSignedResponse) {
  const signature = response.device_signature;
  
  if (!signature) {
    return {
      present: false,
      device_pubkey: null,
      timestamp: null,
      age_ms: null,
      version: null,
    };
  }

  return {
    present: true,
    device_pubkey: signature.device_pubkey,
    timestamp: signature.timestamp,
    age_ms: Date.now() - signature.timestamp,
    version: signature.version,
    valid: hasValidDeviceSignature(response),
  };
}

/**
 * Console logging helper for device signature debugging
 */
export function logDeviceSignature(response: DeviceSignedResponse, context: string = '') {
  const info = getSignatureInfo(response);
  const prefix = context ? `[${context}] ` : '';
  
  if (info.present) {
    console.log(`${prefix}Device signature:`, {
      device: info.device_pubkey,
      age: `${Math.round((info.age_ms || 0) / 1000)}s`,
      valid: info.valid,
      version: info.version,
    });
  } else {
    console.warn(`${prefix}No device signature present`);
  }
}

/**
 * Future: Cryptographic signature verification (requires crypto library)
 * TODO: Implement actual Ed25519 signature verification
 */
export async function verifyCryptographicSignature(
  response: DeviceSignedResponse,
  originalPayload?: object
): Promise<boolean> {
  // This would require implementing Ed25519 signature verification
  // using a library like tweetnacl or similar
  console.warn('Cryptographic signature verification not yet implemented');
  return hasValidDeviceSignature(response);
}