// Re-export IDL from JSON file
// This ensures the backend uses the same IDL as the Solana program
import IDL_JSON from './idl.json';

export const IDL = IDL_JSON as any;
export type CameraNetwork = typeof IDL_JSON;
