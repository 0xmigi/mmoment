// src/anchor/setup.ts
import { useAnchorWallet } from '@solana/wallet-adapter-react';
import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { clusterApiUrl, Connection, PublicKey } from "@solana/web3.js";
import { IDL } from "./idl";
import * as buffer from "buffer";

window.Buffer = buffer.Buffer;

// More permissive process type that won't conflict
declare global {
  interface Window {
    process?: any;  // Make it completely flexible for browser environment
  }
}

if (typeof window !== 'undefined') {
  // Minimal process implementation needed for our app
  window.process = {
    env: {},
    version: '1.0.0',
    platform: 'browser'
  } as any;  // Type assertion to avoid process conflicts
}

const programID = new PublicKey("5HFxUPQ7aZPZF8UqRSSHsoamq3X4VU1K9euMXuuGcUfj");
const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

export const useProgram = () => {
  const wallet = useAnchorWallet();
  if (!wallet) return null;
  const provider = new AnchorProvider(connection, wallet, {});
  return new Program(IDL, programID, provider);
};