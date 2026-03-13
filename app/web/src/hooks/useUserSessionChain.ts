/**
 * useUserSessionChain Hook & Utilities
 *
 * Manages the user's session chain PDA on Solana.
 * The session chain stores encrypted access keys for the user's historical sessions.
 *
 * Exports:
 * - useUserSessionChain: React hook for components that need reactive state
 * - checkUserHasSessionChain: Standalone async function to check if user has chain
 * - createUserSessionChain: Standalone async function to create chain (used in check-in flow)
 */

import { useState, useEffect, useCallback } from 'react';
import { useProgram, findUserSessionChainPDA } from '../anchor/setup';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { PublicKey, Connection, Transaction } from '@solana/web3.js';
import { useConnection } from '@solana/wallet-adapter-react';
import { Program } from '@coral-xyz/anchor';
import { CONFIG } from '../core/config';

export interface UserSessionChainStatus {
  hasSessionChain: boolean;
  sessionChainPda: string | null;
  sessionCount: number;
  isLoading: boolean;
  error: string | null;
  lastChecked: Date | null;
  refetch: () => Promise<void>;
  createSessionChain: () => Promise<boolean>;
  isCreating: boolean;
}

/**
 * Fetch the mmoment authority (cron bot) public key from the backend.
 * This is stored in the session chain so both user AND cron bot can write to it.
 */
export async function fetchAuthorityPublicKey(): Promise<PublicKey | null> {
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/api/cleanup-cron/status`);
    if (!response.ok) {
      console.error('[SessionChain] Failed to fetch authority:', response.status);
      return null;
    }
    const data = await response.json();
    if (data.cronBot) {
      return new PublicKey(data.cronBot);
    }
    console.error('[SessionChain] No cronBot in status response');
    return null;
  } catch (error) {
    console.error('[SessionChain] Error fetching authority:', error);
    return null;
  }
}

/**
 * Check if a user has a session chain PDA (standalone function).
 * Use this when you need a one-time check without reactive state.
 */
export async function checkUserHasSessionChain(
  userPubkey: PublicKey,
  program: Program<any>
): Promise<{ exists: boolean; sessionCount: number; pda: PublicKey }> {
  const [sessionChainPda] = findUserSessionChainPDA(userPubkey);

  try {
    const chainAccount = await (program.account as any).userSessionChain.fetch(sessionChainPda);
    return {
      exists: !!chainAccount,
      sessionCount: chainAccount?.sessionCount?.toNumber?.() || 0,
      pda: sessionChainPda,
    };
  } catch {
    // Account doesn't exist
    return {
      exists: false,
      sessionCount: 0,
      pda: sessionChainPda,
    };
  }
}

/**
 * Create a user's session chain (standalone function).
 * Uses a sponsored flow: backend (cron bot) pays rent + tx fees,
 * user only signs to authorize creation of their chain.
 *
 * Flow:
 * 1. Frontend requests a partially-signed tx from backend (cron bot signs as fee_payer)
 * 2. User signs the tx (authorizes chain creation)
 * 3. Frontend submits the fully-signed tx
 *
 * @param userPubkey - User's wallet public key
 * @param wallet - Dynamic wallet object (must be Solana wallet with getSigner)
 * @param connection - Solana connection
 * @returns Transaction signature on success, null on failure
 */
export async function createUserSessionChain(
  userPubkey: PublicKey,
  wallet: any,
  _program: Program<any>,
  connection: Connection
): Promise<string | null> {
  try {
    console.log('[SessionChain] Creating session chain (sponsored)...');
    console.log('[SessionChain] User:', userPubkey.toString());

    // Step 1: Request a sponsored, partially-signed tx from backend
    const response = await fetch(`${CONFIG.BACKEND_URL}/api/session/build-create-chain-tx`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_pubkey: userPubkey.toString() }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      if (response.status === 409) {
        // Chain already exists — not an error
        console.log('[SessionChain] Session chain already exists');
        return 'already_exists';
      }
      throw new Error(errorData.error || `Backend returned ${response.status}`);
    }

    const { transaction: txBase64 } = await response.json();

    // Step 2: Deserialize and have user co-sign
    const txBuffer = Buffer.from(txBase64, 'base64');
    const tx = Transaction.from(txBuffer);

    console.log('[SessionChain] Requesting wallet signature...');
    const signer = await wallet.getSigner();
    const signedTx = await signer.signTransaction(tx);

    // Step 3: Submit the fully-signed tx
    console.log('[SessionChain] Sending transaction...');
    const signature = await connection.sendRawTransaction(signedTx.serialize());
    await connection.confirmTransaction(signature, 'confirmed');

    console.log('[SessionChain] Session chain created! Tx:', signature);
    return signature;
  } catch (error: any) {
    console.error('[SessionChain] Failed to create session chain:', error);
    return null;
  }
}

/**
 * React hook for components that need reactive session chain state.
 * For one-time checks (like in check-in flow), use the standalone functions instead.
 */
export function useUserSessionChain(): UserSessionChainStatus {
  const [status, setStatus] = useState<Omit<UserSessionChainStatus, 'refetch' | 'createSessionChain'>>({
    hasSessionChain: false,
    sessionChainPda: null,
    sessionCount: 0,
    isLoading: true,
    error: null,
    lastChecked: null,
    isCreating: false,
  });

  const { program } = useProgram();
  const { connection } = useConnection();
  const { primaryWallet } = useDynamicContext();

  const checkSessionChainStatus = useCallback(async () => {
    if (!primaryWallet?.address || !program) {
      setStatus(prev => ({
        ...prev,
        isLoading: false,
        error: null, // Don't show error, just not ready yet
        lastChecked: new Date(),
      }));
      return;
    }

    try {
      setStatus(prev => ({ ...prev, isLoading: true, error: null }));

      const userPublicKey = new PublicKey(primaryWallet.address);
      const result = await checkUserHasSessionChain(userPublicKey, program);

      setStatus({
        hasSessionChain: result.exists,
        sessionChainPda: result.pda.toString(),
        sessionCount: result.sessionCount,
        isLoading: false,
        error: null,
        lastChecked: new Date(),
        isCreating: false,
      });
    } catch (error) {
      console.log('[useUserSessionChain] Error:', error);
      const userPublicKey = new PublicKey(primaryWallet.address);
      const [sessionChainPda] = findUserSessionChainPDA(userPublicKey);

      setStatus({
        hasSessionChain: false,
        sessionChainPda: sessionChainPda.toString(),
        sessionCount: 0,
        isLoading: false,
        error: null,
        lastChecked: new Date(),
        isCreating: false,
      });
    }
  }, [primaryWallet?.address, program]);

  const handleCreateSessionChain = useCallback(async (): Promise<boolean> => {
    if (!primaryWallet?.address || !program) {
      setStatus(prev => ({ ...prev, error: 'Wallet or program not available' }));
      return false;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setStatus(prev => ({ ...prev, error: 'Not a Solana wallet' }));
      return false;
    }

    setStatus(prev => ({ ...prev, isCreating: true, error: null }));

    const userPublicKey = new PublicKey(primaryWallet.address);
    const signature = await createUserSessionChain(userPublicKey, primaryWallet, program, connection);

    if (signature) {
      await checkSessionChainStatus();
      return true;
    } else {
      setStatus(prev => ({
        ...prev,
        isCreating: false,
        error: 'Failed to create session chain',
      }));
      return false;
    }
  }, [primaryWallet, program, connection, checkSessionChainStatus]);

  useEffect(() => {
    checkSessionChainStatus();
  }, [checkSessionChainStatus]);

  return {
    ...status,
    refetch: checkSessionChainStatus,
    createSessionChain: handleCreateSessionChain,
  };
}
