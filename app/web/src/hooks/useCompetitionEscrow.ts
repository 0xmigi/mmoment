/**
 * Competition Escrow Hook
 *
 * Provides all operations for creating and managing competition escrows
 * with on-chain stakes and settlements.
 */

import { useState, useCallback } from 'react';
import { PublicKey, SystemProgram } from '@solana/web3.js';
import { useConnection } from '@solana/wallet-adapter-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { BN } from '@coral-xyz/anchor';
import {
  useCompetitionEscrowProgram,
  findCompetitionEscrowPDA,
  parseCompetitionStatus,
  lamportsToSol,
  solToLamports,
  type CompetitionStatusType,
} from '../anchor/setup';

export interface CompetitionConfig {
  cameraDevicePubkey: string;
  invitees: string[];
  stakePerPersonSol: number;
  payoutRule: 'winnerTakesAll' | 'thresholdSplit';
  thresholdMinReps?: number;
  inviteTimeoutSecs?: number;
  initiatorParticipates: boolean;
}

export interface ActiveCompetition {
  escrowPda: string;
  camera: string;
  initiator: string;
  status: CompetitionStatusType;
  stakePerPerson: number; // in SOL
  totalPool: number; // in SOL
  participants: string[];
  pendingInvites: string[];
  winners: string[];
  createdAt: number;
  inviteTimeoutSecs: number;
}

interface UseCompetitionEscrowReturn {
  // State
  loading: boolean;
  error: string | null;
  activeCompetition: ActiveCompetition | null;

  // Operations
  createCompetition: (config: CompetitionConfig) => Promise<{ escrowPda: string; createdAt: number } | null>;
  joinCompetition: (escrowPda: string) => Promise<boolean>;
  declineCompetition: (escrowPda: string) => Promise<boolean>;
  startCompetition: (escrowPda: string) => Promise<boolean>;
  cancelCompetition: (escrowPda: string, reason: string) => Promise<boolean>;
  fetchCompetition: (escrowPda: string) => Promise<ActiveCompetition | null>;
  fetchPendingInvites: (userAddress: string, cameraDevicePubkey: string) => Promise<ActiveCompetition[]>;

  // Helpers
  clearError: () => void;
  isInitiator: (competition: ActiveCompetition) => boolean;
  isParticipant: (competition: ActiveCompetition) => boolean;
  isPendingInvite: (competition: ActiveCompetition) => boolean;
}

export function useCompetitionEscrow(): UseCompetitionEscrowReturn {
  const { connection } = useConnection();
  const { primaryWallet } = useDynamicContext();
  const { program, loading: programLoading, error: programError } = useCompetitionEscrowProgram();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeCompetition, setActiveCompetition] = useState<ActiveCompetition | null>(null);

  const walletAddress = primaryWallet?.address;

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Parse on-chain account to ActiveCompetition
  // Note: Account field names are snake_case from the IDL
  const parseCompetitionAccount = useCallback((account: any, escrowPda: PublicKey): ActiveCompetition => {
    return {
      escrowPda: escrowPda.toString(),
      camera: account.camera.toString(),
      initiator: account.initiator.toString(),
      status: parseCompetitionStatus(account.status),
      stakePerPerson: lamportsToSol(account.stake_per_person ?? account.stakePerPerson),
      totalPool: lamportsToSol(account.total_pool ?? account.totalPool),
      participants: account.participants.map((p: PublicKey) => p.toString()),
      pendingInvites: (account.pending_invites ?? account.pendingInvites).map((p: PublicKey) => p.toString()),
      winners: account.winners.map((p: PublicKey) => p.toString()),
      createdAt: (account.created_at ?? account.createdAt).toNumber(),
      inviteTimeoutSecs: account.invite_timeout_secs ?? account.inviteTimeoutSecs,
    };
  }, []);

  // Create a new competition
  const createCompetition = useCallback(async (config: CompetitionConfig): Promise<{ escrowPda: string; createdAt: number } | null> => {
    if (!program || !primaryWallet || !walletAddress) {
      setError('Wallet not connected');
      return null;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setError('Not a Solana wallet');
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const createdAt = Math.floor(Date.now() / 1000);
      const cameraPubkey = new PublicKey(config.cameraDevicePubkey);
      const initiatorPubkey = new PublicKey(walletAddress);

      // Derive PDA
      const [escrowPda] = findCompetitionEscrowPDA(cameraPubkey, createdAt);

      // Build payout rule - Anchor 0.29 converts IDL variant names to camelCase internally
      // So even though IDL says "WinnerTakesAll", the encoder expects "winnerTakesAll"
      let payoutRule: any;
      if (config.payoutRule === 'winnerTakesAll') {
        payoutRule = { winnerTakesAll: {} };
      } else {
        payoutRule = { thresholdSplit: { minReps: config.thresholdMinReps || 10 } };
      }

      // Build args with camelCase field names to match TypeScript IDL
      const args = {
        invitees: config.invitees.map(addr => new PublicKey(addr)),
        initiatorParticipates: config.initiatorParticipates,
        stakePerPerson: solToLamports(config.stakePerPersonSol),
        payoutRule: payoutRule,
        inviteTimeoutSecs: config.inviteTimeoutSecs || 60,
      };

      console.log('[CompetitionEscrow] Creating competition:', {
        escrowPda: escrowPda.toString(),
        camera: cameraPubkey.toString(),
        createdAt,
        args,
      });

      // Build transaction
      const tx = await program.methods
        .createCompetition(args, new BN(createdAt))
        .accounts({
          initiator: initiatorPubkey,
          camera: cameraPubkey,
          escrow: escrowPda,
          systemProgram: SystemProgram.programId,
        })
        .transaction();

      // Set recent blockhash and fee payer
      tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      tx.feePayer = initiatorPubkey;

      // Sign with Dynamic wallet
      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);

      // Send transaction
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');

      console.log('[CompetitionEscrow] Competition created:', signature);

      // Fetch and set the active competition
      const account = await program.account.competitionEscrow.fetch(escrowPda);
      const competition = parseCompetitionAccount(account, escrowPda);
      setActiveCompetition(competition);

      return { escrowPda: escrowPda.toString(), createdAt };
    } catch (err: any) {
      console.error('[CompetitionEscrow] Error creating competition:', err);
      setError(err.message || 'Failed to create competition');
      return null;
    } finally {
      setLoading(false);
    }
  }, [program, primaryWallet, walletAddress, connection, parseCompetitionAccount]);

  // Join a competition (accept invite)
  const joinCompetition = useCallback(async (escrowPdaStr: string): Promise<boolean> => {
    if (!program || !primaryWallet || !walletAddress) {
      setError('Wallet not connected');
      return false;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setError('Not a Solana wallet');
      return false;
    }

    setLoading(true);
    setError(null);

    try {
      const escrowPda = new PublicKey(escrowPdaStr);
      const participantPubkey = new PublicKey(walletAddress);

      const tx = await program.methods
        .joinCompetition()
        .accounts({
          participant: participantPubkey,
          escrow: escrowPda,
          systemProgram: SystemProgram.programId,
        })
        .transaction();

      tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      tx.feePayer = participantPubkey;

      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);

      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');

      console.log('[CompetitionEscrow] Joined competition:', signature);

      // Refresh competition state
      const account = await program.account.competitionEscrow.fetch(escrowPda);
      const competition = parseCompetitionAccount(account, escrowPda);
      setActiveCompetition(competition);

      return true;
    } catch (err: any) {
      console.error('[CompetitionEscrow] Error joining competition:', err);
      setError(err.message || 'Failed to join competition');
      return false;
    } finally {
      setLoading(false);
    }
  }, [program, primaryWallet, walletAddress, connection, parseCompetitionAccount]);

  // Decline a competition invite
  const declineCompetition = useCallback(async (escrowPdaStr: string): Promise<boolean> => {
    if (!program || !primaryWallet || !walletAddress) {
      setError('Wallet not connected');
      return false;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setError('Not a Solana wallet');
      return false;
    }

    setLoading(true);
    setError(null);

    try {
      const escrowPda = new PublicKey(escrowPdaStr);
      const participantPubkey = new PublicKey(walletAddress);

      const tx = await program.methods
        .declineCompetition()
        .accounts({
          participant: participantPubkey,
          escrow: escrowPda,
        })
        .transaction();

      tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      tx.feePayer = participantPubkey;

      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);

      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');

      console.log('[CompetitionEscrow] Declined competition:', signature);

      return true;
    } catch (err: any) {
      console.error('[CompetitionEscrow] Error declining competition:', err);
      setError(err.message || 'Failed to decline competition');
      return false;
    } finally {
      setLoading(false);
    }
  }, [program, primaryWallet, walletAddress, connection]);

  // Start the competition
  const startCompetition = useCallback(async (escrowPdaStr: string): Promise<boolean> => {
    if (!program || !primaryWallet || !walletAddress) {
      setError('Wallet not connected');
      return false;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setError('Not a Solana wallet');
      return false;
    }

    setLoading(true);
    setError(null);

    try {
      const escrowPda = new PublicKey(escrowPdaStr);
      const authorityPubkey = new PublicKey(walletAddress);

      const tx = await program.methods
        .startCompetition()
        .accounts({
          authority: authorityPubkey,
          escrow: escrowPda,
        })
        .transaction();

      tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      tx.feePayer = authorityPubkey;

      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);

      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');

      console.log('[CompetitionEscrow] Started competition:', signature);

      // Refresh competition state
      const account = await program.account.competitionEscrow.fetch(escrowPda);
      const competition = parseCompetitionAccount(account, escrowPda);
      setActiveCompetition(competition);

      return true;
    } catch (err: any) {
      console.error('[CompetitionEscrow] Error starting competition:', err);
      setError(err.message || 'Failed to start competition');
      return false;
    } finally {
      setLoading(false);
    }
  }, [program, primaryWallet, walletAddress, connection, parseCompetitionAccount]);

  // Cancel the competition (initiator only)
  const cancelCompetition = useCallback(async (escrowPdaStr: string, reason: string): Promise<boolean> => {
    if (!program || !primaryWallet || !walletAddress) {
      setError('Wallet not connected');
      return false;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setError('Not a Solana wallet');
      return false;
    }

    setLoading(true);
    setError(null);

    try {
      const escrowPda = new PublicKey(escrowPdaStr);
      const initiatorPubkey = new PublicKey(walletAddress);

      // Fetch escrow to get participants for remaining accounts
      const escrowAccount = await program.account.competitionEscrow.fetch(escrowPda) as any;
      const participants = escrowAccount.participants as PublicKey[];

      const tx = await program.methods
        .cancelCompetition(reason)
        .accounts({
          initiator: initiatorPubkey,
          escrow: escrowPda,
          systemProgram: SystemProgram.programId,
        })
        .remainingAccounts(
          participants.map(p => ({
            pubkey: p,
            isWritable: true,
            isSigner: false,
          }))
        )
        .transaction();

      tx.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      tx.feePayer = initiatorPubkey;

      const signer = await primaryWallet.getSigner();
      const signedTx = await signer.signTransaction(tx);

      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction(signature, 'confirmed');

      console.log('[CompetitionEscrow] Cancelled competition:', signature);

      setActiveCompetition(null);

      return true;
    } catch (err: any) {
      console.error('[CompetitionEscrow] Error cancelling competition:', err);
      setError(err.message || 'Failed to cancel competition');
      return false;
    } finally {
      setLoading(false);
    }
  }, [program, primaryWallet, walletAddress, connection]);

  // Fetch a competition by PDA
  const fetchCompetition = useCallback(async (escrowPdaStr: string): Promise<ActiveCompetition | null> => {
    if (!program) {
      return null;
    }

    try {
      const escrowPda = new PublicKey(escrowPdaStr);
      const account = await program.account.competitionEscrow.fetch(escrowPda);
      return parseCompetitionAccount(account, escrowPda);
    } catch (err) {
      console.error('[CompetitionEscrow] Error fetching competition:', err);
      return null;
    }
  }, [program, parseCompetitionAccount]);

  // Fetch pending invites for a user
  const fetchPendingInvites = useCallback(async (userAddress: string, cameraDevicePubkey: string): Promise<ActiveCompetition[]> => {
    if (!program) {
      return [];
    }

    try {
      // Fetch all competition escrows for this camera
      // Note: In production, you'd want to use getProgramAccounts with filters
      // or an indexer for better performance
      const accounts = await program.account.competitionEscrow.all([
        {
          memcmp: {
            offset: 8 + 32, // After discriminator + initiator
            bytes: cameraDevicePubkey,
          },
        },
      ]);

      const pendingInvites: ActiveCompetition[] = [];

      for (const { account, publicKey } of accounts) {
        const competition = parseCompetitionAccount(account, publicKey);
        if (competition.status === 'pending' && competition.pendingInvites.includes(userAddress)) {
          pendingInvites.push(competition);
        }
      }

      return pendingInvites;
    } catch (err) {
      console.error('[CompetitionEscrow] Error fetching pending invites:', err);
      return [];
    }
  }, [program, parseCompetitionAccount]);

  // Helper: check if current user is initiator
  const isInitiator = useCallback((competition: ActiveCompetition): boolean => {
    return walletAddress?.toLowerCase() === competition.initiator.toLowerCase();
  }, [walletAddress]);

  // Helper: check if current user is participant
  const isParticipant = useCallback((competition: ActiveCompetition): boolean => {
    return competition.participants.some(p => p.toLowerCase() === walletAddress?.toLowerCase());
  }, [walletAddress]);

  // Helper: check if current user has pending invite
  const isPendingInvite = useCallback((competition: ActiveCompetition): boolean => {
    return competition.pendingInvites.some(p => p.toLowerCase() === walletAddress?.toLowerCase());
  }, [walletAddress]);

  return {
    loading: loading || programLoading,
    error: error || (programError ? programError.message : null),
    activeCompetition,
    createCompetition,
    joinCompetition,
    declineCompetition,
    startCompetition,
    cancelCompetition,
    fetchCompetition,
    fetchPendingInvites,
    clearError,
    isInitiator,
    isParticipant,
    isPendingInvite,
  };
}

export default useCompetitionEscrow;
