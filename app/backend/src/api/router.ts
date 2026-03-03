import { Router } from 'express';
import crypto from 'crypto';
import { Connection, PublicKey } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import { Keypair } from '@solana/web3.js';
import { IDL } from '../idl';
import { apiKeyAuth, AuthenticatedRequest } from './middleware/api-key-auth';
import { generateApiKey, hashApiKey, getKeyPrefix } from './keys/api-key-gen';
import {
  createApiKey,
  getApiKeysForWallet,
  revokeApiKey,
  getUserProfile,
  getWalrusFilesForWallet,
  getDatabaseStats,
} from '../database';

const PROGRAM_ID = new PublicKey('E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL');

export function createApiRouter(): Router {
  const router = Router();

  // Solana connection for on-chain reads (read-only, no keypair needed for fetching)
  const connection = new Connection(
    process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
    'confirmed'
  );
  // Anchor requires a wallet for Provider but we only read — use a throwaway keypair
  const readOnlyWallet = new Wallet(Keypair.generate());
  const provider = new AnchorProvider(connection, readOnlyWallet, { commitment: 'confirmed' });
  const program = new Program(IDL as any, PROGRAM_ID, provider);

  // ============================================================================
  // KEY MANAGEMENT (wallet-based, no API key needed to create)
  // ============================================================================

  router.post('/v1/keys', async (req, res) => {
    try {
      const { wallet_address, name } = req.body;

      if (!wallet_address) {
        return res.status(400).json({ error: 'wallet_address is required' });
      }

      const id = crypto.randomUUID();
      const rawKey = generateApiKey();
      const keyHash = hashApiKey(rawKey);
      const keyPrefix = getKeyPrefix(rawKey);

      await createApiKey(id, keyHash, keyPrefix, wallet_address, name || 'default');

      // Return the raw key ONCE — it's never stored
      res.status(201).json({
        data: {
          id,
          key: rawKey,
          key_prefix: keyPrefix,
          wallet_address,
          name: name || 'default',
          created_at: Date.now(),
        },
        meta: { warning: 'Save this key now. It cannot be retrieved again.' }
      });
    } catch (err) {
      console.error('[API] Failed to create key:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to create API key' });
    }
  });

  // All routes below require API key auth
  router.use('/v1', apiKeyAuth);

  router.get('/v1/keys', async (req: AuthenticatedRequest, res) => {
    try {
      const keys = await getApiKeysForWallet(req.apiKey!.walletAddress);
      res.json({
        data: keys.map(k => ({
          id: k.id,
          key_prefix: k.keyPrefix,
          name: k.name,
          created_at: k.createdAt,
          last_used_at: k.lastUsedAt,
          revoked_at: k.revokedAt,
        }))
      });
    } catch (err) {
      console.error('[API] Failed to list keys:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to list keys' });
    }
  });

  router.delete('/v1/keys/:keyId', async (req: AuthenticatedRequest, res) => {
    try {
      const revoked = await revokeApiKey(req.params.keyId, req.apiKey!.walletAddress);
      if (!revoked) {
        return res.status(404).json({ error: 'not_found', message: 'Key not found or not owned by you' });
      }
      res.json({ data: { revoked: true } });
    } catch (err) {
      console.error('[API] Failed to revoke key:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to revoke key' });
    }
  });

  // ============================================================================
  // ON-CHAIN DATA ENDPOINTS (reads directly from Solana blockchain)
  // ============================================================================

  // User session chain — encrypted session keys stored on-chain
  // The caller decrypts with their wallet key to access session data
  router.get('/v1/users/:wallet/sessions', async (req, res) => {
    try {
      let userPubkey: PublicKey;
      try {
        userPubkey = new PublicKey(req.params.wallet);
      } catch {
        return res.status(400).json({ error: 'bad_request', message: 'Invalid wallet address' });
      }

      const [sessionChainPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('user-session-chain'), userPubkey.toBuffer()],
        PROGRAM_ID
      );

      try {
        const chain = await (program.account as any).userSessionChain.fetch(sessionChainPda);
        res.json({
          data: {
            user: chain.user.toString(),
            session_count: chain.sessionCount.toNumber(),
            encrypted_keys: chain.encryptedKeys.map((k: any) => ({
              key_ciphertext: Buffer.from(k.keyCiphertext).toString('base64'),
              nonce: Buffer.from(k.nonce).toString('base64'),
              timestamp: k.timestamp.toNumber(),
            })),
          }
        });
      } catch {
        // Account doesn't exist — user has no session chain yet
        res.json({
          data: {
            user: req.params.wallet,
            session_count: 0,
            encrypted_keys: [],
          }
        });
      }
    } catch (err) {
      console.error('[API] Failed to get user session chain:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to read session chain from blockchain' });
    }
  });

  // Camera timeline — anonymous encrypted activities stored on-chain
  // No user identification in this data. Caller decrypts if they have access grants.
  router.get('/v1/cameras/:cameraId/timeline', async (req, res) => {
    try {
      let cameraPubkey: PublicKey;
      try {
        cameraPubkey = new PublicKey(req.params.cameraId);
      } catch {
        return res.status(400).json({ error: 'bad_request', message: 'Invalid camera ID (must be a Solana public key)' });
      }

      const [timelinePda] = PublicKey.findProgramAddressSync(
        [Buffer.from('camera-timeline'), cameraPubkey.toBuffer()],
        PROGRAM_ID
      );

      try {
        const timeline = await (program.account as any).cameraTimeline.fetch(timelinePda);
        res.json({
          data: {
            camera: timeline.camera.toString(),
            activity_count: timeline.activityCount.toNumber(),
            encrypted_activities: timeline.encryptedActivities.map((a: any) => ({
              timestamp: a.timestamp.toNumber(),
              activity_type: a.activityType,
              encrypted_content: Buffer.from(a.encryptedContent).toString('base64'),
              nonce: Buffer.from(a.nonce).toString('base64'),
              access_grants: a.accessGrants.map((g: any) => Buffer.from(g).toString('base64')),
            })),
          }
        });
      } catch {
        // No timeline exists for this camera yet
        res.json({
          data: {
            camera: req.params.cameraId,
            activity_count: 0,
            encrypted_activities: [],
          }
        });
      }
    } catch (err) {
      console.error('[API] Failed to get camera timeline:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to read camera timeline from blockchain' });
    }
  });

  // ============================================================================
  // BACKEND DATA ENDPOINTS (data that legitimately lives in the backend)
  // ============================================================================

  // User profile (plaintext, not sensitive — display name, socials, pfp)
  router.get('/v1/users/:wallet/profile', async (req, res) => {
    try {
      const profile = await getUserProfile(req.params.wallet);
      if (!profile) {
        return res.status(404).json({ error: 'not_found', message: 'Profile not found' });
      }
      res.json({
        data: {
          wallet_address: profile.walletAddress,
          display_name: profile.displayName,
          username: profile.username,
          profile_image: profile.profileImage,
          provider: profile.provider,
        }
      });
    } catch (err) {
      console.error('[API] Failed to get profile:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get profile' });
    }
  });

  // User media (Walrus files, encrypted with their own access grants)
  router.get('/v1/users/:wallet/media', async (req, res) => {
    try {
      const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);
      const files = await getWalrusFilesForWallet(req.params.wallet, limit);
      res.json({
        data: files.map(f => ({
          blob_id: f.blobId,
          wallet_address: f.walletAddress,
          download_url: f.downloadUrl,
          camera_id: f.cameraId,
          file_type: f.fileType,
          timestamp: f.timestamp,
          original_size: f.originalSize,
          encrypted_size: f.encryptedSize,
          nonce: f.nonce,
          access_grants: JSON.parse(f.accessGrants || '[]'),
          created_at: f.createdAt.toISOString(),
        }))
      });
    } catch (err) {
      console.error('[API] Failed to get user media:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get media' });
    }
  });

  // Network stats (aggregate, no user-identifying data)
  router.get('/v1/stats', async (_req, res) => {
    try {
      const stats = await getDatabaseStats();
      res.json({ data: stats });
    } catch (err) {
      console.error('[API] Failed to get stats:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get stats' });
    }
  });

  return router;
}
