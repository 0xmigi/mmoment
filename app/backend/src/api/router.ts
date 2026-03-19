import { Router } from 'express';
import crypto from 'crypto';
import { Connection, PublicKey, Keypair, ComputeBudgetProgram, SystemProgram, VersionedTransaction, TransactionMessage, TransactionInstruction } from '@solana/web3.js';
import { Program, AnchorProvider, Wallet } from '@coral-xyz/anchor';
import BN from 'bn.js';
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
  getCameraEvent,
  computeEventStatus,
} from '../database';
import { getQueueState, joinQueue, leaveQueue, skipCurrent, updateConfig as updateQueueConfig } from '../queue-manager';
import { getActiveUsersForCamera, getRecentTimelineEvents, getCheckedInUsers, getWalletCamera, markApiCapture } from '../camera-state';
import {
  createRpc,
  bn,
  deriveAddressSeed,
  deriveAddress,
  getRegisteredProgramPda,
  getAccountCompressionAuthority,
  sendAndConfirmTx,
  packNewAddressParams,
  toAccountMetas,
  lightSystemProgram,
  noopProgram,
  accountCompressionProgram,
} from '@lightprotocol/stateless.js';

const PROGRAM_ID = new PublicKey('E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL');

// V1 address tree — must match Rust v1::derive_address for consistent addresses
const ADDRESS_TREE = new PublicKey('amt1Ayt45jfbdw5YSo7iz6WZxUmnZsQTYXy82hVwyC2');
const ADDRESS_QUEUE = new PublicKey('aq1S9z4reTSQAdgWHGD2zDaS39sjGrAxbR31vxJ2F4F');
// V2 batch state tree for output — V1 trees give StateMerkleTreeAccountDiscriminatorMismatch
const BATCH_STATE_QUEUE = new PublicKey('oq1na8gojfdUhsfCpyjNt6h4JaDWtHf1yQj4koBWfto');

// CPI signer PDA: findProgramAddress(["cpi_authority"], PROGRAM_ID)
const [CPI_SIGNER_PDA] = PublicKey.findProgramAddressSync(
  [Buffer.from('cpi_authority')],
  PROGRAM_ID
);

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
  const program = new Program(IDL as any, provider);

  // Helius RPC for compressed account operations
  const heliusRpcUrl = process.env.HELIUS_RPC_URL;

  // ============================================================================
  // KEY MANAGEMENT (wallet-based, no API key needed to create)
  // ============================================================================

  // List all API keys for a wallet (public — only returns prefixes, not full keys)
  router.get('/v1/keys/wallet/:walletAddress', async (req, res) => {
    try {
      const keys = await getApiKeysForWallet(req.params.walletAddress);
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

  // Revoke all keys for a wallet
  router.delete('/v1/keys/wallet/:walletAddress', async (req, res) => {
    try {
      const keys = await getApiKeysForWallet(req.params.walletAddress);
      let count = 0;
      for (const key of keys) {
        if (!key.revokedAt) {
          await revokeApiKey(key.id, req.params.walletAddress);
          count++;
        }
      }
      res.json({ data: { revoked: count } });
    } catch (err) {
      console.error('[API] Failed to revoke all keys:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to revoke keys' });
    }
  });

  // Revoke a key by ID for a wallet
  router.delete('/v1/keys/wallet/:walletAddress/:keyId', async (req, res) => {
    try {
      const revoked = await revokeApiKey(req.params.keyId, req.params.walletAddress);
      if (!revoked) {
        return res.status(404).json({ error: 'not_found', message: 'Key not found' });
      }
      res.json({ data: { revoked: true } });
    } catch (err) {
      console.error('[API] Failed to revoke key:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to revoke key' });
    }
  });

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

  // Queue state — public (displayed on camera TV/desktop view)
  router.get('/v1/cameras/:cameraId/queue', async (req, res) => {
    try {
      const state = await getQueueState(req.params.cameraId);
      res.json({ data: state });
    } catch (err) {
      console.error('[API] Failed to get queue state:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get queue state' });
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

  // Camera timeline — reads compressed TimelineEntry accounts via Helius
  // No user identification in this data. Caller decrypts if they have access grants.
  router.get('/v1/cameras/:cameraId/timeline', async (req, res) => {
    try {
      if (!heliusRpcUrl) {
        return res.status(503).json({ error: 'service_unavailable', message: 'HELIUS_RPC_URL not configured' });
      }

      let cameraPubkey: PublicKey;
      try {
        cameraPubkey = new PublicKey(req.params.cameraId);
      } catch {
        return res.status(400).json({ error: 'bad_request', message: 'Invalid camera ID (must be a Solana public key)' });
      }

      const rpc = createRpc(heliusRpcUrl);

      // Fetch all compressed accounts owned by our program, filtered by camera pubkey
      // In the compressed account data layout:
      // [8 bytes discriminator][32 bytes camera pubkey][8 bytes entry_index]...
      const accounts = await rpc.getCompressedAccountsByOwner(PROGRAM_ID, {
        filters: [
          { memcmp: { offset: 8, bytes: cameraPubkey.toBase58() } },
        ],
      });

      const entries = accounts.items.map(account => {
        try {
          if (!account.data) return null;
          const data = Buffer.from(account.data.data);
          // Manual decode: discriminator(8) + camera(32) + entry_index(8) + timestamp(8) +
          // activity_count(1) + encrypted_payload(4+len) + nonce(12) + access_grants_blob(4+len)
          let offset = 8; // skip discriminator
          offset += 32; // skip camera (already filtered)
          const entryIndex = Number(data.readBigUInt64LE(offset));
          offset += 8;
          const timestamp = Number(data.readBigInt64LE(offset));
          offset += 8;
          const activityCount = data.readUInt8(offset);
          offset += 1;
          const payloadLen = data.readUInt32LE(offset);
          offset += 4;
          const encryptedPayload = data.subarray(offset, offset + payloadLen);
          offset += payloadLen;
          const nonce = data.subarray(offset, offset + 12);
          offset += 12;
          const grantsLen = data.readUInt32LE(offset);
          offset += 4;
          const accessGrantsBlob = data.subarray(offset, offset + grantsLen);

          return {
            entry_index: entryIndex,
            timestamp,
            activity_count: activityCount,
            encrypted_payload: Buffer.from(encryptedPayload).toString('base64'),
            nonce: Buffer.from(nonce).toString('base64'),
            access_grants_blob: Buffer.from(accessGrantsBlob).toString('base64'),
          };
        } catch (e) {
          console.error('[API] Failed to decode timeline entry:', e);
          return null;
        }
      }).filter(Boolean);

      entries.sort((a: any, b: any) => a.entry_index - b.entry_index);

      res.json({
        data: {
          camera: req.params.cameraId,
          entries,
          total: entries.length,
        }
      });
    } catch (err) {
      console.error('[API] Failed to get camera timeline:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to read camera timeline from blockchain' });
    }
  });

  // ============================================================================
  // RELAY ENDPOINTS — two-phase compressed timeline entry creation
  // Phase 1: Backend builds tx, signs with payer, returns serialized message
  // Phase 2: Jetson signs message with device key, sends signature back
  // Device private key NEVER leaves the Jetson.
  // ============================================================================

  router.post('/relay/prepare-timeline-entry', async (req, res) => {
    try {
      if (!heliusRpcUrl) {
        return res.status(503).json({ error: 'service_unavailable', message: 'HELIUS_RPC_URL not configured' });
      }

      const {
        camera_address,
        device_pubkey,
        encrypted_payload,
        nonce,
        access_grants_blob,
        activity_count,
        chunk_index = 0,
        total_chunks = 1,
      } = req.body;

      if (!camera_address || !device_pubkey || !encrypted_payload || !nonce ||
          !access_grants_blob || activity_count === undefined) {
        return res.status(400).json({
          error: 'bad_request',
          message: 'Required: camera_address, device_pubkey, encrypted_payload, nonce, access_grants_blob, activity_count'
        });
      }

      const cameraPubkey = new PublicKey(camera_address);
      const devicePubkey = new PublicKey(device_pubkey);

      const feePayerSecret = process.env.FEE_PAYER_SECRET_KEY;
      if (!feePayerSecret) {
        return res.status(503).json({ error: 'service_unavailable', message: 'Fee payer not configured' });
      }
      const payerKeypair = Keypair.fromSecretKey(new Uint8Array(JSON.parse(feePayerSecret)));

      const rpc = createRpc(heliusRpcUrl);

      // Fetch camera to get current activity_counter
      const camera = await (program.account as any).cameraAccount.fetch(cameraPubkey);
      const entryIndex = camera.activityCounter as BN;

      // Use V1 address tree — must match Rust v1::derive_address
      const addressTreePubkey = ADDRESS_TREE;
      const addressQueuePubkey = ADDRESS_QUEUE;
      const addressSeed = deriveAddressSeed(
        [
          Buffer.from('timeline-entry'),
          cameraPubkey.toBuffer(),
          entryIndex.toArrayLike(Buffer, 'le', 8),
        ],
        PROGRAM_ID,
      );
      const address = deriveAddress(addressSeed, addressTreePubkey);

      // Get validity proof for the new address
      const proofResult = await rpc.getValidityProofV0(
        [],
        [{
          address: bn(address.toBytes()),
          tree: addressTreePubkey,
          queue: addressQueuePubkey,
        }]
      );

      // Build remaining accounts matching Rust CpiAccounts layout
      const systemAccounts: PublicKey[] = [
        new PublicKey(lightSystemProgram),     // [0] light_system_program
        CPI_SIGNER_PDA,                      // [1] authority (cpi signer)
        getRegisteredProgramPda(),            // [2] registered_program_pda
        new PublicKey(noopProgram),           // [3] noop_program
        getAccountCompressionAuthority(),     // [4] account_compression_authority
        new PublicKey(accountCompressionProgram), // [5] account_compression_program
        PROGRAM_ID,                          // [6] invoking_program
        SystemProgram.programId,             // [7] system_program
      ];

      // Pack address params into remaining accounts
      const { newAddressParamsPacked, remainingAccounts: remainingPubkeys } = packNewAddressParams(
        [{
          seed: addressSeed,
          addressMerkleTreeRootIndex: proofResult.rootIndices[0],
          addressMerkleTreePubkey: addressTreePubkey,
          addressQueuePubkey: addressQueuePubkey,
        }],
        systemAccounts,
      );

      // Add output state tree queue
      const outputTreeIndex = remainingPubkeys.length;
      remainingPubkeys.push(BATCH_STATE_QUEUE);

      // Convert to AccountMetas with proper isSigner/isWritable flags
      const remainingAccounts = toAccountMetas(remainingPubkeys);

      // Build instruction data manually — Anchor's BorshInstructionCoder uses
      // Buffer.alloc(1000) which overflows with large encrypted payloads.
      const encryptedPayloadBuf = Buffer.from(encrypted_payload, 'base64');
      const nonceBuf = Buffer.from(nonce, 'base64');
      const accessGrantsBlobBuf = Buffer.from(access_grants_blob, 'base64');

      // Discriminator from IDL: create_timeline_entry
      const discriminator = Buffer.from([243, 124, 212, 170, 45, 118, 145, 86]);

      // Calculate total size and allocate
      const proof = proofResult.compressedProof;
      const proofA = proof ? Buffer.from(proof.a) : null;
      const proofB = proof ? Buffer.from(proof.b) : null;
      const proofC = proof ? Buffer.from(proof.c) : null;
      const proofSize = proof ? 1 + proofA!.length + proofB!.length + proofC!.length : 1;
      const totalSize = 8 // discriminator
        + proofSize
        + 4 // PackedAddressTreeInfo: u8 + u8 + u16
        + 1 // output_merkle_tree_index: u8
        + 4 + encryptedPayloadBuf.length // Vec<u8>: 4-byte len + data
        + 12 // nonce: [u8; 12]
        + 4 + accessGrantsBlobBuf.length // Vec<u8>: 4-byte len + data
        + 1 // activity_count: u8
        + 1 // chunk_index: u8
        + 1; // total_chunks: u8

      const ixData = Buffer.alloc(totalSize);
      let offset = 0;

      // Discriminator
      discriminator.copy(ixData, offset); offset += 8;

      // ValidityProof: Option<CompressedProof>
      if (proof && proofA && proofB && proofC) {
        ixData.writeUInt8(1, offset); offset += 1; // Some
        proofA.copy(ixData, offset); offset += proofA.length;
        proofB.copy(ixData, offset); offset += proofB.length;
        proofC.copy(ixData, offset); offset += proofC.length;
      } else {
        ixData.writeUInt8(0, offset); offset += 1; // None
      }

      // PackedAddressTreeInfo — indices must be relative to tree_accounts
      // (i.e., offset by system accounts length), not absolute remaining account indices.
      // The Rust SDK's CpiAccounts splits remaining_accounts into system_accounts[0..8]
      // and tree_accounts[8..], and get_tree_pubkey indexes into tree_accounts.
      const SYSTEM_ACCOUNTS_LEN = systemAccounts.length;
      ixData.writeUInt8(newAddressParamsPacked[0].addressMerkleTreeAccountIndex - SYSTEM_ACCOUNTS_LEN, offset); offset += 1;
      ixData.writeUInt8(newAddressParamsPacked[0].addressQueueAccountIndex - SYSTEM_ACCOUNTS_LEN, offset); offset += 1;
      ixData.writeUInt16LE(proofResult.rootIndices[0], offset); offset += 2;

      // output_merkle_tree_index: u8 — also relative to tree_accounts
      ixData.writeUInt8(outputTreeIndex - SYSTEM_ACCOUNTS_LEN, offset); offset += 1;

      // encrypted_payload: Vec<u8>
      ixData.writeUInt32LE(encryptedPayloadBuf.length, offset); offset += 4;
      encryptedPayloadBuf.copy(ixData, offset); offset += encryptedPayloadBuf.length;

      // nonce: [u8; 12]
      nonceBuf.copy(ixData, offset); offset += 12;

      // access_grants_blob: Vec<u8>
      ixData.writeUInt32LE(accessGrantsBlobBuf.length, offset); offset += 4;
      accessGrantsBlobBuf.copy(ixData, offset); offset += accessGrantsBlobBuf.length;

      // activity_count: u8
      ixData.writeUInt8(activity_count, offset); offset += 1;

      // chunk_index: u8
      ixData.writeUInt8(chunk_index, offset); offset += 1;

      // total_chunks: u8
      ixData.writeUInt8(total_chunks, offset); offset += 1;

      // Build account metas
      const keys = [
        { pubkey: payerKeypair.publicKey, isSigner: true, isWritable: true },
        { pubkey: devicePubkey, isSigner: true, isWritable: false },
        { pubkey: cameraPubkey, isSigner: false, isWritable: true },
        ...remainingAccounts,
      ];

      const ix = new TransactionInstruction({
        programId: PROGRAM_ID,
        keys,
        data: ixData,
      });

      const computeIx = ComputeBudgetProgram.setComputeUnitLimit({ units: 1_000_000 });
      const { blockhash } = await rpc.getLatestBlockhash();

      // Build the versioned message with both payer and device as signers
      const message = new TransactionMessage({
        payerKey: payerKeypair.publicKey,
        recentBlockhash: blockhash,
        instructions: [computeIx, ix],
      }).compileToV0Message();

      // Create tx and sign with payer ONLY — device signs on Jetson
      const tx = new VersionedTransaction(message);
      tx.sign([payerKeypair]);

      // Find device key's signer index in the message
      const accountKeys = message.staticAccountKeys;
      const deviceSignerIndex = accountKeys.findIndex(
        (key: PublicKey) => key.equals(devicePubkey)
      );

      const serializedTx = Buffer.from(tx.serialize()).toString('base64');
      const messageBytes = Buffer.from(message.serialize()).toString('base64');

      console.log(`[Relay] Prepared timeline entry for camera ${camera_address}, entry_index=${entryIndex.toString()}, device_signer_index=${deviceSignerIndex}`);

      res.json({
        transaction: serializedTx,
        message_bytes: messageBytes,
        device_signer_index: deviceSignerIndex,
        entry_index: entryIndex.toString(),
      });
    } catch (err: any) {
      console.error('[Relay] Failed to prepare timeline entry:', err);
      res.status(500).json({
        error: 'relay_failed',
        message: err.message || 'Failed to prepare timeline entry',
      });
    }
  });

  router.post('/relay/submit-timeline-entry', async (req, res) => {
    try {
      if (!heliusRpcUrl) {
        return res.status(503).json({ error: 'service_unavailable', message: 'HELIUS_RPC_URL not configured' });
      }

      const { signed_transaction } = req.body;

      if (!signed_transaction) {
        return res.status(400).json({
          error: 'bad_request',
          message: 'Required: signed_transaction (base64 serialized VersionedTransaction)'
        });
      }

      const rpc = createRpc(heliusRpcUrl);

      const txBytes = Buffer.from(signed_transaction, 'base64');
      const tx = VersionedTransaction.deserialize(txBytes);

      const signature = await sendAndConfirmTx(rpc, tx);

      console.log(`[Relay] Timeline entry submitted, sig=${signature}`);

      res.json({
        success: true,
        signature,
      });
    } catch (err: any) {
      console.error('[Relay] Failed to submit timeline entry:', err);
      res.status(500).json({
        error: 'relay_failed',
        message: err.message || 'Failed to submit timeline entry',
      });
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

  // ============================================================================
  // DEVELOPER API — IDENTITY & DISCOVERY
  // ============================================================================

  // Who am I? Returns wallet address and camera currently checked in at (if any).
  // This reads from a per-wallet record set by the check-in event — no scanning.
  router.get('/v1/me', async (req: AuthenticatedRequest, res) => {
    try {
      const walletAddress = req.apiKey!.walletAddress;
      const cameraId = getWalletCamera(walletAddress);
      const profile = await getUserProfile(walletAddress);

      res.json({
        data: {
          wallet: walletAddress,
          display_name: profile?.displayName || null,
          username: profile?.username || null,
          checked_in_at: cameraId,
        }
      });
    } catch (err) {
      console.error('[API] Failed to get identity:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get identity' });
    }
  });

  // ============================================================================
  // DEVELOPER API — CAMERA ENDPOINTS
  // Physical-first access model:
  //   - Not present: only aggregate people count (nothing else)
  //   - Checked in: full access to event, presence, activities, capture
  //   - Camera owner: full remote access (checked via camera service)
  // ============================================================================

  // Helper: check if wallet is checked in at a camera
  function isWalletPresent(cameraId: string, walletAddress: string): boolean {
    // Check both sources: timeline events (real-time) and wallet sessions (survives restarts)
    const checkedIn = getCheckedInUsers(cameraId);
    if (checkedIn.some(u => u.address === walletAddress)) return true;
    // Fallback: check wallet sessions map (hydrated from DB on startup)
    const sessionCamera = getWalletCamera(walletAddress);
    return sessionCamera === cameraId;
  }

  // Helper: check if wallet is camera owner (via camera service)
  async function isWalletCameraOwner(cameraId: string, walletAddress: string): Promise<boolean> {
    try {
      const cameraUrl = `https://${cameraId.toLowerCase()}.mmoment.xyz`;
      const res = await fetch(`${cameraUrl}/api/camera/info`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) return false;
      const data = await res.json();
      // Camera info includes owner wallet from on-chain data
      return data?.data?.owner === walletAddress || data?.owner === walletAddress;
    } catch {
      return false;
    }
  }

  // Camera status — physical-first access
  // Not present: camera_id + people_present count only
  // Present or owner: full status with event info, streaming, etc.
  router.get('/v1/cameras/:cameraId/status', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const checkedIn = getCheckedInUsers(cameraId);
      const present = isWalletPresent(cameraId, walletAddress);
      const owner = !present ? await isWalletCameraOwner(cameraId, walletAddress) : false;
      const hasAccess = present || owner;

      // Everyone gets aggregate count
      if (!hasAccess) {
        return res.json({
          data: {
            camera_id: cameraId,
            people_present: checkedIn.length,
            access: 'limited',
            message: 'Check in at the camera for full access',
          }
        });
      }

      // Full access for present users and owners
      const event = await getCameraEvent(cameraId);
      let online = false;
      let streaming = false;
      try {
        const cameraUrl = `https://${cameraId.toLowerCase()}.mmoment.xyz`;
        const healthRes = await fetch(`${cameraUrl}/api/health`, { signal: AbortSignal.timeout(3000) });
        if (healthRes.ok) {
          online = true;
          const streamRes = await fetch(`${cameraUrl}/api/stream/info`, { signal: AbortSignal.timeout(3000) });
          if (streamRes.ok) {
            const streamData = await streamRes.json();
            streaming = streamData?.data?.isActive || false;
          }
        }
      } catch (_) {}

      res.json({
        data: {
          camera_id: cameraId,
          online,
          streaming,
          people_present: checkedIn.length,
          access: 'full',
          event: event ? {
            name: event.eventName,
            status: event.eventStatus,
            start_time: event.eventStartTime,
            end_time: event.eventEndTime,
            type: event.eventType,
          } : null,
          checked_at: Date.now(),
        }
      });
    } catch (err) {
      console.error('[API] Failed to get camera status:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get camera status' });
    }
  });

  // Camera event — requires presence or ownership
  router.get('/v1/cameras/:cameraId/event', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const present = isWalletPresent(cameraId, walletAddress);
      const owner = !present ? await isWalletCameraOwner(cameraId, walletAddress) : false;

      if (!present && !owner) {
        return res.status(403).json({
          error: 'not_present',
          message: 'You must be checked in at the camera to view event details',
        });
      }

      const event = await getCameraEvent(cameraId);
      if (!event) {
        return res.json({ data: null });
      }

      const recentEvents = getRecentTimelineEvents(cameraId, 500);
      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);
      const todayMs = todayStart.getTime();
      const todayEvents = recentEvents.filter(e => e.timestamp >= todayMs);
      const checkedIn = getCheckedInUsers(cameraId);

      res.json({
        data: {
          camera_id: cameraId,
          event_name: event.eventName,
          event_description: event.eventDescription,
          event_start_time: event.eventStartTime,
          event_end_time: event.eventEndTime,
          event_type: event.eventType,
          event_location: event.eventLocation,
          event_status: event.eventStatus,
          stats: {
            check_ins_today: todayEvents.filter(e => e.type === 'check_in').length,
            photos_captured: todayEvents.filter(e => e.type === 'photo_captured' || e.type === 'video_recorded').length,
            active_now: checkedIn.length,
          },
        }
      });
    } catch (err) {
      console.error('[API] Failed to get camera event:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get camera event' });
    }
  });

  // Camera presence — requires presence or ownership
  router.get('/v1/cameras/:cameraId/presence', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const checkedIn = getCheckedInUsers(cameraId);
      const present = isWalletPresent(cameraId, walletAddress);
      const owner = !present ? await isWalletCameraOwner(cameraId, walletAddress) : false;

      if (!present && !owner) {
        return res.json({
          data: {
            camera_id: cameraId,
            count: checkedIn.length,
            users: null,
            access: 'limited',
            message: 'Check in at the camera to see who is present',
          }
        });
      }

      res.json({
        data: {
          camera_id: cameraId,
          count: checkedIn.length,
          access: 'full',
          users: checkedIn.map(u => ({
            wallet: u.address,
            display_name: u.displayName || null,
            username: u.username || null,
            profile_image: u.pfpUrl || null,
          })),
        }
      });
    } catch (err) {
      console.error('[API] Failed to get camera presence:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get camera presence' });
    }
  });

  // Camera capture — requires presence (no remote capture, even for owner)
  router.post('/v1/cameras/:cameraId/capture', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;

      if (!isWalletPresent(cameraId, walletAddress)) {
        return res.status(403).json({
          error: 'not_present',
          message: 'You must be physically checked in at the camera to capture',
        });
      }

      // Mark this capture as API-triggered so the timeline event gets tagged
      markApiCapture(walletAddress, cameraId);

      const cameraUrl = `https://${cameraId.toLowerCase()}.mmoment.xyz`;
      const captureRes = await fetch(`${cameraUrl}/api/capture`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: walletAddress,
          triggered_by: 'api',
        }),
        signal: AbortSignal.timeout(10000),
      });

      if (!captureRes.ok) {
        const errBody = await captureRes.json().catch(() => ({}));
        return res.status(captureRes.status).json({
          error: 'capture_failed',
          message: (errBody as any).error || 'Camera rejected capture request',
        });
      }

      const captureData = await captureRes.json();

      // Build a direct photo URL the agent can share
      const filename = captureData.filename;
      const photoUrl = filename ? `${cameraUrl}/photos/${filename}` : null;

      res.json({
        data: {
          success: true,
          photo_url: photoUrl,
          filename,
          timestamp: captureData.timestamp,
          width: captureData.width,
          height: captureData.height,
        }
      });
    } catch (err: any) {
      if (err.name === 'TimeoutError') {
        return res.status(504).json({ error: 'timeout', message: 'Camera did not respond in time' });
      }
      console.error('[API] Failed to trigger capture:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to trigger capture' });
    }
  });

  // Video recording — requires presence
  router.post('/v1/cameras/:cameraId/record/start', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;

      if (!isWalletPresent(cameraId, walletAddress)) {
        return res.status(403).json({ error: 'not_present', message: 'You must be physically checked in at the camera to record' });
      }

      const duration = req.body?.duration || 0; // 0 = until stopped
      const cameraUrl = `https://${cameraId.toLowerCase()}.mmoment.xyz`;
      const recordRes = await fetch(`${cameraUrl}/api/record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet_address: walletAddress, action: 'start', duration }),
        signal: AbortSignal.timeout(10000),
      });

      if (!recordRes.ok) {
        const errBody = await recordRes.json().catch(() => ({}));
        return res.status(recordRes.status).json({ error: 'record_failed', message: (errBody as any).error || 'Camera rejected record request' });
      }

      const data = await recordRes.json();
      res.json({ data: { success: true, recording: true, duration: duration || 'until_stopped' } });
    } catch (err: any) {
      if (err.name === 'TimeoutError') return res.status(504).json({ error: 'timeout', message: 'Camera did not respond in time' });
      console.error('[API] Failed to start recording:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to start recording' });
    }
  });

  router.post('/v1/cameras/:cameraId/record/stop', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;

      if (!isWalletPresent(cameraId, walletAddress)) {
        return res.status(403).json({ error: 'not_present', message: 'You must be physically checked in at the camera to stop recording' });
      }

      const cameraUrl = `https://${cameraId.toLowerCase()}.mmoment.xyz`;
      const recordRes = await fetch(`${cameraUrl}/api/record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet_address: walletAddress, action: 'stop' }),
        signal: AbortSignal.timeout(10000),
      });

      if (!recordRes.ok) {
        const errBody = await recordRes.json().catch(() => ({}));
        return res.status(recordRes.status).json({ error: 'record_failed', message: (errBody as any).error || 'Camera rejected stop request' });
      }

      const data = await recordRes.json();
      const filename = data.filename;
      const videoUrl = filename ? `${cameraUrl}/videos/${filename}` : null;

      res.json({
        data: {
          success: true,
          recording: false,
          video_url: videoUrl,
          filename,
          duration_seconds: data.duration,
        }
      });
    } catch (err: any) {
      if (err.name === 'TimeoutError') return res.status(504).json({ error: 'timeout', message: 'Camera did not respond in time' });
      console.error('[API] Failed to stop recording:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to stop recording' });
    }
  });

  // Camera activities — requires presence or ownership
  router.get('/v1/cameras/:cameraId/activities', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const present = isWalletPresent(cameraId, walletAddress);
      const owner = !present ? await isWalletCameraOwner(cameraId, walletAddress) : false;

      if (!present && !owner) {
        return res.status(403).json({
          error: 'not_present',
          message: 'You must be checked in at the camera to view activities',
        });
      }

      const limit = Math.min(parseInt(req.query.limit as string) || 50, 200);
      const events = getRecentTimelineEvents(cameraId, limit);

      res.json({
        data: {
          camera_id: cameraId,
          count: events.length,
          activities: events.map(e => ({
            id: e.id,
            type: e.type,
            timestamp: e.timestamp,
            user: {
              display_name: e.user.displayName || null,
              username: e.user.username || null,
            },
            transaction_id: e.transactionId || null,
          })),
        }
      });
    } catch (err) {
      console.error('[API] Failed to get camera activities:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to get camera activities' });
    }
  });

  // ============================================================================
  // QUEUE OPERATIONS (authenticated)
  // ============================================================================

  // Join queue — any authenticated user
  router.post('/v1/cameras/:cameraId/queue/join', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const { duration, display_name, title } = req.body || {};

      if (!duration || typeof duration !== 'number') {
        return res.status(400).json({ error: 'bad_request', message: 'duration (seconds) is required' });
      }

      const entry = await joinQueue(cameraId, walletAddress, duration, display_name, title);
      res.json({ data: entry });
    } catch (err: any) {
      if (err.message === 'Already in queue' || err.message?.includes('Duration must be') || err.message === 'Queue is not enabled for this camera') {
        return res.status(400).json({ error: 'bad_request', message: err.message });
      }
      console.error('[API] Failed to join queue:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to join queue' });
    }
  });

  // Leave queue — own entry only
  router.delete('/v1/cameras/:cameraId/queue/leave', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const removed = await leaveQueue(cameraId, walletAddress);

      if (!removed) {
        return res.status(404).json({ error: 'not_found', message: 'You are not in this queue' });
      }
      res.json({ data: { left: true } });
    } catch (err) {
      console.error('[API] Failed to leave queue:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to leave queue' });
    }
  });

  // Update queue config — camera owner only
  router.put('/v1/cameras/:cameraId/queue/config', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const owner = await isWalletCameraOwner(cameraId, walletAddress);

      if (!owner) {
        return res.status(403).json({ error: 'forbidden', message: 'Only the camera owner can configure the queue' });
      }

      const { max_slot_duration, min_slot_duration, enabled } = req.body || {};
      const config = await updateQueueConfig(cameraId, {
        enabled: enabled !== undefined ? !!enabled : undefined,
        maxSlotDuration: max_slot_duration,
        minSlotDuration: min_slot_duration,
      });
      res.json({ data: config });
    } catch (err) {
      console.error('[API] Failed to update queue config:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to update queue config' });
    }
  });

  // Skip current — camera owner only
  router.post('/v1/cameras/:cameraId/queue/skip', async (req: AuthenticatedRequest, res) => {
    try {
      const { cameraId } = req.params;
      const walletAddress = req.apiKey!.walletAddress;
      const owner = await isWalletCameraOwner(cameraId, walletAddress);

      if (!owner) {
        return res.status(403).json({ error: 'forbidden', message: 'Only the camera owner can skip the current user' });
      }

      const skipped = await skipCurrent(cameraId);
      if (!skipped) {
        return res.status(404).json({ error: 'not_found', message: 'No active slot to skip' });
      }
      res.json({ data: { skipped: true } });
    } catch (err) {
      console.error('[API] Failed to skip queue:', err);
      res.status(500).json({ error: 'internal', message: 'Failed to skip queue entry' });
    }
  });

  return router;
}
