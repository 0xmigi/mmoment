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
} from '../database';
import {
  createRpc,
  bn,
  deriveAddressSeed,
  deriveAddress,
  getDefaultAddressTreeInfo,
  selectStateTreeInfo,
  getRegisteredProgramPda,
  getAccountCompressionAuthority,
  sendAndConfirmTx,
  packNewAddressParams,
} from '@lightprotocol/stateless.js';

const PROGRAM_ID = new PublicKey('E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL');
const LIGHT_SYSTEM_PROGRAM = new PublicKey('SySTEM1eSU2p4BGQfQpimFEWWSC1XDFeun3Nqzz3rT7');
const ACCOUNT_COMPRESSION_PROGRAM = new PublicKey('compr6CUsB5m2jS4Y3831ztGSTnDpnKJTKS95d64XVq');
const NOOP_PROGRAM = new PublicKey('noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV');

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

      // Get address tree info and derive compressed account address
      const addressTreeInfo = getDefaultAddressTreeInfo();
      const addressSeed = deriveAddressSeed(
        [
          Buffer.from('timeline-entry'),
          cameraPubkey.toBuffer(),
          entryIndex.toArrayLike(Buffer, 'le', 8),
        ],
        PROGRAM_ID,
      );
      const address = deriveAddress(addressSeed, addressTreeInfo.tree);

      // Get validity proof for the new address
      const proofResult = await rpc.getValidityProofV0(
        [],
        [{
          address: bn(address.toBytes()),
          tree: addressTreeInfo.tree,
          queue: addressTreeInfo.queue,
        }]
      );

      // Get state tree for output
      const stateTreeInfos = await rpc.getStateTreeInfos();
      const outputStateTreeInfo = selectStateTreeInfo(stateTreeInfos);

      // Build remaining accounts matching Rust CpiAccounts layout
      const systemAccounts: PublicKey[] = [
        LIGHT_SYSTEM_PROGRAM,                // [0] light_system_program
        CPI_SIGNER_PDA,                      // [1] authority (cpi signer)
        getRegisteredProgramPda(),            // [2] registered_program_pda
        NOOP_PROGRAM,                        // [3] noop_program
        getAccountCompressionAuthority(),     // [4] account_compression_authority
        ACCOUNT_COMPRESSION_PROGRAM,          // [5] account_compression_program
        PROGRAM_ID,                          // [6] invoking_program
        SystemProgram.programId,             // [7] system_program
      ];

      // Pack address params into remaining accounts
      const { newAddressParamsPacked, remainingAccounts } = packNewAddressParams(
        [{
          seed: addressSeed,
          addressMerkleTreeRootIndex: proofResult.rootIndices[0],
          addressMerkleTreePubkey: addressTreeInfo.tree,
          addressQueuePubkey: addressTreeInfo.queue,
        }],
        systemAccounts,
      );

      // Add output state tree queue
      const outputTreeIndex = remainingAccounts.length;
      remainingAccounts.push(outputStateTreeInfo.queue);

      // Build instruction data manually — Anchor's BorshInstructionCoder uses
      // Buffer.alloc(1000) which overflows with large encrypted payloads.
      const encryptedPayloadBuf = Buffer.from(encrypted_payload, 'base64');
      const nonceBuf = Buffer.from(nonce, 'base64');
      const accessGrantsBlobBuf = Buffer.from(access_grants_blob, 'base64');

      // Discriminator from IDL: create_timeline_entry
      const discriminator = Buffer.from([243, 124, 212, 170, 45, 118, 145, 86]);

      // Calculate total size and allocate
      const proof = proofResult.compressedProof;
      const proofSize = proof ? 1 + 128 : 1; // Option tag + CompressedProof {a:[u8;32], b:[u8;64], c:[u8;32]}
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
      if (proof) {
        ixData.writeUInt8(1, offset); offset += 1; // Some
        Buffer.from(proof.a).copy(ixData, offset); offset += 32;
        Buffer.from(proof.b).copy(ixData, offset); offset += 64;
        Buffer.from(proof.c).copy(ixData, offset); offset += 32;
      } else {
        ixData.writeUInt8(0, offset); offset += 1; // None
      }

      // PackedAddressTreeInfo
      ixData.writeUInt8(newAddressParamsPacked[0].addressMerkleTreeAccountIndex, offset); offset += 1;
      ixData.writeUInt8(newAddressParamsPacked[0].addressQueueAccountIndex, offset); offset += 1;
      ixData.writeUInt16LE(proofResult.rootIndices[0], offset); offset += 2;

      // output_merkle_tree_index: u8
      ixData.writeUInt8(outputTreeIndex, offset); offset += 1;

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
        ...remainingAccounts.map((pubkey: PublicKey) => ({
          pubkey, isSigner: false, isWritable: false,
        })),
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

      console.log(`[Relay] Prepared timeline entry for camera ${camera_address}, entry_index=${entryIndex.toString()}, device_signer_index=${deviceSignerIndex}`);

      res.json({
        transaction: serializedTx,
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

  return router;
}
