# Compressed CameraTimeline Implementation Spec

## Overview

Replace the fixed-size `CameraTimeline` PDA (10KB, ~20 activities max) with **compressed accounts via Light Protocol / ZK Compression**. Each checkout creates an immutable compressed account — no size limits, no realloc, ~100x cheaper, unlimited history per camera.

## Prerequisites

### Anchor Upgrade: 0.29.0 → 0.31.1

Light SDK 0.23.0 **requires** Anchor 0.31.1. This is a hard dependency, no workaround.

Key breaking changes to address:
- IDL format changed (new spec in 0.30)
- Account resolution updated
- Import paths may shift
- `idl-build` feature flag syntax changed

**Do the Anchor upgrade first, verify all existing instructions still compile and work, THEN add Light SDK.**

### Dependencies

Update `programs/camera-network/Cargo.toml`:

```toml
[dependencies]
anchor-lang = { version = "0.31.1", features = ["init-if-needed"] }
anchor-spl = "0.31.1"
light-hasher = "5.0.0"
light-sdk = { version = "0.23.0", features = ["anchor", "cpi-context"] }

[features]
default = ["idl-build"]
idl-build = ["anchor-lang/idl-build", "anchor-spl/idl-build", "light-sdk/idl-build"]
```

Remove `solana-program = "1.17.0"` (Anchor 0.31 bundles its own version).

### NPM Dependencies (backend + frontend)

```json
{
  "@lightprotocol/stateless.js": "^0.22.1",
  "@coral-xyz/anchor": "^0.31.1"
}
```

### Infrastructure

- **Helius RPC** with ZK Compression support (devnet free tier works)
  - Devnet: `https://devnet.helius-rpc.com/?api-key=<KEY>`
  - Provides: Solana RPC + Photon indexer + Prover (all three behind one URL)
- No custom indexer needed

---

## Program Changes

### 1. Remove

- **`CameraTimeline` account** from `state.rs` (the struct with `Vec<EncryptedActivity>`)
- **`EncryptedActivity` struct** from `state.rs` (replaced by `TimelineEntry`)
- **`write_to_camera_timeline.rs`** instruction file
- **`write_to_camera_timeline` module** from `instructions/mod.rs` and `lib.rs`
- **`TimelineUpdated` event** from `write_to_camera_timeline.rs`

### 2. Add to `state.rs`

```rust
use light_sdk::{LightDiscriminator, LightHasher};

/// A single timeline entry — one per checkout, stored as a compressed account.
/// Contains all encrypted activities from that session at this camera.
///
/// Think of it like a git commit: immutable, append-only, tied to the camera.
/// No user identity — just anonymous encrypted blobs with access grants.
#[event]
#[derive(Clone, Debug, Default, LightDiscriminator, LightHasher)]
pub struct TimelineEntry {
    /// Which camera this entry belongs to (for querying all entries by camera)
    #[hash]
    pub camera: Pubkey,

    /// Sequential index (camera.activity_counter at time of write)
    pub entry_index: u64,

    /// When this checkout occurred
    pub timestamp: i64,

    /// Number of activities in this entry
    pub activity_count: u8,

    /// All activities from this session, serialized and encrypted together.
    /// Format: AES-256-GCM encrypted JSON array of activities.
    /// Each activity has: { type, timestamp, content }
    /// Encrypted as a single blob rather than individual activities to save space.
    #[hash]
    pub encrypted_payload: Vec<u8>,

    /// AES-GCM nonce for decrypting encrypted_payload
    pub nonce: [u8; 12],

    /// One encrypted AES key per user who was present during the session.
    /// Each grant is a NaCl SealedBox (~48 bytes) that the user can decrypt
    /// with their private key to get the AES key for encrypted_payload.
    ///
    /// NOTE: Vec<Vec<u8>> cannot use #[hash] with LightHasher's Poseidon mode.
    /// We serialize the grants into encrypted_payload instead, OR use LightHasherSha.
    /// See "Hashing Constraint" section below.
    #[hash]
    pub access_grants_blob: Vec<u8>,
}
```

**Hashing Constraint**: `Vec<Vec<u8>>` is NOT supported by `LightHasher` (Poseidon). Two options:

**Option A (recommended)**: Flatten `access_grants` into a single `Vec<u8>` (`access_grants_blob`). Prefix each grant with a 2-byte length. The client knows how to parse this format.

```
access_grants_blob = [len1_hi, len1_lo, grant1_bytes..., len2_hi, len2_lo, grant2_bytes..., ...]
```

**Option B**: Use `#[derive(LightHasherSha)]` instead of `LightHasher`. SHA256 mode serializes the entire struct via Borsh — no field restrictions. But it may not be compatible with some Light SDK features. Needs testing.

### 3. Add `instructions/create_timeline_entry.rs`

```rust
use anchor_lang::prelude::*;
use light_sdk::{
    account::LightAccount,
    address::v2::derive_address,
    cpi::{v2::CpiAccounts, CpiSigner},
    derive_light_cpi_signer,
    instruction::{PackedAddressTreeInfo, ValidityProof},
    LightDiscriminator, LightHasher, PackedAddressTreeInfoExt,
};
use light_sdk::cpi::{
    v2::LightSystemProgramCpi, InvokeLightSystemProgram, LightCpiInstruction,
};
use light_sdk::constants::ADDRESS_TREE_V2;
use crate::state::{CameraAccount, TimelineEntry, ActivityType};
use crate::error::CameraNetworkError;

// Replace with your actual program ID
pub const LIGHT_CPI_SIGNER: CpiSigner =
    derive_light_cpi_signer!("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL");

/// Event emitted when a timeline entry is created (no user info)
#[event]
pub struct TimelineEntryCreated {
    pub camera: Pubkey,
    pub entry_index: u64,
    pub activity_count: u8,
    pub timestamp: i64,
}

#[derive(Accounts)]
pub struct CreateTimelineEntry<'info> {
    /// Fee payer — backend pays gas
    #[account(mut)]
    pub payer: Signer<'info>,

    /// Device authenticator — must be camera's device key or owner
    pub device: Signer<'info>,

    /// Camera account — verified against device signer
    #[account(
        mut,
        constraint = (
            camera.device_pubkey == Some(device.key()) ||
            camera.owner == device.key()
        ) @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler<'info>(
    ctx: Context<'_, '_, '_, 'info, CreateTimelineEntry<'info>>,
    // Light SDK params
    proof: ValidityProof,
    address_tree_info: PackedAddressTreeInfo,
    output_state_tree_index: u8,
    // Timeline data
    encrypted_payload: Vec<u8>,
    nonce: [u8; 12],
    access_grants_blob: Vec<u8>,
    activity_count: u8,
) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    let now = Clock::get()?.unix_timestamp;

    require!(!encrypted_payload.is_empty(), CameraNetworkError::InvalidCameraData);

    // Build Light CPI accounts from remaining_accounts
    let light_cpi_accounts = CpiAccounts::new(
        ctx.accounts.payer.as_ref(),
        ctx.remaining_accounts,
        LIGHT_CPI_SIGNER,
    );

    // Validate address tree
    let address_tree_pubkey = address_tree_info
        .get_tree_pubkey(&light_cpi_accounts)
        .map_err(|_| ErrorCode::AccountNotEnoughKeys)?;
    if address_tree_pubkey.to_bytes() != ADDRESS_TREE_V2 {
        return Err(ProgramError::InvalidAccountData.into());
    }

    // Derive deterministic address: ["timeline-entry", camera_key, entry_index]
    let entry_index = camera.activity_counter;
    let (address, address_seed) = derive_address(
        &[
            b"timeline-entry",
            camera.key().as_ref(),
            &entry_index.to_le_bytes(),
        ],
        &address_tree_pubkey,
        &crate::ID,
    );

    let new_address_params = address_tree_info
        .into_new_address_params_assigned_packed(address_seed, Some(0));

    // Create the compressed account
    let mut entry = LightAccount::<TimelineEntry>::new_init(
        &crate::ID,
        Some(address),
        output_state_tree_index,
    );
    entry.camera = camera.key();
    entry.entry_index = entry_index;
    entry.timestamp = now;
    entry.activity_count = activity_count;
    entry.encrypted_payload = encrypted_payload;
    entry.nonce = nonce;
    entry.access_grants_blob = access_grants_blob;

    // CPI to light-system-program
    LightSystemProgramCpi::new_cpi(LIGHT_CPI_SIGNER, proof)
        .with_light_account(entry)?
        .with_new_addresses(&[new_address_params])
        .invoke(light_cpi_accounts)?;

    // Update camera stats (same as before)
    camera.activity_counter = camera.activity_counter.saturating_add(1);
    camera.last_activity_at = now;
    camera.last_activity_type = ActivityType::CheckOut as u8;

    emit!(TimelineEntryCreated {
        camera: camera.key(),
        entry_index,
        activity_count,
        timestamp: now,
    });

    Ok(())
}
```

### 4. Update `instructions/mod.rs`

```rust
// Remove:
pub mod write_to_camera_timeline;
pub use write_to_camera_timeline::*;

// Add:
pub mod create_timeline_entry;
pub use create_timeline_entry::*;
```

### 5. Update `lib.rs`

```rust
// Remove:
pub fn write_to_camera_timeline(
    ctx: Context<WriteToCameraTimeline>,
    activities: Vec<ActivityData>,
) -> Result<()> {
    instructions::write_to_camera_timeline::handler(ctx, activities)
}

// Add:
/// Create a compressed timeline entry for a camera session
/// Called at checkout — device signs, backend pays gas
pub fn create_timeline_entry<'info>(
    ctx: Context<'_, '_, '_, 'info, CreateTimelineEntry<'info>>,
    proof: ValidityProof,
    address_tree_info: PackedAddressTreeInfo,
    output_state_tree_index: u8,
    encrypted_payload: Vec<u8>,
    nonce: [u8; 12],
    access_grants_blob: Vec<u8>,
    activity_count: u8,
) -> Result<()> {
    instructions::create_timeline_entry::handler(
        ctx, proof, address_tree_info, output_state_tree_index,
        encrypted_payload, nonce, access_grants_blob, activity_count,
    )
}
```

### 6. Clean up `state.rs`

Remove:
- `CameraTimeline` struct
- `EncryptedActivity` struct
- `ActivityData` struct (no longer needed as instruction input — data comes as flat fields now)

Keep everything else unchanged.

---

## Data Format Change

### Before (old `ActivityData` per activity)

Each activity was its own struct in a Vec:
```
{ timestamp, activity_type, encrypted_content, nonce, access_grants[] }
```

### After (single `TimelineEntry` per checkout)

All activities from a session are encrypted together into one blob:
```
encrypted_payload = AES-256-GCM({
  session_id: "...",
  activities: [
    { type: "check_in", timestamp: 1234567890, data: {...} },
    { type: "photo_capture", timestamp: 1234567900, data: {...} },
    { type: "check_out", timestamp: 1234567950, data: {...} }
  ]
})
nonce = [12 bytes]
access_grants_blob = [len1, grant1, len2, grant2, ...]
```

This is actually simpler — one encryption operation per checkout instead of per activity.

### Access Grants Blob Format

Since `Vec<Vec<u8>>` can't be used with LightHasher, flatten to a single `Vec<u8>`:

```
Format: [num_grants (2 bytes BE)] + for each grant: [grant_len (2 bytes BE)] + [grant_bytes]

Example with 2 grants of 48 bytes each:
[0x00, 0x02, 0x00, 0x30, ...48 bytes..., 0x00, 0x30, ...48 bytes...]

Total: 2 + 2*(2+48) = 102 bytes for 2 users
```

Parsing (TypeScript):
```typescript
function parseAccessGrants(blob: Buffer): Buffer[] {
  const grants: Buffer[] = [];
  const numGrants = blob.readUInt16BE(0);
  let offset = 2;
  for (let i = 0; i < numGrants; i++) {
    const len = blob.readUInt16BE(offset);
    offset += 2;
    grants.push(blob.subarray(offset, offset + len));
    offset += len;
  }
  return grants;
}

function serializeAccessGrants(grants: Buffer[]): Buffer {
  const parts: Buffer[] = [Buffer.alloc(2)];
  parts[0].writeUInt16BE(grants.length);
  for (const grant of grants) {
    const lenBuf = Buffer.alloc(2);
    lenBuf.writeUInt16BE(grant.length);
    parts.push(lenBuf, grant);
  }
  return Buffer.concat(parts);
}
```

---

## Solana Middleware Changes (Jetson side)

### `app/orin_nano/services/camera-service/services/timeline_writer.py`

Currently builds a `writeToCameraTimeline` transaction with Borsh-serialized activities.

**Changes needed:**

1. **Encrypt all activities as a single blob** instead of individually:
   ```python
   # Before: each activity encrypted separately
   # After: all activities encrypted together as JSON array
   payload = json.dumps({"session_id": sid, "activities": activities_list})
   encrypted_payload, nonce = aes_encrypt(payload, session_aes_key)
   ```

2. **Flatten access grants** into the blob format described above.

3. **Call backend's new relay endpoint** that builds the Light SDK transaction:
   ```python
   # The Jetson can't build Light SDK transactions directly (needs validity proof from Helius).
   # Two options:
   #
   # Option A: Jetson calls backend, backend builds full tx, Jetson signs, backend submits
   # Option B: Backend handles everything after receiving encrypted data
   #
   # Option A preserves device attestation (device signs the tx).
   # Option B is simpler but loses the "device signed this data" property.
   ```

**Recommended flow (Option A — preserves device attestation):**

```
1. Jetson encrypts all activities into single payload
2. Jetson sends to backend: { encrypted_payload, nonce, access_grants_blob, activity_count }
3. Backend fetches validity proof from Helius
4. Backend builds the createTimelineEntry transaction (unsigned)
5. Backend sends partial tx back to Jetson
6. Jetson signs with device key
7. Jetson sends signed tx back to backend
8. Backend adds payer signature and submits
```

**Simpler flow (Option B — backend signs as device proxy):**

Not recommended long-term, but for initial testing:
```
1. Jetson encrypts all activities, sends to backend
2. Backend builds and signs full transaction (as both payer and device)
3. This requires the device constraint to allow backend/authority as signer
```

### Encryption Change

Currently each activity is encrypted individually. The new approach encrypts all activities together:

```python
# timeline_activity_service.py changes:
# Instead of encrypting each activity individually and storing encrypted blobs,
# at checkout time, collect all raw activities and encrypt once:

def prepare_timeline_payload(activities: list, present_wallets: list) -> dict:
    """Prepare a single encrypted payload for all session activities."""
    # Generate a session AES key
    session_key = os.urandom(32)

    # Encrypt all activities as one blob
    payload = json.dumps({
        "activities": [
            {"type": a["type"], "timestamp": a["timestamp"], "data": a["data"]}
            for a in activities
        ]
    }).encode()

    nonce = os.urandom(12)
    encrypted = aes_gcm_encrypt(payload, session_key, nonce)

    # Create access grants — one sealed box per present user
    grants = []
    for wallet in present_wallets:
        user_pubkey = get_user_x25519_pubkey(wallet)  # from their on-chain profile or handshake
        sealed = nacl_seal(session_key, user_pubkey)
        grants.append(sealed)

    return {
        "encrypted_payload": encrypted,
        "nonce": nonce,
        "access_grants_blob": serialize_grants(grants),
        "activity_count": len(activities),
    }
```

---

## Backend API Changes

### Relay Endpoint for Timeline Writes

Add to `solana-middleware` or backend:

```typescript
// POST /relay/create-timeline-entry
// Called by Jetson at checkout
async function createTimelineEntry(req, res) {
  const { cameraAddress, encryptedPayload, nonce, accessGrantsBlob, activityCount, deviceSignature } = req.body;

  // 1. Create Helius-backed RPC
  const rpc = createRpc(HELIUS_RPC_URL);

  // 2. Get address tree and state tree info
  const addressTree = new PublicKey(batchAddressTree);
  const stateTreeInfos = await rpc.getStateTreeInfos();
  const stateTreeInfo = selectStateTreeInfo(stateTreeInfos);

  // 3. Get camera's current activity_counter for the entry index
  const camera = await program.account.cameraAccount.fetch(cameraAddress);
  const entryIndex = camera.activityCounter;

  // 4. Derive compressed account address
  const seed = deriveAddressSeedV2([
    new TextEncoder().encode("timeline-entry"),
    new PublicKey(cameraAddress).toBytes(),
    new BN(entryIndex).toArrayLike(Buffer, "le", 8),
  ]);
  const address = deriveAddressV2(seed, addressTree, program.programId);

  // 5. Get validity proof
  const proofResult = await rpc.getValidityProofV0(
    [],
    [{ tree: addressTree, queue: addressTree, address: bn(address.toBytes()) }]
  );

  // 6. Build remaining accounts
  const systemAccountConfig = SystemAccountMetaConfig.new(program.programId);
  const remainingAccounts = PackedAccounts.newWithSystemAccountsV2(systemAccountConfig);
  const addressMerkleTreePubkeyIndex = remainingAccounts.insertOrGet(addressTree);
  const outputMerkleTreeIndex = remainingAccounts.insertOrGet(stateTreeInfo.queue);

  const packedAddressTreeInfo = {
    rootIndex: proofResult.rootIndices[0],
    addressMerkleTreePubkeyIndex,
    addressQueuePubkeyIndex: addressMerkleTreePubkeyIndex,
  };

  const proof = { 0: proofResult.compressedProof };

  // 7. Build transaction
  const tx = await program.methods
    .createTimelineEntry(
      proof,
      packedAddressTreeInfo,
      outputMerkleTreeIndex,
      Buffer.from(encryptedPayload),
      Array.from(nonce),
      Buffer.from(accessGrantsBlob),
      activityCount,
    )
    .accounts({
      payer: backendWallet.publicKey,
      device: devicePubkey,
      camera: new PublicKey(cameraAddress),
    })
    .preInstructions([ComputeBudgetProgram.setComputeUnitLimit({ units: 1_000_000 })])
    .remainingAccounts(remainingAccounts.toAccountMetas().remainingAccounts)
    .transaction();

  // 8. Sign with payer, device signature added by Jetson
  // (exact signing flow depends on Option A vs B above)
  tx.sign(backendWallet);
  const sig = await rpc.sendTransaction(tx);

  res.json({ success: true, signature: sig });
}
```

### Reading Timeline (Consumer API)

Update `/v1/cameras/:cameraId/timeline`:

```typescript
// GET /v1/cameras/:cameraId/timeline
async function getCameraTimeline(req, res) {
  const { cameraId } = req.params;
  const rpc = createRpc(HELIUS_RPC_URL);

  // Fetch all compressed accounts owned by our program
  // Filter by camera address in the data
  const accounts = await rpc.getCompressedAccountsByOwner(program.programId, {
    filters: [
      // Filter by TimelineEntry discriminator (first 8 bytes)
      { memcmp: { offset: 0, bytes: TIMELINE_ENTRY_DISCRIMINATOR } },
      // Filter by camera pubkey (bytes 8-40)
      { memcmp: { offset: 8, bytes: new PublicKey(cameraId).toBase58() } },
    ],
  });

  // Decode each entry
  const entries = accounts.items.map(account => {
    const decoded = coder.types.decode("TimelineEntry", account.data.data);
    return {
      entryIndex: decoded.entryIndex.toNumber(),
      timestamp: decoded.timestamp.toNumber(),
      activityCount: decoded.activityCount,
      encryptedPayload: Buffer.from(decoded.encryptedPayload).toString("base64"),
      nonce: Buffer.from(decoded.nonce).toString("base64"),
      accessGrantsBlob: Buffer.from(decoded.accessGrantsBlob).toString("base64"),
    };
  });

  // Sort by entry_index
  entries.sort((a, b) => a.entryIndex - b.entryIndex);

  res.json({ camera: cameraId, entries, total: entries.length });
}
```

---

## Migration Plan

### Phase 1: Anchor Upgrade (do first, independently)

1. Update `Cargo.toml` to Anchor 0.31.1
2. Fix any breaking changes in existing instructions
3. Build, deploy, verify all existing functionality still works
4. Copy new IDL to `app/backend/src/idl.json` and `app/web/src/anchor/idl.json`

### Phase 2: Add Light SDK + TimelineEntry

1. Add `light-sdk` dependency
2. Add `TimelineEntry` struct to `state.rs`
3. Add `create_timeline_entry` instruction
4. Remove old `CameraTimeline` + `write_to_camera_timeline`
5. Build, deploy

### Phase 3: Update Middleware + Backend

1. Update `timeline_writer.py` on Jetson — encrypt activities as single blob
2. Add relay endpoint to backend for building Light SDK transactions
3. Update consumer API to read compressed accounts from Helius
4. Test full check-in → check-out → read-back flow

### Phase 4: Cleanup

1. The old `CameraTimeline` PDA (account `F9o48sPQ...`) can be closed to reclaim ~0.07 SOL rent
2. Remove any dead code referencing old timeline format
3. Update frontend to use new timeline API response format

---

## Cost Summary

| Item | Cost |
|---|---|
| Per checkout (compressed account creation) | ~0.00002 SOL |
| Helius RPC (devnet) | Free |
| Helius RPC (production) | ~$50-100/month |
| Old CameraTimeline PDA rent (reclaimable) | ~0.07 SOL |

---

## Key Differences from Current Implementation

| Aspect | Current (Vec PDA) | New (Compressed) |
|---|---|---|
| Storage limit | 10KB (~20 activities) | Unlimited |
| Cost per camera | 0.07 SOL rent (fixed) | ~0.00002 SOL per checkout |
| Encryption | Per-activity | Per-session (all activities in one blob) |
| Reading data | `getAccountInfo` on PDA | Helius `getCompressedAccountsByOwner` |
| Batching | Multiple txs for many activities | Single tx per checkout |
| Account lifecycle | Mutable, grows until full | Immutable once created |
| Infrastructure | Standard Solana RPC | Helius with ZK Compression support |

---

## Open Questions

1. **Device signing flow**: Option A (Jetson signs Light tx) vs Option B (backend signs as proxy). Option A is more secure but requires round-trip. Decide based on latency tolerance.

2. **LightHasher vs LightHasherSha**: If `access_grants_blob` as `Vec<u8>` with `#[hash]` causes issues, switch to `LightHasherSha` (SHA256 mode, no field restrictions).

3. **Existing timeline data**: The old PDA has ~10KB of test activities. These won't be migrated — they're test data. Production would start fresh with compressed accounts.

4. **Helius API key**: Need a Helius account. Free tier for devnet, paid for mainnet.
