# Walrus Storage Migration Plan

This document outlines the migration from Pipe Network to Walrus for decentralized content storage with true user ownership.

## Current Architecture (What Exists Today)

### On-Chain (Solana - Encrypted)

1. **UserSessionChain** - PDA: `["user-session-chain", user.key()]`
   ```rust
   pub struct UserSessionChain {
       pub user: Pubkey,                               // Owner
       pub authority: Pubkey,                          // mmoment cron bot (fallback writer)
       pub encrypted_keys: Vec<EncryptedSessionKey>,   // AES keys for decrypting camera timeline
       pub session_count: u64,
       pub bump: u8,
   }
   ```

2. **RecognitionToken** - PDA: `["recognition-token", user.key()]`
   ```rust
   pub struct RecognitionToken {
       pub user: Pubkey,
       pub encrypted_embedding: Vec<u8>,  // Fernet-encrypted facial embedding
       pub display_name: Option<String>,
       pub source: u8,                    // 0=phone_selfie, 1=jetson_capture, 2=imported
       // ...
   }
   ```

3. **CameraTimeline** - PDA: `["camera-timeline", camera.key()]`
   - Stores encrypted activity entries (photos, videos, check-ins, etc.)

### Off-Chain (Pipe Network - UNENCRYPTED)

- **Photos/Videos**: Raw files uploaded directly to Pipe (no encryption)
- **Backend tracks**: `file_mappings` table maps device signature → Pipe fileId/fileName
- **Problem**: Pipe has ~10KB upload limit (broken), unreliable

### Off-Chain (Backend SQLite)

- `file_mappings`: signature → Pipe file info
- `user_profiles`: display names, usernames, profile images
- `session_activity_buffers`: encrypted activities (buffered before on-chain write)

---

## Target Architecture (With Walrus)

### Goals

1. **Replace Pipe with Walrus** for photo/video storage
2. **True user ownership** - Users own Blob objects on Sui
3. **mmoment pays storage** - Seamless UX, no Sui funding required
4. **Sui keypair in UserSessionChain** - Permanent, tied to Solana identity
5. **Optional: Encrypt content before upload** for privacy

### New Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER'S SOLANA WALLET                          │
│                         (Primary identity - exists)                     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    │ Signs: "mmoment:sui-storage-key:v1"
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND                                     │
│                                                                          │
│  1. Generate Sui Ed25519 keypair                                        │
│  2. Derive encryption key from Solana signature                         │
│  3. Encrypt Sui private key with derived key                            │
│  4. Store in UserSessionChain (on-chain, permanent):                    │
│     - sui_storage_address: "0x123..."        (public)                   │
│     - encrypted_sui_private_key: [bytes]     (only user can decrypt)    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    │ Sui address used for Walrus uploads
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         JETSON CAMERA (Capture)                         │
│                                                                          │
│  1. Capture photo/video                                                 │
│  2. Sign capture event with device key                                  │
│  3. Upload to Walrus with send_object_to={user_sui_address}            │
│  4. mmoment publisher pays WAL tokens                                   │
│  5. User OWNS the Blob object on Sui (costs them $0)                   │
│  6. Notify backend with blobId for tracking                            │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WALRUS NETWORK                                   │
│                                                                          │
│  - Blob stored across 50+ storage nodes                                 │
│  - User owns Blob object on Sui (transferable)                          │
│  - Instant download via any aggregator                                  │
│  - URL: https://aggregator.walrus-mainnet.walrus.space/v1/blobs/{blobId}│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Solana Program Update (UserSessionChain)

**Location**: `programs/camera-network/src/state.rs`

**Changes to UserSessionChain**:
```rust
#[account]
pub struct UserSessionChain {
    pub user: Pubkey,                               // 32 bytes - owner
    pub authority: Pubkey,                          // 32 bytes - mmoment cron bot
    pub encrypted_keys: Vec<EncryptedSessionKey>,   // Dynamic - existing
    pub session_count: u64,                         // 8 bytes
    pub bump: u8,                                   // 1 byte
    // NEW FIELDS FOR WALRUS:
    pub sui_storage_address: Option<[u8; 32]>,      // 33 bytes - User's Sui address (32 bytes + 1 for Option tag)
    pub encrypted_sui_key: Option<Vec<u8>>,         // ~100 bytes - Encrypted Sui private key
}
```

**New Instruction**: `initialize_sui_storage`
```rust
pub fn initialize_sui_storage(
    ctx: Context<InitializeSuiStorage>,
    sui_address: [u8; 32],
    encrypted_sui_key: Vec<u8>,
) -> Result<()> {
    let chain = &mut ctx.accounts.user_session_chain;

    // Only allow setting once (immutable after set)
    require!(chain.sui_storage_address.is_none(), CameraNetworkError::SuiStorageAlreadyInitialized);

    chain.sui_storage_address = Some(sui_address);
    chain.encrypted_sui_key = Some(encrypted_sui_key);

    Ok(())
}
```

**Migration Strategy**:
- Add fields as `Option<>` for backward compatibility
- Existing UserSessionChains continue to work
- New instruction to initialize Sui storage (one-time)
- Can be called during first camera check-in or explicitly by user

### Phase 2: Backend - Sui Keypair Generation

**Location**: `app/backend/src/sui-storage.ts` (new file)

```typescript
import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';
import { sha256 } from '@noble/hashes/sha256';
import * as nacl from 'tweetnacl';

// Message for deterministic signature
const SUI_KEY_DERIVATION_MESSAGE = "mmoment:sui-storage-key:v1";

interface SuiStorageInitResult {
  suiAddress: string;           // "0x..." format
  encryptedPrivateKey: Uint8Array;  // AES-256 encrypted
}

/**
 * Generate Sui keypair and encrypt it with user's Solana signature
 */
export async function initializeSuiStorage(
  solanaSignature: Uint8Array  // User signs SUI_KEY_DERIVATION_MESSAGE
): Promise<SuiStorageInitResult> {
  // 1. Derive encryption key from Solana signature
  const encryptionKey = sha256(solanaSignature);  // 32 bytes

  // 2. Generate new Sui Ed25519 keypair
  const suiKeypair = new Ed25519Keypair();
  const suiAddress = suiKeypair.getPublicKey().toSuiAddress();
  const suiPrivateKey = suiKeypair.getSecretKey();  // 32 bytes

  // 3. Encrypt Sui private key with derived key
  const nonce = nacl.randomBytes(24);  // 24 bytes for secretbox
  const encrypted = nacl.secretbox(suiPrivateKey, nonce, encryptionKey);

  // 4. Combine nonce + ciphertext for storage
  const encryptedPrivateKey = new Uint8Array(nonce.length + encrypted.length);
  encryptedPrivateKey.set(nonce);
  encryptedPrivateKey.set(encrypted, nonce.length);

  return {
    suiAddress,
    encryptedPrivateKey,  // Store this on-chain in UserSessionChain
  };
}

/**
 * Decrypt Sui private key (for user export or transfer operations)
 */
export async function decryptSuiPrivateKey(
  encryptedPrivateKey: Uint8Array,
  solanaSignature: Uint8Array
): Promise<Ed25519Keypair> {
  // 1. Derive encryption key
  const encryptionKey = sha256(solanaSignature);

  // 2. Extract nonce and ciphertext
  const nonce = encryptedPrivateKey.slice(0, 24);
  const ciphertext = encryptedPrivateKey.slice(24);

  // 3. Decrypt
  const privateKey = nacl.secretbox.open(ciphertext, nonce, encryptionKey);
  if (!privateKey) {
    throw new Error('Failed to decrypt Sui private key');
  }

  // 4. Reconstruct keypair
  return Ed25519Keypair.fromSecretKey(privateKey);
}
```

**API Endpoint**: `POST /api/user/initialize-sui-storage`
```typescript
// Request: { solanaSignature: string (base58) }
// Response: { success: true, suiAddress: string, transactionId: string }

app.post('/api/user/initialize-sui-storage', async (req, res) => {
  const { walletAddress, solanaSignature } = req.body;

  // 1. Verify signature is for correct message
  // 2. Generate Sui keypair and encrypt
  const result = await initializeSuiStorage(bs58.decode(solanaSignature));

  // 3. Call Solana program to store in UserSessionChain
  const tx = await program.methods.initializeSuiStorage(
    Array.from(bs58.decode(result.suiAddress)),
    Array.from(result.encryptedPrivateKey)
  ).accounts({...}).rpc();

  res.json({
    success: true,
    suiAddress: result.suiAddress,
    transactionId: tx,
  });
});
```

### Phase 3: Jetson - Walrus Upload Service

**Location**: `app/orin_nano/services/camera-service/services/walrus_upload_service.py` (new file)

```python
"""
Walrus Upload Service

Uploads photos/videos to Walrus with user ownership via send_object_to parameter.
mmoment pays storage costs, user owns the Blob object on Sui.
"""

import requests
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("WalrusUploadService")

# Walrus endpoints
WALRUS_PUBLISHER = "https://publisher.walrus-mainnet.walrus.space"
WALRUS_AGGREGATOR = "https://aggregator.walrus-mainnet.walrus.space"

# Default storage duration (in epochs, ~1 epoch = 1 day on mainnet)
DEFAULT_EPOCHS = 365  # 1 year


class WalrusUploadService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.backend_url = "http://192.168.1.232:3001"  # Your backend
        logger.info("WalrusUploadService initialized")

    def upload_file(
        self,
        file_path: str,
        user_sui_address: Optional[str] = None,
        epochs: int = DEFAULT_EPOCHS,
        deletable: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload file to Walrus with optional user ownership.

        Args:
            file_path: Path to file to upload
            user_sui_address: If provided, Blob ownership transfers to user
            epochs: Storage duration in epochs
            deletable: Whether blob can be deleted by owner

        Returns:
            Dict with blobId, downloadUrl, success status
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        # Build query params
        params = {
            "epochs": epochs,
            "deletable": str(deletable).lower(),
        }

        # Transfer ownership to user if Sui address provided
        if user_sui_address:
            params["send_object_to"] = user_sui_address
            logger.info(f"Uploading with ownership transfer to {user_sui_address[:16]}...")

        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    f"{WALRUS_PUBLISHER}/v1/blobs",
                    params=params,
                    data=f,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=300  # 5 minutes for large files
                )
                response.raise_for_status()

            result = response.json()

            # Extract blobId from response
            if "newlyCreated" in result:
                blob_info = result["newlyCreated"]["blobObject"]
                blob_id = blob_info["blobId"]
                object_id = blob_info["id"]
            elif "alreadyCertified" in result:
                blob_id = result["alreadyCertified"]["blobId"]
                object_id = result["alreadyCertified"].get("event", {}).get("blobId")
            else:
                return {"success": False, "error": "Unexpected response format", "raw": result}

            download_url = f"{WALRUS_AGGREGATOR}/v1/blobs/{blob_id}"

            logger.info(f"Uploaded to Walrus: {blob_id}")
            logger.info(f"Download URL: {download_url}")

            return {
                "success": True,
                "blobId": blob_id,
                "objectId": object_id,
                "downloadUrl": download_url,
                "owner": user_sui_address or "mmoment",
                "epochs": epochs,
                "size": file_path.stat().st_size,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Walrus upload failed: {e}")
            return {"success": False, "error": str(e)}

    def upload_photo(
        self,
        wallet_address: str,
        photo_path: str,
        camera_id: str,
        device_signature: str,
        timestamp: int = None,
        user_sui_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload photo to Walrus and notify backend.

        Args:
            wallet_address: User's Solana wallet (for backend tracking)
            photo_path: Path to photo file
            camera_id: Camera PDA that captured this
            device_signature: Device-signed capture event
            timestamp: Capture timestamp
            user_sui_address: User's Sui address for ownership (from UserSessionChain)
        """
        # 1. Upload to Walrus
        upload_result = self.upload_file(
            file_path=photo_path,
            user_sui_address=user_sui_address,
        )

        if not upload_result["success"]:
            return upload_result

        # 2. Notify backend for tracking
        try:
            notify_data = {
                "walletAddress": wallet_address,
                "blobId": upload_result["blobId"],
                "downloadUrl": upload_result["downloadUrl"],
                "cameraId": camera_id,
                "deviceSignature": device_signature,
                "fileType": "photo",
                "timestamp": timestamp,
                "size": upload_result["size"],
                "suiOwner": user_sui_address,
            }

            response = requests.post(
                f"{self.backend_url}/api/walrus/upload-complete",
                json=notify_data,
                timeout=30
            )
            response.raise_for_status()

            logger.info(f"Backend notified of Walrus upload: {upload_result['blobId']}")

        except Exception as e:
            logger.warning(f"Failed to notify backend: {e}")
            # Don't fail the upload if backend notification fails

        return upload_result

    def upload_video(
        self,
        wallet_address: str,
        video_path: str,
        camera_id: str,
        device_signature: str,
        duration: int = None,
        timestamp: int = None,
        user_sui_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload video to Walrus and notify backend."""
        # Same as upload_photo but with video-specific metadata
        upload_result = self.upload_file(
            file_path=video_path,
            user_sui_address=user_sui_address,
        )

        if not upload_result["success"]:
            return upload_result

        # Notify backend
        try:
            notify_data = {
                "walletAddress": wallet_address,
                "blobId": upload_result["blobId"],
                "downloadUrl": upload_result["downloadUrl"],
                "cameraId": camera_id,
                "deviceSignature": device_signature,
                "fileType": "video",
                "timestamp": timestamp,
                "size": upload_result["size"],
                "duration": duration,
                "suiOwner": user_sui_address,
            }

            response = requests.post(
                f"{self.backend_url}/api/walrus/upload-complete",
                json=notify_data,
                timeout=30
            )
            response.raise_for_status()

        except Exception as e:
            logger.warning(f"Failed to notify backend: {e}")

        return upload_result

    def get_download_url(self, blob_id: str) -> str:
        """Get download URL for a blob."""
        return f"{WALRUS_AGGREGATOR}/v1/blobs/{blob_id}"


def get_walrus_upload_service() -> WalrusUploadService:
    """Get singleton WalrusUploadService instance."""
    return WalrusUploadService()
```

### Phase 4: Backend - Walrus File Tracking

**Update `database.ts`**:
```typescript
// Add new interface
export interface WalrusFileMapping {
  blobId: string;           // Primary key - Walrus blob ID
  walletAddress: string;    // Solana wallet (for lookup)
  downloadUrl: string;      // Aggregator URL
  cameraId: string;
  deviceSignature: string;
  fileType: 'photo' | 'video';
  timestamp: number;
  size: number;
  suiOwner?: string;        // Sui address that owns this blob
  createdAt: Date;
}

// Add table creation
await runQuery(`
  CREATE TABLE IF NOT EXISTS walrus_files (
    blob_id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,
    download_url TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    device_signature TEXT NOT NULL,
    file_type TEXT NOT NULL,
    timestamp INTEGER,
    size INTEGER,
    sui_owner TEXT,
    created_at INTEGER NOT NULL
  )
`);
await runQuery(`CREATE INDEX IF NOT EXISTS idx_walrus_wallet ON walrus_files(wallet_address)`);
```

**New API endpoint**: `POST /api/walrus/upload-complete`
```typescript
app.post('/api/walrus/upload-complete', async (req, res) => {
  const { blobId, walletAddress, downloadUrl, cameraId, deviceSignature, fileType, timestamp, size, suiOwner } = req.body;

  await saveWalrusFile({
    blobId,
    walletAddress,
    downloadUrl,
    cameraId,
    deviceSignature,
    fileType,
    timestamp,
    size,
    suiOwner,
    createdAt: new Date(),
  });

  res.json({ success: true });
});
```

**Gallery API update**: `GET /api/user/:wallet/gallery`
```typescript
app.get('/api/user/:wallet/gallery', async (req, res) => {
  const files = await getWalrusFilesForWallet(req.params.wallet);

  res.json({
    success: true,
    files: files.map(f => ({
      blobId: f.blobId,
      url: f.downloadUrl,  // Direct Walrus aggregator URL
      type: f.fileType,
      timestamp: f.timestamp,
      cameraId: f.cameraId,
      owned: !!f.suiOwner,  // True if user owns on Sui
    })),
  });
});
```

### Phase 5: Frontend - Display Walrus Media

**Update media URLs**:
```typescript
// Before (Pipe)
const mediaUrl = `/api/pipe/download/${fileId}`;

// After (Walrus)
const mediaUrl = file.downloadUrl;  // Direct aggregator URL
// e.g., https://aggregator.walrus-mainnet.walrus.space/v1/blobs/{blobId}
```

**Gallery component**:
```typescript
const GalleryItem: React.FC<{ file: WalrusFile }> = ({ file }) => {
  return (
    <div className="gallery-item">
      {file.type === 'photo' ? (
        <img src={file.url} alt="Captured photo" />
      ) : (
        <video src={file.url} controls />
      )}
      {file.owned && (
        <span className="ownership-badge">
          Owned on Sui
        </span>
      )}
    </div>
  );
};
```

---

## Flow Diagrams

### First-Time User Setup (Sui Storage Initialization)

```
1. User connects Solana wallet
2. User creates UserSessionChain (existing flow)
3. Frontend prompts: "Initialize decentralized storage?"
4. User signs message: "mmoment:sui-storage-key:v1"
5. Backend:
   a. Generates Sui Ed25519 keypair
   b. Encrypts private key with signature-derived key
   c. Calls Solana program: initializeSuiStorage(suiAddress, encryptedKey)
6. UserSessionChain now has sui_storage_address
7. All future uploads transfer blob ownership to this address
```

### Photo Capture Flow (With Walrus)

```
1. User presses capture button
2. Jetson captures frame, saves to disk
3. Jetson signs capture event with device key
4. Jetson fetches user's sui_storage_address from backend
5. Jetson uploads to Walrus:
   PUT /v1/blobs?epochs=365&send_object_to={sui_address}
6. Walrus returns blobId, user now owns Blob on Sui
7. Jetson notifies backend: POST /api/walrus/upload-complete
8. Backend stores mapping: blobId → wallet, camera, etc.
9. Frontend fetches gallery, displays via direct aggregator URLs
```

### User Export Flow (Claim Sui Wallet)

```
1. User wants to export their Sui wallet (to transfer blobs, etc.)
2. Frontend prompts: "Export your Sui wallet?"
3. User signs message: "mmoment:sui-storage-key:v1"
4. Backend:
   a. Fetches encrypted_sui_key from UserSessionChain
   b. Decrypts with signature-derived key
   c. Returns Sui seed phrase or private key
5. User imports into Sui Wallet extension
6. User can now transfer/manage blobs independently
```

---

## Migration Checklist

### Prerequisites
- [ ] Acquire WAL tokens for mmoment publisher wallet
- [ ] Set up Sui wallet for mmoment (payment source)
- [ ] Test Walrus uploads on testnet

### Phase 1: Solana Program
- [ ] Add `sui_storage_address` field to UserSessionChain
- [ ] Add `encrypted_sui_key` field to UserSessionChain
- [ ] Create `initialize_sui_storage` instruction
- [ ] Update IDL
- [ ] Deploy to devnet for testing
- [ ] Deploy to mainnet

### Phase 2: Backend
- [ ] Create `sui-storage.ts` with keypair generation
- [ ] Add `/api/user/initialize-sui-storage` endpoint
- [ ] Add `walrus_files` table to database
- [ ] Add `/api/walrus/upload-complete` endpoint
- [ ] Update gallery API to return Walrus URLs
- [ ] Add `/api/user/:wallet/sui-address` endpoint

### Phase 3: Jetson
- [ ] Create `walrus_upload_service.py`
- [ ] Update `routes.py` to use Walrus instead of Pipe
- [ ] Add endpoint to fetch user's Sui address from backend
- [ ] Test photo upload flow
- [ ] Test video upload flow

### Phase 4: Frontend
- [ ] Add Sui storage initialization UI
- [ ] Update gallery to use Walrus URLs
- [ ] Add ownership badge for user-owned content
- [ ] Add "Export Sui Wallet" feature

### Phase 5: Pipe Isolation (Keep but Disable)
- [ ] Add `STORAGE_PROVIDER` env var to Jetson config
- [ ] Wrap Pipe upload calls with feature flag check
- [ ] Comment out Pipe imports when `STORAGE_PROVIDER=walrus`
- [ ] Keep all Pipe code for potential future use
- [ ] Update documentation

---

## Cost Estimates

### Walrus Storage Costs (paid by mmoment)

| Content Type | Avg Size | WAL Cost (est) | Annual (100 users, 10 photos/day) |
|--------------|----------|----------------|-----------------------------------|
| Photo        | 200 KB   | ~0.001 WAL     | ~365 WAL                          |
| Video (30s)  | 10 MB    | ~0.05 WAL      | ~18,250 WAL (if daily)           |

*Note: Actual costs depend on WAL token price and network parameters*

### Solana Costs (one-time, paid by user)

- UserSessionChain creation: ~0.003 SOL (existing)
- Initialize Sui Storage: ~0.001 SOL (new, one-time)
- Account rent for new fields: ~0.0005 SOL

---

## Security Considerations

1. **Sui Private Key Security**
   - Encrypted with AES-256 (via NaCl secretbox)
   - Encryption key derived from Solana signature
   - Only user with Solana private key can decrypt
   - Stored on-chain in UserSessionChain (encrypted)

2. **Content Privacy (Optional Enhancement)**
   - By default, Walrus blobs are PUBLIC
   - For private content: encrypt before upload
   - Use existing camera session keys for encryption
   - Only session participants can decrypt

3. **Blob Ownership**
   - User owns Sui Blob object
   - mmoment cannot delete or transfer user's blobs
   - User can transfer/delete if they export Sui wallet

---

## Design Decisions (Finalized)

### 1. Storage Duration: 183 epochs (~1 year maximum)

**Decision**: Store for maximum duration (183 epochs ≈ 1 year)

**Rationale**:
- No true "permanent" option exists - "permanent" means non-deletable, not infinite
- Maximum is 183 epochs (~1 year) per transaction
- Cost difference is linear (6 months = half the cost)
- Can add renewal mechanism later if needed
- Users can use "shared blobs" for community-funded extensions

**Cost per 200KB photo**: ~0.0001 WAL per year

### 2. Privacy: Encrypt content BEFORE upload

**Decision**: Yes, encrypt all photos/videos before Walrus upload

**Implementation**:
- Use AES-256-GCM encryption
- Key derived from user's session (same pattern as timeline activities)
- Only users with decryption key can view content
- Encrypted blob is public on Walrus, but content is protected

### 3. Existing Content: Start fresh, no migration

**Decision**: Do NOT migrate existing Pipe content

**Rationale**:
- Simplifies implementation
- Existing content still accessible (Pipe code preserved)
- New content goes to Walrus

### 4. Pipe Network: Keep code but isolate with feature flag

**Decision**: Keep Pipe code for potential future use, but disable by default

**Implementation**:
```python
# In routes.py or config
STORAGE_PROVIDER = os.environ.get("STORAGE_PROVIDER", "walrus")

if STORAGE_PROVIDER == "walrus":
    from services.walrus_upload_service import get_walrus_upload_service
    upload_service = get_walrus_upload_service()
elif STORAGE_PROVIDER == "pipe":
    from services.direct_pipe_upload import get_direct_pipe_upload_service
    upload_service = get_direct_pipe_upload_service()
```

If Pipe improves their Firestarter storage, can switch back with env var change.

---

## Updated Implementation: Encrypted Uploads

Since content will be encrypted before upload, here's the updated Jetson service:

### Jetson - Encrypted Walrus Upload Service

**Location**: `app/orin_nano/services/camera-service/services/walrus_upload_service.py`

```python
"""
Walrus Upload Service with Pre-Upload Encryption

Encrypts content with session key before uploading to Walrus.
Only users with the decryption key can view content.
mmoment pays storage, user owns the encrypted Blob on Sui.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("WalrusUploadService")

WALRUS_PUBLISHER = "https://publisher.walrus-mainnet.walrus.space"
WALRUS_AGGREGATOR = "https://aggregator.walrus-mainnet.walrus.space"
DEFAULT_EPOCHS = 183  # Maximum (~1 year)


class WalrusUploadService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.backend_url = os.environ.get("BACKEND_URL", "http://192.168.1.232:3001")
        logger.info("WalrusUploadService initialized (with encryption)")

    def _encrypt_file(self, file_path: str, encryption_key: bytes) -> tuple[bytes, bytes]:
        """
        Encrypt file content with AES-256-GCM.

        Args:
            file_path: Path to file to encrypt
            encryption_key: 32-byte AES key

        Returns:
            (encrypted_data, nonce) - nonce prepended to ciphertext
        """
        with open(file_path, 'rb') as f:
            plaintext = f.read()

        # Generate random 12-byte nonce
        nonce = os.urandom(12)

        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(encryption_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Prepend nonce to ciphertext
        encrypted_data = nonce + ciphertext

        logger.info(f"Encrypted {len(plaintext)} bytes → {len(encrypted_data)} bytes")
        return encrypted_data, nonce

    def upload_file_encrypted(
        self,
        file_path: str,
        encryption_key: bytes,
        user_sui_address: Optional[str] = None,
        epochs: int = DEFAULT_EPOCHS,
    ) -> Dict[str, Any]:
        """
        Encrypt file and upload to Walrus.

        Args:
            file_path: Path to file
            encryption_key: 32-byte AES key
            user_sui_address: Sui address for blob ownership
            epochs: Storage duration (183 = max = ~1 year)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            # 1. Encrypt the file
            encrypted_data, nonce = self._encrypt_file(str(file_path), encryption_key)

            # 2. Build query params
            params = {
                "epochs": epochs,
                "deletable": "true",
            }

            if user_sui_address:
                params["send_object_to"] = user_sui_address

            # 3. Upload encrypted data
            response = requests.put(
                f"{WALRUS_PUBLISHER}/v1/blobs",
                params=params,
                data=encrypted_data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300
            )
            response.raise_for_status()

            result = response.json()

            # Extract blobId
            if "newlyCreated" in result:
                blob_info = result["newlyCreated"]["blobObject"]
                blob_id = blob_info["blobId"]
                object_id = blob_info["id"]
            elif "alreadyCertified" in result:
                blob_id = result["alreadyCertified"]["blobId"]
                object_id = None
            else:
                return {"success": False, "error": "Unexpected response", "raw": result}

            return {
                "success": True,
                "blobId": blob_id,
                "objectId": object_id,
                "downloadUrl": f"{WALRUS_AGGREGATOR}/v1/blobs/{blob_id}",
                "owner": user_sui_address or "mmoment",
                "encrypted": True,
                "nonce": nonce.hex(),
                "originalSize": file_path.stat().st_size,
                "encryptedSize": len(encrypted_data),
            }

        except Exception as e:
            logger.error(f"Walrus upload failed: {e}")
            return {"success": False, "error": str(e)}

    def upload_photo(
        self,
        wallet_address: str,
        photo_path: str,
        camera_id: str,
        device_signature: str,
        encryption_key: bytes,
        timestamp: int = None,
        user_sui_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload encrypted photo to Walrus and notify backend."""

        upload_result = self.upload_file_encrypted(
            file_path=photo_path,
            encryption_key=encryption_key,
            user_sui_address=user_sui_address,
        )

        if not upload_result["success"]:
            return upload_result

        # Notify backend
        try:
            notify_data = {
                "walletAddress": wallet_address,
                "blobId": upload_result["blobId"],
                "downloadUrl": upload_result["downloadUrl"],
                "cameraId": camera_id,
                "deviceSignature": device_signature,
                "fileType": "photo",
                "timestamp": timestamp,
                "originalSize": upload_result["originalSize"],
                "encryptedSize": upload_result["encryptedSize"],
                "encrypted": True,
                "nonce": upload_result["nonce"],
                "suiOwner": user_sui_address,
            }

            requests.post(
                f"{self.backend_url}/api/walrus/upload-complete",
                json=notify_data,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Backend notify failed: {e}")

        return upload_result


def get_walrus_upload_service() -> WalrusUploadService:
    return WalrusUploadService()
```

### Frontend - Decrypt and Display

```typescript
/**
 * Decrypt Walrus blob using session key
 */
async function decryptWalrusBlob(
  downloadUrl: string,
  encryptionKey: Uint8Array,  // 32 bytes from session
): Promise<Blob> {
  // 1. Fetch encrypted blob
  const response = await fetch(downloadUrl);
  const encryptedData = await response.arrayBuffer();

  // 2. Extract nonce (first 12 bytes) and ciphertext
  const nonce = new Uint8Array(encryptedData.slice(0, 12));
  const ciphertext = new Uint8Array(encryptedData.slice(12));

  // 3. Decrypt with AES-GCM
  const key = await crypto.subtle.importKey(
    'raw', encryptionKey, { name: 'AES-GCM' }, false, ['decrypt']
  );

  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: nonce }, key, ciphertext
  );

  return new Blob([decrypted]);
}
```

---

## Summary of Final Design

| Aspect | Decision |
|--------|----------|
| **Storage Duration** | 183 epochs (~1 year max) |
| **Content Privacy** | Encrypted with AES-256-GCM before upload |
| **Existing Content** | No migration, start fresh |
| **Pipe Network** | Keep code, isolate with `STORAGE_PROVIDER` env var |
| **Blob Ownership** | Transfer to user's Sui address |
| **Payment** | mmoment pays WAL tokens |

---

## Updated Cost Estimates

### Per-Photo Cost (encrypted, 1 year storage)

| Item | Cost |
|------|------|
| 200KB photo → ~250KB encrypted → ~1.25MB encoded | ~0.0001 WAL |
| Sui gas for blob object transfer | ~0.001 SUI |

### Monthly Estimate (100 users, 10 photos/day)

| Item | Monthly |
|------|---------|
| Storage (30,000 photos) | ~3 WAL |
| Sui gas | ~30 SUI |

*Costs based on testnet pricing, mainnet may vary*
