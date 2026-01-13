# Solana Middleware Service

Real Solana blockchain integration for the mmoment camera system. This service handles all blockchain operations including session management, on-chain identity verification, and camera network interactions.

## Features

- **Real Solana Integration**: Connects to Solana devnet with actual program interactions
- **On-Chain Identity**: Store encrypted facial embeddings on-chain for identity verification
- **Session Management**: Secure wallet-based sessions with encryption
- **Camera Network**: Integration with deployed camera network program
- **Biometric Security**: Encrypted storage and retrieval of facial data

## Configuration

Create a `.env` file or set environment variables:

```bash
# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com

# Your deployed Solana program ID
CAMERA_PROGRAM_ID=Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S

# Camera PDA address for this specific camera
CAMERA_REGISTRY_ADDRESS=WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD

# Camera Configuration
CAMERA_NAME=jetson-camera-01
CAMERA_OWNER_WALLET=your_wallet_address_here

# Service URLs (using localhost for Jetson kernel limitations)
CAMERA_SERVICE_URL=http://localhost:5002
BIOMETRIC_SERVICE_URL=http://localhost:5003
```

## 🎯 Architecture Overview

```
Frontend Wallet (Signs) ← Transaction Data ← Solana Middleware ← Camera Service
                                ↓
                        Solana Devnet/Mainnet
                        (Your Camera Network Program)
```

**Key Features:**
- ✅ Real Solana program integration using your camera network IDL
- ✅ PDA derivation for face NFTs, sessions, and cameras  
- ✅ Transaction building for frontend wallet signing
- ✅ On-chain verification of submitted transactions
- ✅ No private keys stored on Jetson (frontend signs everything)

## 🔧 Setup Instructions

### 1. Update Configuration

Edit your environment variables in `docker-compose.yml` or create a `.env` file:

```bash
# Your deployed Solana program ID
CAMERA_PROGRAM_ID=Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S

# Camera registry PDA address  
CAMERA_REGISTRY_ADDRESS=WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD

# Solana network (devnet recommended for testing)
SOLANA_RPC_URL=https://api.devnet.solana.com

# Camera identification
CAMERA_NAME=jetson-camera-01
CAMERA_OWNER_WALLET=your_wallet_address_here
```

### 2. Deploy Your Solana Program

Make sure your camera network program is deployed on devnet with the IDL provided. The program should support these instructions:

- `initialize` - Initialize camera registry (admin only)
- `registerCamera` - Register this Jetson camera
- `checkIn` - User check-in to camera session
- `checkOut` - User check-out from camera session  
- `enrollFace` - Store encrypted face embedding as NFT

### 3. Initialize Registry (One-time)

Your program's registry needs to be initialized once by an admin wallet.

### 4. Register Camera (One-time per camera)

Each Jetson camera needs to be registered in the program with its metadata.

## 📡 API Endpoints

### Session Management

```javascript
// Create session
POST /api/session/connect
{
  "wallet_address": "user_wallet_address"
}

// Check session status  
GET /api/session/status?session_id=session_id

// Disconnect session
POST /api/session/disconnect
{
  "session_id": "session_id",
  "wallet_address": "user_wallet_address"
}
```

### Face Enrollment (Real Solana)

```javascript
// Prepare face enrollment transaction
POST /api/blockchain/enroll-face
{
  "wallet_address": "user_wallet_address",
  "session_id": "session_id",
  "face_embedding": "encrypted_embedding_data" // optional
}

// Response contains transaction data for frontend signing
{
  "success": true,
  "transaction_buffer": "base64_encoded_transaction_data",
  "face_id": "generated_face_id",
  "metadata": {
    "recognition_token_pda": "derived_pda_address",
    "instruction": "enrollFace",
    "program_id": "your_program_id"
  }
}

// Confirm enrollment after transaction is signed
POST /api/face/enroll/confirm  
{
  "session_id": "session_id",
  "wallet_address": "user_wallet_address", 
  "transaction_signature": "solana_tx_signature",
  "face_id": "face_id_from_prepare"
}
```

### Session Transactions (Real Solana)

```javascript
// Prepare check-in transaction
POST /api/blockchain/prepare-checkin
{
  "wallet_address": "user_wallet_address",
  "session_id": "session_id",
  "camera_name": "jetson-camera-01",
  "use_face_recognition": true
}

// Prepare check-out transaction
POST /api/blockchain/prepare-checkout
{
  "wallet_address": "user_wallet_address", 
  "session_id": "session_id",
  "camera_name": "jetson-camera-01"
}
```

### Wallet Status (Real Solana)

```javascript
// Get real wallet status from Solana
GET /api/wallet/status?wallet_address=user_wallet_address

// Response with real on-chain data
{
  "success": true,
  "connected": true,
  "network": "devnet",
  "balance": "1.2345 SOL",
  "balance_lamports": 1234500000,
  "address": "user_wallet_address",
  "has_recognition_token": true,
  "recognition_token_pda": "derived_recognition_token_address"
}
```

## 🔐 Transaction Flow

### Face Enrollment Flow

1. **Frontend calls** `/api/blockchain/enroll-face`
2. **Middleware derives** recognition token PDA using seeds: `["recognition-token", user_wallet]`
3. **Middleware builds** `upsertRecognitionToken` instruction with encrypted embedding
4. **Middleware returns** serialized transaction data to frontend
5. **Frontend wallet signs** and submits transaction to Solana
6. **Frontend calls** `/api/face/enroll/confirm` with transaction signature
7. **Middleware verifies** transaction on-chain (optional)

### Session Flow

1. **Frontend calls** `/api/blockchain/prepare-checkin`
2. **Middleware derives** session PDA using seeds: `["session", user_wallet, camera_pda]`
3. **Middleware builds** `checkIn` instruction
4. **Frontend signs** and submits transaction
5. **User interacts** with camera (face recognition, etc.)
6. **Frontend calls** `/api/blockchain/prepare-checkout`
7. **Frontend signs** and submits `checkOut` transaction

## 🧱 PDA Derivation

The middleware correctly derives Program Derived Addresses (PDAs) according to your program:

```python
# Camera PDA
seeds = [b"camera", camera_name.encode(), user_wallet_bytes]

# Session PDA
seeds = [b"session", user_wallet_bytes, camera_pda_bytes]

# Recognition Token PDA
seeds = [b"recognition-token", user_wallet_bytes]
```

## 🔍 Testing & Verification

### Health Check
```bash
curl http://localhost:5001/api/health
```

Should return:
```json
{
  "status": "ok",
  "solana_connected": true,
  "solana_network": "https://api.devnet.solana.com",
  "program_id": "Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S",
  "idl_loaded": true
}
```

### Test Session Creation
```bash
curl -X POST http://localhost:5001/api/session/connect \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "your_test_wallet"}'
```

### Test Transaction Building
```bash
curl -X POST http://localhost:5001/api/blockchain/enroll-face \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_address": "your_test_wallet",
    "session_id": "session_from_connect"
  }'
```

## 🚀 Frontend Integration

Your frontend needs to:

1. **Decode transaction data** from `transaction_buffer`
2. **Build Solana transaction** using the provided accounts and instruction data
3. **Sign with user wallet** (Phantom, Solflare, etc.)
4. **Submit to Solana** network
5. **Call confirm endpoint** with transaction signature

Example frontend code:
```javascript
// 1. Get transaction data from middleware
const response = await fetch('/api/blockchain/enroll-face', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: wallet.publicKey.toString(),
    session_id: sessionId
  })
});

const { transaction_buffer, face_id } = await response.json();

// 2. Decode and build transaction
const transactionData = JSON.parse(atob(transaction_buffer));
const transaction = buildTransactionFromData(transactionData);

// 3. Sign with wallet
const signedTx = await wallet.signTransaction(transaction);

// 4. Submit to Solana
const signature = await connection.sendRawTransaction(signedTx.serialize());

// 5. Confirm with middleware
await fetch('/api/face/enroll/confirm', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    wallet_address: wallet.publicKey.toString(),
    transaction_signature: signature,
    face_id: face_id
  })
});
```

## 🔧 Troubleshooting

### "Program ID not set" Warning
Update `CAMERA_PROGRAM_ID` environment variable with your deployed program ID.

### "IDL not loaded" Error  
Ensure `camera_network_idl.json` exists in the service directory.

### "Solana not connected" Error
Check your `SOLANA_RPC_URL` and network connectivity.

### PDA Derivation Errors
Verify your program's seed constants match the middleware implementation.

## 📝 Notes

- **No private keys** are stored on the Jetson
- **All transactions** must be signed by frontend wallets
- **Session management** is local to the middleware (not on-chain)
- **Face embeddings** are encrypted before being stored on-chain
- **Camera registration** may need to be done separately by camera owner

This setup provides real blockchain integration while maintaining security by keeping all signing operations in the frontend wallet. 