#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import time
import logging
import os
import json
import base64
import hashlib
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Solana imports for blockchain reading and writing
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.message import Message
from solana.rpc.types import MemcmpOpts, TxOpts
from solana.rpc.commitment import Confirmed
import base58
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('solana_middleware.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Solana configuration
SOLANA_RPC_URL = os.getenv('SOLANA_RPC_URL', 'https://api.devnet.solana.com')
CAMERA_PROGRAM_ID = os.getenv('CAMERA_PROGRAM_ID', 'YourCameraProgramIdHere')  # Replace with actual program ID
def get_camera_pda():
    """Get camera PDA dynamically from config file or environment"""
    # First try environment variable (for backward compatibility)
    env_pda = os.getenv('CAMERA_PDA')
    if env_pda and env_pda != 'YourCameraPDAHere':
        return env_pda
    
    # Then try config file (written by DeviceRegistrationService)
    config_path = '/app/config/device_config.json'
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('camera_pda')
    except Exception as e:
        logger.debug(f"Could not read device config: {e}")
    
    return None

# Initialize Solana client
solana_client = Client(SOLANA_RPC_URL)

# Simulate database with in-memory dictionary
sessions = {}
# Store for encryption keys
encryption_keys = {}

def generate_encryption_key(wallet_address, camera_pda):
    """Generate a deterministic encryption key based on wallet and camera"""
    # In production, this would use a real Solana NFT for key derivation
    # For now, we'll use a simple deterministic approach
    salt = b'mmoment-facial-recognition'  # Fixed salt
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    # Combine wallet and camera PDA for unique key per user per camera
    key_material = f"{wallet_address}-{camera_pda}".encode()
    key = base64.urlsafe_b64encode(kdf.derive(key_material))
    return key

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to confirm API is working"""
    return jsonify({
        "message": "Solana Middleware API is running",
        "status": "ok",
        "timestamp": int(time.time() * 1000)
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Solana Middleware is healthy",
        "timestamp": int(time.time() * 1000)
    })

# Standardized API endpoint
@app.route('/api/health', methods=['GET'])
def api_health():
    """Standardized health check endpoint"""
    return health()

# Standardized session management endpoints
@app.route('/api/session/connect', methods=['POST'])
def api_session_connect():
    """Standardized session connect endpoint"""
    return connect_wallet()

@app.route('/connect-wallet', methods=['POST'])
def connect_wallet():
    """Connect a wallet to the system and create a session"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        
        if not wallet_address:
            return jsonify({"error": "Wallet address is required"}), 400
        
        # Generate a session ID
        session_id = str(uuid.uuid4().hex)[:16]
        
        # Store session info
        sessions[session_id] = {
            "wallet_address": wallet_address,
            "created_at": time.time(),
            "active": True
        }
        
        logger.info(f"Wallet connected: {wallet_address} with session ID: {session_id}")
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Wallet connected successfully"
        })
    
    except Exception as e:
        logger.error(f"Error connecting wallet: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/disconnect-wallet', methods=['POST'])
def disconnect_wallet():
    """Disconnect a wallet and invalidate the session"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        
        if not wallet_address or not session_id:
            return jsonify({"error": "Wallet address and session ID are required"}), 400
        
        # Check if session exists
        if session_id not in sessions:
            return jsonify({"error": "Invalid session ID"}), 404
        
        # Check if wallet address matches
        if sessions[session_id]["wallet_address"] != wallet_address:
            return jsonify({"error": "Wallet address does not match session"}), 403
        
        # Mark session as inactive
        sessions[session_id]["active"] = False
        
        logger.info(f"Wallet disconnected: {wallet_address} with session ID: {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Wallet disconnected successfully"
        })
    
    except Exception as e:
        logger.error(f"Error disconnecting wallet: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/encrypt-face-embedding', methods=['POST'])
def encrypt_face_embedding():
    """Encrypt a face embedding using the wallet's NFT-based key"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        face_embedding = data.get('face_embedding')
        camera_pda = data.get('camera_pda')
        
        if not wallet_address or not face_embedding:
            return jsonify({"error": "Wallet address and face embedding are required"}), 400
        
        # In production, verify NFT ownership here
        # For testing, we'll simulate successful verification
        
        # Generate encryption key
        key = generate_encryption_key(wallet_address, camera_pda)
        encryption_keys[wallet_address] = key
        cipher = Fernet(key)
        
        # Convert embedding to JSON and encrypt
        embedding_json = json.dumps(face_embedding).encode()
        encrypted_data = cipher.encrypt(embedding_json)
        encrypted_string = base64.b64encode(encrypted_data).decode()
        
        logger.info(f"Encrypted face embedding for wallet: {wallet_address}")
        
        return jsonify({
            "success": True,
            "encrypted_embedding": encrypted_string,
            "message": "Face embedding encrypted successfully"
        })
    
    except Exception as e:
        logger.error(f"Error encrypting face embedding: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/decrypt-face-embedding', methods=['POST'])
def decrypt_face_embedding():
    """Decrypt a face embedding using the wallet's NFT-based key"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        encrypted_embedding = data.get('encrypted_embedding')
        camera_pda = data.get('camera_pda')
        
        if not wallet_address or not encrypted_embedding:
            return jsonify({"error": "Wallet address and encrypted embedding are required"}), 400
        
        # Get or generate encryption key
        if wallet_address in encryption_keys:
            key = encryption_keys[wallet_address]
        else:
            key = generate_encryption_key(wallet_address, camera_pda)
            encryption_keys[wallet_address] = key
            
        cipher = Fernet(key)
        
        # Decode and decrypt
        encrypted_data = base64.b64decode(encrypted_embedding)
        decrypted_json = cipher.decrypt(encrypted_data)
        face_embedding = json.loads(decrypted_json)
        
        logger.info(f"Decrypted face embedding for wallet: {wallet_address}")
        
        return jsonify({
            "success": True,
            "face_embedding": face_embedding,
            "message": "Face embedding decrypted successfully"
        })
    
    except Exception as e:
        logger.error(f"Error decrypting face embedding: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/verify-nft-ownership', methods=['POST'])
def verify_nft_ownership():
    """Verify that a wallet owns a valid recognition token"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        camera_pda = data.get('camera_pda')
        
        if not wallet_address:
            return jsonify({"error": "Wallet address is required"}), 400
        
        # In production, verify NFT ownership on Solana here
        # For testing, simulate successful verification
        # This would check if the wallet owns an NFT from the specified collection
        
        # Always return true for development
        has_nft = True
        
        logger.info(f"Verified NFT ownership for wallet: {wallet_address}, result: {has_nft}")
        
        return jsonify({
            "success": True,
            "has_valid_nft": has_nft,
            "message": "NFT ownership verified"
        })
    
    except Exception as e:
        logger.error(f"Error verifying NFT ownership: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/mint-moment', methods=['POST'])
def mint_moment():
    """Simulate minting a moment as NFT"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        session_id = data.get('session_id')
        image_url = data.get('image_url')
        
        if not wallet_address or not session_id:
            return jsonify({"error": "Wallet address and session ID are required"}), 400
        
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400
        
        # Check if session exists and is active
        if session_id not in sessions or not sessions[session_id]["active"]:
            return jsonify({"error": "Invalid or inactive session ID"}), 404
        
        # Check if wallet address matches
        if sessions[session_id]["wallet_address"] != wallet_address:
            return jsonify({"error": "Wallet address does not match session"}), 403
        
        # Simulate minting process
        nft_data = {
            "mint": f"mint{uuid.uuid4().hex[:8]}",
            "metadata": f"meta{uuid.uuid4().hex[:8]}",
            "transaction": f"tx{uuid.uuid4().hex[:16]}"
        }
        
        logger.info(f"Moment minted for wallet: {wallet_address} - NFT: {nft_data['mint']}")
        
        return jsonify({
            "success": True,
            "nft": nft_data,
            "message": "Moment minted successfully"
        })
    
    except Exception as e:
        logger.error(f"Error minting moment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/session-status', methods=['GET'])
def session_status():
    """Get status of all active sessions"""
    active_sessions = {
        sid: data for sid, data in sessions.items() 
        if data.get("active", False)
    }
    
    return jsonify({
        "success": True,
        "active_sessions": len(active_sessions),
        "sessions": active_sessions
    })

# ========================================
# STANDARDIZED API ROUTES (Pi5 Compatible)
# ========================================

@app.route('/api/session/disconnect', methods=['POST'])
def api_session_disconnect():
    """Standardized session disconnect endpoint"""
    return disconnect_wallet()

@app.route('/api/blockchain/encrypt-face', methods=['POST'])
def api_encrypt_face():
    """Standardized face encryption endpoint"""
    return encrypt_face_embedding()

@app.route('/api/blockchain/decrypt-face', methods=['POST'])
def api_decrypt_face():
    """Standardized face decryption endpoint"""
    return decrypt_face_embedding()

@app.route('/api/blockchain/verify-nft', methods=['POST'])
def api_verify_nft():
    """Standardized NFT verification endpoint"""
    return verify_nft_ownership()

@app.route('/api/blockchain/mint-moment', methods=['POST'])
def api_mint_moment():
    """Standardized moment minting endpoint"""
    return mint_moment()

@app.route('/api/session/status', methods=['GET'])
def api_session_status():
    """Standardized session status endpoint"""
    return session_status()

@app.route('/api/wallet/status', methods=['GET'])
def api_wallet_status():
    """Get wallet connection status and basic info"""
    try:
        # For testing purposes, return a mock wallet status
        # In production, this would check actual Solana wallet connection
        return jsonify({
            'success': True,
            'connected': True,
            'network': 'devnet',
            'balance': '1.5 SOL',
            'address': 'TestWallet123ABC...',
            'nft_count': 3,
            'message': 'Wallet connected to devnet'
        })
    except Exception as e:
        logger.error(f"Error getting wallet status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/blockchain/mint-recognition-token', methods=['POST'])
def mint_recognition_token():
    """Process encrypted facial embedding and prepare transaction for frontend signing"""
    try:
        from solders.instruction import Instruction, AccountMeta
        from solders.message import Message
        from solders.transaction import Transaction as SoldersTransaction
        import struct

        data = request.json
        wallet_address = data.get('wallet_address')
        face_embedding = data.get('face_embedding')  # This is the encrypted NFT package
        biometric_session_id = data.get('biometric_session_id')

        if not wallet_address or not face_embedding:
            return jsonify({"error": "wallet_address and face_embedding are required"}), 400

        # The face_embedding is already the encrypted NFT package from the camera service
        nft_package = face_embedding

        # Generate a face ID for tracking
        face_id = f"face_{uuid.uuid4().hex[:16]}"

        # Get the encrypted embedding data
        encrypted_embedding_b64 = nft_package.get("encrypted_embedding", "")
        encrypted_embedding_bytes = base64.b64decode(encrypted_embedding_b64)

        logger.info(f"[TRANSACTION] Building Solana transaction for wallet {wallet_address}")
        logger.info(f"[TRANSACTION] Encrypted embedding size: {len(encrypted_embedding_bytes)} bytes")

        # Solana transaction size limit: 1232 bytes total
        # With int8 quantization: 512 bytes raw + Fernet overhead (~300-400 bytes) = ~900 bytes
        # Transaction overhead (signatures, accounts, discriminator): ~250 bytes
        # Safe limit for instruction data: 950 bytes
        MAX_EMBEDDING_SIZE = 950

        if len(encrypted_embedding_bytes) > MAX_EMBEDDING_SIZE:
            logger.error(f"[TRANSACTION] Embedding too large ({len(encrypted_embedding_bytes)} bytes), exceeds max {MAX_EMBEDDING_SIZE} bytes")
            return jsonify({"error": f"Encrypted embedding too large: {len(encrypted_embedding_bytes)} bytes (max {MAX_EMBEDDING_SIZE})"}), 400

        logger.info(f"[TRANSACTION] Final embedding size for transaction: {len(encrypted_embedding_bytes)} bytes")

        # Build the Solana instruction data
        # Anchor instruction format: 8-byte discriminator + instruction data
        # For upsert_recognition_token: sighash(global:upsert_recognition_token) + args
        # Args: encrypted_embedding: Vec<u8>, display_name: Option<String>, source: u8

        # Anchor discriminator for upsert_recognition_token
        discriminator = hashlib.sha256(b"global:upsert_recognition_token").digest()[:8]

        # Serialize arguments (Borsh format):
        # 1. Vec<u8>: u32 length + bytes
        # 2. Option<String>: 1 byte (0=None, 1=Some) + (if Some: u32 length + UTF-8 bytes)
        # 3. u8: 1 byte

        # Argument 1: encrypted_embedding (Vec<u8>)
        vec_length = struct.pack('<I', len(encrypted_embedding_bytes))

        # Argument 2: display_name (Option<String>) - use "Phone Selfie"
        display_name = "Phone Selfie"
        display_name_bytes = display_name.encode('utf-8')
        display_name_data = b'\x01' + struct.pack('<I', len(display_name_bytes)) + display_name_bytes

        # Argument 3: source (u8) - 0=phone_selfie, 1=jetson_capture, 2=imported
        source = b'\x00'  # 0 = phone_selfie

        instruction_data = discriminator + vec_length + encrypted_embedding_bytes + display_name_data + source

        logger.info(f"[TRANSACTION] Instruction data size: {len(instruction_data)} bytes")
        logger.info(f"[TRANSACTION] Discriminator: {discriminator.hex()}")

        # Convert wallet address to Pubkey
        user_pubkey = Pubkey.from_string(wallet_address)
        program_id = Pubkey.from_string(CAMERA_PROGRAM_ID)
        system_program = Pubkey.from_string("11111111111111111111111111111111")

        # ✅ NEW: Derive recognition_token PDA (changed from face-nft)
        recognition_token_seeds = [b"recognition-token", bytes(user_pubkey)]
        recognition_token_pda, bump = Pubkey.find_program_address(recognition_token_seeds, program_id)

        logger.info(f"[TRANSACTION] User: {user_pubkey}")
        logger.info(f"[TRANSACTION] Recognition Token PDA: {recognition_token_pda}")
        logger.info(f"[TRANSACTION] Program ID: {program_id}")

        # Create instruction with accounts in the correct order
        accounts = [
            AccountMeta(pubkey=user_pubkey, is_signer=True, is_writable=True),  # user (signer, mut)
            AccountMeta(pubkey=recognition_token_pda, is_signer=False, is_writable=True),  # recognition_token (mut)
            AccountMeta(pubkey=system_program, is_signer=False, is_writable=False),  # system_program
        ]

        instruction = Instruction(
            program_id=program_id,
            accounts=accounts,
            data=bytes(instruction_data)
        )

        # Get recent blockhash
        response = solana_client.get_latest_blockhash()
        recent_blockhash = response.value.blockhash

        logger.info(f"[TRANSACTION] Recent blockhash: {recent_blockhash}")

        # Create transaction
        message = Message.new_with_blockhash(
            [instruction],
            user_pubkey,  # Fee payer
            recent_blockhash
        )

        transaction = SoldersTransaction.new_unsigned(message)

        # Serialize transaction to bytes and encode as base64
        transaction_bytes = bytes(transaction)
        transaction_buffer = base64.b64encode(transaction_bytes).decode('utf-8')

        logger.info(f"[TRANSACTION] Transaction built successfully")
        logger.info(f"[TRANSACTION] Transaction buffer size: {len(transaction_buffer)} chars")
        logger.info(f"[TRANSACTION] Transaction buffer preview: {transaction_buffer[:100]}...")

        return jsonify({
            "success": True,
            "transaction_buffer": transaction_buffer,
            "face_id": face_id,
            "recognition_token_pda": str(recognition_token_pda),
            "message": "Transaction prepared for signing"
        })

    except Exception as e:
        logger.error(f"Error preparing recognition token transaction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/face/enroll/confirm', methods=['POST'])
def confirm_face_enrollment():
    """Confirm face enrollment transaction and finalize the process"""
    try:
        data = request.json
        wallet_address = data.get('wallet_address')
        transaction_signature = data.get('transaction_signature')
        face_id = data.get('face_id')
        
        if not wallet_address or not transaction_signature or not face_id:
            return jsonify({"error": "wallet_address, transaction_signature, and face_id are required"}), 400
        
        logger.info(f"Confirming face enrollment transaction for wallet {wallet_address}")
        logger.info(f"Transaction signature: {transaction_signature}")
        logger.info(f"Face ID: {face_id}")
        
        # In a real implementation, this would:
        # 1. Verify the transaction signature on Solana
        # 2. Confirm the transaction was successful
        # 3. Update any local state/database
        
        # For now, we'll simulate successful confirmation
        transaction_id = transaction_signature  # Use the signature as the transaction ID
        
        logger.info(f"Face enrollment confirmed successfully for wallet {wallet_address}")
        
        return jsonify({
            "success": True,
            "transaction_id": transaction_id,
            "face_id": face_id,
            "wallet_address": wallet_address,
            "status": "confirmed",
            "message": "Face enrollment confirmed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error confirming face enrollment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/blockchain/purge-session', methods=['POST'])
def purge_biometric_session():
    """Purge a biometric session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"error": "Session ID is required"}), 400
        
        # In a real implementation, this would securely delete the session
        # For now, we'll just log it
        logger.info(f"Purging biometric session: {session_id}")
        
        return jsonify({
            "success": True,
            "message": "Biometric session purged successfully"
        })
    
    except Exception as e:
        logger.error(f"Error purging biometric session: {e}")
        return jsonify({"error": str(e)}), 500

# New blockchain reading endpoints
@app.route('/api/blockchain/checked-in-users', methods=['GET'])
def get_checked_in_users_api():
    """Get all users currently checked in to the camera"""
    try:
        camera_pubkey = request.args.get('camera_pubkey', get_camera_pda())

        if not camera_pubkey or camera_pubkey == 'YourCameraPDAHere':
            return jsonify({
                "error": "Camera PDA not configured. Please complete device registration first."
            }), 400

        checked_in_users = get_checked_in_users(camera_pubkey)

        return jsonify({
            "success": True,
            "camera_pubkey": camera_pubkey,
            "checked_in_users": checked_in_users,
            "count": len(checked_in_users)
        })

    except Exception as e:
        logger.error(f"Error getting checked-in users: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/blockchain/get-recognition-token', methods=['GET'])
def get_recognition_token_api():
    """Get a user's recognition token (encrypted embedding) from on-chain"""
    try:
        from solders.pubkey import Pubkey

        wallet_address = request.args.get('wallet_address')

        if not wallet_address:
            return jsonify({"error": "wallet_address is required"}), 400

        logger.info(f"🔍 Fetching recognition token for wallet: {wallet_address}")

        # Convert wallet address to Pubkey
        user_pubkey = Pubkey.from_string(wallet_address)

        # ✅ Derive the RecognitionToken PDA: seeds = ["recognition-token", user_pubkey]
        program_id = Pubkey.from_string(CAMERA_PROGRAM_ID)
        recognition_token_pda, bump = Pubkey.find_program_address(
            [b"recognition-token", bytes(user_pubkey)],
            program_id
        )

        logger.info(f"📍 Recognition Token PDA: {recognition_token_pda}")

        # Fetch the account data from Solana
        account_info = solana_client.get_account_info(recognition_token_pda)

        if not account_info.value:
            logger.warning(f"⚠️  No recognition token found for wallet {wallet_address}")
            return jsonify({
                "success": False,
                "error": "No recognition token found on-chain for this wallet"
            }), 404

        # The account data contains the encrypted embedding
        # Anchor account layout: 8-byte discriminator + Borsh-serialized RecognitionToken struct
        account_data = account_info.value.data

        # Parse account data based on encoding
        if isinstance(account_data, list) and len(account_data) > 0:
            if isinstance(account_data[0], str):
                # Base64 encoded
                account_bytes = base64.b64decode(account_data[0])
            else:
                # Already bytes
                account_bytes = bytes(account_data)
        elif isinstance(account_data, bytes):
            account_bytes = account_data
        else:
            logger.error(f"❌ Unknown account data format: {type(account_data)}")
            return jsonify({"error": "Failed to parse account data"}), 500

        # Skip the 8-byte Anchor discriminator
        data_offset = 8

        # Parse Borsh-encoded RecognitionToken:
        # - user: Pubkey (32 bytes)
        # - encrypted_embedding: Vec<u8> (4 bytes length + data)
        # - created_at: i64 (8 bytes)
        # - version: u8 (1 byte)
        # - bump: u8 (1 byte)
        # - display_name: Option<String>
        # - source: u8 (1 byte)

        # Skip user pubkey (32 bytes)
        data_offset += 32

        # Read Vec<u8> length (4 bytes, little-endian)
        vec_length = int.from_bytes(account_bytes[data_offset:data_offset+4], byteorder='little')
        data_offset += 4

        logger.info(f"📊 Vec<u8> length from Borsh: {vec_length} bytes")

        # Read the encrypted embedding bytes
        encrypted_data = account_bytes[data_offset:data_offset+vec_length]

        logger.info(f"✅ Found recognition token for {wallet_address}, embedding size: {len(encrypted_data)} bytes, base64 will be: {len(base64.b64encode(encrypted_data).decode('utf-8'))} chars")

        # Convert to base64 for JSON transport
        encrypted_embedding_b64 = base64.b64encode(encrypted_data).decode('utf-8')

        # Return token package format (matching what biometric service expects)
        token_package = {
            "encrypted_embedding": encrypted_embedding_b64,
            "wallet_address": wallet_address,
            "source": "on_chain",
            "pda": str(recognition_token_pda)
        }

        return jsonify({
            "success": True,
            "wallet_address": wallet_address,
            "token_package": token_package,
            "recognition_token_pda": str(recognition_token_pda)
        })

    except Exception as e:
        logger.error(f"❌ Error fetching recognition token: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/blockchain/check-user-status', methods=['POST'])
def check_user_status_api():
    """Check if a specific user is checked in to the camera"""
    try:
        data = request.json
        user_pubkey = data.get('user_pubkey')
        camera_pubkey = data.get('camera_pubkey', get_camera_pda())
        
        if not user_pubkey:
            return jsonify({"error": "User public key is required"}), 400
        
        if not camera_pubkey or camera_pubkey == 'YourCameraPDAHere':
            return jsonify({
                "error": "Camera PDA not configured. Please complete device registration first."
            }), 400
        
        is_checked_in = is_user_checked_in(user_pubkey, camera_pubkey)
        session_pda = derive_session_pda(user_pubkey, camera_pubkey, CAMERA_PROGRAM_ID)
        
        return jsonify({
            "success": True,
            "user_pubkey": user_pubkey,
            "camera_pubkey": camera_pubkey,
            "is_checked_in": is_checked_in,
            "session_pda": session_pda
        })
    
    except Exception as e:
        logger.error(f"Error checking user status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/blockchain/session-pda', methods=['POST'])
def derive_session_pda_api():
    """Derive the session PDA for a user and camera"""
    try:
        data = request.json
        user_pubkey = data.get('user_pubkey')
        camera_pubkey = data.get('camera_pubkey', get_camera_pda())
        program_id = data.get('program_id', CAMERA_PROGRAM_ID)
        
        if not user_pubkey:
            return jsonify({"error": "User public key is required"}), 400
        
        session_pda = derive_session_pda(user_pubkey, camera_pubkey, program_id)
        
        if not session_pda:
            return jsonify({"error": "Failed to derive session PDA"}), 500
        
        return jsonify({
            "success": True,
            "user_pubkey": user_pubkey,
            "camera_pubkey": camera_pubkey,
            "program_id": program_id,
            "session_pda": session_pda
        })
    
    except Exception as e:
        logger.error(f"Error deriving session PDA: {e}")
        return jsonify({"error": str(e)}), 500

def derive_session_pda(user_pubkey: str, camera_pubkey: str, program_id: str) -> str:
    """Derive the session PDA for a user and camera"""
    try:
        user_pk = Pubkey.from_string(user_pubkey)
        camera_pk = Pubkey.from_string(camera_pubkey)
        program_pk = Pubkey.from_string(program_id)
        
        # Derive PDA using seeds: ["session", user_pubkey, camera_pubkey]
        seeds = [
            b"session",
            bytes(user_pk),
            bytes(camera_pk)
        ]
        
        pda, bump = Pubkey.find_program_address(seeds, program_pk)
        return str(pda)
    except Exception as e:
        logger.error(f"Error deriving session PDA: {e}")
        return None

def get_checked_in_users(camera_pubkey: str) -> list:
    """Get all users currently checked in to a specific camera"""
    try:
        camera_pk = Pubkey.from_string(camera_pubkey)
        program_pk = Pubkey.from_string(CAMERA_PROGRAM_ID)
        
        logger.info(f"Searching for sessions for camera: {camera_pubkey}")
        logger.info(f"Using program ID: {CAMERA_PROGRAM_ID}")
        
        # For now, get all program accounts and filter in Python
        response = solana_client.get_program_accounts(
            program_pk,
            encoding="base64"
        )
        
        logger.info(f"Got {len(response.value) if response.value else 0} total program accounts")
        
        checked_in_users = []
        
        if response.value:
            for i, account_info in enumerate(response.value):
                try:
                    logger.info(f"Processing account {i+1}/{len(response.value)}: {account_info.pubkey}")
                    
                    # Debug the data format
                    logger.info(f"Account data type: {type(account_info.account.data)}")
                    
                    # Handle different data formats
                    if isinstance(account_info.account.data, bytes):
                        # Direct bytes format
                        account_data = account_info.account.data
                        logger.info(f"Using direct bytes format")
                    elif isinstance(account_info.account.data, list) and len(account_info.account.data) > 0:
                        if isinstance(account_info.account.data[0], str):
                            # Base64 string format
                            account_data = base64.b64decode(account_info.account.data[0])
                            logger.info(f"Using base64 string format")
                        elif isinstance(account_info.account.data[0], int):
                            # Already decoded bytes format
                            account_data = bytes(account_info.account.data)
                            logger.info(f"Using int list format")
                        else:
                            logger.warning(f"Unknown data format: {type(account_info.account.data[0])}")
                            continue
                    elif hasattr(account_info.account.data, 'data'):
                        # Some libraries use .data attribute
                        account_data = account_info.account.data.data
                        logger.info(f"Using .data attribute format")
                    else:
                        logger.warning(f"Cannot parse account data format: {type(account_info.account.data)}")
                        continue
                    
                    logger.info(f"Account data length: {len(account_data)} bytes")
                    
                    # Log first 100 bytes of account data for debugging
                    if len(account_data) >= 100:
                        first_100_hex = account_data[:100].hex()
                        logger.info(f"First 100 bytes (hex): {first_100_hex}")
                    
                    # Check if this account belongs to our camera
                    # The camera pubkey is at offset 40 (32 bytes)
                    if len(account_data) >= 72:  # Minimum size for a session account
                        camera_bytes = account_data[40:72]  # 32 bytes for camera pubkey
                        logger.info(f"Camera bytes from account (hex): {camera_bytes.hex()}")
                        logger.info(f"Expected camera bytes (hex): {bytes(camera_pk).hex()}")
                        
                        account_camera_pk = Pubkey(camera_bytes)
                        logger.info(f"Account camera pubkey: {account_camera_pk}")
                        logger.info(f"Target camera pubkey: {camera_pk}")
                        
                        # If this session belongs to our camera
                        if str(account_camera_pk) == str(camera_pk):
                            logger.info(f"✅ Found matching camera account!")
                            
                            # Extract user pubkey (at offset 8, 32 bytes)
                            user_bytes = account_data[8:40]
                            user_pubkey = Pubkey(user_bytes)
                            logger.info(f"User pubkey: {user_pubkey}")
                            
                            # Extract check-in time (at offset 72, 8 bytes)
                            checkin_time_bytes = account_data[72:80]
                            checkin_time = int.from_bytes(checkin_time_bytes, byteorder='little', signed=True)
                            logger.info(f"Check-in time: {checkin_time}")
                            
                            checked_in_users.append({
                                'user': str(user_pubkey),
                                'camera': str(account_camera_pk),
                                'checkin_time': checkin_time,
                                'account': str(account_info.pubkey)
                            })
                        else:
                            logger.info(f"❌ Camera mismatch - skipping account")
                    else:
                        logger.info(f"❌ Account too small ({len(account_data)} bytes) - skipping")
                            
                except Exception as e:
                    logger.error(f"Error processing account {account_info.pubkey}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        logger.info(f"Found {len(checked_in_users)} users checked in to camera {camera_pubkey}")
        return checked_in_users
        
    except Exception as e:
        logger.error(f"Error getting checked-in users: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def is_user_checked_in(user_pubkey: str, camera_pubkey: str) -> bool:
    """Check if a specific user is checked in to a specific camera"""
    try:
        # Derive the session PDA
        session_pda = derive_session_pda(user_pubkey, camera_pubkey, CAMERA_PROGRAM_ID)
        if not session_pda:
            return False

        # Try to fetch the account
        response = solana_client.get_account_info(Pubkey.from_string(session_pda))

        if response.value and response.value.data:
            # Account exists, decode to check if user is actually checked in
            account_data = base64.b64decode(response.value.data[0])

            # Extract check-in time (8 bytes starting at offset 72)
            checkin_time_bytes = account_data[72:80]
            checkin_time = int.from_bytes(checkin_time_bytes, byteorder='little', signed=True)

            # If checkin_time > 0, user is checked in
            return checkin_time > 0

        return False

    except Exception as e:
        logger.error(f"Error checking if user {user_pubkey} is checked in: {e}")
        return False


# ============================================================================
# Phase 3 Privacy Architecture: Write to Camera Timeline
# ============================================================================

# Device keypair for signing transactions
_device_keypair = None

def get_device_keypair() -> Keypair:
    """Load the device keypair for signing transactions."""
    global _device_keypair
    if _device_keypair is not None:
        return _device_keypair

    # Try production path first, then local path
    paths = [
        '/opt/mmoment/device/device-keypair.enc',
        os.path.expanduser('~/.mmoment/device-keypair.enc')
    ]

    for keypair_path in paths:
        if os.path.exists(keypair_path):
            try:
                # Get hardware key for decryption
                hardware_key = _get_hardware_key()
                cipher = Fernet(hardware_key)

                with open(keypair_path, 'rb') as f:
                    encrypted_data = f.read()

                decrypted_data = cipher.decrypt(encrypted_data)
                keypair_data = json.loads(decrypted_data.decode())

                _device_keypair = Keypair.from_bytes(bytes(keypair_data['private_key']))
                logger.info(f"Loaded device keypair: {_device_keypair.pubkey()}")
                return _device_keypair

            except Exception as e:
                logger.warning(f"Failed to load keypair from {keypair_path}: {e}")
                continue

    raise ValueError("Device keypair not found - device may not be registered")


def _get_hardware_key():
    """Generate hardware-bound encryption key (same as DeviceSigner)"""
    try:
        serial = None

        # First try shared machine-id (created by camera service at startup)
        # This ensures both services use the same key derivation
        shared_id_path = '/app/config/shared-machine-id'
        if os.path.exists(shared_id_path):
            serial = open(shared_id_path).read().strip()
            logger.info(f"Using shared machine-id for key derivation: {serial[:8]}...")

        # Fallback to standard locations
        if not serial:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    for line in cpu_info.split('\n'):
                        if 'Serial' in line:
                            serial = line.split(':')[1].strip()
                            break
            except:
                pass

        if not serial or serial == '0000000000000000':
            try:
                serial = open('/sys/class/net/eth0/address').read().strip()
            except:
                try:
                    serial = open('/etc/machine-id').read().strip()
                except:
                    pass

        if not serial:
            raise ValueError("No hardware identifier found")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'mmoment_depin_device_v1',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(serial.encode()))
        return key

    except Exception as e:
        logger.warning(f"Hardware key generation failed: {e}, using dev fallback")
        return base64.urlsafe_b64encode(b'mmoment_dev_key_32_bytes_long!!')


def derive_camera_timeline_pda(camera_pubkey: str) -> tuple:
    """Derive the CameraTimeline PDA for a camera."""
    try:
        camera_pk = Pubkey.from_string(camera_pubkey)
        program_pk = Pubkey.from_string(CAMERA_PROGRAM_ID)

        seeds = [b"camera-timeline", bytes(camera_pk)]
        pda, bump = Pubkey.find_program_address(seeds, program_pk)
        return str(pda), bump

    except Exception as e:
        logger.error(f"Error deriving camera timeline PDA: {e}")
        return None, None


# =============================================================================
# REMOVED: write-camera-timeline endpoint and helpers (Jan 2026)
#
# The code that was here INCORRECTLY submitted transactions directly to Solana
# from the Jetson device. This violates the CORRECT architecture:
#
# CORRECT PATTERN (applies to ALL on-chain submissions):
#   1. Jetson packages data + partial-signs with device key
#   2. Jetson POSTs to backend
#   3. Backend adds payer signature + submits to Solana
#   4. Backend pays ALL transaction fees
#
# The Jetson NEVER submits directly to Solana. Ever.
#
# If you need to implement timeline writing, follow the pattern in:
#   - competition_settlement.py (camera-service)
#   - checkout_service.py (camera-service)
# =============================================================================


if __name__ == "__main__":
    # Ensure the middleware has the required Python packages
    try:
        from cryptography.fernet import Fernet
        import requests
    except ImportError:
        logger.error("Required packages not found. Installing...")
        os.system("pip install cryptography requests")
        
    logger.info(f"Starting Solana middleware service on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001) 