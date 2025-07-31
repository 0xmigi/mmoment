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

# Solana imports for blockchain reading
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solana.rpc.types import MemcmpOpts
import base58

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
    """Verify that a wallet owns a valid NFT for facial recognition"""
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

@app.route('/api/blockchain/mint-facial-nft', methods=['POST'])
def mint_facial_nft():
    """Process encrypted facial embedding and prepare transaction for frontend signing"""
    try:
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
        
        # Create the transaction buffer in the format expected by the frontend wallet signing
        # The frontend expects: {"args": {"embedding": "<EMBEDDING_DATA>"}}
        transaction_data = {
            "args": {
                "embedding": nft_package.get("encrypted_embedding", "")  # This is the Base64 encrypted embedding
            }
        }
        
        # Convert to JSON string (not base64 encoded)
        transaction_buffer = json.dumps(transaction_data)
        
        logger.info(f"Prepared facial NFT transaction for wallet {wallet_address}")
        
        # Debug logging as suggested
        embedding_data = nft_package.get("encrypted_embedding", "")
        logger.info(f"[DEBUG] Sending transaction_buffer: {transaction_buffer[:200]}...")
        logger.info(f"[DEBUG] Embedding data type: {type(embedding_data)}")
        logger.info(f"[DEBUG] Embedding size: {len(embedding_data)} chars")
        logger.info(f"[DEBUG] Transaction buffer size: {len(transaction_buffer)} chars")
        
        return jsonify({
            "success": True,
            "transaction_buffer": transaction_buffer,
            "face_id": face_id,
            "message": "Transaction prepared for signing"
        })
        
    except Exception as e:
        logger.error(f"Error preparing facial NFT transaction: {e}")
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