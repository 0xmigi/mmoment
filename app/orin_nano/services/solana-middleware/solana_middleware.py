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
    """Trigger secure purging of biometric data after session end"""
    try:
        data = request.json
        session_id = data.get('session_id')
        wallet_address = data.get('wallet_address')
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        # Trigger purge in biometric security service
        biometric_service_url = os.getenv('BIOMETRIC_SERVICE_URL', 'http://localhost:5003')
        
        try:
            response = requests.post(f'{biometric_service_url}/api/biometric/purge',
                                    json={'session_id': session_id})
            
            if response.status_code == 200:
                # Also clean up our session data
                if session_id in sessions:
                    del sessions[session_id]
                
                # Remove encryption keys
                if wallet_address and wallet_address in encryption_keys:
                    del encryption_keys[wallet_address]
                
                logger.info(f"Session {session_id} purged successfully")
                
                return jsonify({
                    "success": True,
                    "message": "Session data securely purged from all services"
                })
            else:
                return jsonify({"error": "Failed to purge biometric data"}), 500
                
        except requests.exceptions.RequestException:
            # If biometric service is unavailable, still clean up local data
            if session_id in sessions:
                del sessions[session_id]
            if wallet_address and wallet_address in encryption_keys:
                del encryption_keys[wallet_address]
            
            logger.warning(f"Biometric service unavailable, cleaned up local data for session {session_id}")
            return jsonify({
                "success": True,
                "message": "Local session data purged (biometric service unavailable)"
            })
            
    except Exception as e:
        logger.error(f"Error purging session: {e}")
        return jsonify({"error": str(e)}), 500

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