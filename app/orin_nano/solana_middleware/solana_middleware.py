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

if __name__ == "__main__":
    # Ensure the middleware has the required Python packages
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.error("Cryptography package not found. Installing...")
        os.system("pip install cryptography")
        
    logger.info(f"Starting Solana middleware service on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001) 