#!/usr/bin/env python3
"""
mmoment Biometric Security Service

Handles:
- Facial embedding encryption for NFT storage
- Session-based encryption key management  
- Secure purging of biometric data
- NFT package creation for blockchain minting

This service operates independently and communicates with:
- Camera Service (for embedding data)
- Solana Middleware (for NFT operations)
"""

import os
import json
import time
import logging
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our services
from services.encryption_service import BiometricEncryptionService
from services.session_manager import SessionManager
from services.secure_storage import SecureStorage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/biometric_security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BiometricSecurity")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
encryption_service = BiometricEncryptionService()
session_manager = SessionManager()
secure_storage = SecureStorage()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'service': 'biometric-security',
            'version': '1.0.0',
            'timestamp': int(time.time() * 1000),
            'active_sessions': len(session_manager.get_active_sessions()),
            'storage_status': secure_storage.get_status()
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Encryption endpoints
@app.route('/api/biometric/encrypt-embedding', methods=['POST'])
def encrypt_embedding():
    """
    Encrypt a facial embedding for NFT storage
    
    Expected payload:
    {
        "embedding": [array of floats],
        "wallet_address": "string",
        "session_id": "string",
        "metadata": {} (optional)
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['embedding', 'wallet_address', 'session_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate session
        if not session_manager.validate_session(data['session_id'], data['wallet_address']):
            return jsonify({'error': 'Invalid session'}), 403
        
        # Encrypt the embedding
        result = encryption_service.encrypt_embedding(
            embedding=data['embedding'],
            wallet_address=data['wallet_address'],
            session_id=data['session_id'],
            metadata=data.get('metadata', {})
        )
        
        # Store for NFT creation
        session_manager.store_encrypted_embedding(
            session_id=data['session_id'],
            wallet_address=data['wallet_address'],
            encrypted_data=result
        )
        
        logger.info(f"Encrypted embedding for wallet: {data['wallet_address']}")

        return jsonify({
            'success': True,
            'token_package': result,
            'wallet_address': data['wallet_address']
        }), 200
        
    except Exception as e:
        logger.error(f"Error encrypting embedding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/decrypt-for-session', methods=['POST'])
def decrypt_for_session():
    """
    Decrypt facial embedding for active session recognition
    
    Expected payload:
    {
        "nft_package": {},
        "wallet_address": "string", 
        "session_id": "string"
    }
    """
    try:
        data = request.json
        
        # Validate required fields - accept both old 'nft_package' and new 'token_package' for compatibility
        token_package = data.get('token_package') or data.get('nft_package')
        if not token_package:
            return jsonify({'error': 'Missing required field: token_package'}), 400

        if 'wallet_address' not in data or 'session_id' not in data:
            return jsonify({'error': 'Missing required fields: wallet_address, session_id'}), 400

        # Validate session
        if not session_manager.validate_session(data['session_id'], data['wallet_address']):
            return jsonify({'error': 'Invalid session'}), 403

        # Decrypt the embedding
        decrypted_embedding = encryption_service.decrypt_embedding(
            token_package=token_package,
            wallet_address=data['wallet_address'],
            session_id=data['session_id']
        )
        
        logger.info(f"Decrypted embedding for session: {data['session_id']}")
        
        return jsonify({
            'success': True,
            'embedding': decrypted_embedding.tolist(),  # Convert numpy array to list
            'wallet_address': data['wallet_address']
        }), 200
        
    except Exception as e:
        logger.error(f"Error decrypting embedding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/get-nft-package', methods=['POST'])
def get_nft_package():
    """
    Get NFT package for a session
    
    Expected payload:
    {
        "wallet_address": "string",
        "session_id": "string"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['wallet_address', 'session_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get NFT package from session
        nft_package = session_manager.get_nft_package(
            session_id=data['session_id'],
            wallet_address=data['wallet_address']
        )
        
        if not nft_package:
            return jsonify({'error': 'No NFT package found for this session'}), 404
        
        return jsonify({
            'success': True,
            'nft_package': nft_package,
            'wallet_address': data['wallet_address']
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting NFT package: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/create-session', methods=['POST'])
def create_session():
    """
    Create a new biometric session
    
    Expected payload:
    {
        "wallet_address": "string",
        "session_duration": 7200 (optional, default 2 hours)
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'wallet_address' not in data:
            return jsonify({'error': 'Missing wallet_address'}), 400
        
        # Create session
        session_info = session_manager.create_session(
            wallet_address=data['wallet_address'],
            duration=data.get('session_duration', 7200)  # Default 2 hours
        )
        
        logger.info(f"Created biometric session for wallet: {data['wallet_address']}")
        
        return jsonify({
            'success': True,
            'session_id': session_info['session_id'],
            'wallet_address': data['wallet_address'],
            'expires_at': session_info['expires_at']
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/purge-session', methods=['POST'])
def purge_session():
    """
    Securely purge all data for a session
    
    Expected payload:
    {
        "session_id": "string"
    }
    """
    try:
        data = request.json
        
        if 'session_id' not in data:
            return jsonify({'error': 'Missing session_id'}), 400
        
        # Perform secure purge
        success = session_manager.purge_session(data['session_id'])
        
        if success:
            logger.info(f"Successfully purged session: {data['session_id']}")
            return jsonify({
                'success': True,
                'message': 'Session data securely purged'
            }), 200
        else:
            return jsonify({'error': 'Failed to purge session'}), 500
        
    except Exception as e:
        logger.error(f"Error purging session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric/status', methods=['GET'])
def service_status():
    """Get detailed service status"""
    try:
        status = {
            'service': 'biometric-security',
            'version': '1.0.0',
            'encryption_service': encryption_service.get_status(),
            'session_manager': session_manager.get_status(),
            'secure_storage': secure_storage.get_status(),
            'timestamp': int(time.time() * 1000)
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting mmoment Biometric Security Service")
    logger.info(f"Service running on port 5003")
    
    # Create required directories
    os.makedirs('/app/logs', exist_ok=True)
    os.makedirs('/app/secure_data', exist_ok=True)
    os.makedirs('/app/temp_embeddings', exist_ok=True)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5003, debug=False) 