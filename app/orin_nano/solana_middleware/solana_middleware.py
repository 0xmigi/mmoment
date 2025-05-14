#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import time
import logging
import os

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

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to confirm API is working"""
    return jsonify({
        "message": "Solana Middleware API is running",
        "status": "ok",
        "timestamp": int(time.time() * 1000)
    })

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

if __name__ == "__main__":
    logger.info(f"Starting Solana middleware service on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001) 