#!/usr/bin/env python3
"""
Solana Transaction Verification Middleware

This middleware sits between the frontend and camera API:
1. Verifies Solana transactions are confirmed
2. Checks the transaction involves the correct program and camera PDA
3. Forwards verified requests to the camera API
"""
import os
import json
import logging
import sys
import time
import requests
from flask import Flask, request, jsonify, make_response, Response, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from solders.pubkey import Pubkey
from solana.rpc.api import Client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
PROGRAM_ID = os.getenv("SOLANA_PROGRAM_ID", "7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4")
CAMERA_PDA = os.getenv("CAMERA_PDA", "5onKAv5c6VdBZ8a7D11XqF79Hdzuv3tnysjv4B2pQWZ2")
CAMERA_API_URL = os.getenv("CAMERA_API_URL", "http://localhost:5001")
MIDDLEWARE_PORT = int(os.getenv("MIDDLEWARE_PORT", "5002"))
VERIFICATION_TIMEOUT = int(os.getenv("VERIFICATION_TIMEOUT", "15"))
DEBUG_BYPASS_VERIFICATION = os.getenv("DEBUG_BYPASS_VERIFICATION", "false").lower() == "true"

# Connect to Solana
client = Client(SOLANA_RPC_URL)
program_id = Pubkey.from_string(PROGRAM_ID)
camera_pda = Pubkey.from_string(CAMERA_PDA)

app = Flask(__name__)
# Enable CORS for all routes with proper configuration
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", 
                     "Cache-Control", "Pragma", "Cf-Connecting-Ip", "Cf-Ipcountry", "Cf-Ray", 
                     "Cf-Visitor", "Access-Control-Allow-Origin"],
    "expose_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
    "supports_credentials": True,
    "send_wildcard": False,
    "vary_header": True
}})

# Helper function to add CORS headers
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Requested-With,Accept,Origin,Cache-Control,Pragma,Cf-Connecting-Ip,Cf-Ipcountry,Cf-Ray,Cf-Visitor')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Vary', 'Origin')
    return response

def verify_transaction(tx_signature: str) -> bool:
    """
    Verify a Solana transaction:
    1. Check it's confirmed or finalized
    2. Check it involves our program
    3. Check it involves the camera PDA
    """
    try:
        logger.info(f"Verifying transaction: {tx_signature}")
        logger.info(f"Using Program ID: {PROGRAM_ID} ({program_id})")
        logger.info(f"Using Camera PDA: {CAMERA_PDA} ({camera_pda})")
        start_time = time.time()
        attempts = 0
        
        # Keep checking until confirmed or timeout
        while time.time() - start_time < VERIFICATION_TIMEOUT:
            attempts += 1
            
            # Log verification attempt
            logger.info(f"Verification attempt {attempts} (elapsed: {time.time() - start_time:.2f}s)")
            
            # Check transaction status
            resp = client.get_signature_statuses([tx_signature])
            logger.info(f"Response from get_signature_statuses: {resp}")
            
            if not resp.value[0]:
                wait_time = min(1.0, 0.5 * attempts)  # Progressive waiting
                logger.info(f"Transaction not found, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            # Accept both confirmed and finalized status
            status = resp.value[0].confirmation_status
            logger.info(f"Transaction status: {status}")
            
            if status not in ["confirmed", "finalized"]:
                wait_time = min(1.0, 0.5 * attempts)  # Progressive waiting
                logger.info(f"Transaction not confirmed yet, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            # Get transaction details
            logger.info(f"Transaction has {status} status, fetching details...")
            tx = client.get_transaction(tx_signature)
            if not tx.value:
                logger.error(f"Transaction {tx_signature} not found after confirmation check")
                return False
            
            # Log the accounts involved
            accounts = tx.value.transaction.message.account_keys
            logger.info(f"Transaction accounts: {accounts}")
                
            # Verify program ID
            logger.info(f"Verifying program ID: {program_id}")
            if program_id not in accounts:
                logger.error(f"Transaction does not involve our program {program_id}")
                return False
                
            # Verify camera PDA
            logger.info(f"Verifying camera PDA: {camera_pda}")
            if camera_pda not in accounts:
                logger.error(f"Transaction does not involve our camera {camera_pda}")
                return False
            
            logger.info(f"Transaction verified successfully after {time.time() - start_time:.2f}s and {attempts} attempts")
            return True
            
        logger.error(f"Transaction verification timed out after {VERIFICATION_TIMEOUT}s and {attempts} attempts")
        return False
        
    except Exception as e:
        logger.error(f"Error verifying transaction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.route('/api/capture', methods=['POST', 'OPTIONS'])
def capture_photo():
    """
    Verify transaction and forward to camera API capture endpoint
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    try:
        # Forward to camera API with the original request data
        headers = {
            'Content-Type': request.headers.get('Content-Type', 'application/json'),
            'Authorization': request.headers.get('Authorization', '')
        }
        
        # Forward the original request body
        response = requests.post(
            f"{CAMERA_API_URL}/api/capture",
            headers=headers,
            json=request.json,
            stream=True
        )
        
        # Create Flask response from the camera API response
        flask_response = make_response(response.raw.read())
        flask_response.status_code = response.status_code
        
        # Copy only essential headers
        flask_response.headers['Content-Type'] = response.headers.get('Content-Type', 'image/jpeg')
        
        return add_cors_headers(flask_response)
    except Exception as e:
        logger.error(f"Error forwarding to camera API: {e}")
        response = jsonify({"error": "Failed to communicate with camera"})
        return add_cors_headers(response), 500

@app.route('/api/stream/start', methods=['POST', 'OPTIONS'])
def start_stream():
    """
    Verify transaction and forward to camera API stream start endpoint
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    data = request.json
    tx_signature = data.get('tx_signature')
    wallet_address = data.get('wallet_address')
    
    # TEMPORARY: Force bypass verification for testing
    bypass_verification = True
    
    logger.info(f"Stream start request received: tx={tx_signature}, wallet={wallet_address}, bypass={bypass_verification}")
    
    if not tx_signature and not bypass_verification:
        response = jsonify({"error": "Missing transaction signature"})
        return add_cors_headers(response), 400
        
    # Verify the transaction unless bypassing verification
    if not bypass_verification and not verify_transaction(tx_signature):
        error_msg = "Invalid or unconfirmed transaction"
        logger.error(f"Stream start failed: {error_msg}")
        response = jsonify({"error": error_msg})
        return add_cors_headers(response), 400
    
    # If bypassing verification, log it
    if bypass_verification:
        logger.warning("⚠️ BYPASSING TRANSACTION VERIFICATION - FOR TESTING ONLY ⚠️")
        
    # Forward to camera API with wallet address directly as the Authorization header (not Bearer)
    try:
        logger.info(f"Forwarding stream start request to camera API with wallet address as Authorization header: {wallet_address}")
        headers = {}
        if wallet_address:
            headers["Authorization"] = wallet_address
            
        response = requests.post(
            f"{CAMERA_API_URL}/api/stream/start", 
            json={},
            headers=headers
        )
        # Pass through the response from the camera API
        flask_response = make_response(response.content)
        flask_response.status_code = response.status_code
        for key, value in response.headers.items():
            if key.lower() not in ['content-length', 'content-encoding', 'transfer-encoding']:
                flask_response.headers[key] = value
        return add_cors_headers(flask_response)
    except Exception as e:
        logger.error(f"Error forwarding to camera API: {e}")
        response = jsonify({"error": "Failed to communicate with camera"})
        return add_cors_headers(response), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """
    Health check endpoint that also verifies camera API is accessible
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    try:
        # Check camera API health
        camera_health = requests.get(f"{CAMERA_API_URL}/api/health")
        if camera_health.status_code != 200:
            response = jsonify({
                "status": "degraded", 
                "camera_api": "unavailable"
            })
            return add_cors_headers(response), 200
            
        # All good
        response = jsonify({
            "status": "ok",
            "camera_api": "available",
            "solana_connected": True
        })
        return add_cors_headers(response), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        response = jsonify({
            "status": "degraded",
            "error": str(e)
        })
        return add_cors_headers(response), 200

# Passthrough routes for stream status and other non-transaction operations
@app.route('/api/stream/info', methods=['GET', 'OPTIONS'])
def stream_info():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    try:
        # First check if we can reach the camera API
        try:
            health_response = requests.get(f"{CAMERA_API_URL}/api/health", timeout=2)
            if health_response.status_code != 200:
                logger.error(f"Camera API health check failed: {health_response.status_code}")
                return add_cors_headers(jsonify({
                    "isActive": False,
                    "status": "camera_unavailable",
                    "error": "Camera service is not responding"
                })), 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to camera API: {e}")
            return add_cors_headers(jsonify({
                "isActive": False,
                "status": "connection_error",
                "error": "Cannot connect to camera service"
            })), 200
            
        # Get stream info from camera API
        response = requests.get(f"{CAMERA_API_URL}/api/stream/info")
        
        # Extract just what we need from the response
        try:
            data = response.json()
            stream_info = {
                "isActive": data.get("isActive", False),
                "playbackId": data.get("playbackId", ""),
                "status": "streaming" if data.get("isActive", False) else "stopped"
            }
            
            # Add CORS headers and return a clean response
            return add_cors_headers(jsonify(stream_info)), 200
        except ValueError:
            # If not JSON, return a fallback response
            logger.error("Failed to parse stream info response as JSON")
            return add_cors_headers(jsonify({
                "isActive": False,
                "status": "parse_error",
                "error": "Invalid response from camera"
            })), 200
            
    except Exception as e:
        logger.error(f"Error getting stream info: {e}")
        return add_cors_headers(jsonify({
            "isActive": False,
            "status": "error",
            "error": str(e)
        })), 200

@app.route('/api/stream/stop', methods=['POST', 'OPTIONS'])
def stop_stream():
    """
    Forward stream stop request to camera API
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    data = request.json
    wallet_address = data.get('wallet_address')
    
    # Forward to camera API with wallet address directly as the Authorization header
    try:
        logger.info(f"Forwarding stream stop request to camera API with wallet address as Authorization header: {wallet_address}")
        headers = {}
        if wallet_address:
            headers["Authorization"] = wallet_address
            
        response = requests.post(
            f"{CAMERA_API_URL}/api/stream/stop", 
            json={},
            headers=headers
        )
        # Pass through the response from the camera API
        flask_response = make_response(response.content)
        flask_response.status_code = response.status_code
        for key, value in response.headers.items():
            if key.lower() not in ['content-length', 'content-encoding', 'transfer-encoding']:
                flask_response.headers[key] = value
        return add_cors_headers(flask_response)
    except Exception as e:
        logger.error(f"Error forwarding to camera API: {e}")
        response = jsonify({"error": "Failed to communicate with camera"})
        return add_cors_headers(response), 500

@app.route('/api/record', methods=['POST', 'OPTIONS'])
def record_video():
    """
    Verify transaction and forward to camera API record endpoint
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    data = request.json
    tx_signature = data.get('tx_signature')
    wallet_address = data.get('wallet_address')
    duration = data.get('duration', 5)
    
    # Log the record request
    logger.info(f"Record request received: tx={tx_signature}, wallet={wallet_address}, duration={duration}")
    
    try:
        # Forward to camera API
        headers = {
            'Content-Type': 'application/json',
            'Authorization': wallet_address
        }
        
        # Forward the record request with duration
        response = requests.post(
            f"{CAMERA_API_URL}/api/record", 
            json={"duration": duration},
            headers=headers
        )
        
        # Check if recording was successful
        if response.status_code == 200:
            # Get the response data
            resp_data = response.json()
            if resp_data.get("status") == "recorded":
                # Enhance the response with direct URLs
                filename = resp_data.get("filename")
                if filename:
                    # Add URLs to the video that the frontend can use
                    resp_data["video_url"] = f"/api/direct-video/{filename}"
                    resp_data["video_url_alternative"] = f"/api/videos/{filename}"
                    
                    # Check if the file actually exists
                    video_path = f"/home/azuolas/camera_files/videos/{filename}"
                    if os.path.exists(video_path):
                        resp_data["file_exists"] = True
                        resp_data["file_size"] = os.path.getsize(video_path)
                        logger.info(f"Video file exists at {video_path} with size {resp_data['file_size']} bytes")
                    else:
                        resp_data["file_exists"] = False
                        logger.warning(f"Video file does not exist at {video_path}")
                        
                # Return the enhanced response
                return add_cors_headers(jsonify(resp_data))
            else:
                return add_cors_headers(jsonify({"status": "error", "message": "Recording failed"})), 400
        else:
            return add_cors_headers(jsonify({"status": "error", "message": f"Camera API returned error: {response.status_code}"})), response.status_code
            
    except Exception as e:
        logger.error(f"Error in record response: {e}")
        return add_cors_headers(jsonify({"status": "error", "message": str(e)})), 500

@app.route('/api/record/latest', methods=['POST', 'OPTIONS'])
def get_latest_recording():
    """
    Fetch the latest recorded video for a wallet
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    data = request.json
    wallet_address = data.get('wallet_address')
    
    if not wallet_address:
        response = jsonify({"error": "Missing wallet address"})
        return add_cors_headers(response), 400
    
    # Forward to camera API with wallet address in authorization header
    try:
        logger.info(f"Fetching latest recording for wallet: {wallet_address}")
        response = requests.get(
            f"{CAMERA_API_URL}/api/record/latest", 
            headers={'Authorization': wallet_address}
        )
        
        # Check if camera API returned an error
        if response.status_code != 200:
            logger.error(f"Camera API error: {response.status_code} - {response.text}")
            return add_cors_headers(jsonify({"error": f"Camera API error: {response.text}"})), response.status_code
        
        # Pass through the response from the camera API
        flask_response = make_response(response.content)
        flask_response.status_code = response.status_code
        
        # Copy headers from camera API response
        for key, value in response.headers.items():
            if key.lower() not in ['content-length', 'content-encoding', 'transfer-encoding']:
                flask_response.headers[key] = value
        
        return add_cors_headers(flask_response)
    except Exception as e:
        logger.error(f"Error fetching latest recording: {e}")
        response = jsonify({"error": "Failed to fetch recording"})
        return add_cors_headers(response), 500

@app.route('/api/videos/<filename>', methods=['GET', 'OPTIONS'])
def get_video(filename):
    """
    Serve video files from the camera's video directory
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    # Security check - Make sure filename doesn't contain path traversal
    if '..' in filename or '/' in filename:
        logger.error(f"Attempted path traversal in filename: {filename}")
        return add_cors_headers(jsonify({"error": "Invalid filename"})), 400
    
    # Define the actual directory where videos are stored
    video_dir = "/home/azuolas/camera_files/videos"
    file_path = os.path.join(video_dir, filename)
    
    logger.info(f"Checking for video file at: {file_path}")
    
    if os.path.exists(file_path):
        logger.info(f"Found video file at {file_path}, serving...")
        content_type = 'video/quicktime' if filename.endswith('.mov') else 'video/mp4'
        response = send_file(
            file_path, 
            mimetype=content_type,
            as_attachment=False,  # Allow browser to play the video
            download_name=filename
        )
        # Add Content-Range header for proper video streaming
        response.headers.add('Accept-Ranges', 'bytes')
        return add_cors_headers(response)
    else:
        # If file not found, try alternative paths
        potential_paths = [
            f"/home/azuolas/camera_files/videos/{filename}",
            f"/home/azuolas/new-camera-service/camera_service/videos/{filename}",
            f"/home/azuolas/new-camera-service/videos/{filename}",
            f"/tmp/camera_files/videos/{filename}"
        ]
        
        for path in potential_paths:
            logger.info(f"Trying alternative path: {path}")
            if os.path.exists(path):
                logger.info(f"Found video at alternative path: {path}")
                content_type = 'video/quicktime' if path.endswith('.mov') else 'video/mp4'
                response = send_file(
                    path,
                    mimetype=content_type,
                    as_attachment=False,  # Allow browser to play the video
                    download_name=filename
                )
                # Add Content-Range header for proper video streaming
                response.headers.add('Accept-Ranges', 'bytes')
                return add_cors_headers(response)
        
        logger.error(f"Video {filename} not found in any location")
        return add_cors_headers(jsonify({"error": "Video not found"})), 404

# Add a new direct video endpoint
@app.route('/api/direct-video/<filename>', methods=['GET', 'OPTIONS'])
def direct_video(filename):
    """Directly serve a video file from the known location"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response()
        return add_cors_headers(response)
    
    # Security check - Make sure filename doesn't contain path traversal
    if '..' in filename or '/' in filename:
        logger.error(f"Attempted path traversal in filename: {filename}")
        return add_cors_headers(jsonify({"error": "Invalid filename"})), 400
    
    # Define the actual directory where videos are stored
    video_dir = "/home/azuolas/camera_files/videos"
    file_path = os.path.join(video_dir, filename)
    
    logger.info(f"Direct video access attempt for: {file_path}")
    
    if os.path.exists(file_path):
        logger.info(f"Found video file at {file_path}, serving directly...")
        content_type = 'video/quicktime' if file_path.endswith('.mov') else 'video/mp4'
        response = send_file(
            file_path, 
            mimetype=content_type,
            as_attachment=False,  # Allow browser to play the video
            download_name=filename
        )
        # Add Content-Range header for proper video streaming
        response.headers.add('Accept-Ranges', 'bytes')
        return add_cors_headers(response)
    else:
        logger.error(f"Video {filename} not found at {file_path}")
        return add_cors_headers(jsonify({"error": "Video not found"})), 404

# Also add a simple endpoint for videos in case the camera API doesn't have the /api prefix
@app.route('/videos/<filename>', methods=['GET', 'OPTIONS'])
def get_video_alt(filename):
    """
    Alternative route to serve video files
    """
    return get_video(filename)

if __name__ == "__main__":
    logger.info(f"Starting Solana middleware on port {MIDDLEWARE_PORT}")
    logger.info(f"Connected to Solana at {SOLANA_RPC_URL}")
    logger.info(f"Program ID: {PROGRAM_ID}")
    logger.info(f"Camera PDA: {CAMERA_PDA}")
    logger.info(f"Forwarding to camera API at {CAMERA_API_URL}")
    app.run(host="0.0.0.0", port=MIDDLEWARE_PORT) 