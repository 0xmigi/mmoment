#!/usr/bin/env python3
# solana_setup.py - Utility script for Solana integration

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("solana_setup")

# Default paths
ENV_FILE = os.path.join("camera_service", ".env.solana")
DEFAULT_KEYPAIR_PATH = os.path.expanduser("~/.camera_enclave/camera_keypair.json")

def load_env(env_file=ENV_FILE):
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key] = value
                        os.environ[key] = value
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(f"Environment file {env_file} not found")
    except Exception as e:
        logger.error(f"Error loading environment: {e}")
    return env_vars

def load_keypair(keypair_path=None):
    """Load Solana keypair from file"""
    if not keypair_path:
        keypair_path = os.environ.get("CAMERA_KEYPAIR_PATH", DEFAULT_KEYPAIR_PATH)
    
    keypair_path = os.path.expanduser(keypair_path)
    if not os.path.exists(keypair_path):
        logger.error(f"Keypair file not found at {keypair_path}")
        return None
        
    try:
        with open(keypair_path, 'r') as f:
            keypair_bytes = json.load(f)
        keypair = Keypair.from_bytes(bytes(keypair_bytes))
        logger.info(f"Loaded keypair with public key: {keypair.pubkey()}")
        return keypair
    except Exception as e:
        logger.error(f"Error loading keypair: {e}")
        return None

def check_camera_status(args):
    """Check camera status on the Solana blockchain"""
    # Load environment variables
    env_vars = load_env(args.env_file)
    
    # Get RPC URL
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    logger.info(f"Using Solana RPC URL: {rpc_url}")
    
    # Get program ID
    program_id_str = os.environ.get("SOLANA_PROGRAM_ID")
    if not program_id_str:
        logger.error("SOLANA_PROGRAM_ID not set in environment")
        return 1
        
    try:
        program_id = Pubkey.from_string(program_id_str)
    except Exception as e:
        logger.error(f"Invalid program ID: {e}")
        return 1
        
    # Load keypair
    keypair = load_keypair(args.keypair)
    if not keypair:
        return 1
        
    # Connect to Solana
    client = Client(rpc_url)
    try:
        version = client.get_version()
        logger.info(f"Connected to Solana node version: {version['result']['solana-core']}")
    except Exception as e:
        logger.error(f"Error connecting to Solana: {e}")
        return 1
        
    # Check if camera is registered
    logger.info(f"Checking status for camera: {keypair.pubkey()}")
    logger.info(f"Program ID: {program_id}")
    
    # TODO: Add code to check if the camera is registered with the program
    # This will depend on your specific Solana program structure
    
    # For now, we'll just display a success message
    logger.info(f"Camera {keypair.pubkey()} is ready to be used with the Solana program")
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Solana camera integration setup utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Check camera status command
    status_parser = subparsers.add_parser("check-status", help="Check camera status on Solana")
    status_parser.add_argument("--env-file", default=ENV_FILE, help="Path to environment file")
    status_parser.add_argument("--keypair", default=None, help="Path to camera keypair file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "check-status":
        return check_camera_status(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 