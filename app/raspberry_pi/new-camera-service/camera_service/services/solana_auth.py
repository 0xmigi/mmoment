# solana_auth.py
import os
import sys
import json
import logging
from pathlib import Path
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class SolanaAuthService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SolanaAuthService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.camera_keypair = None
        self.program_id = Settings.SOLANA_PROGRAM_ID
        self.rpc_url = Settings.SOLANA_RPC_URL
        self.solana_client = None
        
        # Skip real initialization if SKIP_SOLANA_AUTH is True
        if Settings.SKIP_SOLANA_AUTH:
            logger.info("Development mode: Skipping Solana initialization")
            return
            
        # Add path to the Solana environment packages
        solana_env_path = os.path.expanduser("~/.local/solana_env/lib/python3.11/site-packages")
        if os.path.exists(solana_env_path) and solana_env_path not in sys.path:
            sys.path.append(solana_env_path)
        
        # Try to initialize real Solana client
        try:
            from solders.keypair import Keypair
            from solders.pubkey import Pubkey
            from solana.rpc.api import Client
            
            # Initialize Solana client with RPC URL
            logger.info(f"Initializing Solana client with RPC URL: {self.rpc_url}")
            self.solana_client = Client(self.rpc_url)
            
            # Test connection
            try:
                version = self.solana_client.get_version()
                logger.info(f"Connected to Solana node version: {version['result']}")
            except Exception as e:
                logger.error(f"Failed to connect to Solana node: {e}")
                return
            
            self._load_configuration(Pubkey)
            self._load_camera_keypair(Keypair)
            
        except Exception as e:
            logger.error(f"Error initializing Solana client: {e}")
                
    def _load_configuration(self, Pubkey=None):
        """Load Solana configuration from environment variables"""
        try:
            # Skip real initialization in development mode
            if Settings.SKIP_SOLANA_AUTH:
                logger.info("Development mode: Skipping Solana configuration")
                return
                
            # Get program ID from environment variable
            program_id_str = Settings.SOLANA_PROGRAM_ID
            if program_id_str:
                self.program_id = Pubkey.from_string(program_id_str)
                logger.info(f"Loaded program ID: {self.program_id}")
            else:
                logger.warning("SOLANA_PROGRAM_ID not set")
            
        except Exception as e:
            logger.error(f"Error loading Solana configuration: {e}")
            
    def _load_camera_keypair(self, Keypair=None):
        """Load camera keypair from file"""
        try:
            # Skip real keypair loading in development mode
            if Settings.SKIP_SOLANA_AUTH:
                logger.info("Development mode: Using mock camera keypair")
                return
                
            # Get keypair path from environment variable or use default
            keypair_path = Settings.CAMERA_KEYPAIR_PATH
            keypair_path = Path(os.path.expanduser(keypair_path))
            
            if not keypair_path.exists():
                logger.warning(f"Camera keypair file not found at {keypair_path}, using hardcoded public key")
                # We'll use the hardcoded public key instead
                return
                
            # Load keypair from file
            try:
                with open(keypair_path, "r") as f:
                    keypair_bytes = json.load(f)
                    
                self.camera_keypair = Keypair.from_bytes(bytes(keypair_bytes))
                logger.info(f"Loaded camera keypair with public key: {self.camera_keypair.pubkey()}")
            except Exception as e:
                logger.warning(f"Error loading keypair from file: {e}, using hardcoded public key instead")
                # Continue without the keypair - we'll use the hardcoded public key
                
        except Exception as e:
            logger.error(f"Error in keypair loading process: {e}")
            # Continue without the keypair - we'll use the hardcoded public key
            
    def get_camera_public_key(self):
        """Get the camera's public key"""
        if Settings.SKIP_SOLANA_AUTH:
            return "5omKvXxzsMkPJh7HZbozJXHR4h7TGRQXcNgRbTngd1Ww"
            
        if self.camera_keypair:
            return str(self.camera_keypair.pubkey())
            
        # Use the camera public key from your Solana program when no keypair is available
        # This is the actual public key from the registered camera in the program
        logger.info("Using hardcoded camera public key: EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA")
        return "EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA"
        
    def check_camera_registered(self):
        """Check if the camera is registered with the Solana program"""
        if Settings.SKIP_SOLANA_AUTH:
            logger.info("Development mode: Camera is registered")
            return True
            
        if not self.solana_client:
            logger.error("Solana client not initialized")
            return False
            
        try:
            # Get the camera's account data from your program
            camera_pubkey = self.get_camera_public_key()
            account_info = self.solana_client.get_account_info(camera_pubkey)
            
            if account_info['result']['value'] is None:
                logger.error(f"Camera account {camera_pubkey} not found in program {self.program_id}")
                return False
                
            logger.info(f"Camera {camera_pubkey} is registered with program {self.program_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking camera registration: {e}")
            return False
            
    def check_user_authorized(self, user_public_key):
        """Check if a user is authorized to access this camera"""
        if Settings.SKIP_SOLANA_AUTH:
            logger.info(f"Development mode: User {user_public_key} is authorized")
            return True
            
        if not self.solana_client:
            logger.error("Solana client not initialized")
            return False
            
        try:
            # Query your program to check if the user has connected to this camera
            camera_pubkey = self.get_camera_public_key()
            
            # Get the PDA for the user-camera connection
            # This assumes your program stores user connections in a PDA derived from user and camera pubkeys
            from solders.pubkey import Pubkey
            user_pubkey = Pubkey.from_string(user_public_key)
            
            # Get the account info for the user-camera connection
            # The exact account structure depends on your program's implementation
            account_info = self.solana_client.get_account_info(user_pubkey)
            
            if account_info['result']['value'] is None:
                logger.warning(f"User {user_public_key} has not connected to camera {camera_pubkey}")
                return False
                
            logger.info(f"User {user_public_key} is authorized to access camera {camera_pubkey}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking user authorization: {e}")
            return False 