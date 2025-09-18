# camera_service/config/settings.py
import logging
import os
import stat

# Remove the asyncio logger import and set up proper logging
logger = logging.getLogger(__name__)

# Base paths - Use the camera_files directory
BASE_DIR = os.path.expanduser("~/camera_files")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# Create videos directory if it doesn't exist
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Ensure videos directory has correct permissions
try:
    os.chmod(VIDEOS_DIR, 0o755)
except Exception as e:
    print(f"Warning: Could not set permissions on videos directory: {e}")

# Environment variables
SKIP_SOLANA_AUTH = os.getenv("SKIP_SOLANA_AUTH", "false").lower() == "true"
LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")
LIVEPEER_STREAM_KEY = os.getenv("LIVEPEER_STREAM_KEY")
LIVEPEER_INGEST_URL = os.getenv("LIVEPEER_INGEST_URL")

# Print configuration for debugging
print(f"Videos directory: {VIDEOS_DIR}")
print(f"Skip Solana Auth: {SKIP_SOLANA_AUTH}")
print(f"Livepeer configured: {bool(LIVEPEER_API_KEY and LIVEPEER_STREAM_KEY and LIVEPEER_INGEST_URL)}")

class Settings:
    CAMERA_API_PORT = 5001
    CAMERA_API_HOST = "0.0.0.0"
    BASE_DIR = BASE_DIR
    VIDEOS_DIR = VIDEOS_DIR
    LIVEPEER_API_KEY = LIVEPEER_API_KEY
    LIVEPEER_STREAM_KEY = LIVEPEER_STREAM_KEY
    LIVEPEER_INGEST_URL = LIVEPEER_INGEST_URL
    
    # Solana configuration
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    SOLANA_PROGRAM_ID = os.getenv("SOLANA_PROGRAM_ID", "7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4")
    CAMERA_KEYPAIR_PATH = os.getenv("CAMERA_KEYPAIR_PATH", "~/.camera_enclave/camera_keypair.json")
    SKIP_SOLANA_AUTH = SKIP_SOLANA_AUTH

    @classmethod
    def setup(cls):
        """Ensure required directories exist with proper permissions"""
        # Create directories if they don't exist
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(cls.VIDEOS_DIR, exist_ok=True)
        
        # Create camera keypair directory if it doesn't exist
        keypair_dir = os.path.dirname(os.path.expanduser(cls.CAMERA_KEYPAIR_PATH))
        os.makedirs(keypair_dir, exist_ok=True)

        # Set permissions (755 = rwxr-xr-x)
        os.chmod(cls.BASE_DIR, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        os.chmod(cls.VIDEOS_DIR, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        # Debug log the directories and permissions
        logger.debug("=== Directory Setup ===")
        logger.debug(f"BASE_DIR: {cls.BASE_DIR}")
        logger.debug(f"BASE_DIR exists: {os.path.exists(cls.BASE_DIR)}")
        logger.debug(f"BASE_DIR permissions: {oct(os.stat(cls.BASE_DIR).st_mode)}")
        logger.debug(f"VIDEOS_DIR: {cls.VIDEOS_DIR}")
        logger.debug(f"VIDEOS_DIR exists: {os.path.exists(cls.VIDEOS_DIR)}")
        logger.debug(f"VIDEOS_DIR permissions: {oct(os.stat(cls.VIDEOS_DIR).st_mode)}")

        # Debug log the environment variables
        logger.debug("=== Environment Variables ===")
        logger.debug(f"LIVEPEER_API_KEY: {'Set' if cls.LIVEPEER_API_KEY else 'Not Set'}")
        logger.debug(f"LIVEPEER_STREAM_KEY: {'Set' if cls.LIVEPEER_STREAM_KEY else 'Not Set'}")
        logger.debug(f"LIVEPEER_INGEST_URL: {'Set' if cls.LIVEPEER_INGEST_URL else 'Not Set'}")
        
        # Debug log Solana configuration
        logger.debug("=== Solana Configuration ===")
        logger.debug(f"SOLANA_RPC_URL: {cls.SOLANA_RPC_URL}")
        if cls.SOLANA_PROGRAM_ID:
            logger.debug(f"SOLANA_PROGRAM_ID: {cls.SOLANA_PROGRAM_ID}")
        else:
            logger.debug("SOLANA_PROGRAM_ID: Not Set")
        logger.debug(f"CAMERA_KEYPAIR_PATH: {cls.CAMERA_KEYPAIR_PATH}")
        logger.debug(f"SKIP_SOLANA_AUTH: {cls.SKIP_SOLANA_AUTH}")
        
        # Update settings from environment variables that were loaded
        cls.SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", cls.SOLANA_RPC_URL)
        cls.SOLANA_PROGRAM_ID = os.getenv("SOLANA_PROGRAM_ID", cls.SOLANA_PROGRAM_ID)
        cls.CAMERA_KEYPAIR_PATH = os.getenv("CAMERA_KEYPAIR_PATH", cls.CAMERA_KEYPAIR_PATH)
        cls.SKIP_SOLANA_AUTH = os.getenv("SKIP_SOLANA_AUTH", "true" if cls.SKIP_SOLANA_AUTH else "false").lower() == "true"
        
    @classmethod
    def load_env_file(cls, env_file_path):
        """Load environment variables from a .env file"""
        if not os.path.exists(env_file_path):
            logger.warning(f"Environment file not found: {env_file_path}")
            return
            
        logger.info(f"Loading environment from: {env_file_path}")
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    os.environ[key] = value
                    logger.debug(f"Set environment variable: {key}={value}")

    @classmethod
    def _parse_bool(cls, value):
        """Parse a string as boolean"""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', 'yes', '1', 'y')
        
    @classmethod
    def _load_solana_settings(cls):
        """Load Solana-specific settings"""
        cls.SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
        
        # Program ID from environment variable
        cls.SOLANA_PROGRAM_ID = os.getenv("SOLANA_PROGRAM_ID", "7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4")
        
        # Camera keypair path
        cls.CAMERA_KEYPAIR_PATH = os.getenv("CAMERA_KEYPAIR_PATH", "~/.camera_enclave/camera_keypair.json")
        
        # Skip authentication flag - IMPORTANT: This should be false for production
        cls.SKIP_SOLANA_AUTH = cls._parse_bool(os.getenv("SKIP_SOLANA_AUTH", "false"))
        
        logger.debug("=== Solana Configuration ===")
        logger.debug(f"SOLANA_RPC_URL: {cls.SOLANA_RPC_URL}")
        if cls.SOLANA_PROGRAM_ID:
            logger.debug(f"SOLANA_PROGRAM_ID: {cls.SOLANA_PROGRAM_ID}")
        else:
            logger.debug("SOLANA_PROGRAM_ID: Not Set")
        logger.debug(f"CAMERA_KEYPAIR_PATH: {cls.CAMERA_KEYPAIR_PATH}")
        logger.debug(f"SKIP_SOLANA_AUTH: {cls.SKIP_SOLANA_AUTH}")
        
    @classmethod
    def initialize(cls):
        """Initialize the settings from environment variables"""
        # Load base configuration
        cls._load_base_directory()
        cls._load_environment()
        cls._setup_videos_directory()
        
        # Load service-specific configurations
        cls._load_server_settings()
        cls._load_system_settings()
        cls._load_camera_settings()
        cls._load_stream_settings()
        cls._load_solana_settings()