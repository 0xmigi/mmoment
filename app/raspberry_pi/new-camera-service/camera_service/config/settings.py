# camera_service/config/settings.py
import logging
import os
import stat

# Remove the asyncio logger import and set up proper logging
logger = logging.getLogger(__name__)

class Settings:
    CAMERA_API_PORT = 5001
    CAMERA_API_HOST = "0.0.0.0"
    BASE_DIR = os.path.expanduser("~/camera_files")
    VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
    LIVEPEER_API_KEY = os.getenv("LIVEPEER_API_KEY")
    LIVEPEER_STREAM_KEY = os.getenv("LIVEPEER_STREAM_KEY")
    LIVEPEER_INGEST_URL = os.getenv("LIVEPEER_INGEST_URL")

    @classmethod
    def setup(cls):
        """Ensure required directories exist with proper permissions"""
        # Create directories if they don't exist
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(cls.VIDEOS_DIR, exist_ok=True)

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