import os
import stat

BASE_DIR = os.path.expanduser("~/camera_files")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")


class Settings:
    PORT = int(os.getenv("CAMERA_PORT", "5002"))
    HOST = "0.0.0.0"

    BASE_DIR = BASE_DIR
    VIDEOS_DIR = VIDEOS_DIR

    # Backend
    BACKEND_URL = os.getenv("BACKEND_URL", "https://api.mmoment.xyz")

    # MediaMTX relay (Oracle VPS)
    MEDIAMTX_URL = os.getenv("MEDIAMTX_URL", "http://129.80.99.75:8889")

    # Device identity
    KEYPAIR_PATH = os.getenv("KEYPAIR_PATH", os.path.expanduser("~/.mmoment/device-keypair.enc"))

    # Solana
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    CAMERA_PROGRAM_ID = os.getenv("CAMERA_PROGRAM_ID", "E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL")

    # Auth
    SKIP_AUTH = os.getenv("SKIP_AUTH", "false").lower() == "true"

    @classmethod
    def setup(cls):
        os.makedirs(cls.VIDEOS_DIR, exist_ok=True)
        os.chmod(cls.BASE_DIR, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        os.chmod(cls.VIDEOS_DIR, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        keypair_dir = os.path.dirname(cls.KEYPAIR_PATH)
        os.makedirs(keypair_dir, exist_ok=True)
