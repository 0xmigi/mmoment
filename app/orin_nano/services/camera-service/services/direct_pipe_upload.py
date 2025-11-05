"""
Direct Pipe Upload Service for Camera Captures

Integrates pipe_upload_helper.py with camera service for direct Jetson → Pipe uploads.
Uses device-signed capture events instead of blockchain transactions for instant UX.
"""

import sys
import logging
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import pipe_upload_helper
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from pipe_upload_helper import PipeUploader

logger = logging.getLogger(__name__)


class DirectPipeUploadService:
    """
    Service for uploading camera captures directly to Pipe Network.

    Flow:
    1. Camera captures video/photo
    2. Calculate file hash (Blake3)
    3. Sign capture event with device key (instant, no blockchain wait)
    4. Upload directly from Jetson to Pipe with device signature
    5. Backend tracks device_signature → file mapping for gallery retrieval
    """

    def __init__(self, backend_url: str = "http://192.168.1.232:3001"):
        """
        Initialize direct Pipe upload service.

        Args:
            backend_url: URL of MMOMENT backend server (on host network)
        """
        self.backend_url = backend_url
        self.uploader = PipeUploader(backend_url=backend_url)
        logger.info(f"✅ Direct Pipe upload service initialized (backend: {backend_url})")

    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate Blake3 hash of file for content addressing.
        Falls back to SHA256 if blake3 not available.

        Args:
            file_path: Path to file

        Returns:
            Hex string of hash, or None if failed
        """
        try:
            # Try blake3 first (pip install blake3)
            import blake3
            hasher = blake3.blake3()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except ImportError:
            # Fallback to SHA256
            logger.warning("⚠️  blake3 not installed, using SHA256 instead")
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"❌ Hash calculation failed: {e}")
            return None

    def upload_capture(
        self,
        wallet_address: str,
        file_path: str,
        camera_id: str,
        device_signature: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a camera capture directly to Pipe with device signature.

        Args:
            wallet_address: User's wallet address
            file_path: Path to the captured file (video or photo)
            camera_id: Camera ID (PDA)
            device_signature: Device-signed capture event signature
            metadata: Optional metadata dict

        Returns:
            Dict with upload result
        """
        try:
            logger.info(f"📤 Starting direct Pipe upload for {wallet_address[:8]}...")
            logger.info(f"   File: {file_path}")
            logger.info(f"   Camera: {camera_id}")
            logger.info(f"   Device Signature: {device_signature[:16]}...")

            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                return {
                    'success': False,
                    'error': 'File not found',
                    'provider': 'pipe'
                }

            # Upload directly to Pipe with device signature
            logger.info(f"📤 Uploading to Pipe Network...")

            result = self.uploader.upload_file(
                file_path=file_path,
                tx_signature=device_signature,  # Using device signature instead of blockchain tx
                camera_id=camera_id,
                metadata=metadata
            )

            if result['success']:
                logger.info(f"✅ Direct Pipe upload successful!")
                logger.info(f"   File ID: {result['fileId']}")
                logger.info(f"   File Name: {result['fileName']}")
                logger.info(f"   Blake3: {result.get('blake3Hash', 'N/A')[:16]}...")
                logger.info(f"   Size: {result['size']} bytes")

                return {
                    'success': True,
                    'provider': 'pipe',
                    'file_id': result['fileId'],
                    'file_name': result['fileName'],
                    'device_signature': device_signature,
                    'blake3_hash': result.get('blake3Hash'),
                    'size': result['size'],
                    'upload_type': 'direct_jetson_device_signed'
                }
            else:
                logger.error(f"❌ Direct Pipe upload failed: {result}")
                return {
                    'success': False,
                    'error': 'Upload to Pipe failed',
                    'provider': 'pipe',
                    'details': result
                }

        except Exception as e:
            logger.error(f"❌ Direct Pipe upload error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'provider': 'pipe'
            }

    def upload_video(
        self,
        wallet_address: str,
        video_path: str,
        camera_id: str,
        device_signature: str,
        duration: int = 0,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for uploading videos.

        Args:
            wallet_address: User's wallet address
            video_path: Path to the video file
            camera_id: Camera ID (PDA)
            device_signature: Device-signed capture event signature
            duration: Video duration in seconds
            timestamp: Optional timestamp string

        Returns:
            Upload result dict
        """
        metadata = {
            'type': 'video',
            'camera_id': camera_id,
            'duration': duration,
            'timestamp': timestamp or '',
            'capture_type': 'video',
            'user_wallet': wallet_address
        }

        return self.upload_capture(
            wallet_address=wallet_address,
            file_path=video_path,
            camera_id=camera_id,
            device_signature=device_signature,
            metadata=metadata
        )

    def upload_photo(
        self,
        wallet_address: str,
        photo_path: str,
        camera_id: str,
        device_signature: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for uploading photos.

        Args:
            wallet_address: User's wallet address
            photo_path: Path to the photo file
            camera_id: Camera ID (PDA)
            device_signature: Device-signed capture event signature
            timestamp: Optional timestamp string

        Returns:
            Upload result dict
        """
        metadata = {
            'type': 'photo',
            'camera_id': camera_id,
            'timestamp': timestamp or '',
            'capture_type': 'photo',
            'user_wallet': wallet_address
        }

        return self.upload_capture(
            wallet_address=wallet_address,
            file_path=photo_path,
            camera_id=camera_id,
            device_signature=device_signature,
            metadata=metadata
        )


# Singleton instance for camera service
_direct_pipe_upload_service = None


def get_direct_pipe_upload_service(backend_url: str = "http://192.168.1.232:3001") -> DirectPipeUploadService:
    """Get or create the singleton direct Pipe upload service instance."""
    global _direct_pipe_upload_service

    if _direct_pipe_upload_service is None:
        _direct_pipe_upload_service = DirectPipeUploadService(backend_url=backend_url)

    return _direct_pipe_upload_service


# For testing
if __name__ == "__main__":
    service = get_direct_pipe_upload_service()

    # Test with a dummy file
    test_wallet = "RsLjCiEiHq3dyWeDpp1M8jSmAhpmaGamcVK32sJkdLT"
    test_file = "/tmp/test_video.mp4"

    # Create a test file
    Path(test_file).write_bytes(b"fake video data for testing")

    # Create a test device signature
    test_signature = "device_sig_abc123_test"

    result = service.upload_video(
        wallet_address=test_wallet,
        video_path=test_file,
        camera_id="jetson_01",
        device_signature=test_signature,
        duration=30
    )

    print(f"Upload result: {result}")

    # Cleanup
    Path(test_file).unlink()
