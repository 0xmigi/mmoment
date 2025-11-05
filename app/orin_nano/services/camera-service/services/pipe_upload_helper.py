"""
Jetson Pipe Upload Helper

This module handles direct uploads from Jetson to Pipe Network,
then notifies the backend for tracking. Designed for large video files.

Usage:
    from pipe_upload_helper import PipeUploader

    uploader = PipeUploader(backend_url="http://your-backend:3001")
    result = uploader.upload_file("video.mp4", metadata={"cameraId": "jetson_01"})
"""

import requests
import hashlib
import os
from typing import Dict, Optional, Any
from pathlib import Path


class PipeUploader:
    def __init__(self, backend_url: str = "http://localhost:3001"):
        """
        Initialize Pipe uploader with backend URL.

        Args:
            backend_url: URL of your MMOMENT backend server
        """
        self.backend_url = backend_url.rstrip('/')
        self.credentials = None
        self.base_url = None

    def _get_credentials(self) -> Dict[str, str]:
        """Fetch Pipe credentials from backend."""
        if self.credentials:
            return self.credentials

        print("🔑 Fetching Pipe credentials from backend...")
        response = requests.get(f"{self.backend_url}/api/pipe/jetson/credentials")
        response.raise_for_status()

        data = response.json()
        self.credentials = {
            'user_id': data['userId'],
            'user_app_key': data['userAppKey']
        }
        self.base_url = data['baseUrl']

        print(f"✅ Got credentials for user: {self.credentials['user_id'][:20]}...")
        return self.credentials

    def _calculate_blake3(self, file_path: str) -> Optional[str]:
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
            print("⚠️  blake3 not installed, using SHA256 instead")
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"❌ Hash calculation failed: {e}")
            return None

    def upload_file(
        self,
        file_path: str,
        tx_signature: str,
        camera_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload file directly to Pipe Network, then notify backend.

        Args:
            file_path: Path to file to upload
            tx_signature: On-chain transaction signature (for ownership)
            camera_id: Camera ID that captured this
            metadata: Optional metadata dict

        Returns:
            Dict with success status and file info
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get credentials
        creds = self._get_credentials()

        # Calculate hash for content addressing
        print(f"📊 Calculating Blake3 hash for {file_path.name}...")
        blake3_hash = self._calculate_blake3(str(file_path))

        # Get file size
        file_size = file_path.stat().st_size
        print(f"📤 Uploading {file_path.name} ({file_size / 1024 / 1024:.2f} MB)...")

        # Upload to Pipe Network with priority endpoint
        upload_url = f"{self.base_url}/priorityUpload"

        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}

            # Build params and headers based on auth type
            params = {'file_name': file_path.name}
            headers = {}

            # Support both user_app_key (new accounts) and JWT (legacy accounts)
            if creds.get('user_app_key'):
                # New account with user_app_key
                headers['X-User-Id'] = creds['user_id']
                headers['X-User-App-Key'] = creds['user_app_key']
            elif creds.get('access_token'):
                # Legacy account with JWT
                headers['Authorization'] = f"Bearer {creds['access_token']}"
            else:
                raise ValueError("Missing authentication: need either user_app_key or access_token")

            # Upload with timeout suitable for large files
            response = requests.post(
                upload_url,
                files=files,
                params=params,
                headers=headers,
                timeout=600  # 10 minutes for large videos
            )
            response.raise_for_status()

            # Pipe returns filename as response
            stored_filename = response.text.strip()
            print(f"✅ Uploaded to Pipe: {stored_filename}")

        # Determine file type
        file_type = 'video' if file_path.suffix.lower() in ['.mp4', '.mov', '.avi'] else 'photo'

        # Notify backend for tracking (with tx signature for ownership)
        print(f"📝 Notifying backend with tx signature: {tx_signature[:16]}...")
        notify_data = {
            'txSignature': tx_signature,
            'fileName': stored_filename,
            'fileId': blake3_hash or stored_filename,
            'blake3Hash': blake3_hash,
            'size': file_size,
            'cameraId': camera_id,
            'fileType': file_type,
            'metadata': metadata or {}
        }

        notify_response = requests.post(
            f"{self.backend_url}/api/pipe/jetson/upload-complete",
            json=notify_data
        )
        notify_response.raise_for_status()

        print(f"✅ Upload complete and tracked!")

        return {
            'success': True,
            'fileName': stored_filename,
            'fileId': blake3_hash or stored_filename,
            'size': file_size,
            'blake3Hash': blake3_hash
        }

    def upload_directory(
        self,
        directory: str,
        camera_id: str,
        pattern: str = "*.mp4",
        tx_signature_fn: Optional[callable] = None,
        metadata_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Upload all files matching pattern in directory.

        Args:
            directory: Directory to scan
            camera_id: Camera ID for all uploads
            pattern: Glob pattern for files (default: *.mp4)
            tx_signature_fn: Optional function that takes filepath and returns tx signature
            metadata_fn: Optional function that takes filepath and returns metadata dict

        Returns:
            Dict with upload results
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        print(f"📁 Found {len(files)} files matching '{pattern}' in {directory}")

        results = {
            'success': [],
            'failed': []
        }

        for file_path in files:
            try:
                # Get tx signature (required - no fallback)
                if not tx_signature_fn:
                    raise ValueError("tx_signature_fn is required for batch uploads")
                tx_sig = tx_signature_fn(file_path)
                metadata = metadata_fn(file_path) if metadata_fn else {}
                result = self.upload_file(str(file_path), tx_sig, camera_id, metadata)
                results['success'].append(result)
            except Exception as e:
                print(f"❌ Failed to upload {file_path.name}: {e}")
                results['failed'].append({
                    'fileName': file_path.name,
                    'error': str(e)
                })

        print(f"\n📊 Upload summary:")
        print(f"   ✅ Success: {len(results['success'])}")
        print(f"   ❌ Failed: {len(results['failed'])}")

        return results


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python pipe_upload_helper.py <file_path> <tx_signature> <camera_id>")
        print("Example: python pipe_upload_helper.py video.mp4 5xK2j... jetson_01")
        sys.exit(1)

    file_path = sys.argv[1]
    tx_signature = sys.argv[2]
    camera_id = sys.argv[3]

    uploader = PipeUploader()

    # Upload single file with real transaction signature
    result = uploader.upload_file(
        file_path,
        tx_signature=tx_signature,
        camera_id=camera_id,
        metadata={"source": "command_line"}
    )
    print(f"\n🎉 Result: {result}")
