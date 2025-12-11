"""
Walrus Upload Service for Camera Captures

Uploads photos/videos to Walrus with pre-upload AES-256-GCM encryption.
Creates access grants so all checked-in users can decrypt content.
Optionally transfers blob ownership to user's Sui address.

mmoment pays storage costs, user owns the encrypted Blob on Sui.
"""

import os
import json
import base64
import logging
import hashlib
import requests
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# Walrus endpoints (mainnet)
WALRUS_PUBLISHER_URL = os.environ.get(
    "WALRUS_PUBLISHER_URL", "https://walrus-mainnet-publisher-1.staketab.org"
)
WALRUS_AGGREGATOR_URL = os.environ.get(
    "WALRUS_AGGREGATOR_URL", "https://aggregator.walrus-mainnet.walrus.space"
)

# Default storage duration (5 epochs for testing, increase for production)
DEFAULT_EPOCHS = 5


class WalrusUploadService:
    """
    Service for uploading camera captures to Walrus with encryption.

    Flow:
    1. Camera captures photo/video
    2. Generate random AES-256 key, encrypt content
    3. Create access grants for all checked-in users (sealed box encryption)
    4. Upload encrypted data to Walrus with send_object_to for user ownership
    5. Notify backend with blobId + access grants for gallery tracking
    """

    def __init__(self, backend_url: str = None):
        """
        Initialize Walrus upload service.

        Args:
            backend_url: URL of MMOMENT backend server
        """
        self.backend_url = backend_url or os.environ.get(
            "BACKEND_URL", "https://mmoment-production.up.railway.app"
        )
        # Check if backend relay is available
        self.relay_enabled = self._check_relay_status()
        logger.info(f"Walrus upload service initialized")
        logger.info(f"  Publisher: {WALRUS_PUBLISHER_URL}")
        logger.info(f"  Aggregator: {WALRUS_AGGREGATOR_URL}")
        logger.info(f"  Backend: {self.backend_url}")
        logger.info(f"  Relay enabled: {self.relay_enabled}")

    def _check_relay_status(self) -> bool:
        """
        Check if backend upload relay is available.
        """
        try:
            response = requests.get(
                f"{self.backend_url}/api/walrus/relay-status",
                timeout=5
            )
            if response.ok:
                data = response.json()
                return data.get("relayEnabled", False)
            return False
        except Exception as e:
            logger.debug(f"Relay status check failed: {e}")
            return False

    def _upload_via_relay(
        self,
        encrypted_data: bytes,
        wallet_address: str,
        camera_id: str,
        device_signature: str,
        file_type: str,
        timestamp: int,
        original_size: int,
        nonce: str,
        access_grants: List[Dict],
        sui_owner: Optional[str] = None,
        epochs: int = DEFAULT_EPOCHS,
    ) -> Dict[str, Any]:
        """
        Upload encrypted data via backend relay (fast path).

        The backend uses the TypeScript SDK with upload relay for ~10x faster uploads.
        Backend handles saving to DB and notifying websocket clients.

        Returns:
            Dict with blobId, downloadUrl, uploadDurationMs
        """
        import time
        start_time = time.time()

        try:
            headers = {
                "Content-Type": "application/octet-stream",
                "X-Wallet-Address": wallet_address,
                "X-Camera-Id": camera_id,
                "X-Device-Signature": device_signature,
                "X-File-Type": file_type,
                "X-Timestamp": str(timestamp or int(time.time() * 1000)),
                "X-Original-Size": str(original_size),
                "X-Nonce": nonce,
                "X-Access-Grants": json.dumps(access_grants),
                "X-Epochs": str(epochs),
            }

            if sui_owner:
                headers["X-Sui-Owner"] = sui_owner

            logger.info(f"Uploading {len(encrypted_data)} bytes via backend relay...")

            response = requests.post(
                f"{self.backend_url}/api/walrus/upload",
                data=encrypted_data,
                headers=headers,
                timeout=120  # 2 minutes for relay upload
            )

            duration_ms = int((time.time() - start_time) * 1000)

            if response.ok:
                data = response.json()
                logger.info(f"Relay upload complete in {duration_ms}ms (backend reported: {data.get('uploadDurationMs', 'N/A')}ms)")
                return {
                    "success": True,
                    "blobId": data["blobId"],
                    "downloadUrl": data["downloadUrl"],
                    "uploadMethod": "relay",
                    "uploadDurationMs": data.get("uploadDurationMs", duration_ms),
                    "totalDurationMs": data.get("totalDurationMs", duration_ms),
                }
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    pass
                logger.warning(f"Relay upload failed ({response.status_code}): {error_msg}")
                return {
                    "success": False,
                    "error": f"Relay upload failed: {response.status_code} - {error_msg}",
                    "uploadMethod": "relay",
                }

        except requests.exceptions.Timeout:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Relay upload timed out after {duration_ms}ms")
            return {
                "success": False,
                "error": "Relay upload timed out",
                "uploadMethod": "relay",
            }
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Relay upload error after {duration_ms}ms: {e}")
            return {
                "success": False,
                "error": str(e),
                "uploadMethod": "relay",
            }

    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate Blake3 hash of file for content addressing.
        Falls back to SHA256 if blake3 not available.
        """
        try:
            import blake3
            hasher = blake3.blake3()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except ImportError:
            logger.warning("blake3 not installed, using SHA256")
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return None

    def _encrypt_file(self, file_path: str) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt file content with random AES-256-GCM key.

        Args:
            file_path: Path to file to encrypt

        Returns:
            Tuple of (encrypted_data, content_key, nonce)
        """
        # Generate random 32-byte key for this file
        content_key = os.urandom(32)

        # Generate random 12-byte nonce for AES-GCM
        nonce = os.urandom(12)

        # Read file content
        with open(file_path, 'rb') as f:
            plaintext = f.read()

        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(content_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Prepend nonce to ciphertext for easy decryption
        encrypted_data = nonce + ciphertext

        logger.info(f"Encrypted {len(plaintext)} bytes -> {len(encrypted_data)} bytes")
        return encrypted_data, content_key, nonce

    def _get_sui_x25519_pubkey(self, wallet_address: str) -> Optional[bytes]:
        """
        Fetch user's Sui X25519 public key from backend.

        The backend manages Sui keypairs for users. We fetch the X25519 public key
        which is derived from the Sui Ed25519 key, used for sealed box encryption.

        Args:
            wallet_address: User's Solana wallet address

        Returns:
            X25519 public key bytes (32 bytes) or None if not found
        """
        try:
            response = requests.get(
                f"{self.backend_url}/api/walrus/sui-pubkey/{wallet_address}",
                timeout=10
            )

            if not response.ok:
                logger.warning(f"Failed to fetch Sui pubkey for {wallet_address[:8]}...: {response.status_code}")
                return None

            data = response.json()
            if not data.get("success"):
                logger.warning(f"Backend returned error for Sui pubkey: {data.get('error')}")
                return None

            x25519_pubkey_b64 = data.get("x25519PublicKey")
            if not x25519_pubkey_b64:
                logger.warning(f"No x25519PublicKey in response for {wallet_address[:8]}...")
                return None

            return base64.b64decode(x25519_pubkey_b64)

        except Exception as e:
            logger.error(f"Error fetching Sui pubkey for {wallet_address[:8]}...: {e}")
            return None

    def _encrypt_key_for_user(self, content_key: bytes, user_pubkey: str) -> bytes:
        """
        Encrypt the content key for a specific user using their Sui X25519 public key.

        Fetches the user's Sui X25519 public key from backend and uses NaCl sealed box
        encryption. The backend can then decrypt using the corresponding Sui private key.

        Args:
            content_key: The 32-byte AES key to encrypt
            user_pubkey: User's Solana wallet address (used to look up Sui key)

        Returns:
            Encrypted key bytes (sealed box format)
        """
        try:
            from nacl.public import SealedBox, PublicKey

            # Fetch Sui X25519 public key from backend
            x25519_pubkey_bytes = self._get_sui_x25519_pubkey(user_pubkey)

            if x25519_pubkey_bytes is None:
                logger.warning(f"No Sui pubkey found for {user_pubkey[:8]}..., using fallback")
                return self._fallback_encrypt_key(content_key, user_pubkey)

            # Create X25519 public key object
            x25519_pubkey = PublicKey(x25519_pubkey_bytes)

            # Use sealed box to encrypt (anonymous sender)
            sealed_box = SealedBox(x25519_pubkey)
            encrypted_key = sealed_box.encrypt(content_key)

            logger.info(f"Encrypted content key with Sui pubkey for {user_pubkey[:8]}...")
            return encrypted_key

        except ImportError:
            logger.warning("PyNaCl not installed, using fallback encryption")
            return self._fallback_encrypt_key(content_key, user_pubkey)
        except Exception as e:
            logger.warning(f"NaCl encryption failed for {user_pubkey[:8]}...: {e}")
            return self._fallback_encrypt_key(content_key, user_pubkey)

    def _fallback_encrypt_key(self, content_key: bytes, user_pubkey: str) -> bytes:
        """
        Fallback key encryption using HKDF derivation.
        Same pattern as activity_encryption_service.py.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        salt = b'mmoment-walrus-encryption-v1'

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b'walrus-content-key-wrap'
        )

        wrapping_key = hkdf.derive(user_pubkey.encode('utf-8'))
        encrypted = bytes(a ^ b for a, b in zip(content_key, wrapping_key))

        return encrypted

    def _create_access_grants(
        self,
        content_key: bytes,
        users: List[str],
    ) -> List[Dict[str, str]]:
        """
        Create access grants for all users who should be able to decrypt.

        Args:
            content_key: The 32-byte AES key used to encrypt content
            users: List of wallet addresses

        Returns:
            List of {pubkey, encryptedKey} dicts
        """
        access_grants = []
        for user_pubkey in users:
            try:
                encrypted_key = self._encrypt_key_for_user(content_key, user_pubkey)
                access_grants.append({
                    'pubkey': user_pubkey,
                    'encryptedKey': base64.b64encode(encrypted_key).decode('utf-8')
                })
            except Exception as e:
                logger.warning(f"Failed to create access grant for {user_pubkey[:8]}...: {e}")

        return access_grants

    def _upload_to_walrus(
        self,
        encrypted_data: bytes,
        user_sui_address: Optional[str] = None,
        epochs: int = DEFAULT_EPOCHS,
        deletable: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload encrypted data to Walrus.

        Args:
            encrypted_data: Encrypted file content
            user_sui_address: If provided, transfers blob ownership to this Sui address
            epochs: Storage duration (183 = ~1 year max)
            deletable: Whether blob can be deleted by owner

        Returns:
            Dict with blobId, objectId, downloadUrl
        """
        params = {
            "epochs": epochs,
        }

        if user_sui_address:
            params["send_object_to"] = user_sui_address
            logger.info(f"Uploading with ownership transfer to {user_sui_address[:16]}...")

        try:
            response = requests.put(
                f"{WALRUS_PUBLISHER_URL}/v1/blobs",
                params=params,
                data=encrypted_data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300  # 5 minutes for large files
            )
            response.raise_for_status()

            result = response.json()

            # Extract blobId from response
            if "newlyCreated" in result:
                blob_info = result["newlyCreated"]["blobObject"]
                blob_id = blob_info["blobId"]
                object_id = blob_info.get("id")
            elif "alreadyCertified" in result:
                blob_id = result["alreadyCertified"]["blobId"]
                object_id = result["alreadyCertified"].get("event", {}).get("objectId")
            else:
                return {"success": False, "error": "Unexpected response format", "raw": result}

            download_url = f"{WALRUS_AGGREGATOR_URL}/v1/blobs/{blob_id}"

            logger.info(f"Uploaded to Walrus: {blob_id}")
            logger.info(f"Download URL: {download_url}")

            return {
                "success": True,
                "blobId": blob_id,
                "objectId": object_id,
                "downloadUrl": download_url,
                "owner": user_sui_address or "mmoment",
                "epochs": epochs,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Walrus upload failed: {e}")
            return {"success": False, "error": str(e)}

    def _notify_backend(
        self,
        wallet_address: str,
        blob_id: str,
        download_url: str,
        camera_id: str,
        device_signature: str,
        file_type: str,
        timestamp: Optional[int],
        original_size: int,
        encrypted_size: int,
        nonce: str,
        access_grants: List[Dict],
        sui_owner: Optional[str] = None,
    ) -> bool:
        """
        Notify backend of completed Walrus upload.

        Args:
            All the file metadata and access grants

        Returns:
            True if notification successful
        """
        try:
            notify_data = {
                "walletAddress": wallet_address,
                "blobId": blob_id,
                "downloadUrl": download_url,
                "cameraId": camera_id,
                "deviceSignature": device_signature,
                "fileType": file_type,
                "timestamp": timestamp,
                "originalSize": original_size,
                "encryptedSize": encrypted_size,
                "nonce": nonce,
                "accessGrants": access_grants,
                "suiOwner": sui_owner,
            }

            response = requests.post(
                f"{self.backend_url}/api/walrus/upload-complete",
                json=notify_data,
                timeout=30
            )
            response.raise_for_status()

            logger.info(f"Backend notified of Walrus upload: {blob_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to notify backend: {e}")
            return False

    def upload_capture(
        self,
        wallet_address: str,
        file_path: str,
        camera_id: str,
        device_signature: str,
        checked_in_users: List[str],
        file_type: str = "photo",
        timestamp: Optional[int] = None,
        user_sui_address: Optional[str] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """
        Upload a camera capture to Walrus with encryption.

        Args:
            wallet_address: User's wallet address (owner)
            file_path: Path to the captured file
            camera_id: Camera ID (PDA)
            device_signature: Device-signed capture event signature
            checked_in_users: List of wallet addresses for access grants
            file_type: 'photo' or 'video'
            timestamp: Capture timestamp
            user_sui_address: User's Sui address for blob ownership
            private: If True, only owner can decrypt (no access grants for others)

        Returns:
            Dict with upload result
        """
        try:
            logger.info(f"Starting Walrus upload for {wallet_address[:8]}...")
            logger.info(f"  File: {file_path}")
            logger.info(f"  Camera: {camera_id}")
            logger.info(f"  Private: {private}")

            # Verify file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {
                    'success': False,
                    'error': 'File not found',
                    'provider': 'walrus'
                }

            original_size = os.path.getsize(file_path)

            # 1. Encrypt the file
            encrypted_data, content_key, nonce = self._encrypt_file(file_path)
            encrypted_size = len(encrypted_data)

            # 2. Create access grants
            if private:
                # Only owner can decrypt
                grant_users = [wallet_address]
            else:
                # All checked-in users can decrypt
                grant_users = list(checked_in_users)
                if wallet_address not in grant_users:
                    grant_users.append(wallet_address)

            access_grants = self._create_access_grants(content_key, grant_users)
            logger.info(f"Created {len(access_grants)} access grants")

            nonce_hex = base64.b64encode(nonce).decode('utf-8')
            upload_result = None
            upload_method = "unknown"

            # 3. Try relay first (fast path), then fall back to direct publisher
            if self.relay_enabled:
                logger.info("Attempting upload via backend relay (fast path)...")
                upload_result = self._upload_via_relay(
                    encrypted_data=encrypted_data,
                    wallet_address=wallet_address,
                    camera_id=camera_id,
                    device_signature=device_signature,
                    file_type=file_type,
                    timestamp=timestamp,
                    original_size=original_size,
                    nonce=nonce_hex,
                    access_grants=access_grants,
                    sui_owner=user_sui_address,
                )

                if upload_result["success"]:
                    upload_method = "relay"
                    # Backend already saved to DB and notified websocket clients
                    logger.info(f"Relay upload succeeded in {upload_result.get('uploadDurationMs', 'N/A')}ms")
                else:
                    logger.warning(f"Relay upload failed: {upload_result.get('error')}, falling back to direct publisher")
                    upload_result = None  # Reset to try direct publisher

            # Fall back to direct publisher if relay failed or not enabled
            if upload_result is None:
                logger.info("Using direct HTTP publisher (slow path)...")
                upload_result = self._upload_to_walrus(
                    encrypted_data=encrypted_data,
                    user_sui_address=user_sui_address,
                )
                upload_method = "direct"

                if not upload_result["success"]:
                    return {
                        'success': False,
                        'error': upload_result.get('error', 'Upload failed'),
                        'provider': 'walrus'
                    }

                # 4. Notify backend (only needed for direct publisher path)
                self._notify_backend(
                    wallet_address=wallet_address,
                    blob_id=upload_result["blobId"],
                    download_url=upload_result["downloadUrl"],
                    camera_id=camera_id,
                    device_signature=device_signature,
                    file_type=file_type,
                    timestamp=timestamp,
                    original_size=original_size,
                    encrypted_size=encrypted_size,
                    nonce=nonce_hex,
                    access_grants=access_grants,
                    sui_owner=user_sui_address,
                )

            logger.info(f"Walrus upload complete: {upload_result['blobId']}")

            return {
                'success': True,
                'provider': 'walrus',
                'blob_id': upload_result["blobId"],
                'object_id': upload_result.get("objectId"),
                'download_url': upload_result["downloadUrl"],
                'device_signature': device_signature,
                'encrypted': True,
                'nonce': nonce_hex,
                'original_size': original_size,
                'encrypted_size': encrypted_size,
                'access_grants_count': len(access_grants),
                'sui_owner': user_sui_address,
                'upload_type': 'walrus_encrypted',
                'upload_method': upload_method,
                'upload_duration_ms': upload_result.get("uploadDurationMs"),
            }

        except Exception as e:
            logger.error(f"Walrus upload error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'provider': 'walrus'
            }

    def upload_photo(
        self,
        wallet_address: str,
        photo_path: str,
        camera_id: str,
        device_signature: str,
        checked_in_users: List[str],
        timestamp: Optional[int] = None,
        user_sui_address: Optional[str] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method for uploading photos.
        """
        return self.upload_capture(
            wallet_address=wallet_address,
            file_path=photo_path,
            camera_id=camera_id,
            device_signature=device_signature,
            checked_in_users=checked_in_users,
            file_type="photo",
            timestamp=timestamp,
            user_sui_address=user_sui_address,
            private=private,
        )

    def upload_video(
        self,
        wallet_address: str,
        video_path: str,
        camera_id: str,
        device_signature: str,
        checked_in_users: List[str],
        duration: int = 0,
        timestamp: Optional[int] = None,
        user_sui_address: Optional[str] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method for uploading videos.
        """
        return self.upload_capture(
            wallet_address=wallet_address,
            file_path=video_path,
            camera_id=camera_id,
            device_signature=device_signature,
            checked_in_users=checked_in_users,
            file_type="video",
            timestamp=timestamp,
            user_sui_address=user_sui_address,
            private=private,
        )

    def get_download_url(self, blob_id: str) -> str:
        """Get download URL for a blob."""
        return f"{WALRUS_AGGREGATOR_URL}/v1/blobs/{blob_id}"


# Singleton instance
_walrus_upload_service = None


def get_walrus_upload_service(backend_url: str = None) -> WalrusUploadService:
    """Get or create the singleton Walrus upload service instance."""
    global _walrus_upload_service

    if _walrus_upload_service is None:
        _walrus_upload_service = WalrusUploadService(backend_url=backend_url)

    return _walrus_upload_service
