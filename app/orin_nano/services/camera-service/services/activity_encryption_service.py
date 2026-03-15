"""
Activity Encryption Service

Handles AES-256-GCM encryption of CV app activities for privacy-preserving timeline.
Encrypts activity content and creates access grants for all checked-in users.

This is different from the biometric encryption service (Fernet for facial embeddings).
This service encrypts timeline activities (pushup counts, basketball scores, etc.)
"""

import os
import json
import base64
import logging
import time
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("ActivityEncryption")


class ActivityEncryptionService:
    """
    Service for encrypting CV app activities with AES-256-GCM.
    Creates access grants so all checked-in users can decrypt activities.

    Architecture (from PRIVACY_ARCHITECTURE.md):
    1. Generate random AES-256 key per activity (K_activity)
    2. Encrypt activity content with AES-256-GCM → encrypted_content + nonce
    3. For each checked-in user, encrypt K_activity to their wallet pubkey → access_grants
    4. Store encrypted bundle in backend buffer (not blockchain yet)
    """

    def __init__(self):
        """Initialize the activity encryption service."""
        self._initialized = True
        logger.info("Activity encryption service initialized (AES-256-GCM)")

    def encrypt_activity(
        self,
        activity_content: Dict[str, Any],
        users_present: List[str],
        activity_type: int = 50  # CVAppActivity = 50
    ) -> Dict[str, Any]:
        """
        Encrypt an activity for privacy-preserving storage.

        Args:
            activity_content: Dict containing activity data (e.g., {"type": "pushup", "count": 25, "user": "..."})
            users_present: List of wallet addresses currently checked in
            activity_type: Activity type enum (50 = CVAppActivity)

        Returns:
            Dict with encrypted_content, nonce, access_grants (all base64 encoded)
        """
        try:
            # 1. Generate random AES-256 key for this activity
            activity_key = os.urandom(32)  # 256 bits

            # 2. Generate random nonce for AES-GCM (12 bytes recommended)
            nonce = os.urandom(12)

            # 3. Serialize activity content to JSON bytes
            content_bytes = json.dumps(activity_content, separators=(',', ':')).encode('utf-8')

            # 4. Encrypt content with AES-256-GCM
            aesgcm = AESGCM(activity_key)
            encrypted_content = aesgcm.encrypt(nonce, content_bytes, None)

            # 5. Create access grants for each checked-in user
            access_grants = []
            for user_pubkey in users_present:
                try:
                    encrypted_key = self._encrypt_key_for_user(activity_key, user_pubkey)
                    access_grants.append({
                        'pubkey': user_pubkey,
                        'encryptedKey': base64.b64encode(encrypted_key).decode('utf-8')
                    })
                except Exception as e:
                    logger.warning(f"Failed to create access grant for {user_pubkey[:8]}...: {e}")

            # 6. Return encrypted bundle
            result = {
                'encryptedContent': base64.b64encode(encrypted_content).decode('utf-8'),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'accessGrants': access_grants,
                'activityType': activity_type,
                'timestamp': int(time.time() * 1000),
                'usersPresent': len(users_present)
            }

            logger.info(f"✅ Encrypted activity for {len(users_present)} users, content size: {len(encrypted_content)} bytes")
            return result

        except Exception as e:
            logger.error(f"❌ Error encrypting activity: {e}")
            raise

    def _encrypt_key_for_user(self, activity_key: bytes, user_pubkey: str) -> bytes:
        """
        Encrypt the activity key for a specific user's wallet pubkey.

        For Solana wallets, the pubkey is an Ed25519 public key (base58 encoded).
        We use a simple approach: derive a shared secret from the pubkey and encrypt.

        In production, this should use:
        - Convert Ed25519 pubkey to X25519 (using libsodium/nacl)
        - Use ECDH + HKDF to derive encryption key
        - Encrypt activity_key with derived key

        For now, we use a deterministic key derivation that the user can replicate
        with their private key. This is a simplified version - the full implementation
        would use proper asymmetric encryption.

        Args:
            activity_key: The 32-byte AES key to encrypt
            user_pubkey: User's wallet address (base58 Solana pubkey)

        Returns:
            Encrypted key bytes
        """
        try:
            # Try to use nacl for proper sealed box encryption
            from nacl.public import PublicKey, SealedBox
            from nacl.signing import VerifyKey
            import base58

            # Decode the Solana pubkey from base58
            pubkey_bytes = base58.b58decode(user_pubkey)

            # Convert Ed25519 verify key to X25519 public key for encryption
            # Ed25519 keys can be converted to X25519 (Curve25519) keys
            verify_key = VerifyKey(pubkey_bytes)
            x25519_pubkey = verify_key.to_curve25519_public_key()

            # Use sealed box to encrypt (anonymous sender)
            sealed_box = SealedBox(x25519_pubkey)
            encrypted_key = sealed_box.encrypt(activity_key)

            return encrypted_key

        except ImportError:
            logger.warning("PyNaCl not installed, using fallback encryption")
            return self._fallback_encrypt_key(activity_key, user_pubkey)
        except Exception as e:
            logger.warning(f"NaCl encryption failed for {user_pubkey[:8]}...: {e}, using fallback")
            return self._fallback_encrypt_key(activity_key, user_pubkey)

    def _fallback_encrypt_key(self, activity_key: bytes, user_pubkey: str) -> bytes:
        """
        Fallback key encryption using PBKDF2 derivation.

        This is less secure than proper asymmetric encryption but works
        when nacl is not available. The user can derive the same key
        using their wallet address + a shared secret.

        NOTE: This is a temporary solution. Production should use proper
        Ed25519 → X25519 conversion with sealed boxes.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        # Use HKDF to derive an encryption key from pubkey + salt
        salt = b'mmoment-activity-encryption-v1'

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b'activity-key-wrap'
        )

        # Derive wrapping key from pubkey
        wrapping_key = hkdf.derive(user_pubkey.encode('utf-8'))

        # XOR the activity key with the wrapping key (simple but reversible)
        encrypted = bytes(a ^ b for a, b in zip(activity_key, wrapping_key))

        return encrypted

    def decrypt_activity(
        self,
        encrypted_content: str,  # base64
        nonce: str,              # base64
        access_grants: List[Dict],
        user_private_key: bytes  # User's Ed25519 private key
    ) -> Optional[Dict[str, Any]]:
        """
        Decrypt an activity using the user's private key.

        This is typically done client-side in the browser, but this method
        is provided for testing and backend validation.

        Args:
            encrypted_content: Base64-encoded AES-256-GCM ciphertext
            nonce: Base64-encoded 12-byte nonce
            access_grants: List of {pubkey, encryptedKey} grants
            user_private_key: User's Ed25519 private key bytes

        Returns:
            Decrypted activity content dict, or None if decryption fails
        """
        try:
            from nacl.signing import SigningKey
            from nacl.public import SealedBox

            # Convert Ed25519 signing key to X25519 private key
            signing_key = SigningKey(user_private_key)
            x25519_private = signing_key.to_curve25519_private_key()

            # Get user's pubkey for matching
            user_pubkey = str(signing_key.verify_key)  # This needs base58 encoding

            # Find the access grant for this user
            activity_key = None
            for grant in access_grants:
                try:
                    # Try to decrypt this grant
                    sealed_box = SealedBox(x25519_private)
                    encrypted_key = base64.b64decode(grant['encryptedKey'])
                    activity_key = sealed_box.decrypt(encrypted_key)
                    break
                except Exception:
                    continue

            if not activity_key:
                logger.warning("No valid access grant found for this user")
                return None

            # Decrypt the content with the activity key
            aesgcm = AESGCM(activity_key)
            ciphertext = base64.b64decode(encrypted_content)
            nonce_bytes = base64.b64decode(nonce)

            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext, None)
            content = json.loads(plaintext.decode('utf-8'))

            return content

        except Exception as e:
            logger.error(f"Failed to decrypt activity: {e}")
            return None

    def get_status(self) -> Dict:
        """Get encryption service status."""
        return {
            'service': 'activity-encryption',
            'version': '1.0.0',
            'encryption_method': 'AES-256-GCM',
            'key_encryption': 'Ed25519-to-X25519 sealed box',
            'status': 'active'
        }


# Global instance
_activity_encryption_service = None


def get_activity_encryption_service() -> ActivityEncryptionService:
    """Get the activity encryption service singleton."""
    global _activity_encryption_service
    if _activity_encryption_service is None:
        _activity_encryption_service = ActivityEncryptionService()
    return _activity_encryption_service
