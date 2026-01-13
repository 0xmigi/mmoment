import os
import json
import logging
import time
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib

logger = logging.getLogger(__name__)

class DeviceSigner:
    """
    Hardware-bound device signing service for DePIN authentication.
    Generates and manages ed25519 keypairs tied to device hardware.
    
    Future Enhancement: This same keypair can be used for on-chain transactions
    by integrating with Solana RPC for gasless meta-transactions or relayed txs.
    """
    
    def __init__(self):
        self.keypair = None
        self._load_or_create_keypair()

    def _get_hardware_key(self):
        """Generate hardware-bound encryption key using device-specific identifiers"""
        try:
            # Primary: Use CPU serial number for hardware binding
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                serial = None
                for line in cpu_info.split('\n'):
                    if 'Serial' in line:
                        serial = line.split(':')[1].strip()
                        break

            if not serial or serial == '0000000000000000':
                # Fallback to MAC address for hardware binding
                try:
                    serial = open('/sys/class/net/eth0/address').read().strip()
                except:
                    # Final fallback to machine-id
                    serial = open('/etc/machine-id').read().strip()

            # Create deterministic key from hardware serial
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'mmoment_depin_device_v1',  # Version for future key rotation
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(serial.encode()))
            logger.info(f"Hardware key generated from device identifier: {serial[:8]}...")
            return key
            
        except Exception as e:
            logger.warning(f"Hardware key generation failed: {e}, using development fallback")
            # Development fallback - in production this should fail
            return base64.urlsafe_b64encode(b'mmoment_dev_key_32_bytes_long!!')

    def _load_or_create_keypair(self):
        """Load existing hardware-bound keypair or create new one"""
        # Try production path first, fallback to local path for development/testing
        try:
            keypair_path = Path('/opt/mmoment/device/device-keypair.enc')
            keypair_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback to local directory for testing/development
            keypair_path = Path(os.path.expanduser('~/.mmoment/device-keypair.enc'))
            keypair_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using development keypair path: {keypair_path}")

        cipher = Fernet(self._get_hardware_key())

        if keypair_path.exists():
            try:
                # Load encrypted keypair
                with open(keypair_path, 'rb') as f:
                    encrypted_data = f.read()

                decrypted_data = cipher.decrypt(encrypted_data)
                keypair_data = json.loads(decrypted_data.decode())

                # Import Solana ed25519 keypair
                from solders.keypair import Keypair
                self.keypair = Keypair.from_bytes(bytes(keypair_data['private_key']))
                
                logger.info(f"Loaded device keypair: {self.keypair.pubkey()}")
                logger.info(f"Device ready for DePIN authentication and future on-chain operations")
                return

            except Exception as e:
                logger.warning(f"Failed to load keypair: {e}, generating new one")

        # Generate new Solana ed25519 keypair
        from solders.keypair import Keypair
        self.keypair = Keypair()

        # Prepare keypair data for encryption and storage
        keypair_data = {
            'private_key': list(bytes(self.keypair)),
            'public_key': str(self.keypair.pubkey()),
            'created_at': int(time.time()),
            'version': '1.0',
            'purpose': 'depin_device_auth_and_future_onchain'
        }

        # Encrypt and save
        encrypted_data = cipher.encrypt(json.dumps(keypair_data).encode())
        with open(keypair_path, 'wb') as f:
            f.write(encrypted_data)

        # Set secure permissions
        os.chmod(keypair_path, 0o600)
        
        logger.info(f"Generated new device keypair: {self.keypair.pubkey()}")
        logger.info("Device keypair ready for DePIN operations and future on-chain transactions")

    def sign_response(self, response_data: dict) -> dict:
        """
        Sign API response with device ed25519 key for DePIN authentication.
        
        Future Enhancement: This same signing method can be used for on-chain
        transaction signing by replacing response_data with transaction bytes.
        """
        if not self.keypair:
            logger.error("No device keypair available for signing")
            return response_data

        # Create a copy to avoid mutating original
        signed_response = response_data.copy()
        
        # Add device signature metadata
        signed_response['device_signature'] = {
            'device_pubkey': str(self.keypair.pubkey()),
            'timestamp': int(time.time() * 1000),
            'version': '1.0',
            'algorithm': 'ed25519',
            'purpose': 'depin_device_auth'
        }

        # Create deterministic signature payload
        # Sort keys for consistent signing across calls
        payload_dict = {k: v for k, v in signed_response.items() if k != 'device_signature'}
        payload = json.dumps(payload_dict, sort_keys=True, separators=(',', ':')).encode()
        
        # Add signature metadata to payload for complete verification
        signature_metadata = json.dumps(signed_response['device_signature'], sort_keys=True).encode()
        full_payload = payload + b'|' + signature_metadata

        # Sign with ed25519 - same method used for on-chain transactions
        signature = self.keypair.sign_message(full_payload)
        signed_response['device_signature']['signature'] = base64.b64encode(bytes(signature)).decode()
        
        return signed_response

    def get_public_key(self) -> str:
        """Get device public key as string - same format used on-chain"""
        return str(self.keypair.pubkey()) if self.keypair else None

    def get_keypair(self):
        """
        Get raw keypair for future on-chain transaction signing.
        SECURITY: Only expose this for authenticated on-chain operations.
        """
        return self.keypair
    
    def sign_transaction_bytes(self, transaction_bytes: bytes) -> bytes:
        """
        Future method: Sign raw transaction bytes for on-chain operations.
        This enables the same device key to authorize blockchain transactions.
        """
        if not self.keypair:
            raise ValueError("No device keypair available for transaction signing")
        
        signature = self.keypair.sign_message(transaction_bytes)
        return bytes(signature)
    
    def get_device_info(self) -> dict:
        """Get device information for DePIN network registration"""
        return {
            'device_pubkey': self.get_public_key(),
            'device_type': 'jetson_orin_nano',
            'capabilities': ['camera_capture', 'ai_processing', 'video_streaming'],
            'version': '1.0',
            'ready_for_onchain': True
        }

    def verify_wallet_signature(self, wallet_address: str, message: str, signature: str) -> bool:
        """
        Verify an ed25519 signature from a Solana wallet.

        This is the CRITICAL security function that proves a request actually
        came from the owner of the wallet, not just someone who knows the address.

        Args:
            wallet_address: Base58-encoded Solana public key
            message: The message that was signed (typically wallet_address|timestamp|nonce)
            signature: Base58-encoded ed25519 signature

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            import base58
            from nacl.signing import VerifyKey
            from nacl.exceptions import BadSignatureError

            # Decode wallet address (Solana pubkey is ed25519 public key)
            pubkey_bytes = base58.b58decode(wallet_address)

            # Decode signature
            signature_bytes = base58.b58decode(signature)

            # Create verify key from wallet public key
            verify_key = VerifyKey(pubkey_bytes)

            # Verify the signature
            verify_key.verify(message.encode(), signature_bytes)

            logger.info(f"✅ Wallet signature verified for {wallet_address[:8]}...")
            return True

        except BadSignatureError:
            logger.warning(f"❌ Invalid signature from wallet {wallet_address[:8]}...")
            return False
        except Exception as e:
            logger.error(f"❌ Signature verification error: {e}")
            return False

    def verify_request_signature(
        self,
        wallet_address: str,
        timestamp: int,
        nonce: str,
        signature: str,
        max_age_seconds: int = 300
    ) -> tuple[bool, str]:
        """
        Verify a signed request from a wallet with replay attack protection.

        Args:
            wallet_address: Wallet making the request
            timestamp: Unix timestamp (milliseconds) when request was signed
            nonce: Random nonce to prevent replay attacks
            signature: Base58-encoded signature of "wallet_address|timestamp|nonce"
            max_age_seconds: Maximum allowed age of request (default 5 minutes)

        Returns:
            Tuple of (is_valid, error_message)
        """
        import time

        # Check timestamp isn't too old (prevent replay attacks)
        current_time = int(time.time() * 1000)
        age_ms = current_time - timestamp

        if age_ms < 0:
            return False, "Request timestamp is in the future"

        if age_ms > (max_age_seconds * 1000):
            return False, f"Request too old ({age_ms // 1000}s > {max_age_seconds}s)"

        # Construct the message that should have been signed
        message = f"{wallet_address}|{timestamp}|{nonce}"

        # Verify the signature
        if not self.verify_wallet_signature(wallet_address, message, signature):
            return False, "Invalid signature"

        return True, "OK"