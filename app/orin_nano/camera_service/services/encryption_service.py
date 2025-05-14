import os
import base64
import logging
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class EncryptionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncryptionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.public_key = None
        self.private_key = None
        
        # Try to load existing keys or generate new ones
        if not self._load_keys():
            self._generate_keys()
    
    def _load_keys(self):
        """Attempt to load existing keys"""
        public_key_path = Settings.NETWORK_PUBLIC_KEY_PATH
        private_key_path = Settings.NETWORK_PRIVATE_KEY_PATH
        logger.info(f"EncryptionService: Attempting to load public key from: {os.path.abspath(public_key_path)}")
        logger.info(f"EncryptionService: Attempting to load private key from: {os.path.abspath(private_key_path)}")
        try:
            if os.path.exists(public_key_path) and os.path.exists(private_key_path):
                # Load public key
                with open(public_key_path, "rb") as key_file:
                    self.public_key = serialization.load_pem_public_key(
                        key_file.read(),
                        backend=default_backend()
                    )
                
                # Load private key
                with open(private_key_path, "rb") as key_file:
                    self.private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend()
                    )
                
                logger.info("Successfully loaded encryption keys")
                return True
        except Exception as e:
            logger.error(f"Failed to load encryption keys: {e}")
        
        return False
    
    def _generate_keys(self):
        """Generate new RSA key pair"""
        try:
            # Generate a new RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            # Serialize and save the private key
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Serialize and save the public key
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Save keys to files
            with open(Settings.NETWORK_PRIVATE_KEY_PATH, "wb") as key_file:
                key_file.write(private_pem)
            
            with open(Settings.NETWORK_PUBLIC_KEY_PATH, "wb") as key_file:
                key_file.write(public_pem)
            
            logger.info("Successfully generated and saved new encryption keys")
            return True
        except Exception as e:
            logger.error(f"Failed to generate encryption keys: {e}")
            return False
    
    def encrypt_data(self, data):
        """Encrypt data using the public key"""
        try:
            if not self.public_key:
                raise ValueError("Public key not available")
            
            # Generate a random AES key for symmetric encryption
            aes_key = os.urandom(32)  # 256-bit key
            iv = os.urandom(16)       # 128-bit IV
            
            # Encrypt the data with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad the data to be a multiple of the block size (16 bytes for AES)
            padded_data = self._pad_data(data)
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Encrypt the AES key with RSA
            encrypted_key = self.public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine everything and encode as base64
            result = {
                "encrypted_key": base64.b64encode(encrypted_key).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8')
            }
            
            return result
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt_data(self, encrypted_data_obj):
        """Decrypt data using the private key"""
        try:
            if not self.private_key:
                raise ValueError("Private key not available")
            
            # Decode base64 components
            encrypted_key = base64.b64decode(encrypted_data_obj["encrypted_key"])
            iv = base64.b64decode(encrypted_data_obj["iv"])
            encrypted_data = base64.b64decode(encrypted_data_obj["encrypted_data"])
            
            # Decrypt the AES key with RSA
            aes_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt the data with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            data = self._unpad_data(padded_data)
            
            return data
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None
    
    def _pad_data(self, data):
        """PKCS#7 padding for AES encryption"""
        if not isinstance(data, bytes):
            data = data if isinstance(data, bytearray) else bytes(data)
        
        block_size = 16  # AES block size
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)
        return padded_data
    
    def _unpad_data(self, padded_data):
        """Remove PKCS#7 padding"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length] 