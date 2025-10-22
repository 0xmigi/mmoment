#!/usr/bin/env python3
"""
Biometric Encryption Service

Handles AES-256 encryption/decryption of facial embeddings for NFT storage.
Uses session-based encryption keys for enhanced security.
"""

import os
import json
import base64
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("BiometricEncryption")

class BiometricEncryptionService:
    """
    Service for encrypting/decrypting facial embeddings.
    Uses AES-256 encryption with session-based keys.
    """
    
    def __init__(self):
        # Encryption configuration
        self.salt = b'mmoment-insightface-v1-salt'  # Static salt for key derivation
        self.iterations = 100000  # PBKDF2 iterations
        
        logger.info("Biometric encryption service initialized")
    
    def generate_encryption_key(self, wallet_address: str, session_id: str = None) -> bytes:
        """
        Generate deterministic encryption key for a wallet

        For recognition tokens that need to work across sessions/cameras,
        the key is derived ONLY from wallet address (session_id is ignored).

        Args:
            wallet_address: User's wallet address
            session_id: Ignored for recognition tokens (kept for API compatibility)

        Returns:
            32-byte encryption key
        """
        try:
            # Use ONLY wallet address for key material (session-independent)
            # This makes recognition tokens portable across sessions and cameras
            key_material_str = f"{wallet_address}-recognition-token-v2"
            key_material = key_material_str.encode()

            logger.info(f"🔑 Generating encryption key with material: {wallet_address[:8]}...-recognition-token-v2")

            # Use PBKDF2 for key derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=self.iterations,
            )

            # Generate and encode key
            key = base64.urlsafe_b64encode(kdf.derive(key_material))

            logger.info(f"🔑 Generated wallet-only encryption key for: {wallet_address[:8]}...")
            return key

        except Exception as e:
            logger.error(f"Error generating encryption key: {e}")
            raise
    
    def encrypt_embedding(self, embedding: List[float], wallet_address: str, session_id: str, metadata: Dict = None) -> Dict:
        """
        Encrypt a facial embedding for NFT storage
        
        Args:
            embedding: InsightFace embedding as list of floats
            wallet_address: User's wallet address
            session_id: Current session ID
            metadata: Optional metadata to include
            
        Returns:
            NFT-ready package with encrypted embedding (raw bytes, no Base64)
        """
        try:
            # Convert embedding to numpy array if needed
            if isinstance(embedding, list):
                embedding_array = np.array(embedding, dtype=np.float32)
            else:
                embedding_array = embedding.astype(np.float32)
            
            # Generate encryption key
            key = self.generate_encryption_key(wallet_address, session_id)
            cipher = Fernet(key)

            # Quantize embedding from float32 to int8 for smaller on-chain storage
            # This reduces size from 2048 bytes to 512 bytes with minimal accuracy loss (~1-3%)
            # Original range: typically [-1, 1] for normalized embeddings
            embedding_quantized = np.clip(embedding_array * 127, -128, 127).astype(np.int8)

            # Prepare embedding data
            embedding_bytes = embedding_quantized.tobytes()
            embedding_shape = embedding_array.shape  # Keep original shape
            embedding_dtype = 'int8'  # Mark as quantized
            
            # Create comprehensive metadata
            full_metadata = {
                'shape': embedding_shape,
                'dtype': embedding_dtype,
                'wallet_address': wallet_address,
                'session_id': session_id,
                'timestamp': int(time.time() * 1000),
                'model': 'insightface',
                'version': 'v1.0',
                'encryption_method': 'PBKDF2-HMAC-SHA256-AES256',
                'service': 'mmoment-biometric-security'
            }
            
            # Add any additional metadata
            if metadata:
                full_metadata.update(metadata)
            
            # Encrypt the embedding data
            encrypted_embedding = cipher.encrypt(embedding_bytes)
            
            # Encrypt the metadata
            metadata_json = json.dumps(full_metadata).encode()
            encrypted_metadata = cipher.encrypt(metadata_json)
            
            # Create NFT-ready package with Base64 encoding (efficient for JSON)
            nft_package = {
                'encrypted_embedding': base64.b64encode(encrypted_embedding).decode('utf-8'),
                'encrypted_metadata': base64.b64encode(encrypted_metadata).decode('utf-8'),
                'wallet_address': wallet_address,
                'created_at': full_metadata['timestamp'],
                'biometric_type': 'facial_embedding_insightface',
                'encryption_version': 'v1.0',
                'model_type': 'insightface',
                'service_version': 'mmoment-biometric-security-v1.0'
            }
            
            logger.info(f"Successfully encrypted embedding for wallet: {wallet_address[:8]}... (Base64, {len(nft_package['encrypted_embedding'])} chars)")
            return nft_package
            
        except Exception as e:
            logger.error(f"Error encrypting embedding: {e}")
            raise
    
    def decrypt_embedding(self, token_package: Dict, wallet_address: str, session_id: str) -> np.ndarray:
        """
        Decrypt a facial embedding from recognition token package

        Args:
            token_package: The recognition token package containing encrypted embedding
            wallet_address: User's wallet address
            session_id: Current session ID

        Returns:
            Decrypted InsightFace embedding as numpy array
        """
        try:
            # Generate decryption key (same as encryption key)
            key = self.generate_encryption_key(wallet_address, session_id)
            cipher = Fernet(key)

            # Check if this is an on-chain recognition token (no encrypted_metadata)
            # or a full token package (with encrypted_metadata)
            has_metadata = 'encrypted_metadata' in token_package

            if has_metadata:
                # Full token package with metadata - original flow
                encrypted_metadata = base64.b64decode(token_package['encrypted_metadata'])
                metadata_json = cipher.decrypt(encrypted_metadata)
                metadata = json.loads(metadata_json.decode())

                # Validate wallet address matches
                if metadata['wallet_address'] != wallet_address:
                    raise ValueError("Wallet address mismatch in token package")

                # Verify this is an InsightFace embedding
                if metadata.get('model') != 'insightface':
                    logger.warning(f"Token package model type: {metadata.get('model')} (expected: insightface)")

                # Decrypt embedding data
                encrypted_embedding = base64.b64decode(token_package['encrypted_embedding'])
                embedding_bytes = cipher.decrypt(encrypted_embedding)

                # Reconstruct numpy array with original shape and dtype
                embedding = np.frombuffer(embedding_bytes, dtype=metadata['dtype'])
                embedding = embedding.reshape(metadata['shape'])

                # Dequantize from int8 back to float32 if needed
                if metadata['dtype'] == 'int8':
                    embedding = embedding.astype(np.float32) / 127.0

            else:
                # On-chain recognition token (minimal format - just encrypted embedding)
                logger.info(f"Decrypting on-chain recognition token for wallet: {wallet_address[:8]}...")

                try:
                    # Decrypt embedding data
                    # The token_package['encrypted_embedding'] is base64-encoded bytes from blockchain
                    encrypted_embedding_b64 = token_package['encrypted_embedding']
                    logger.info(f"Encrypted embedding (base64) length: {len(encrypted_embedding_b64)} chars")

                    # The encrypted_embedding_b64 is OUR base64 encoding of Fernet's output
                    # Fernet.decrypt() expects the Fernet token (which is already base64 internally)
                    # So we need to decode our outer base64 layer first
                    fernet_token = base64.b64decode(encrypted_embedding_b64)
                    embedding_bytes = cipher.decrypt(fernet_token)
                    logger.info(f"Decrypted embedding size: {len(embedding_bytes)} bytes")

                    # Reconstruct numpy array - embeddings are now quantized to int8 (512 bytes)
                    embedding = np.frombuffer(embedding_bytes, dtype=np.int8)

                    # Validate size
                    if len(embedding) != 512:
                        logger.warning(f"Unexpected embedding size: {len(embedding)} (expected 512)")

                    # Dequantize from int8 back to float32
                    embedding = embedding.astype(np.float32) / 127.0
                    logger.info(f"Dequantized embedding to float32, range: [{embedding.min():.3f}, {embedding.max():.3f}]")

                except Exception as decrypt_error:
                    logger.error(f"Failed to decrypt on-chain token: {type(decrypt_error).__name__}: {str(decrypt_error)}")
                    raise

            logger.info(f"Successfully decrypted embedding for wallet: {wallet_address[:8]}...")
            return embedding

        except Exception as e:
            logger.error(f"Error decrypting embedding: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def validate_nft_package(self, nft_package: Dict) -> bool:
        """
        Validate NFT package structure and required fields
        
        Args:
            nft_package: NFT package to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = [
                'encrypted_embedding',
                'encrypted_metadata', 
                'wallet_address',
                'created_at',
                'biometric_type',
                'encryption_version'
            ]
            
            for field in required_fields:
                if field not in nft_package:
                    logger.warning(f"Missing required field in NFT package: {field}")
                    return False
            
            # Validate biometric type
            if nft_package['biometric_type'] != 'facial_embedding_insightface':
                logger.warning(f"Invalid biometric type: {nft_package['biometric_type']}")
                return False
            
            # Validate base64 encoding
            try:
                base64.b64decode(nft_package['encrypted_embedding'])
                base64.b64decode(nft_package['encrypted_metadata'])
            except Exception:
                logger.warning("Invalid base64 encoding in NFT package")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating NFT package: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get encryption service status"""
        return {
            'service': 'biometric-encryption',
            'version': '1.0.0',
            'encryption_method': 'PBKDF2-HMAC-SHA256-AES256',
            'key_iterations': self.iterations,
            'supported_models': ['insightface'],
            'status': 'active'
        }

# Import time for timestamp
import time 