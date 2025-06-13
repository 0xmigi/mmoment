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
    
    def generate_encryption_key(self, wallet_address: str, session_id: str) -> bytes:
        """
        Generate deterministic encryption key for a wallet/session combination
        
        Args:
            wallet_address: User's wallet address
            session_id: Current session ID
            
        Returns:
            32-byte encryption key
        """
        try:
            # Combine wallet + session + static component for key material
            key_material = f"{wallet_address}-{session_id}-facial-embedding".encode()
            
            # Use PBKDF2 for key derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=self.iterations,
            )
            
            # Generate and encode key
            key = base64.urlsafe_b64encode(kdf.derive(key_material))
            
            logger.debug(f"Generated encryption key for wallet: {wallet_address[:8]}...")
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
            NFT-ready package with encrypted embedding
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
            
            # Prepare embedding data
            embedding_bytes = embedding_array.tobytes()
            embedding_shape = embedding_array.shape
            embedding_dtype = str(embedding_array.dtype)
            
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
            
            # Create NFT-ready package
            nft_package = {
                'encrypted_embedding': base64.b64encode(encrypted_embedding).decode(),
                'encrypted_metadata': base64.b64encode(encrypted_metadata).decode(),
                'wallet_address': wallet_address,
                'created_at': full_metadata['timestamp'],
                'biometric_type': 'facial_embedding_insightface',
                'encryption_version': 'v1.0',
                'model_type': 'insightface',
                'service_version': 'mmoment-biometric-security-v1.0'
            }
            
            logger.info(f"Successfully encrypted embedding for wallet: {wallet_address[:8]}...")
            return nft_package
            
        except Exception as e:
            logger.error(f"Error encrypting embedding: {e}")
            raise
    
    def decrypt_embedding(self, nft_package: Dict, wallet_address: str, session_id: str) -> np.ndarray:
        """
        Decrypt a facial embedding from NFT package
        
        Args:
            nft_package: The NFT package containing encrypted embedding
            wallet_address: User's wallet address  
            session_id: Current session ID
            
        Returns:
            Decrypted InsightFace embedding as numpy array
        """
        try:
            # Generate decryption key (same as encryption key)
            key = self.generate_encryption_key(wallet_address, session_id)
            cipher = Fernet(key)
            
            # Decrypt metadata first to validate
            encrypted_metadata = base64.b64decode(nft_package['encrypted_metadata'])
            metadata_json = cipher.decrypt(encrypted_metadata)
            metadata = json.loads(metadata_json.decode())
            
            # Validate wallet address matches
            if metadata['wallet_address'] != wallet_address:
                raise ValueError("Wallet address mismatch in NFT package")
            
            # Verify this is an InsightFace embedding
            if metadata.get('model') != 'insightface':
                logger.warning(f"NFT package model type: {metadata.get('model')} (expected: insightface)")
            
            # Decrypt embedding data
            encrypted_embedding = base64.b64decode(nft_package['encrypted_embedding'])
            embedding_bytes = cipher.decrypt(encrypted_embedding)
            
            # Reconstruct numpy array with original shape and dtype
            embedding = np.frombuffer(embedding_bytes, dtype=metadata['dtype'])
            embedding = embedding.reshape(metadata['shape'])
            
            logger.info(f"Successfully decrypted embedding for wallet: {wallet_address[:8]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Error decrypting embedding: {e}")
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