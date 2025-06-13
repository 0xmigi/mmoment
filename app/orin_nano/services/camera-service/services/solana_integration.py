"""
Solana Integration Service

Provides integration with the Solana blockchain for NFT validation
and facial embedding encryption/decryption.
"""

import os
import json
import logging
import requests
import threading
import time
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolanaIntegration")

class SolanaIntegrationService:
    """
    Service for integrating with Solana blockchain via the middleware service.
    Handles NFT validation and face embedding encryption/decryption.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SolanaIntegrationService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize instance
        self._initialized = True
        
        # Middleware connection settings
        self._middleware_url = os.environ.get('SOLANA_MIDDLEWARE_URL', 'http://localhost:5004')
        self._camera_pda = os.environ.get('CAMERA_PDA', 'WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD')
        self._program_id = os.environ.get('PROGRAM_ID', 'Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S')
        
        # Health status
        self._last_health_check = 0
        self._health_status = "Not checked"
        self._is_healthy = False
        
        # Perform initial health check
        self._check_health()
        
        logger.info(f"SolanaIntegrationService initialized with middleware at {self._middleware_url}")
        
    def _check_health(self) -> bool:
        """
        Check the health of the Solana middleware service.
        """
        try:
            response = requests.get(f"{self._middleware_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self._health_status = "Healthy"
                self._is_healthy = True
                self._last_health_check = time.time()
                return True
            else:
                self._health_status = f"Error: {response.status_code}"
                self._is_healthy = False
                return False
        except Exception as e:
            self._health_status = f"Error: {str(e)}"
            self._is_healthy = False
            return False
    
    def is_healthy(self) -> bool:
        """
        Check if the Solana middleware is healthy.
        Will only re-check health every 60 seconds to avoid excessive requests.
        """
        if time.time() - self._last_health_check > 60:  # Check health every 60 seconds
            self._check_health()
        return self._is_healthy
    
    def get_health_status(self) -> Dict:
        """
        Get the health status of the Solana middleware.
        """
        if time.time() - self._last_health_check > 60:  # Check health every 60 seconds
            self._check_health()
            
        return {
            "healthy": self._is_healthy,
            "status": self._health_status,
            "camera_pda": self._camera_pda,
            "program_id": self._program_id,
            "last_check": int(self._last_health_check * 1000) if self._last_health_check > 0 else 0
        }
    
    def encrypt_face_embedding(self, wallet_address: str, face_embedding: List[float]) -> Optional[str]:
        """
        Encrypt a face embedding using the user's NFT-based key.
        
        Args:
            wallet_address: The user's wallet address
            face_embedding: The face embedding as a list of floats
            
        Returns:
            Encrypted embedding string or None if encryption failed
        """
        if not self.is_healthy():
            logger.error("Solana middleware is not healthy, cannot encrypt face embedding")
            return None
            
        try:
            payload = {
                "wallet_address": wallet_address,
                "face_embedding": face_embedding,
                "camera_pda": self._camera_pda
            }
            
            response = requests.post(
                f"{self._middleware_url}/encrypt-face-embedding",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("encrypted_embedding")
                else:
                    logger.error(f"Failed to encrypt face embedding: {data.get('error')}")
                    return None
            else:
                logger.error(f"Failed to encrypt face embedding: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error encrypting face embedding: {str(e)}")
            return None
    
    def decrypt_face_embedding(self, wallet_address: str, encrypted_embedding: str) -> Optional[List[float]]:
        """
        Decrypt a face embedding using the user's NFT-based key.
        
        Args:
            wallet_address: The user's wallet address
            encrypted_embedding: The encrypted face embedding string
            
        Returns:
            Decrypted face embedding as a list of floats or None if decryption failed
        """
        if not self.is_healthy():
            logger.error("Solana middleware is not healthy, cannot decrypt face embedding")
            return None
            
        try:
            payload = {
                "wallet_address": wallet_address,
                "encrypted_embedding": encrypted_embedding,
                "camera_pda": self._camera_pda
            }
            
            response = requests.post(
                f"{self._middleware_url}/decrypt-face-embedding",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("face_embedding")
                else:
                    logger.error(f"Failed to decrypt face embedding: {data.get('error')}")
                    return None
            else:
                logger.error(f"Failed to decrypt face embedding: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error decrypting face embedding: {str(e)}")
            return None
    
    def verify_nft_ownership(self, wallet_address: str) -> bool:
        """
        Verify that a user owns an NFT that is valid for facial recognition.
        
        Args:
            wallet_address: The user's wallet address
            
        Returns:
            True if the user owns a valid NFT, False otherwise
        """
        if not self.is_healthy():
            logger.error("Solana middleware is not healthy, cannot verify NFT ownership")
            return False
            
        try:
            response = requests.get(
                f"{self._middleware_url}/verify-nft-ownership/{wallet_address}",
                params={"camera_pda": self._camera_pda},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("success", False) and data.get("has_valid_nft", False)
            else:
                logger.error(f"Failed to verify NFT ownership: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying NFT ownership: {str(e)}")
            return False

# Global function to get the Solana integration service instance
def get_solana_integration_service() -> SolanaIntegrationService:
    """
    Get the singleton SolanaIntegrationService instance.
    """
    return SolanaIntegrationService() 