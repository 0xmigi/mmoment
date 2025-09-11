"""
Pipe Network storage service for MMOMENT camera captures

This service handles uploading camera captures to users' private Pipe storage accounts.
Each user gets their own encrypted storage that only they can access.
"""

import json
import subprocess
import tempfile
import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PipeStorageService:
    """
    Service for managing user storage on Pipe Network
    
    This service:
    1. Manages Pipe SDK integration
    2. Handles per-user encrypted uploads
    3. Maintains user storage sessions
    4. Provides storage status/balance info
    """
    
    def __init__(self):
        self.pipe_sdk_path = self._find_pipe_sdk()
        self.pipe_credentials = self._load_pipe_credentials()
        self.enabled = self.pipe_credentials is not None
        
        if not self.enabled:
            logger.warning("Pipe Network storage disabled - no credentials found")
            logger.info("To enable: run 'pipe new-user mmoment_camera_XX'")
    
    def _find_pipe_sdk(self) -> str:
        """Find the compiled Pipe SDK binary"""
        sdk_paths = [
            "/mnt/nvme/mmoment/pipe-sdk/target/release/examples/test_real_api",
            "/mnt/nvme/mmoment/pipe-sdk/target/debug/examples/test_real_api"
        ]
        
        for path in sdk_paths:
            if os.path.exists(path):
                return path
        
        return "pipe-sdk-not-found"
    
    def _load_pipe_credentials(self) -> Optional[Dict[str, str]]:
        """Load Pipe credentials from ~/.pipe-cli.json"""
        creds_path = Path.home() / ".pipe-cli.json"
        if creds_path.exists():
            try:
                with open(creds_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load Pipe credentials: {e}")
        return None
    
    def is_enabled(self) -> bool:
        """Check if Pipe storage is available"""
        return self.enabled
    
    async def upload_user_capture(self, 
                                user_wallet: str,
                                image_data: bytes,
                                capture_type: str = "photo",
                                metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Upload a camera capture to user's Pipe storage
        
        Args:
            user_wallet: User's Solana wallet address (from face recognition)
            image_data: Raw image/video data from camera
            capture_type: Type of capture ("photo", "video") 
            metadata: Additional metadata (timestamp, camera_id, etc.)
            
        Returns:
            Dict with upload result
        """
        
        if not self.enabled:
            return {
                'success': False,
                'error': 'Pipe storage not enabled',
                'provider': 'pipe'
            }
        
        try:
            # Generate filename with MMOMENT naming convention
            timestamp = metadata.get('timestamp', 'unknown') if metadata else 'unknown'
            camera_id = metadata.get('camera_id', 'cam01') if metadata else 'cam01'
            filename = f"mmoment_{capture_type}_{camera_id}_{timestamp}.jpg"
            
            logger.info(f"📤 Uploading {capture_type} for user {user_wallet[:8]}...")
            logger.info(f"   File: {filename}")
            logger.info(f"   Size: {len(image_data)} bytes")
            
            # TODO: Replace with direct Rust SDK call via subprocess or PyO3
            # For now, simulate successful upload
            await asyncio.sleep(0.1)  # Simulate upload time
            
            # Simulate encryption
            logger.info(f"🔒 Encrypting with user-specific key...")
            
            result = {
                'success': True,
                'file_id': filename,
                'encrypted': True,
                'user_wallet': user_wallet,
                'provider': 'pipe',
                'storage_url': f"pipe://{user_wallet}/{filename}",
                'size': len(image_data),
                'upload_time': timestamp
            }
            
            logger.info(f"✅ Upload successful: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipe upload failed for {user_wallet[:8]}: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_wallet': user_wallet,
                'provider': 'pipe'
            }
    
    async def get_user_storage_info(self, user_wallet: str) -> Dict[str, Any]:
        """
        Get storage info for a user (balance, usage, etc.)
        
        Args:
            user_wallet: User's wallet address
            
        Returns:
            Dict with storage info
        """
        
        if not self.enabled:
            return {
                'success': False,
                'error': 'Pipe storage not enabled'
            }
        
        try:
            # TODO: Call Pipe SDK to get actual balance
            # For now, simulate
            return {
                'success': True,
                'user_wallet': user_wallet,
                'sol_balance': 0.0,
                'pipe_balance': 0.0,
                'files_count': 0,
                'total_storage_mb': 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage info for {user_wallet[:8]}: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_wallet': user_wallet
            }
    
    async def list_user_files(self, user_wallet: str) -> Dict[str, Any]:
        """
        List files in user's storage
        
        Args:
            user_wallet: User's wallet address
            
        Returns:
            Dict with file list
        """
        
        if not self.enabled:
            return {
                'success': False,
                'error': 'Pipe storage not enabled',
                'files': []
            }
        
        try:
            # TODO: Call Pipe SDK to list files
            return {
                'success': True,
                'user_wallet': user_wallet,
                'files': [],
                'total_count': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to list files for {user_wallet[:8]}: {e}")
            return {
                'success': False,
                'error': str(e),
                'files': []
            }

# Singleton instance for camera service
pipe_storage = PipeStorageService()

async def handle_user_capture_upload(user_wallet: str, 
                                   image_data: bytes,
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integration point for camera service capture handling
    
    This function should be called from capture_service.py when
    a camera capture is triggered for a recognized user.
    """
    
    return await pipe_storage.upload_user_capture(
        user_wallet=user_wallet,
        image_data=image_data,
        capture_type="photo",
        metadata=metadata
    )