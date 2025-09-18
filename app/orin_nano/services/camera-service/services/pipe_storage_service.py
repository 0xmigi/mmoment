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
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

# Add the parent directory to Python path to import pipe_integration
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from pipe_integration import handle_camera_capture, on_user_checkin, PipeStorageManager

# Create pipe manager with host network backend URL
pipe_manager = PipeStorageManager(backend_url="http://192.168.1.232:3001")

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
        # Always enabled with new HTTP-based approach
        self.enabled = True
        logger.info("✅ Pipe Network storage ready for direct uploads")
    
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
            # Generate metadata for the upload
            timestamp = metadata.get('timestamp', 'unknown') if metadata else 'unknown'
            camera_id = metadata.get('camera_id', 'cam01') if metadata else 'cam01'

            logger.info(f"📤 Uploading {capture_type} for user {user_wallet[:8]}...")
            logger.info(f"   Camera: {camera_id}")
            logger.info(f"   Size: {len(image_data)} bytes")

            # Use the new fast direct upload
            upload_metadata = {
                'timestamp': timestamp,
                'camera_id': camera_id
            }

            result = await handle_camera_capture(
                wallet_address=user_wallet,
                image_data=image_data,
                metadata=upload_metadata
            )

            if result['success']:
                logger.info(f"✅ Upload successful: {result.get('filename', 'unknown')}")
                # Add user_wallet and provider for compatibility
                result['user_wallet'] = user_wallet
                result['provider'] = 'pipe'
                return result
            else:
                logger.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
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
            # Get user credentials first
            credentials = await get_or_create_user_credentials(user_wallet)
            if not credentials:
                return {
                    'success': False,
                    'error': 'Failed to get user credentials',
                    'user_wallet': user_wallet
                }

            # Get storage info using real SDK
            storage_info = await pipe_sdk.get_user_storage_info(credentials)

            if storage_info['success']:
                return {
                    'success': True,
                    'user_wallet': user_wallet,
                    'sol_balance': storage_info.get('sol_balance', 0.0),
                    'pipe_balance': storage_info.get('pipe_balance', 0.0),
                    'files_count': storage_info.get('files_count', 0),
                    'total_storage_mb': storage_info.get('total_storage_mb', 0.0)
                }
            else:
                return {
                    'success': False,
                    'error': storage_info.get('error', 'Failed to get storage info'),
                    'user_wallet': user_wallet
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
            # Get user credentials first
            credentials = await get_or_create_user_credentials(user_wallet)
            if not credentials:
                return {
                    'success': False,
                    'error': 'Failed to get user credentials',
                    'files': []
                }

            # List files using real SDK
            files_result = await pipe_sdk.list_user_files(credentials)

            if files_result['success']:
                return {
                    'success': True,
                    'user_wallet': user_wallet,
                    'files': files_result.get('files', []),
                    'total_count': files_result.get('total_count', 0)
                }
            else:
                return {
                    'success': False,
                    'error': files_result.get('error', 'Failed to list files'),
                    'files': []
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

async def pre_authorize_user_session(user_wallet: str) -> bool:
    """
    Pre-authorize a user's Pipe session when they check in with wallet connection.
    This creates a cached session for fast uploads during their visit.

    Called from wallet connection/check-in flow.
    """
    logger.info(f"🔐 Pre-authorizing Pipe session for {user_wallet[:8]}...")
    return await on_user_checkin(user_wallet)

# Upload Event Logging Functions (for future on-chain recording)
def get_upload_events(clear_after_retrieval: bool = False) -> Dict[str, Any]:
    """
    Get pending upload events for on-chain recording

    Args:
        clear_after_retrieval: If True, clears events after retrieval

    Returns:
        Dict with events and statistics
    """
    try:
        events = pipe_manager.get_pending_events(mark_as_retrieved=clear_after_retrieval)
        stats = pipe_manager.get_upload_stats()

        return {
            'success': True,
            'events': events,
            'stats': stats,
            'count': len(events)
        }

    except Exception as e:
        logger.error(f"Failed to get upload events: {e}")
        return {
            'success': False,
            'error': str(e),
            'events': [],
            'stats': {},
            'count': 0
        }

def mark_events_as_recorded(event_hashes: List[str]) -> Dict[str, Any]:
    """
    Mark upload events as recorded on-chain

    Args:
        event_hashes: List of event hashes that were successfully recorded

    Returns:
        Dict with result information
    """
    try:
        if not event_hashes:
            return {
                'success': False,
                'error': 'event_hashes list is required',
                'marked_count': 0
            }

        marked_count = pipe_manager.mark_events_as_recorded_on_chain(event_hashes)

        return {
            'success': True,
            'marked_count': marked_count,
            'message': f'Marked {marked_count} events as recorded on-chain'
        }

    except Exception as e:
        logger.error(f"Failed to mark events as recorded: {e}")
        return {
            'success': False,
            'error': str(e),
            'marked_count': 0
        }