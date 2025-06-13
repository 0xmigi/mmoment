#!/usr/bin/env python3
"""
Secure Storage Service

Handles secure file operations for biometric data:
- Secure file deletion with multiple overwrites
- Temporary file management
- Session-based file cleanup
"""

import os
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("SecureStorage")

class SecureStorage:
    """
    Service for secure storage operations and file management
    """
    
    def __init__(self):
        # Storage configuration
        self.base_dir = Path('/app/secure_data')
        self.temp_dir = Path('/app/temp_embeddings')
        
        # Ensure directories exist with proper permissions
        self._setup_directories()
        
        logger.info("Secure storage service initialized")
    
    def _setup_directories(self):
        """Setup secure directories with proper permissions"""
        try:
            # Create directories
            self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            # Set restrictive permissions
            os.chmod(self.base_dir, 0o700)
            os.chmod(self.temp_dir, 0o700)
            
            logger.info("Secure directories setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup secure directories: {e}")
            raise
    
    def secure_delete_file(self, file_path: Path) -> bool:
        """
        Securely delete a file using multiple overwrites
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if not file_path.exists():
                return True
            
            logger.debug(f"Securely deleting file: {file_path}")
            
            # Try to use system shred command first (more secure)
            if self._shred_file(file_path):
                return True
            
            # Fallback to Python-based secure deletion
            return self._python_secure_delete(file_path)
            
        except Exception as e:
            logger.error(f"Error securely deleting {file_path}: {e}")
            return False
    
    def _shred_file(self, file_path: Path) -> bool:
        """
        Use system shred command for secure deletion
        
        Args:
            file_path: Path to the file to shred
            
        Returns:
            True if shred successful, False otherwise
        """
        try:
            # Use shred with 3 passes, verbose output, and zero final pass
            result = subprocess.run(
                ['shred', '-vfz', '-n', '3', str(file_path)], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                logger.debug(f"Successfully shredded file: {file_path}")
                return True
            else:
                logger.warning(f"Shred failed for {file_path}: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Shred command unavailable or failed: {e}")
            return False
    
    def _python_secure_delete(self, file_path: Path) -> bool:
        """
        Python-based secure file deletion with multiple overwrites
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            
            # Perform 3 passes of random data overwriting
            with open(file_path, 'rb+') as f:
                for pass_num in range(3):
                    f.seek(0)
                    # Write random data
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    logger.debug(f"Completed overwrite pass {pass_num + 1}/3 for {file_path}")
            
            # Finally delete the file
            file_path.unlink()
            logger.debug(f"Python secure delete completed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Python secure delete failed for {file_path}: {e}")
            return False
    
    def purge_session_files(self, session_id: str) -> int:
        """
        Purge all files associated with a session
        
        Args:
            session_id: Session ID to purge files for
            
        Returns:
            Number of files successfully purged
        """
        purged_count = 0
        
        try:
            # Search for files with session ID in name
            for directory in [self.base_dir, self.temp_dir]:
                for file_path in directory.glob(f"*{session_id}*"):
                    if file_path.is_file():
                        if self.secure_delete_file(file_path):
                            purged_count += 1
                            logger.debug(f"Purged session file: {file_path}")
                        else:
                            logger.warning(f"Failed to purge session file: {file_path}")
            
            logger.info(f"Purged {purged_count} files for session: {session_id}")
            return purged_count
            
        except Exception as e:
            logger.error(f"Error purging session files for {session_id}: {e}")
            return purged_count
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age in hours before files are considered old
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        
        try:
            for directory in [self.base_dir, self.temp_dir]:
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        # Check file age
                        file_age = current_time - file_path.stat().st_mtime
                        
                        if file_age > max_age_seconds:
                            if self.secure_delete_file(file_path):
                                cleaned_count += 1
                                logger.debug(f"Cleaned up old file: {file_path}")
                            else:
                                logger.warning(f"Failed to clean up old file: {file_path}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
            return cleaned_count
    
    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                'base_directory': str(self.base_dir),
                'temp_directory': str(self.temp_dir),
                'base_files': 0,
                'temp_files': 0,
                'total_size_bytes': 0
            }
            
            # Count files and calculate sizes
            for directory_name, directory_path in [('base', self.base_dir), ('temp', self.temp_dir)]:
                file_count = 0
                total_size = 0
                
                if directory_path.exists():
                    for file_path in directory_path.iterdir():
                        if file_path.is_file():
                            file_count += 1
                            total_size += file_path.stat().st_size
                
                stats[f'{directory_name}_files'] = file_count
                stats['total_size_bytes'] += total_size
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
    
    def verify_permissions(self) -> Dict:
        """
        Verify directory permissions are correct
        
        Returns:
            Dictionary with permission status
        """
        try:
            permissions = {}
            
            for name, path in [('base_dir', self.base_dir), ('temp_dir', self.temp_dir)]:
                if path.exists():
                    stat_info = path.stat()
                    mode = oct(stat_info.st_mode)[-3:]  # Get last 3 digits of octal mode
                    permissions[name] = {
                        'path': str(path),
                        'exists': True,
                        'mode': mode,
                        'uid': stat_info.st_uid,
                        'gid': stat_info.st_gid,
                        'secure': mode == '700'  # Should be owner-only access
                    }
                else:
                    permissions[name] = {
                        'path': str(path),
                        'exists': False,
                        'secure': False
                    }
            
            return permissions
            
        except Exception as e:
            logger.error(f"Error verifying permissions: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict:
        """Get secure storage status"""
        try:
            storage_stats = self.get_storage_stats()
            permissions = self.verify_permissions()
            
            return {
                'service': 'secure-storage',
                'version': '1.0.0',
                'storage_stats': storage_stats,
                'permissions': permissions,
                'shred_available': self._check_shred_available(),
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting secure storage status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_shred_available(self) -> bool:
        """Check if shred command is available"""
        try:
            result = subprocess.run(['which', 'shred'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False 