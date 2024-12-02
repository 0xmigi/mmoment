# src/nfc_controller.py
import subprocess
import time
import logging
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NFCController')

class NFCController:
    def __init__(self):
        self.camera_id = "cam_001"
        self.images_dir = "captured_images"
        
    def get_activity_stats(self):
        """Get real-time activity stats for the URL"""
        try:
            # Count images from last hour
            now = time.time()
            hour_ago = now - 3600
            
            recent_captures = 0
            if os.path.exists(self.images_dir):
                for filename in os.listdir(self.images_dir):
                    if filename.startswith("capture_"):
                        try:
                            timestamp = int(filename.split("_")[1].split(".")[0])
                            if timestamp > hour_ago:
                                recent_captures += 1
                        except (IndexError, ValueError):
                            continue
            
            return {
                "recent_activity": recent_captures,
                "timestamp": int(now)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"recent_activity": 0, "timestamp": int(time.time())}

    def create_ndef_payload(self):
        """Create a dynamic URL-based NDEF record"""
        try:
            stats = self.get_activity_stats()
            
            # Create dynamic URL with stats
            base_url = "https://f289-2603-7000-9400-2792-b477-72f0-1722-8e6f.ngrok-free.app"  # Update this!
            url = f"{base_url}/quickstart/{self.camera_id}/{stats['timestamp']}?activity={stats['recent_activity']}"
            
            logger.info(f"Generated URL: {url}")
            
            # Create NDEF record structure for URL
            NDEF_RECORD = bytearray([
                0xD1,                # MB=1, ME=1, CF=0, SR=1, IL=0, TNF=1
                0x01,                # Record type length
                len(url) + 1,        # Payload length (+1 for URL prefix)
                0x55,                # 'U' - Record type
                0x04,                # https:// prefix
            ])
            
            # Add URL without https:// prefix
            url_without_prefix = url[8:]  # Remove 'https://'
            NDEF_RECORD.extend(url_without_prefix.encode())
            
            return NDEF_RECORD
            
        except Exception as e:
            logger.error(f"Error creating NDEF payload: {e}")
            raise

    def run(self):
        logger.info("Starting NFC controller with dynamic URL generation...")
        
        while True:
            try:
                # Generate fresh NDEF payload with current stats
                payload = self.create_ndef_payload()
                
                # Write to NFC device
                cmd = ['nfc-mfsetuid', '-f', payload.hex()]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if "UID" in result.stdout:
                    logger.info("Phone detected! Served dynamic URL with current stats")
                    
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    controller = NFCController()
    controller.run()

