import os
import time
import json
import logging
import threading
import requests
from typing import Optional, Dict, Any
from services.device_signer import DeviceSigner
from services.tunnel_manager import get_tunnel_manager

logger = logging.getLogger(__name__)

class DeviceRegistrationService:
    """
    Service to poll backend for device configuration after QR registration.
    Handles the complete device configuration flow including Cloudflare tunnel setup.
    """
    
    def __init__(self):
        self.device_signer = DeviceSigner()
        self.backend_url = self._get_backend_url()
        self.tunnel_manager = get_tunnel_manager()
        self.device_config = None
        self.polling_thread = None
        self.stop_polling = False
        self.config_received = False
        
    def _get_backend_url(self) -> str:
        """Get backend URL from environment or use default"""
        return os.getenv('BACKEND_URL', 'https://mmoment-production.up.railway.app')
        
    def start_polling(self):
        """Start polling for device configuration"""
        if self.polling_thread and self.polling_thread.is_alive():
            logger.info("Device configuration polling already running")
            return
            
        logger.info("Starting device configuration polling...")
        self.stop_polling = False
        self.config_received = False
        self.polling_thread = threading.Thread(target=self._poll_for_config, daemon=True)
        self.polling_thread.start()
        
    def stop_polling_service(self):
        """Stop polling for device configuration"""
        logger.info("Stopping device configuration polling...")
        self.stop_polling = True
        if self.polling_thread:
            self.polling_thread.join(timeout=5)
            
    def _poll_for_config(self):
        """Poll backend for device configuration"""
        device_pubkey = self.device_signer.get_public_key()
        if not device_pubkey:
            logger.error("No device public key available for polling")
            return
            
        poll_interval = 10  # seconds
        max_attempts = 360  # 1 hour max polling
        attempt = 0
        
        logger.info(f"Polling for configuration for device: {device_pubkey}")
        
        while not self.stop_polling and not self.config_received and attempt < max_attempts:
            try:
                # Poll the backend endpoint
                url = f"{self.backend_url}/api/device/{device_pubkey}/config"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    config_data = response.json()
                    logger.info(f"Received device configuration: {config_data}")
                    
                    # Validate the configuration
                    if self._validate_config(config_data):
                        self.device_config = config_data
                        self.config_received = True
                        
                        # Configure Cloudflare tunnel with the PDA subdomain
                        if self._configure_tunnel(config_data):
                            logger.info("Device registration and tunnel configuration completed successfully!")
                        else:
                            logger.error("Failed to configure Cloudflare tunnel")
                        return
                    else:
                        logger.warning("Received invalid configuration data")
                        
                elif response.status_code == 404:
                    # Device not yet configured, continue polling
                    logger.debug(f"Device configuration not yet available (attempt {attempt + 1})")
                    
                else:
                    logger.warning(f"Unexpected response from backend: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Network error while polling: {e}")
            except Exception as e:
                logger.error(f"Error during configuration polling: {e}")
                
            attempt += 1
            time.sleep(poll_interval)
            
        if attempt >= max_attempts:
            logger.warning("Maximum polling attempts reached, device configuration not received")
        elif self.stop_polling:
            logger.info("Configuration polling stopped")
            
    def _validate_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate received configuration data"""
        required_fields = ['device_pubkey', 'camera_pda', 'subdomain', 'full_domain']
        
        for field in required_fields:
            if field not in config_data:
                logger.error(f"Missing required field in configuration: {field}")
                return False
                
        # Verify device public key matches ours
        if config_data['device_pubkey'] != self.device_signer.get_public_key():
            logger.error("Device public key mismatch in configuration")
            return False
            
        return True
        
    def _configure_tunnel(self, config_data: Dict[str, Any]) -> bool:
        """Configure Cloudflare tunnel with PDA subdomain"""
        try:
            full_domain = config_data['full_domain']
            
            logger.info(f"Configuring Cloudflare tunnel for domain: {full_domain}")
            
            # Use tunnel manager to configure the tunnel
            if self.tunnel_manager.configure_tunnel(full_domain, service_port=5002):
                logger.info("Tunnel configured successfully")
                
                # Test tunnel connectivity
                time.sleep(5)  # Give tunnel time to start
                if self.tunnel_manager.test_tunnel_connectivity(full_domain):
                    logger.info("Tunnel connectivity test passed")
                else:
                    logger.warning("Tunnel connectivity test failed, but configuration completed")
                
                # Save configuration to persistent storage
                self._save_device_config(config_data)
                return True
            else:
                logger.error("Failed to configure tunnel")
                return False
            
        except Exception as e:
            logger.error(f"Failed to configure tunnel: {e}")
            return False
            
    def _save_device_config(self, config_data: Dict[str, Any]):
        """Save device configuration and update Docker environment"""
        try:
            # Save config to persistent storage
            config_path = '/home/azuolas/mmoment/app/orin_nano/data/config/device_config.json'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Device configuration saved to {config_path}")
            
            # Update docker-compose.yml with CAMERA_PDA
            self._update_docker_compose_pda(config_data['camera_pda'])
            
            # Restart camera service to pick up new PDA
            self._restart_camera_service()
            
        except Exception as e:
            logger.error(f"Failed to save device configuration: {e}")
            
    def _update_docker_compose_pda(self, camera_pda: str):
        """Update docker-compose.yml with the camera PDA"""
        try:
            compose_path = '/home/azuolas/mmoment/app/orin_nano/docker-compose.yml'
            
            with open(compose_path, 'r') as f:
                content = f.read()
            
            # Replace the CAMERA_PDA comment with actual PDA
            old_line = '      # CAMERA_PDA will be set dynamically after device registration'
            new_line = f'      - CAMERA_PDA={camera_pda}'
            
            if old_line in content:
                content = content.replace(old_line, new_line)
                
                with open(compose_path, 'w') as f:
                    f.write(content)
                    
                logger.info(f"Updated docker-compose.yml with CAMERA_PDA: {camera_pda}")
                return True
            else:
                logger.warning("Could not find CAMERA_PDA comment line in docker-compose.yml")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update docker-compose.yml: {e}")
            return False
            
    def _restart_camera_service(self):
        """Restart camera service to pick up new environment variables"""
        try:
            import subprocess
            
            # Change to docker-compose directory
            compose_dir = '/home/azuolas/mmoment/app/orin_nano'
            
            # Restart camera service
            result = subprocess.run(
                ['docker', 'compose', 'restart', 'camera-service'],
                cwd=compose_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("Camera service restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart camera service: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Camera service restart timed out")
            return False
        except Exception as e:
            logger.error(f"Exception restarting camera service: {e}")
            return False
            
    def get_device_config(self) -> Optional[Dict[str, Any]]:
        """Get current device configuration"""
        return self.device_config
        
    def is_configured(self) -> bool:
        """Check if device is configured and tunnel is active"""
        return self.config_received and self.device_config is not None
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information for API responses"""
        base_info = self.device_signer.get_device_info()
        
        # Add tunnel status
        tunnel_status = self.tunnel_manager.get_tunnel_status()
        
        if self.device_config:
            base_info.update({
                'camera_pda': self.device_config.get('camera_pda'),
                'subdomain': self.device_config.get('subdomain'),
                'full_domain': self.device_config.get('full_domain'),
                'tunnel_configured': True,
                'registration_complete': True
            })
        else:
            base_info.update({
                'tunnel_configured': False,
                'registration_complete': False,
                'status': 'waiting_for_configuration'
            })
            
        # Add tunnel status information
        base_info['tunnel_status'] = tunnel_status
            
        return base_info

# Singleton instance
_device_registration_service = None

def get_device_registration_service() -> DeviceRegistrationService:
    """Get the singleton device registration service instance"""
    global _device_registration_service
    if _device_registration_service is None:
        _device_registration_service = DeviceRegistrationService()
    return _device_registration_service