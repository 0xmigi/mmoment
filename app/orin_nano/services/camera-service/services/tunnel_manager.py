import os
import json
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from services.cloudflare_dns import get_cloudflare_dns_manager

logger = logging.getLogger(__name__)

class TunnelManager:
    """
    Manages Cloudflare tunnel configuration for dynamic PDA-based subdomains.
    Updates tunnel configuration when device receives its assigned PDA from backend.
    """
    
    def __init__(self):
        self.config_path = '/home/azuolas/.cloudflared/config.yml'
        self.tunnel_id = os.getenv('CLOUDFLARE_TUNNEL_ID', '6257e873-7943-4b85-b8a3-72b5b9d0a500')
        self.credentials_file = f'/home/azuolas/.cloudflared/{self.tunnel_id}.json'
        self.dns_manager = get_cloudflare_dns_manager()
        
    def configure_tunnel(self, full_domain: str, service_port: int = 5002) -> bool:
        """
        Configure Cloudflare tunnel with the assigned PDA subdomain and create DNS record.
        
        Args:
            full_domain: The full domain (e.g., '9naul1kfaeckddqymtewaaauoevvwpbxkdhtd86hv3xby.mmoment.xyz')
            service_port: Local service port to tunnel to
            
        Returns:
            bool: True if configuration successful, False otherwise
        """
        try:
            if not self.tunnel_id:
                logger.error("CLOUDFLARE_TUNNEL_ID environment variable not set")
                return False
                
            logger.info(f"Configuring tunnel for domain: {full_domain}")
            
            # Extract subdomain from full domain
            subdomain = full_domain.split('.')[0]
            logger.info(f"PDA subdomain: {subdomain}")
            
            # Step 1: Create DNS record automatically if DNS manager is available
            if self.dns_manager.is_available():
                logger.info("Creating DNS record automatically...")
                if self.dns_manager.create_pda_dns_record(subdomain):
                    logger.info("✅ DNS record created successfully")
                else:
                    logger.warning("❌ Failed to create DNS record - tunnel will still be configured")
            else:
                logger.warning("Cloudflare DNS API not configured - manual DNS setup required")
            
            # Step 2: Create tunnel configuration
            tunnel_config = {
                'tunnel': self.tunnel_id,
                'credentials-file': self.credentials_file,
                'ingress': [
                    {
                        'hostname': full_domain,
                        'service': f'http://localhost:{service_port}'
                    },
                    {
                        'service': 'http_status:404'
                    }
                ]
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Write configuration file
            with open(self.config_path, 'w') as f:
                yaml.dump(tunnel_config, f, default_flow_style=False)
                
            logger.info(f"Tunnel configuration written to {self.config_path}")
            
            # Step 3: Restart cloudflared service
            if self._restart_tunnel_service():
                logger.info("Cloudflare tunnel configured and restarted successfully")
                
                # Step 4: Verify DNS propagation if DNS was created
                if self.dns_manager.is_available():
                    import time
                    logger.info("Waiting for DNS propagation...")
                    time.sleep(10)  # Give DNS time to propagate
                    
                    if self.dns_manager.verify_dns_propagation(full_domain):
                        logger.info("✅ DNS propagation verified - camera is online!")
                    else:
                        logger.info("⏳ DNS still propagating - camera will be online shortly")
                
                return True
            else:
                logger.error("Failed to restart tunnel service")
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure tunnel: {e}")
            return False
            
    def _restart_tunnel_service(self) -> bool:
        """Restart the cloudflared service"""
        try:
            # Try systemctl first (most common)
            try:
                result = subprocess.run(['sudo', 'systemctl', 'restart', 'cloudflared'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("Cloudflared service restarted via systemctl")
                    return True
                else:
                    logger.warning(f"systemctl restart failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning("systemctl restart timed out")
            except FileNotFoundError:
                logger.info("systemctl not found, trying alternative methods")
                
            # Try docker container restart (if running in container mode)
            try:
                result = subprocess.run(['docker', 'restart', 'cloudflared'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("Cloudflared container restarted")
                    return True
                else:
                    logger.warning(f"docker restart failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning("docker restart timed out")
            except FileNotFoundError:
                logger.info("docker not found")
                
            # Try killing and restarting cloudflared process
            try:
                # Kill existing process
                subprocess.run(['sudo', 'pkill', 'cloudflared'], 
                             capture_output=True, text=True, timeout=10)
                
                # Start new process in background
                subprocess.Popen(['sudo', 'cloudflared', 'tunnel', 'run', self.tunnel_id],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                logger.info("Cloudflared process restarted manually")
                return True
                
            except Exception as e:
                logger.warning(f"Manual restart failed: {e}")
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to restart tunnel service: {e}")
            return False
            
    def get_tunnel_status(self) -> Dict[str, Any]:
        """Get current tunnel status and configuration"""
        try:
            status = {
                'configured': False,
                'running': False,
                'config_exists': os.path.exists(self.config_path),
                'tunnel_id': self.tunnel_id,
                'error': None
            }
            
            # Check if config file exists and read it
            if status['config_exists']:
                try:
                    with open(self.config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        status['configured'] = True
                        status['config'] = config_data
                        
                        # Extract hostname from config
                        if 'ingress' in config_data and len(config_data['ingress']) > 0:
                            first_rule = config_data['ingress'][0]
                            if 'hostname' in first_rule:
                                status['hostname'] = first_rule['hostname']
                                
                except Exception as e:
                    status['error'] = f"Failed to read config: {e}"
                    
            # Check if tunnel process is running
            try:
                result = subprocess.run(['pgrep', '-f', 'cloudflared'], 
                                      capture_output=True, text=True, timeout=5)
                status['running'] = result.returncode == 0
            except Exception as e:
                status['error'] = f"Failed to check process: {e}"
                
            return status
            
        except Exception as e:
            logger.error(f"Failed to get tunnel status: {e}")
            return {'error': f"Status check failed: {e}"}
            
    def test_tunnel_connectivity(self, domain: str) -> bool:
        """Test if the tunnel is accessible from the configured domain"""
        try:
            import requests
            
            # Test the health endpoint through the tunnel
            test_url = f"https://{domain}/api/health"
            
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"Tunnel connectivity test successful: {domain}")
                return True
            else:
                logger.warning(f"Tunnel test failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Tunnel connectivity test failed: {e}")
            return False
            
    def backup_current_config(self) -> Optional[str]:
        """Backup current tunnel configuration"""
        try:
            if not os.path.exists(self.config_path):
                return None
                
            backup_path = f"{self.config_path}.backup"
            
            with open(self.config_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
                    
            logger.info(f"Tunnel config backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup config: {e}")
            return None
            
    def restore_config(self, backup_path: str) -> bool:
        """Restore tunnel configuration from backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
                
            with open(backup_path, 'r') as src:
                with open(self.config_path, 'w') as dst:
                    dst.write(src.read())
                    
            logger.info(f"Tunnel config restored from {backup_path}")
            return self._restart_tunnel_service()
            
        except Exception as e:
            logger.error(f"Failed to restore config: {e}")
            return False

# Singleton instance
_tunnel_manager = None

def get_tunnel_manager() -> TunnelManager:
    """Get the singleton tunnel manager instance"""
    global _tunnel_manager
    if _tunnel_manager is None:
        _tunnel_manager = TunnelManager()
    return _tunnel_manager