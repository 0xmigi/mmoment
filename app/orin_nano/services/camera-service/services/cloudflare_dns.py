import os
import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class CloudflareDNSManager:
    """
    Manages Cloudflare DNS records for automatic PDA subdomain creation.
    This enables cameras to automatically get their PDA-based URLs without manual DNS configuration.
    """
    
    def __init__(self):
        self.api_token = os.getenv('CLOUDFLARE_API_TOKEN')
        self.zone_id = os.getenv('CLOUDFLARE_ZONE_ID')  # For mmoment.xyz
        self.tunnel_id = os.getenv('CLOUDFLARE_TUNNEL_ID', '6257e873-7943-4b85-b8a3-72b5b9d0a500')
        self.base_domain = 'mmoment.xyz'
        
        if not self.api_token:
            logger.warning("CLOUDFLARE_API_TOKEN not set - DNS automation disabled")
        if not self.zone_id:
            logger.warning("CLOUDFLARE_ZONE_ID not set - DNS automation disabled")
            
    def is_available(self) -> bool:
        """Check if Cloudflare DNS management is available"""
        return bool(self.api_token and self.zone_id)
        
    def create_pda_dns_record(self, pda_subdomain: str) -> bool:
        """
        Create DNS CNAME record for PDA subdomain pointing to tunnel.
        
        Args:
            pda_subdomain: The PDA-based subdomain (e.g., '9naul1kfaeckddqymtewaaauoevvwpbxkdhtd86hv3xby')
            
        Returns:
            bool: True if DNS record created successfully, False otherwise
        """
        if not self.is_available():
            logger.error("Cloudflare DNS not configured - cannot create DNS record")
            return False
            
        try:
            logger.info(f"Creating DNS record for PDA subdomain: {pda_subdomain}.{self.base_domain}")
            
            # Check if record already exists
            existing_record = self._get_dns_record(pda_subdomain)
            if existing_record:
                logger.info(f"DNS record already exists for {pda_subdomain}.{self.base_domain}")
                return True
            
            # Create new CNAME record
            record_data = {
                'type': 'CNAME',
                'name': pda_subdomain,
                'content': f'{self.tunnel_id}.cfargotunnel.com',
                'ttl': 1,  # Auto TTL
                'proxied': True  # Enable Cloudflare proxy (orange cloud)
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            url = f'https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records'
            response = requests.post(url, json=record_data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    record_id = result['result']['id']
                    logger.info(f"✅ DNS record created successfully: {pda_subdomain}.{self.base_domain} -> {self.tunnel_id}.cfargotunnel.com")
                    logger.info(f"   Record ID: {record_id}")
                    return True
                else:
                    logger.error(f"❌ Cloudflare API error: {result.get('errors', [])}")
                    return False
            else:
                logger.error(f"❌ HTTP error creating DNS record: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception creating DNS record: {e}")
            return False
            
    def _get_dns_record(self, subdomain: str) -> Optional[Dict[str, Any]]:
        """Get existing DNS record for subdomain"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            url = f'https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records'
            params = {
                'name': f'{subdomain}.{self.base_domain}',
                'type': 'CNAME'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and result.get('result'):
                    return result['result'][0]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error checking existing DNS record: {e}")
            return None
            
    def delete_pda_dns_record(self, pda_subdomain: str) -> bool:
        """Delete DNS record for PDA subdomain"""
        if not self.is_available():
            logger.error("Cloudflare DNS not configured")
            return False
            
        try:
            # Get existing record
            existing_record = self._get_dns_record(pda_subdomain)
            if not existing_record:
                logger.info(f"DNS record does not exist for {pda_subdomain}.{self.base_domain}")
                return True
                
            record_id = existing_record['id']
            
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
            
            url = f'https://api.cloudflare.com/client/v4/zones/{self.zone_id}/dns_records/{record_id}'
            response = requests.delete(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.info(f"✅ DNS record deleted: {pda_subdomain}.{self.base_domain}")
                    return True
                else:
                    logger.error(f"❌ Cloudflare API error deleting record: {result.get('errors', [])}")
                    return False
            else:
                logger.error(f"❌ HTTP error deleting DNS record: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception deleting DNS record: {e}")
            return False
            
    def verify_dns_propagation(self, full_domain: str) -> bool:
        """Verify that DNS record has propagated and is accessible"""
        try:
            # Simple HTTP check to see if domain resolves
            test_url = f"https://{full_domain}/api/health"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ DNS propagation verified: {full_domain} is accessible")
                return True
            else:
                logger.info(f"⏳ DNS still propagating: {full_domain} returned {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.info(f"⏳ DNS still propagating: {full_domain} not yet resolvable")
            return False
        except Exception as e:
            logger.warning(f"DNS verification error: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get Cloudflare DNS manager status"""
        return {
            'available': self.is_available(),
            'api_token_set': bool(self.api_token),
            'zone_id_set': bool(self.zone_id),
            'tunnel_id': self.tunnel_id,
            'base_domain': self.base_domain
        }

# Singleton instance
_cloudflare_dns_manager = None

def get_cloudflare_dns_manager() -> CloudflareDNSManager:
    """Get the singleton Cloudflare DNS manager instance"""
    global _cloudflare_dns_manager
    if _cloudflare_dns_manager is None:
        _cloudflare_dns_manager = CloudflareDNSManager()
    return _cloudflare_dns_manager