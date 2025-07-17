"""
Docker management tools for MMOMENT MCP server
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Optional

import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)

class DockerTools:
    """Tools for managing Docker containers on Jetson"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            self.client = None
    
    async def get_jetson_status(self) -> Dict:
        """Get status of all MMOMENT Docker containers"""
        if not self.client:
            return {"error": "Docker client not available"}
        
        try:
            # Get containers from docker-compose
            containers = self.client.containers.list(all=True)
            
            mmoment_containers = []
            for container in containers:
                # Filter for MMOMENT containers
                if any(service in container.name for service in 
                       ['camera-service', 'biometric-security', 'solana-middleware']):
                    
                    # Get container stats
                    stats = {
                        'name': container.name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'created': container.attrs['Created'],
                        'ports': container.attrs['NetworkSettings']['Ports'] if 'NetworkSettings' in container.attrs else {}
                    }
                    
                    # Get resource usage if running
                    if container.status == 'running':
                        try:
                            container_stats = container.stats(stream=False)
                            stats['cpu_percent'] = self._calculate_cpu_percent(container_stats)
                            stats['memory_usage'] = self._calculate_memory_usage(container_stats)
                        except Exception as e:
                            logger.warning(f"Failed to get stats for {container.name}: {e}")
                    
                    mmoment_containers.append(stats)
            
            return {
                "containers": mmoment_containers,
                "total_containers": len(mmoment_containers),
                "running_containers": len([c for c in mmoment_containers if c['status'] == 'running'])
            }
            
        except Exception as e:
            logger.error(f"Error getting Jetson status: {e}")
            return {"error": str(e)}
    
    async def restart_jetson_service(self, service_name: str) -> Dict:
        """Restart a specific Jetson Docker service"""
        if not self.client:
            return {"error": "Docker client not available"}
        
        try:
            # Map service names to container names
            service_mapping = {
                'camera-service': 'camera-service',
                'biometric-security': 'biometric-security',
                'solana-middleware': 'solana-middleware'
            }
            
            if service_name not in service_mapping:
                return {"error": f"Unknown service: {service_name}"}
            
            container_name = service_mapping[service_name]
            
            # Find the container
            containers = self.client.containers.list(all=True)
            target_container = None
            
            for container in containers:
                if container_name in container.name:
                    target_container = container
                    break
            
            if not target_container:
                return {"error": f"Container {container_name} not found"}
            
            # Restart the container
            logger.info(f"Restarting container: {target_container.name}")
            target_container.restart()
            
            # Wait a moment and check status
            await asyncio.sleep(2)
            target_container.reload()
            
            return {
                "service": service_name,
                "container": target_container.name,
                "status": target_container.status,
                "message": f"Service {service_name} restarted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
            return {"error": str(e)}
    
    async def get_container_logs(self, container_name: str, lines: int = 50) -> Dict:
        """Get logs from a specific container"""
        if not self.client:
            return {"error": "Docker client not available"}
        
        try:
            # Find the container
            containers = self.client.containers.list(all=True)
            target_container = None
            
            for container in containers:
                if container_name in container.name:
                    target_container = container
                    break
            
            if not target_container:
                return {"error": f"Container {container_name} not found"}
            
            # Get logs
            logs = target_container.logs(tail=lines, timestamps=True).decode('utf-8')
            
            return {
                "container": target_container.name,
                "lines": lines,
                "logs": logs
            }
            
        except Exception as e:
            logger.error(f"Error getting logs for {container_name}: {e}")
            return {"error": str(e)}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from container stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return round(cpu_percent, 2)
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_memory_usage(self, stats: Dict) -> Dict:
        """Calculate memory usage from container stats"""
        try:
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                "usage_bytes": memory_usage,
                "limit_bytes": memory_limit,
                "usage_mb": round(memory_usage / (1024 * 1024), 2),
                "limit_mb": round(memory_limit / (1024 * 1024), 2),
                "percent": round(memory_percent, 2)
            }
        except KeyError:
            return {"error": "Memory stats not available"}