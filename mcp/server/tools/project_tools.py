"""
Project-specific tools for MMOMENT MCP server
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

class ProjectTools:
    """Tools for MMOMENT project coordination and testing"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.services = {
            'camera-service': 'http://localhost:5002',
            'biometric-security': 'http://localhost:5003',
            'solana-middleware': 'http://localhost:5001'
        }
    
    def _find_project_root(self) -> str:
        """Find the MMOMENT project root directory"""
        current_dir = os.getcwd()
        
        # Look for project markers
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir, 'Anchor.toml')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # Fallback to current directory
        return os.getcwd()
    
    async def test_connection(self, frontend_url: Optional[str], jetson_endpoint: str) -> Dict:
        """Test connection between frontend and Jetson services"""
        try:
            results = {}
            
            # Test Jetson endpoint
            jetson_result = await self._test_endpoint(jetson_endpoint)
            results['jetson_endpoint'] = jetson_result
            
            # Test all MMOMENT services
            for service_name, service_url in self.services.items():
                service_result = await self._test_endpoint(service_url)
                results[service_name] = service_result
            
            # Test frontend if provided
            if frontend_url:
                frontend_result = await self._test_endpoint(frontend_url)
                results['frontend'] = frontend_result
            
            # Overall health check
            healthy_services = sum(1 for result in results.values() if result.get('status') == 'healthy')
            total_services = len(results)
            
            return {
                "connection_test": results,
                "summary": {
                    "healthy_services": healthy_services,
                    "total_services": total_services,
                    "overall_health": "healthy" if healthy_services == total_services else "degraded"
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing connections: {e}")
            return {"error": str(e)}
    
    async def _test_endpoint(self, url: str) -> Dict:
        """Test a single endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    return {
                        "url": url,
                        "status": "healthy",
                        "http_status": response.status,
                        "response_time": response.headers.get('X-Response-Time', 'unknown')
                    }
        except asyncio.TimeoutError:
            return {
                "url": url,
                "status": "timeout",
                "error": "Connection timeout"
            }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }
    
    async def sync_project_state(self) -> Dict:
        """Get git status and recent changes"""
        try:
            # Get git status
            git_status = await self._run_command("git status --porcelain")
            
            # Get recent commits
            git_log = await self._run_command("git log --oneline -10")
            
            # Get current branch
            current_branch = await self._run_command("git rev-parse --abbrev-ref HEAD")
            
            # Get uncommitted changes
            git_diff = await self._run_command("git diff --name-only")
            
            return {
                "git_status": {
                    "branch": current_branch.strip() if current_branch else "unknown",
                    "uncommitted_files": git_status.strip().split('\n') if git_status and git_status.strip() else [],
                    "modified_files": git_diff.strip().split('\n') if git_diff and git_diff.strip() else [],
                    "recent_commits": git_log.strip().split('\n') if git_log else []
                },
                "project_root": self.project_root
            }
            
        except Exception as e:
            logger.error(f"Error syncing project state: {e}")
            return {"error": str(e)}
    
    async def validate_integration(self) -> Dict:
        """Run comprehensive integration tests"""
        try:
            results = {}
            
            # Test Docker services
            docker_test = await self._test_docker_services()
            results['docker_services'] = docker_test
            
            # Test API endpoints
            api_test = await self._test_api_endpoints()
            results['api_endpoints'] = api_test
            
            # Test camera functionality
            camera_test = await self._test_camera_functionality()
            results['camera_functionality'] = camera_test
            
            # Test blockchain connectivity
            blockchain_test = await self._test_blockchain_connectivity()
            results['blockchain_connectivity'] = blockchain_test
            
            # Calculate overall health
            healthy_categories = sum(1 for test in results.values() if test.get('status') == 'healthy')
            total_categories = len(results)
            
            return {
                "integration_tests": results,
                "summary": {
                    "healthy_categories": healthy_categories,
                    "total_categories": total_categories,
                    "overall_status": "healthy" if healthy_categories == total_categories else "degraded"
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating integration: {e}")
            return {"error": str(e)}
    
    async def _test_docker_services(self) -> Dict:
        """Test Docker services are running"""
        try:
            # Check if docker-compose services are up
            result = await self._run_command("docker-compose ps")
            
            if result:
                # Parse docker-compose output
                lines = result.strip().split('\n')[1:]  # Skip header
                services = []
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            services.append({
                                'name': parts[0],
                                'status': parts[1] if len(parts) > 1 else 'unknown'
                            })
                
                return {
                    "status": "healthy",
                    "services": services
                }
            
            return {"status": "error", "message": "No docker-compose services found"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _test_api_endpoints(self) -> Dict:
        """Test all API endpoints"""
        try:
            endpoint_tests = {}
            
            # Test health endpoints for each service
            health_endpoints = {
                'camera-service': 'http://localhost:5002/health',
                'biometric-security': 'http://localhost:5003/health',
                'solana-middleware': 'http://localhost:5001/health'
            }
            
            for service, endpoint in health_endpoints.items():
                test_result = await self._test_endpoint(endpoint)
                endpoint_tests[service] = test_result
            
            # Overall API health
            healthy_apis = sum(1 for test in endpoint_tests.values() if test.get('status') == 'healthy')
            total_apis = len(endpoint_tests)
            
            return {
                "status": "healthy" if healthy_apis == total_apis else "degraded",
                "endpoints": endpoint_tests,
                "summary": f"{healthy_apis}/{total_apis} APIs healthy"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _test_camera_functionality(self) -> Dict:
        """Test camera functionality"""
        try:
            # Test camera device availability
            camera_devices = await self._run_command("ls -la /dev/video*")
            
            # Test camera service API
            camera_status = await self._test_endpoint("http://localhost:5002/status")
            
            return {
                "status": "healthy" if camera_status.get('status') == 'healthy' else "degraded",
                "camera_devices": camera_devices.strip().split('\n') if camera_devices else [],
                "service_status": camera_status
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _test_blockchain_connectivity(self) -> Dict:
        """Test Solana blockchain connectivity"""
        try:
            # Test Solana middleware
            solana_status = await self._test_endpoint("http://localhost:5001/status")
            
            # Test devnet connectivity
            devnet_test = await self._run_command("solana cluster-version --url devnet")
            
            return {
                "status": "healthy" if solana_status.get('status') == 'healthy' else "degraded",
                "middleware_status": solana_status,
                "devnet_connection": devnet_test.strip() if devnet_test else "unavailable"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _run_command(self, command: str) -> Optional[str]:
        """Run a system command and return output"""
        try:
            # Change to project root directory
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                logger.warning(f"Command failed: {command}, stderr: {stderr.decode('utf-8')}")
                return None
                
        except Exception as e:
            logger.error(f"Error running command '{command}': {e}")
            return None