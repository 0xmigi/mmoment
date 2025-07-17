"""
Jetson Orin Nano system monitoring tools
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, Optional

import psutil

logger = logging.getLogger(__name__)

class JetsonTools:
    """Tools for monitoring Jetson Orin Nano system resources"""
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
    
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson device"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return 'tegra' in f.read().lower()
        except FileNotFoundError:
            return False
    
    async def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        try:
            metrics = {
                "cpu": self._get_cpu_metrics(),
                "memory": self._get_memory_metrics(),
                "disk": self._get_disk_metrics(),
                "network": self._get_network_metrics(),
                "temperature": await self._get_temperature_metrics(),
                "gpu": await self._get_gpu_metrics() if self.is_jetson else None,
                "jetson_stats": await self._get_jetson_stats() if self.is_jetson else None
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}
    
    def _get_cpu_metrics(self) -> Dict:
        """Get CPU usage metrics"""
        try:
            return {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "per_cpu": psutil.cpu_percent(interval=1, percpu=True)
            }
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            return {"error": str(e)}
    
    def _get_memory_metrics(self) -> Dict:
        """Get memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "virtual": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2)
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "percent": swap.percent,
                    "total_gb": round(swap.total / (1024**3), 2),
                    "used_gb": round(swap.used / (1024**3), 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            return {"error": str(e)}
    
    def _get_disk_metrics(self) -> Dict:
        """Get disk usage metrics"""
        try:
            disk = psutil.disk_usage('/')
            
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            logger.error(f"Error getting disk metrics: {e}")
            return {"error": str(e)}
    
    def _get_network_metrics(self) -> Dict:
        """Get network interface metrics"""
        try:
            net_io = psutil.net_io_counters()
            
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "mb_sent": round(net_io.bytes_sent / (1024**2), 2),
                "mb_recv": round(net_io.bytes_recv / (1024**2), 2)
            }
        except Exception as e:
            logger.error(f"Error getting network metrics: {e}")
            return {"error": str(e)}
    
    async def _get_temperature_metrics(self) -> Dict:
        """Get temperature metrics"""
        try:
            if self.is_jetson:
                # Jetson-specific temperature monitoring
                result = await self._run_command("cat /sys/class/thermal/thermal_zone*/temp")
                if result:
                    temps = []
                    for line in result.strip().split('\n'):
                        if line.strip():
                            temp_millic = int(line.strip())
                            temp_celsius = temp_millic / 1000.0
                            temps.append(temp_celsius)
                    
                    return {
                        "thermal_zones": temps,
                        "max_temp": max(temps) if temps else None,
                        "avg_temp": sum(temps) / len(temps) if temps else None
                    }
            else:
                # Generic temperature monitoring
                temps = psutil.sensors_temperatures()
                return {"sensors": temps}
                
        except Exception as e:
            logger.error(f"Error getting temperature metrics: {e}")
            return {"error": str(e)}
    
    async def _get_gpu_metrics(self) -> Optional[Dict]:
        """Get GPU metrics (Jetson specific)"""
        if not self.is_jetson:
            return None
        
        try:
            # Try to get GPU stats from tegrastats
            result = await self._run_command("tegrastats --interval 1000 --logfile /tmp/tegrastats.log")
            
            # Also try nvidia-smi if available
            nvidia_smi = await self._run_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
            
            gpu_metrics = {}
            
            if nvidia_smi:
                lines = nvidia_smi.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_metrics[f"gpu_{i}"] = {
                            "utilization": float(parts[0].strip()),
                            "memory_used": float(parts[1].strip()),
                            "memory_total": float(parts[2].strip()),
                            "memory_percent": (float(parts[1].strip()) / float(parts[2].strip())) * 100
                        }
            
            return gpu_metrics if gpu_metrics else {"status": "GPU metrics not available"}
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {"error": str(e)}
    
    async def _get_jetson_stats(self) -> Optional[Dict]:
        """Get Jetson-specific system stats"""
        if not self.is_jetson:
            return None
        
        try:
            # Get Jetson model info
            model_info = await self._run_command("cat /proc/device-tree/model")
            
            # Get L4T version
            l4t_version = await self._run_command("cat /etc/nv_tegra_release")
            
            # Get CUDA version
            cuda_version = await self._run_command("nvcc --version")
            
            return {
                "model": model_info.strip() if model_info else "Unknown",
                "l4t_version": l4t_version.strip() if l4t_version else "Unknown",
                "cuda_version": cuda_version.strip() if cuda_version else "Unknown"
            }
            
        except Exception as e:
            logger.error(f"Error getting Jetson stats: {e}")
            return {"error": str(e)}
    
    async def _run_command(self, command: str) -> Optional[str]:
        """Run a system command and return output"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
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