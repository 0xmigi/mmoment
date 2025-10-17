import psutil
import logging
from functools import wraps
from flask import jsonify
import os

logger = logging.getLogger(__name__)

class SystemChecks:
    # Critical thresholds
    TEMP_CRITICAL = 80.0  # Celsius
    CPU_CRITICAL = 90.0   # Percent
    MEM_CRITICAL = 90.0   # Percent
    
    # Warning thresholds
    TEMP_WARNING = 75.0
    CPU_WARNING = 80.0
    MEM_WARNING = 80.0

    @staticmethod
    def get_cpu_temp():
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0

    @staticmethod
    def get_system_status():
        """Get current system status"""
        try:
            cpu_temp = SystemChecks.get_cpu_temp()
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            
            status = {
                "temperature": cpu_temp,
                "cpu_usage": cpu_percent,
                "memory_usage": mem.percent,
                "is_critical": False,
                "warnings": []
            }
            
            # Check critical thresholds
            if cpu_temp >= SystemChecks.TEMP_CRITICAL:
                status["is_critical"] = True
                status["warnings"].append("Temperature critical")
            elif cpu_temp >= SystemChecks.TEMP_WARNING:
                status["warnings"].append("Temperature high")
                
            if cpu_percent >= SystemChecks.CPU_CRITICAL:
                status["is_critical"] = True
                status["warnings"].append("CPU usage critical")
            elif cpu_percent >= SystemChecks.CPU_WARNING:
                status["warnings"].append("CPU usage high")
                
            if mem.percent >= SystemChecks.MEM_CRITICAL:
                status["is_critical"] = True
                status["warnings"].append("Memory usage critical")
            elif mem.percent >= SystemChecks.MEM_WARNING:
                status["warnings"].append("Memory usage high")
                
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "is_critical": True,
                "warnings": ["System status check failed"]
            }

def check_system_resources(priority_level):
    """
    Decorator to check system resources before handling request
    Priority levels:
    1 = Buffer (Most important)
    2 = Camera control
    3 = Video recording
    4 = Picture taking
    5 = Streaming
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            status = SystemChecks.get_system_status()
            
            # Always allow buffer operations
            if priority_level == 1:
                return f(*args, **kwargs)
                
            # If system is critical, only allow priority 1-2
            if status["is_critical"] and priority_level > 2:
                return jsonify({
                    "error": "System resources critical",
                    "warnings": status["warnings"]
                }), 503
                
            # If high temperature/resources, restrict based on priority
            if status["warnings"] and priority_level > 3:
                return jsonify({
                    "error": "System resources constrained",
                    "warnings": status["warnings"]
                }), 503
                
            return f(*args, **kwargs)
        return wrapped
    return decorator