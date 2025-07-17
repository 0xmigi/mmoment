#!/usr/bin/env python3
"""
MMOMENT MCP Server

Provides cross-environment coordination between Mac and Jetson Claude instances.
Runs on Jetson Orin Nano and provides tools for Docker container management,
system monitoring, and cross-environment testing.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from tools.docker_tools import DockerTools
from tools.jetson_tools import JetsonTools
from tools.project_tools import ProjectTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MMOMENTMCPServer:
    """Main MCP server for MMOMENT project coordination"""
    
    def __init__(self):
        self.server = Server("mmoment-mcp-server")
        self.docker_tools = DockerTools()
        self.jetson_tools = JetsonTools()
        self.project_tools = ProjectTools()
        
        # Register all tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools"""
        
        # Docker Management Tools
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="get_jetson_status",
                    description="Get status of all Jetson Docker containers",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="restart_jetson_service",
                    description="Restart a specific Jetson Docker service",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "Name of the service to restart (camera-service, biometric-security, solana-middleware)"
                            }
                        },
                        "required": ["service_name"]
                    }
                ),
                types.Tool(
                    name="get_container_logs",
                    description="Get logs from a specific container",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "container_name": {
                                "type": "string",
                                "description": "Name of the container to get logs from"
                            },
                            "lines": {
                                "type": "integer",
                                "description": "Number of recent log lines to retrieve",
                                "default": 50
                            }
                        },
                        "required": ["container_name"]
                    }
                ),
                types.Tool(
                    name="get_system_metrics",
                    description="Get Jetson system metrics (GPU, memory, CPU)",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="test_connection",
                    description="Test connection between frontend and Jetson services",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "frontend_url": {
                                "type": "string",
                                "description": "Frontend URL to test from"
                            },
                            "jetson_endpoint": {
                                "type": "string",
                                "description": "Jetson service endpoint to test"
                            }
                        },
                        "required": ["jetson_endpoint"]
                    }
                ),
                types.Tool(
                    name="sync_project_state",
                    description="Get git status and recent changes on Jetson",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="validate_integration",
                    description="Run full stack integration tests",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None
        ) -> list[types.TextContent]:
            """Handle tool calls"""
            
            if arguments is None:
                arguments = {}
                
            try:
                if name == "get_jetson_status":
                    result = await self.docker_tools.get_jetson_status()
                    
                elif name == "restart_jetson_service":
                    service_name = arguments.get("service_name")
                    result = await self.docker_tools.restart_jetson_service(service_name)
                    
                elif name == "get_container_logs":
                    container_name = arguments.get("container_name")
                    lines = arguments.get("lines", 50)
                    result = await self.docker_tools.get_container_logs(container_name, lines)
                    
                elif name == "get_system_metrics":
                    result = await self.jetson_tools.get_system_metrics()
                    
                elif name == "test_connection":
                    frontend_url = arguments.get("frontend_url")
                    jetson_endpoint = arguments.get("jetson_endpoint")
                    result = await self.project_tools.test_connection(frontend_url, jetson_endpoint)
                    
                elif name == "sync_project_state":
                    result = await self.project_tools.sync_project_state()
                    
                elif name == "validate_integration":
                    result = await self.project_tools.validate_integration()
                    
                else:
                    result = f"Unknown tool: {name}"
                    
                return [types.TextContent(type="text", text=str(result))]
                
            except Exception as e:
                logger.error(f"Error calling tool {name}: {str(e)}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main entry point"""
    server = MMOMENTMCPServer()
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mmoment-mcp-server",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())