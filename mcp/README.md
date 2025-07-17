# MMOMENT MCP Integration

This directory contains the Model Context Protocol (MCP) server and client configurations for coordinating Claude instances across the M1 Mac and Jetson Orin Nano environments.

## Overview

The MCP setup enables seamless coordination between Claude instances running on:
- **Mac**: Frontend development (React/TypeScript)
- **Jetson**: Backend services (Docker containers)

## Architecture

```
Mac Claude ←→ MCP Server (Jetson) ←→ Jetson Claude
```

### MCP Server (Jetson)
- Runs on Jetson Orin Nano
- Provides tools for Docker container management
- Monitors system health and logs
- Coordinates cross-environment actions

### MCP Clients (Mac & Jetson)
- Connect to MCP server
- Access shared tools and context
- Enable automatic cross-environment problem solving

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation instructions.

### 1. Install Dependencies (Jetson):
```bash
cd ~/mmoment/mcp/server
pip install -r requirements.txt
```

### 2. Configure Claude (Both Machines):
```bash
# Mac: Copy mac-config/claude_settings.json to ~/.config/claude/settings.json
# Jetson: Copy jetson-config/claude_settings.json to ~/.config/claude/settings.json
```

### 3. Start MCP Server (Jetson):
```bash
python mcp_server.py
```

## Available MCP Tools

### Docker Management
- `get_jetson_status()` - Container health, logs, metrics
- `restart_jetson_service(service_name)` - Restart specific containers
- `get_container_logs(container_name)` - Stream container logs

### Cross-Environment Testing
- `test_connection(frontend_url, jetson_endpoint)` - End-to-end connectivity test
- `sync_project_state()` - Git status and recent changes
- `validate_integration()` - Full stack testing

### System Monitoring
- `get_system_metrics()` - GPU, memory, CPU usage
- `get_service_health()` - All three container status
- `get_error_context()` - Recent errors and debugging info

## Usage

Once configured, use Claude normally. The MCP tools are available transparently:

```
User: "The frontend can't connect to the camera service"
Claude: "Let me check both the frontend code and Jetson container status..."
[Automatically uses MCP tools to diagnose across both environments]
```

## Directory Structure

```
mcp/
├── server/                 # MCP server (runs on Jetson)
├── mac-config/            # Mac-side MCP client config
├── jetson-config/         # Jetson-side MCP client config
└── README.md
```