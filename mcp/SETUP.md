# MMOMENT MCP Setup Guide

This guide will help you set up the Model Context Protocol (MCP) server for coordinating Claude instances between your Mac and Jetson Orin Nano.

## Prerequisites

- Python 3.8+ on Jetson Orin Nano
- SSH access to Jetson from Mac
- Docker running on Jetson
- Claude Code installed on both Mac and Jetson

## Installation Steps

### 1. Install MCP Server Dependencies (Jetson)

```bash
# SSH into your Jetson
ssh jetson.mmoment.xyz

# Navigate to the project
cd ~/mmoment/mcp/server

# Install Python dependencies
pip install -r requirements.txt

# Test the server
python mcp_server.py --test
```

### 2. Configure Claude on Mac

```bash
# Copy the Mac configuration to Claude settings
cp ~/mmoment/mcp/mac-config/claude_settings.json ~/.config/claude/settings.json

# Or merge with existing settings if you have them
```

### 3. Configure Claude on Jetson

```bash
# On Jetson, copy the Jetson configuration
cp ~/mmoment/mcp/jetson-config/claude_settings.json ~/.config/claude/settings.json
```

### 4. Start the MCP Server

```bash
# On Jetson, start the MCP server
cd ~/mmoment/mcp/server
python mcp_server.py
```

The server will run in the background and provide tools to both Claude instances.

## Testing the Setup

### Test 1: Basic Connection
On Mac, ask Claude:
```
"Can you check the status of the Jetson containers?"
```

Claude should be able to run the `get_jetson_status` tool automatically.

### Test 2: Cross-Environment Debugging
```
"The frontend can't connect to the camera service. Can you check both sides?"
```

Claude should check frontend code on Mac and container status on Jetson.

### Test 3: Service Management
```
"Restart the camera service and test the connection"
```

Claude should restart the Docker container and verify connectivity.

## Available MCP Tools

Once configured, these tools are available to both Claude instances:

### Docker Management
- `get_jetson_status()` - Status of all containers
- `restart_jetson_service(service_name)` - Restart specific service
- `get_container_logs(container_name, lines)` - Get container logs

### System Monitoring
- `get_system_metrics()` - GPU, CPU, memory usage
- `sync_project_state()` - Git status and changes
- `validate_integration()` - Full stack health check

### Testing Tools
- `test_connection(frontend_url, jetson_endpoint)` - Connectivity test
- Custom MMOMENT-specific debugging tools

## Troubleshooting

### MCP Server Won't Start
```bash
# Check Python dependencies
pip list | grep mcp

# Check Docker access
docker ps

# Check permissions
ls -la ~/mmoment/mcp/server/
```

### Claude Can't Connect to MCP
```bash
# Verify SSH connection from Mac
ssh jetson.mmoment.xyz "cd ~/mmoment/mcp/server && python mcp_server.py --test"

# Check Claude settings
cat ~/.config/claude/settings.json
```

### Tools Not Available
- Restart Claude Code on both machines
- Check MCP server logs
- Verify network connectivity

## Usage Examples

### Normal Development Workflow
```
You: "I'm getting a 500 error when trying to enroll a face"
Claude: "Let me check the frontend error and also look at the camera service logs on Jetson..."
[Uses MCP tools automatically to diagnose both sides]
```

### Deployment Coordination
```
You: "Deploy the latest changes to Jetson"
Claude: "I'll push the git changes and restart the necessary containers..."
[Coordinates git push and Docker restart across environments]
```

### System Health Check
```
You: "How is the system performing?"
Claude: "Let me check the system metrics and container health..."
[Gets comprehensive status from both Mac and Jetson]
```

## Advanced Configuration

### Custom Tools
You can add custom MCP tools by modifying:
- `mcp/server/tools/project_tools.py` - Project-specific tools
- `mcp/server/tools/docker_tools.py` - Docker management
- `mcp/server/tools/jetson_tools.py` - System monitoring

### Security Considerations
- MCP server runs locally on Jetson
- SSH connections are secured with your existing keys
- No external network access required
- All communication is encrypted via SSH

## Next Steps

Once MCP is working, you can:
1. Add custom tools for your specific workflows
2. Integrate with your CI/CD pipeline
3. Set up monitoring and alerting
4. Create automated deployment scripts

The MCP integration will make your cross-environment development much more seamless!