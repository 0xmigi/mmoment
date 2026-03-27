#!/bin/bash

# Cloudflare Tunnel Restore Script
# This script restores a Cloudflare tunnel from backed up configuration files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Cloudflare Tunnel Restore Script${NC}"
echo "This script will restore your Cloudflare tunnel configuration from backup files"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo -e "${YELLOW}Cloudflared not installed. Installing...${NC}"
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
    sudo dpkg -i cloudflared-linux-arm64.deb
    rm cloudflared-linux-arm64.deb
else
    echo -e "${GREEN}Cloudflared already installed: $(cloudflared version)${NC}"
fi

# Directory for cloudflared config
mkdir -p ~/.cloudflared

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for config file
CONFIG_FILE="${SCRIPT_DIR}/config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found at ${CONFIG_FILE}${NC}"
    echo "Please run setup.sh first to create the configuration files."
    exit 1
fi

# Extract tunnel ID from config
TUNNEL_ID=$(grep -oP 'tunnel: \K[0-9a-f-]+' "$CONFIG_FILE")
CREDS_FILE="${SCRIPT_DIR}/${TUNNEL_ID}.json"

if [ ! -f "$CREDS_FILE" ]; then
    echo -e "${RED}Error: Credentials file not found at ${CREDS_FILE}${NC}"
    echo "Please run setup.sh first to create the configuration files."
    exit 1
fi

echo -e "${YELLOW}Found tunnel configuration with ID: ${TUNNEL_ID}${NC}"

# Copy config to cloudflared directory
echo -e "${YELLOW}Restoring configuration files...${NC}"
cp "$CONFIG_FILE" ~/.cloudflared/config.yml
cp "$CREDS_FILE" ~/.cloudflared/

# Install service file
echo -e "${YELLOW}Installing systemd service...${NC}"
sudo cp "${SCRIPT_DIR}/cloudflare-tunnel.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cloudflare-tunnel.service

# Start service
echo -e "${YELLOW}Starting Cloudflare tunnel service...${NC}"
sudo systemctl restart cloudflare-tunnel.service
sudo systemctl status cloudflare-tunnel.service

# Extract hostname from config
HOSTNAME=$(grep -oP 'hostname: \K[^\s]+' "$CONFIG_FILE")

echo -e "\n${GREEN}Restore complete!${NC}"
echo "Your camera API should now be accessible at: https://${HOSTNAME}"
echo "To check tunnel status: sudo systemctl status cloudflare-tunnel.service"
echo "To restart tunnel: sudo systemctl restart cloudflare-tunnel.service" 