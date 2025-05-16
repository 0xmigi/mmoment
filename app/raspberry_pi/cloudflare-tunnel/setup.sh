#!/bin/bash

# Cloudflare Tunnel Setup Script
# This script automates the setup of a Cloudflare tunnel for the camera API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Cloudflare Tunnel Setup Script${NC}"
echo "This script will help you set up a Cloudflare tunnel for your camera API"
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

# Ask for information
echo -e "\n${YELLOW}Please provide the following information:${NC}"

read -p "Enter your tunnel name (e.g., camera-api-tunnel): " TUNNEL_NAME
read -p "Enter your domain (e.g., yourdomain.com): " DOMAIN
read -p "Enter subdomain for camera API (e.g., camera): " SUBDOMAIN
read -p "Enter API port (default: 5000): " API_PORT
API_PORT=${API_PORT:-5000}

FQDN="${SUBDOMAIN}.${DOMAIN}"

echo -e "\n${YELLOW}Generating Cloudflare login link...${NC}"
LOGIN_URL=$(cloudflared tunnel login 2>&1 | grep -o 'https://dash.cloudflare.com/[^ ]*')

echo -e "\n${GREEN}Please open the following URL in a browser to authenticate with Cloudflare:${NC}"
echo -e "${YELLOW}$LOGIN_URL${NC}"
echo -e "After authenticating, come back here and press Enter to continue..."
read

# Wait for the cert.pem to be created
while [ ! -f ~/.cloudflared/cert.pem ]; do
    echo -e "${YELLOW}Waiting for authentication to complete...${NC}"
    sleep 2
done

echo -e "${GREEN}Authentication successful!${NC}"

# Create tunnel
echo -e "\n${YELLOW}Creating tunnel '${TUNNEL_NAME}'...${NC}"
TUNNEL_ID=$(cloudflared tunnel create "$TUNNEL_NAME" | grep -oP 'Created tunnel [^\s]+ with id \K[0-9a-f-]+')

if [ -z "$TUNNEL_ID" ]; then
    echo -e "${RED}Failed to extract tunnel ID. Please check the output above.${NC}"
    exit 1
fi

echo -e "${GREEN}Tunnel created with ID: ${TUNNEL_ID}${NC}"

# Create DNS record
echo -e "\n${YELLOW}Creating DNS record for ${FQDN}...${NC}"
cloudflared tunnel route dns "$TUNNEL_NAME" "$FQDN"

# Create config file
echo -e "\n${YELLOW}Creating config file...${NC}"
cat > ~/.cloudflared/config.yml << EOL
tunnel: ${TUNNEL_ID}
credentials-file: /home/azuolas/.cloudflared/${TUNNEL_ID}.json
ingress:
  - hostname: ${FQDN}
    service: http://localhost:${API_PORT}
  - service: http_status:404
EOL

# Save the FQDN to a file for reference
echo "$FQDN" > ~/.cloudflared/hostname

# Copy config to monorepo for backup
mkdir -p ~/mmoment/app/raspberry_pi/cloudflare-tunnel
cp ~/.cloudflared/config.yml ~/mmoment/app/raspberry_pi/cloudflare-tunnel/
cp ~/.cloudflared/${TUNNEL_ID}.json ~/mmoment/app/raspberry_pi/cloudflare-tunnel/
cp ~/.cloudflared/hostname ~/mmoment/app/raspberry_pi/cloudflare-tunnel/

# Create service file
echo -e "\n${YELLOW}Creating systemd service file...${NC}"
cat > ~/mmoment/app/raspberry_pi/cloudflare-tunnel/cloudflare-tunnel.service << EOL
[Unit]
Description=Cloudflare Tunnel
After=network.target
Wants=network-online.target

[Service]
User=azuolas
Group=azuolas
WorkingDirectory=/home/azuolas
ExecStart=/usr/bin/cloudflared tunnel run
Restart=always
RestartSec=5
StartLimitInterval=0

[Install]
WantedBy=multi-user.target
EOL

# Install service
sudo cp ~/mmoment/app/raspberry_pi/cloudflare-tunnel/cloudflare-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cloudflare-tunnel.service

# Start service
echo -e "\n${YELLOW}Starting Cloudflare tunnel service...${NC}"
sudo systemctl start cloudflare-tunnel.service
sudo systemctl status cloudflare-tunnel.service

echo -e "\n${GREEN}Setup complete!${NC}"
echo "Your camera API should now be accessible at: https://${FQDN}"
echo "Configuration files have been backed up to: ~/mmoment/app/raspberry_pi/cloudflare-tunnel/"
echo "To check tunnel status: sudo systemctl status cloudflare-tunnel.service"
echo "To restart tunnel: sudo systemctl restart cloudflare-tunnel.service" 