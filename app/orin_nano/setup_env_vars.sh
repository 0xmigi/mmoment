#!/bin/bash

echo "🔧 Setting up environment variables for MMoment Device Registration"

# Check if we're in the docker-compose environment
if [ -f .env ]; then
    echo "Found .env file, updating..."
else
    echo "Creating .env file..."
    touch .env
fi

# Add necessary environment variables
cat >> .env << 'EOF'

# Device Registration Configuration
BACKEND_HOST=192.168.1.80
BACKEND_PORT=3001

# Cloudflare Tunnel Configuration (replace with actual tunnel ID)
# CLOUDFLARE_TUNNEL_ID=your-tunnel-id-here

# Solana Configuration
CAMERA_PROGRAM_ID=E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL
SOLANA_RPC_URL=https://api.devnet.solana.com

EOF

echo "✅ Environment variables added to .env file"
echo ""
echo "📝 Manual steps needed:"
echo "1. Set your actual Cloudflare tunnel ID:"
echo "   CLOUDFLARE_TUNNEL_ID=your-actual-tunnel-id"
echo ""
echo "2. Restart the camera service to pick up new environment variables:"
echo "   docker compose restart camera-service"
echo ""
echo "3. Test the registration flow:"
echo "   python demo_registration_flow.py"