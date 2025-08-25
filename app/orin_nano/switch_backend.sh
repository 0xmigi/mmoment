#!/bin/bash

# Backend Switcher Script for MMOMENT
# Switch between local development backend and Railway production backend

echo "🌐 MMOMENT Backend Switcher"
echo "=========================="

echo "Which backend do you want to use?"
echo "1) Development - Local backend (192.168.1.232:3001)"
echo "2) Production - Railway backend (https://mmoment-production.up.railway.app)"
read -p "Enter choice (1-2): " choice

case $choice in
    1)
        echo "🔧 Configuring for LOCAL DEVELOPMENT backend..."
        
        # Update Docker Compose for local backend
        sed -i 's|BACKEND_HOST=.*|BACKEND_HOST=192.168.1.232|g' docker-compose.yml
        sed -i 's|BACKEND_PORT=.*|BACKEND_PORT=3001|g' docker-compose.yml
        sed -i 's|BACKEND_URL=.*|BACKEND_URL=http://192.168.1.232:3001|g' docker-compose.yml
        
        echo "✅ Configured for local development backend"
        ;;
    2)
        echo "🚀 Configuring for RAILWAY PRODUCTION backend..."
        
        # Update Docker Compose for Railway backend
        sed -i 's|BACKEND_HOST=.*|BACKEND_HOST=mmoment-production.up.railway.app|g' docker-compose.yml
        sed -i 's|BACKEND_PORT=.*|BACKEND_PORT=443|g' docker-compose.yml
        sed -i 's|BACKEND_URL=.*|BACKEND_URL=https://mmoment-production.up.railway.app|g' docker-compose.yml
        
        echo "✅ Configured for Railway production backend"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Restart camera service to apply changes
echo ""
echo "🔄 Restarting camera service to apply backend changes..."
docker compose restart camera-service

echo "⏳ Waiting for service to start..."
sleep 5

# Test backend connectivity
echo ""
echo "🔍 Testing backend connectivity..."
BACKEND_URL=$(grep "BACKEND_URL=" docker-compose.yml | head -1 | cut -d'=' -f2- | tr -d ' ')

if curl -s --max-time 10 "$BACKEND_URL/health" > /dev/null 2>&1; then
    echo "✅ Backend is reachable at: $BACKEND_URL"
else
    echo "⚠️  Backend connectivity test failed for: $BACKEND_URL"
    echo "   This might be normal if the backend isn't running yet"
fi

echo ""
echo "🎉 Backend switch complete!"
echo "Camera service is now configured to use: $BACKEND_URL"