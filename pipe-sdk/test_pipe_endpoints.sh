#!/bin/bash

echo "Testing various Pipe Network endpoints..."

# Test different possible endpoints
endpoints=(
    "https://api.pipenetwork.com"
    "https://devnet.pipenetwork.com"
    "https://testnet.pipenetwork.com"
    "https://api.pipe.network"
    "https://devnet.pipe.network"
    "https://cdn.pipenetwork.com"
    "https://firestarter.pipenetwork.com"
)

for endpoint in "${endpoints[@]}"; do
    echo -n "Testing $endpoint... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null)
    if [ "$response" != "000" ]; then
        echo "✓ HTTP $response"
        # Try creating a user to see if it's the right API
        echo "  Checking /createUser endpoint..."
        curl -X POST "$endpoint/createUser" \
            -H "Content-Type: application/json" \
            -d '{"username":"test"}' \
            -w "\n  Response: %{http_code}\n" \
            2>/dev/null
    else
        echo "✗ Not reachable"
    fi
    echo ""
done