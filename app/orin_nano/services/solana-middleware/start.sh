#!/bin/bash

# Source the Solana configuration if it exists
if [ -f "/app/solana_config.env" ]; then
    echo "Loading Solana configuration from solana_config.env..."
    source /app/solana_config.env
fi

# Start the Python application
echo "Starting Solana Middleware..."
exec python solana_middleware.py 