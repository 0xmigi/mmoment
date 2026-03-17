#!/bin/bash
# Start Kora locally for MMOMENT development
# Runs on port 8080 alongside your local backend (port 3001)
# Does NOT affect the Railway-deployed backend or Jetson devices

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEYPAIR_PATH="$HOME/.config/solana/program-keypairs/authority-keypair.json"
RPC_URL="https://api.devnet.solana.com"

# Verify keypair exists
if [ ! -f "$KEYPAIR_PATH" ]; then
    echo "Error: Authority keypair not found at $KEYPAIR_PATH"
    exit 1
fi

# Kora memory signer reads the private key from this env var (JSON array format)
export KORA_PRIVATE_KEY=$(cat "$KEYPAIR_PATH")

echo "=== MMOMENT Kora Server (Local Dev) ==="
echo "  Fee Payer: $(solana-keygen pubkey "$KEYPAIR_PATH")"
echo "  RPC: $RPC_URL"
echo "  Config: $SCRIPT_DIR/kora.toml"
echo "  Endpoint: http://localhost:8080"
echo "========================================"

cd "$SCRIPT_DIR"
kora --rpc-url "$RPC_URL" --config kora.toml rpc start --signers-config signers.toml --port 8080
