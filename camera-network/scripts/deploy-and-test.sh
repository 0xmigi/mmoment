#!/bin/bash
set -e

# Display script actions
echo "===== Simple Checkin Deploy and Test ====="
echo "1. Installing dependencies"
echo "2. Building program"
echo "3. Checking program ID"
echo "4. Deploying to devnet"
echo "5. Running test client"
echo "========================================"

# Install dependencies
echo -e "\n>> Installing dependencies..."
yarn install

# Build the program
echo -e "\n>> Building the program..."
anchor build

# Check program ID
echo -e "\n>> Checking program ID..."
PROG_ID=$(solana-keygen pubkey ./target/deploy/simple_checkin-keypair.json)
echo "Program ID from keypair: $PROG_ID"

# Get program ID from lib.rs
LIB_PROG_ID=$(grep -o 'declare_id!("[^"]*' programs/simple-checkin/src/lib.rs | cut -d'"' -f2)
echo "Program ID from lib.rs: $LIB_PROG_ID"

if [ "$PROG_ID" != "$LIB_PROG_ID" ]; then
  echo -e "\n⚠️  WARNING: Program IDs don't match!"
  echo "Update the declare_id! in programs/simple-checkin/src/lib.rs to: declare_id!(\"$PROG_ID\");"
  echo "Then rebuild and try again."
  exit 1
fi

# Deploy the program
echo -e "\n>> Deploying to devnet..."
anchor deploy --provider.cluster devnet

# Run the test
echo -e "\n>> Running test client..."
node scripts/simple-client.js 