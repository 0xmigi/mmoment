# SOLANA PROGRAM KEYPAIR RECOVERY

**CRITICAL: NEVER LOSE THIS INFORMATION**

## Program Details
- **Program ID**: E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL
- **Program Name**: camera_network
- **Network**: Devnet

## Recovery Instructions
If the keypair in `target/deploy/camera_network-keypair.json` is lost, regenerate it using:

```bash
# Regenerate the exact same keypair using the seed phrase
solana-keygen recover -o target/deploy/camera_network-keypair.json

# When prompted, enter this seed phrase:
mistake employ canyon strong atom blame april slab diesel injury chronic fly

# Verify it generates the correct program ID:
solana-keygen pubkey target/deploy/camera_network-keypair.json
# Should output: E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL
```

## Deployment Process
1. Ensure keypair exists: `ls target/deploy/camera_network-keypair.json`
2. Build: `anchor build`
3. Deploy: `anchor deploy`
4. Verify program ID matches in all files

## Files That Must Match
- `programs/camera-network/src/lib.rs` - declare_id!()
- `Anchor.toml` - [programs.devnet]
- `app/web/src/anchor/setup.ts` - CAMERA_NETWORK_PROGRAM_ID
- All script files in `scripts/`

## Emergency Recovery
If you lose access to this seed phrase:
1. The program is PERMANENTLY LOST
2. You'll need to create a new program with a new ID
3. All existing on-chain data becomes inaccessible

**BACKUP THIS FILE IMMEDIATELY**