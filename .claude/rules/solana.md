---
paths:
  - "programs/**"
  - "tests/**"
  - "target/**"
  - "app/web/src/blockchain/**"
---
# Solana / Anchor Rules

## Keypair Management
- NEVER delete `target/deploy/camera_network-keypair.json`
- ALWAYS run `anchor keys sync` after any keypair changes
- Backup seed phrase in `target/deploy/KEYPAIR_RECOVERY.md`

## IDL Files
NEVER manually edit IDL files. Always copy exact output from `target/idl/*.json` after `anchor build`. Even small differences (reordered fields, missing events) cause deserialization failures at runtime.

## Anchor Version
Keep anchor-lang at the same version light-sdk depends on (currently ^0.31.1). Do NOT bump to match the CLI version. Two copies of anchor-lang in the binary cause type mismatches that break Light Protocol compressed PDA operations.
