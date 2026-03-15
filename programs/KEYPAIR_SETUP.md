# Solana Program Keypair Setup

Program keypairs are stored outside the repo to prevent accidental deletion or overwriting.

## Keypair Location

```
~/.config/solana/program-keypairs/
```

A symlink in `target/deploy/` points to the actual keypair file.

## Building Programs

Always build individually:
```bash
anchor build -p camera_network
anchor build -p competition_escrow
```

## After Fresh Clone or `anchor clean`

Recreate symlinks before building:
```bash
mkdir -p target/deploy
ln -sf ~/.config/solana/program-keypairs/camera_network-keypair.json target/deploy/camera_network-keypair.json
```

## Verify Keypair

```bash
solana-keygen pubkey ~/.config/solana/program-keypairs/camera_network-keypair.json
```

Compare output to program ID in `Anchor.toml`.

## Recovery

Seed phrase stored securely offline (not in repo).
