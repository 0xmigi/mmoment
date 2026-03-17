# Kora gasless transaction server for MMOMENT
# Deployed as a separate Railway service

FROM rust:1.86-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo install kora-cli

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cargo/bin/kora /usr/local/bin/kora

WORKDIR /app
COPY kora.toml signers.toml ./

# Environment variables required:
#   KORA_PRIVATE_KEY - fee payer keypair (JSON array format)
#   RPC_URL          - Solana RPC endpoint (defaults to devnet)

EXPOSE 8080

# Shell form so env vars expand at runtime
CMD kora --rpc-url "${RPC_URL:-https://api.devnet.solana.com}" --config kora.toml rpc start --signers-config signers.toml --port 8080
