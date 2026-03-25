# CLAUDE.md

MMOMENT — blockchain-integrated camera network for identity-aware content capture at physical interaction points.

## Development Commands

### Web Frontend (app/web/)
```
cd app/web && yarn dev          # Dev server
cd app/web && yarn dev:devnet   # Dev with devnet
cd app/web && yarn build        # Production build
cd app/web && yarn lint         # ESLint
```

### Backend (app/backend/)
```
cd app/backend && yarn dev      # Dev server (ts-node)
cd app/backend && yarn build    # Compile TypeScript
cd app/backend && yarn test     # Jest tests
```

### Solana Programs
```
anchor build                    # Build programs
anchor test                     # Run tests
anchor deploy                   # Deploy to configured cluster
```

### Camera Services (app/orin_nano/, app/raspberry_pi/)
```
cd app/orin_nano && docker-compose up -d    # Start Jetson services
cd app/raspberry_pi/new-camera-service && python -m camera_service.main
```

## Package Management

Always use `yarn`, never `npm`.

## Solana Program

- Program ID: `E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL`
- Network: Devnet

## Product Principles

- **Physical-first**: API/agent access MUST require physical presence (check-in). No remote visibility except aggregate count + camera owner.
- **No fake features**: Never describe capabilities that don't exist in the codebase. If suggesting something new, explicitly flag it as "would need to be built."
- **Stick to existing flow**: Don't inject new UX elements into proposals as if they already exist. Default to what's built.
- **No localStorage for important state**: Persist server-side via authenticated endpoints.

## Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Vite
- **Backend**: Node.js, Express, Socket.IO
- **Smart Contracts**: Anchor framework (Rust)
- **Camera Services**: Python Flask, OpenCV, YOLOv8, InsightFace
