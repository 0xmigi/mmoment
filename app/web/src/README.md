# Mmoment Web App Structure

## Overview

This codebase follows a modular structure optimized for clarity, maintainability, and extensibility. Each module has a clear responsibility and exports a well-defined interface through its `index.ts` file.

## Directory Structure

```
/src
├── core/                           # Core application setup
│   ├── App.tsx                     # Main application entry
│   ├── Router.tsx                  # Application routing
│   ├── config.ts                   # Configuration values
│   └── styles/                     # Global styles
│
├── auth/                           # Authentication
│   ├── components/                 # Auth UI components
│   ├── farcaster/                  # Farcaster integration
│   ├── AuthProvider.tsx            # Auth context
│   ├── WalletProvider.tsx          # Wallet provider selection
│   └── index.ts                    # Exports auth functionality
│
├── camera/                         # Camera module
│   ├── CameraProvider.tsx          # Camera context provider
│   ├── camera-service.ts           # Camera communication service
│   ├── useCameraRegistry.ts        # Camera registry hook
│   ├── useCameraStatus.ts          # Camera status hook
│   └── index.ts                    # Exports all camera functionality
│
├── timeline/                       # Timeline module
│   ├── Timeline.tsx                # Timeline component
│   ├── TimelineEvents.tsx          # Timeline event components
│   ├── timeline-service.ts         # Timeline backend service
│   ├── timeline-types.ts           # Type definitions
│   └── index.ts                    # Exports all timeline functionality
│
├── media/                          # Media handling
│   ├── MediaViewer.tsx             # Media viewing components
│   ├── StreamPlayer.tsx            # Live stream player
│   ├── Gallery.tsx                 # Image/video gallery
│   ├── VideoRecorder.tsx           # Video recording component
│   └── index.ts                    # Exports all media functionality
│
├── storage/                        # Storage solutions
│   ├── ipfs/                       # IPFS implementation
│   ├── walrus/                     # Walrus implementation (future)
│   ├── storage-provider.tsx        # Common storage interface
│   └── index.ts                    # Exports unified storage API
│
├── blockchain/                     # Blockchain integration
│   ├── solana-provider.tsx         # Solana context provider
│   ├── anchor-client.ts            # Anchor interaction wrapper
│   └── index.ts                    # Exports blockchain functionality
│
├── profile/                        # Profile components
│   ├── ProfileModal.tsx            # User profile modal
│   └── index.ts                    # Exports profile functionality
│
├── nfc/                            # NFC functionality
│   ├── NFCUrlGenerator.tsx         # NFC URL generator component
│   └── index.ts                    # Exports NFC functionality
│
├── pages/                          # Page components
│   ├── LandingPage.tsx             # Landing page
│   ├── ProductPage.tsx             # Product page
│   ├── QuickStartView.tsx          # Quick start view
│   └── index.ts                    # Exports all pages
│
├── ui/                             # UI components library
│   ├── layout/                     # Layout components
│   ├── common/                     # Reusable UI components
│   ├── feedback/                   # Toast notifications, alerts etc.
│   ├── debug/                      # Debug components
│   ├── settings/                   # Settings components
│   ├── account/                    # Account components
│   └── index.ts                    # Exports all UI components
│
├── utils/                          # Utility functions
│   └── index.ts                    # Exports all utilities
│
└── anchor/                         # Anchor/Solana related code (original location)
    ├── idl.ts                      # Interface definition language
    └── setup.ts                    # Setup configuration
```

## Module Design Principles

1. **Self-Contained Modules**: Each module exports a complete public API through its index.ts file.
2. **Clear Boundaries**: Modules have clear responsibilities and dependencies.
3. **Pluggable Services**: Services like storage providers are designed to be swappable.
4. **Context Providers**: State is managed through React Context providers where appropriate.
5. **TypeScript Throughout**: Type definitions help maintain code quality.

## Key Providers and Their Responsibilities

- **StorageProvider**: Manages media storage options (IPFS, Walrus, etc.)
- **WalletProvider**: Manages embedded wallet options (Turnkey, Crossmint, etc.)
- **AuthProvider**: Manages user authentication
- **SolanaProvider**: Provides Solana blockchain connectivity
- **CameraProvider**: Manages camera interaction and state
- **NotificationProvider**: Manages notifications and toasts

## How to Add New Features

### Adding a New Storage Provider

1. Create a new implementation in `/storage/your-provider/`
2. Implement the `StorageService` interface
3. Update the `getStorageService` function in `storage-provider.tsx`
4. Export from the module's index.ts

### Adding New Auth Method

1. Create new components in `/auth/components/`
2. Update the relevant providers
3. Export from the module's index.ts

## Usage Guidelines

- Import from module index files rather than direct files
- Use hooks from the respective modules (useStorage, useCamera, etc.)
- Keep UI components focused on presentation, with business logic in hooks and services
- Follow the patterns established in existing modules when adding new functionality 