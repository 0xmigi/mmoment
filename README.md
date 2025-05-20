# MMOMENT: Identity-Aware Camera Content Network

Mmoment solves the friction in capturing valuable social content by creating an identity-aware camera network strategically positioned where social content naturally occurs. This solution transforms predictable interaction points into seamless content-capture opportunities, eliminating the technical barriers between experiencing moments and sharing them—all while generating a social graph of real-world human interactions.

## Project Overview

Despite smartphone ubiquity, recording meaningful moments remains cumbersome—requiring device setup, perfect timing, and navigating complex sharing preferences. Mmoment addresses this by placing intelligent cameras in locations where social activities naturally occur:

- Fitness enthusiasts use the same equipment
- Diners occupy the same tables
- Event attendees gather at the same booths

These predictable interaction points are transformed into effortless content-capture opportunities through the identity-aware camera network.

## Repository Structure

```
mmoment/
├── app/                        # Main application code
│   ├── orin_nano/              # NVIDIA Jetson Orin Nano implementation
│   ├── raspberry_pi/           # Raspberry Pi implementation
│   ├── web/                    # Web frontend application
│   └── backend/                # Backend services
├── programs/                   # Solana smart contracts
│   └── camera-network/         # Main camera network program
├── scripts/                    # Utility scripts
└── tests/                      # Testing framework
```

## Core Components

### 1. Camera Hardware (app/orin_nano, app/raspberry_pi)

The camera network evolved through multiple iterations:

- **Raspberry Pi Zero 2 W** (Initial Prototype)  
  Simple, low-cost device for proof of concept.

- **Raspberry Pi 5** (Enhanced Version)  
  Improved performance and connectivity for production-ready devices.

- **NVIDIA Jetson Orin Nano** (Current Version)  
  Advanced computer vision capabilities with on-device ML for facial recognition, gesture detection, and high-quality video processing.

### 2. Camera Service (app/orin_nano/camera_service_new)

The core vision processing system built on the Jetson platform:

- High-performance frame buffer (30fps)
- Face recognition and gesture detection
- Media capture (photos and videos)
- Real-time video streaming

### 3. Frontend Bridge (app/orin_nano/frontend_bridge)

Middleware API for connecting camera systems to web applications:

- RESTful API for frontend integration
- MJPEG streaming
- Session management
- Cross-origin request handling

### 4. Solana Blockchain Integration (app/orin_nano/solana_middleware)

Decentralized authentication and content ownership:

- Wallet connection management
- NFT-based identity verification
- Moment minting (creating NFTs from captured content)
- Secure content authorization

### 5. Web Application (app/web)

User-facing interface for interacting with the camera network:

- Wallet integration (Solana)
- Camera discovery and connection
- Content browsing and sharing
- User profile management

### 6. Solana Smart Contracts (programs/camera-network)

On-chain logic for the decentralized camera network:

- Camera registry for device authentication
- Content ownership verification
- Access control mechanisms
- Social graph relationships

#### Smart Contract Architecture

The Solana program is built on the Anchor framework and implements:

1. **Camera Registry** - A global PDA that tracks all cameras in the network
2. **User Registry** - Stores user identities and their associated face embeddings (encrypted)
3. **Moment NFTs** - Content ownership with metadata linking to captured moments
4. **Access Control** - Permission system for camera access based on NFT ownership
5. **Social Graph** - On-chain representation of real-world interactions

Example of registering a new camera:
```rust
pub fn register_camera(ctx: Context<RegisterCamera>, camera_id: String) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    camera.owner = ctx.accounts.authority.key();
    camera.camera_id = camera_id;
    camera.active = true;
    camera.created_at = Clock::get()?.unix_timestamp;
    Ok(())
}
```

## Key Features

### Identity Awareness

The system recognizes users through:

- Facial recognition (privacy-preserving with on-device processing)
- Solana wallet authentication
- NFT-based access control

### Gesture Detection

Natural interaction with cameras through:

- Hand gesture recognition
- Pose estimation
- Intent-based capture triggers

### Seamless Content Capture

Automatic content recording based on:

- User presence detection
- Activity recognition
- Social context awareness

### Blockchain-Verified Ownership

All captured content is:

- Verifiably owned by the user
- Optionally mintable as NFTs
- Securely shareable with privacy controls

## Real-World Applications

### Fitness Centers

Mmoment cameras positioned at key equipment stations automatically capture:
- Personal records and achievements
- Form checks and improvements
- Before/after transformation documentation

### Restaurants & Culinary Experiences

Strategically placed cameras allow diners to:
- Capture dish presentations without interrupting the experience
- Record chef's table interactions and cooking demonstrations
- Create food review content without phone distractions

### Event Venues

Cameras at photo-worthy locations enable:
- Automatic photo booth functionality without the booth
- Action shots during activities/experiences
- Group photos without excluding the photographer

### Social Graph Generation

The system creates a privacy-preserving social graph by:
- Mapping real-world interactions between users
- Creating connection points based on shared experiences
- Building communities around common activities and locations

## Getting Started

### Hardware Requirements

For Jetson Orin Nano setup:
- NVIDIA Jetson Orin Nano Developer Kit
- Logitech StreamCam (or compatible USB camera)
- 5V DC, 4A power supply
- Network connectivity

For Raspberry Pi setup:
- Raspberry Pi 5 (or Pi Zero 2 W for simpler deployments)
- Compatible camera module
- Power supply
- Network connectivity

### Software Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/0xmigi/mmoment.git
   cd mmoment
   ```

2. Set up the camera system (Jetson Orin Nano):
   ```bash
   cd app/orin_nano
   ./install_services.sh
   ./start_services.sh
   ```

3. Launch the web application:
   ```bash
   cd app/web
   yarn install
   yarn dev
   ```

4. Connect your Solana wallet to start capturing moments!

## Development Roadmap

### Phase 1: Single Camera Prototype ✅
- Raspberry Pi Zero 2 W implementation
- Basic camera streaming
- Simple web interface

### Phase 2: Enhanced Camera System ✅
- Raspberry Pi 5 implementation
- Improved image quality
- Cloud connectivity

### Phase 3: Computer Vision Integration ✅
- NVIDIA Jetson Orin Nano implementation
- Face recognition
- Gesture detection
- Real-time processing

### Phase 4: Decentralized Identity ✅
- Solana wallet integration
- NFT-based authentication
- On-chain content verification

### Phase 5: Deployment & Scaling 🚧
- Production-ready hardware
- Multi-camera network support
- Commercial venue partnerships

## Future Vision

### Enhanced Computer Vision

- **Emotion Recognition**: Automatically capture content during peak emotional moments
- **Activity Classification**: Smart recording triggered by specific activities or achievements
- **Multi-Person Interactions**: Detect group dynamics and social connections

### Decentralized Camera Network

- **Tokenized Camera Access**: Economic model for commercial camera deployments
- **Content Monetization**: Creator revenue sharing for venue-captured content

### Privacy Innovations

- **Selective Disclosure**: Users control exactly what data is shared and when
- **Self-Sovereign Identity**: Complete user ownership of all identity data

## Technical Documentation

- [API Endpoints](app/orin_nano/docs/API_ENDPOINTS.md) - Complete reference of all available API endpoints
- [System Setup](app/orin_nano/docs/SYSTEM_SETUP.md) - How to set up the system from scratch
- [Frontend Integration](app/orin_nano/docs/FRONTEND_INTEGRATION.md) - Guide for frontend developers
- [Hardware Setup](app/orin_nano/docs/HARDWARE.md) - Information about hardware requirements and setup


## License

This project is proprietary software owned by mmoment. All rights reserved.