# Solana Camera Network with Face Recognition

A Solana program that creates a network of cameras with face recognition capabilities, allowing users to check in and out of locations securely.

## Features

- Camera registration and management
- User check-in/check-out functionality
- Facial recognition for secure authentication
- Support for multiple cameras in a registry
- Camera activity tracking

## Prerequisites

- Node.js (v16 or higher)
- Solana CLI tools
- Anchor framework
- Yarn

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   yarn install
   ```
3. Build the program:
   ```
   yarn build
   ```
4. Deploy to devnet:
   ```
   yarn deploy
   ```

## Using a Persistent Wallet

A wallet file has been generated at `test-wallet.json`. This wallet is used by the test scripts to avoid hitting airdrop limits.

### Funding Your Wallet

The test wallet needs to be funded before running the scripts. You can fund it at:
- https://solfaucet.com/
- Enter the wallet address: `BRmmwLEvvfGYUsJEzPD7PTzvwKqzPXaBhGxPrN3Lvoed`

## Running the Demo Scripts

### Camera Network Client

Basic camera network functionality:

```
yarn camera-network
```

The script will:
1. Initialize the camera registry (if needed)
2. Register a camera with various features
3. Create a test user and fund it
4. Check in the test user to the camera
5. Check out the test user

### Face Recognition Test

Test the face recognition enrollment:

```
yarn face-test
```

The script will:
1. Connect to the Solana devnet
2. Enroll test facial data for the current user
3. Verify the face data account exists

### Face Recognition + Check-In

Complete face recognition and check-in flow:

```
yarn face-checkin
```

The script will:
1. Enroll facial data for user authentication
2. Ensure camera registry exists
3. Register a camera with face recognition capability
4. Check in with face recognition enabled
5. Check out and close the session

## Program Overview

The Camera Network program provides several main functions:

1. `initialize` - Initialize the camera registry
2. `registerCamera` - Register a new camera with various features
3. `enrollFace` - Enroll facial data for a user
4. `checkIn` - Check in a user to a camera (with optional face recognition)
5. `checkOut` - Check out a user from a camera

## Program Addresses

The program is deployed on Solana devnet at:
```
Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S
```

## Account Structure

The program uses several account types:

1. `CameraRegistry` - Tracks all cameras in the network
2. `CameraAccount` - Individual camera data and features
3. `FaceData` - User's facial recognition data (stores a hash, not raw data)
4. `UserSession` - Active user session with a camera

## License

MIT 