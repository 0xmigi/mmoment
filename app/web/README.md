# Solana Project Web Scripts

This directory contains scripts for interacting with your Solana program.

## Module Format Issues

When working with Node.js v23.0.0 and Solana/Anchor libraries, you may encounter issues with module formats (CommonJS vs ES Modules). Here's how to resolve them:

### Solution 1: Use CommonJS (.cjs) Files

The most reliable approach is to use CommonJS format with `.cjs` file extensions:

```javascript
// Example: get_pda.cjs
const { PublicKey } = require('@solana/web3.js');

// Self-executing async function
(async () => {
  const programId = new PublicKey('7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4');
  const [pda, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("camera-registry")],
    programId
  );
  console.log('CameraRegistry PDA:', pda.toString());
  console.log('Bump:', bump);
})().catch(err => {
  console.error('Error:', err);
});
```

### Solution 2: ES Modules with package.json Configuration

If you prefer ES Modules, ensure your package.json has `"type": "module"` and use the correct import syntax:

```javascript
// Example: get_pda.js
import { PublicKey } from '@solana/web3.js';

(async () => {
  const programId = new PublicKey('7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4');
  const [pda, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("camera-registry")],
    programId
  );
  console.log('CameraRegistry PDA:', pda.toString());
  console.log('Bump:', bump);
})();
```

## Working with Anchor Programs

When initializing Anchor programs, you may encounter compatibility issues between @solana/web3.js and @coral-xyz/anchor. Here are some approaches:

### Using Direct Transactions

For more reliable interaction, you can use direct transactions with the correct instruction format:

```javascript
const instructionData = Buffer.from([
  175, 175, 109, 31, 13, 152, 155, 237, // "initialize" discriminator for Anchor
  // Additional data if needed
]);

const instruction = new TransactionInstruction({
  keys: [
    { pubkey: keypair.publicKey, isSigner: true, isWritable: true },
    { pubkey: registryPDA, isSigner: false, isWritable: true },
    { pubkey: SystemProgram.programId, isSigner: false, isWritable: false }
  ],
  programId: programId,
  data: instructionData
});
```

## Available Scripts

- `get_pda.cjs` - Get the Program Derived Address (PDA) for your camera registry
- `simple_initialize.cjs` - Display PDA information and instructions for manual initialization
- `direct_initialize.cjs` - Attempt to initialize the program using direct transaction creation

## Running Scripts

```bash
node get_pda.cjs
node simple_initialize.cjs
node direct_initialize.cjs
```

## Troubleshooting

If you encounter errors with the Anchor Program class, try using the direct transaction approach or downgrade Node.js to a version that's more compatible with the Solana libraries.
