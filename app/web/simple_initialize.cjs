const { Connection, Keypair, PublicKey } = require('@solana/web3.js');
const fs = require('fs');

// Load your keypair from ~/.config/solana/id.json
const secretKey = Uint8Array.from(require('/Users/azuolascompy/.config/solana/id.json'));
const keypair = Keypair.fromSecretKey(secretKey);

// Create a proper PublicKey instance for the program ID
const programIdString = '7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4';
const programId = new PublicKey(programIdString);

console.log('Program ID:', programId.toString());
console.log('Your public key:', keypair.publicKey.toString());

// Calculate the PDA
const [registryPDA, bump] = PublicKey.findProgramAddressSync(
  [Buffer.from('camera-registry')],
  programId
);

console.log('Registry PDA:', registryPDA.toString());
console.log('Bump:', bump);

// Log instructions for manual initialization
console.log('\nTo initialize the program, run the following command in your Anchor project:');
console.log(`anchor deploy --program-id ${programId.toString()}`);
console.log('\nOr use the Solana CLI:');
console.log(`solana program deploy --program-id ${programId.toString()} <PATH_TO_PROGRAM_SO_FILE>`); 