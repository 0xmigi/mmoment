const { Keypair } = require('@solana/web3.js');
const bs58 = require('bs58').default;
const fs = require('fs');

// Read the keypair from the test-wallet.json file
const keypairData = JSON.parse(fs.readFileSync('./camera-network/test-wallet.json', 'utf8'));
const secretKey = new Uint8Array(keypairData);
const keypair = Keypair.fromSecretKey(secretKey);

// Get the base58 encoded private key (for Phantom import)
const privateKeyBase58 = bs58.encode(Buffer.from(secretKey));

// Show the wallet's public key to verify it's the correct one
console.log('Public Key:', keypair.publicKey.toString());
console.log('\nPrivate Key (base58, can be imported to Phantom):\n');
console.log(privateKeyBase58);
console.log('\nNote: This key controls funds and should be kept private!'); 