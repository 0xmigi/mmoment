const { 
  Connection, 
  Keypair, 
  PublicKey, 
  SystemProgram,
  Transaction,
  TransactionInstruction,
  sendAndConfirmTransaction
} = require('@solana/web3.js');
const fs = require('fs');

// Load your keypair from ~/.config/solana/id.json
const secretKey = Uint8Array.from(require('/Users/azuolascompy/.config/solana/id.json'));
const keypair = Keypair.fromSecretKey(secretKey);

// Create a connection to the Solana network
const connection = new Connection('https://api.devnet.solana.com', 'confirmed');

// Program ID
const programIdString = '7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4';
const programId = new PublicKey(programIdString);

// Calculate the PDA
const [registryPDA, bump] = PublicKey.findProgramAddressSync(
  [Buffer.from('camera-registry')],
  programId
);

console.log('Program ID:', programId.toString());
console.log('Your public key:', keypair.publicKey.toString());
console.log('Registry PDA:', registryPDA.toString());
console.log('Bump:', bump);

// Function to initialize the program
async function initializeProgram() {
  try {
    // Create a transaction
    const transaction = new Transaction();
    
    // Create an instruction with the correct Anchor format
    // For Anchor programs, the first 8 bytes are the instruction discriminator
    // The discriminator is the first 8 bytes of the SHA256 hash of the instruction name
    // For "initialize", this is typically [175, 175, 109, 31, 13, 152, 155, 237]
    const instructionData = Buffer.from([
      175, 175, 109, 31, 13, 152, 155, 237, // "initialize" discriminator
      // Additional data would go here if needed
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
    
    // Add the instruction to the transaction
    transaction.add(instruction);
    
    // Send and confirm the transaction
    const signature = await sendAndConfirmTransaction(
      connection,
      transaction,
      [keypair]
    );
    
    console.log('Transaction successful!');
    console.log('Signature:', signature);
    
    return signature;
  } catch (error) {
    console.error('Error initializing program:', error);
    throw error;
  }
}

// Execute the initialization
initializeProgram().then(() => {
  console.log('Program initialized successfully!');
}).catch(err => {
  console.error('Failed to initialize program:', err);
}); 