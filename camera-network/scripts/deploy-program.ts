import { 
  Connection, 
  Keypair, 
  PublicKey, 
  sendAndConfirmTransaction, 
  Transaction, 
  TransactionInstruction,
  SystemProgram,
  LAMPORTS_PER_SOL,
  BpfLoader
} from '@solana/web3.js';
import fs from 'fs';
import path from 'path';

// Path to the compiled program
const PROGRAM_PATH = path.join(__dirname, '../target/deploy/simple_checkin.so');
// Path to the program keypair
const PROGRAM_KEYPAIR_PATH = path.join(__dirname, '../target/deploy/simple_checkin-keypair.json');

async function main() {
  // Connect to Devnet
  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  
  // Read program keypair
  console.log('Reading program keypair...');
  const programKeyPairJson = JSON.parse(fs.readFileSync(PROGRAM_KEYPAIR_PATH, 'utf-8'));
  const programKeypair = Keypair.fromSecretKey(
    Uint8Array.from(programKeyPairJson)
  );
  
  console.log('Program ID:', programKeypair.publicKey.toString());
  
  // Read the BPF program
  console.log('Reading program binary...');
  const programData = fs.readFileSync(PROGRAM_PATH);
  console.log(`Program size: ${programData.length} bytes`);
  
  // Get balance of the deployer
  console.log('Checking wallet balance...');
  try {
    const balance = await connection.getBalance(programKeypair.publicKey);
    console.log(`Program account balance: ${balance / LAMPORTS_PER_SOL} SOL`);
  } catch (e) {
    console.log('Program account not yet created');
  }
  
  console.log(`Deployment process would require uploading the program binary to the Solana blockchain.`);
  console.log(`This would create an executable account at: ${programKeypair.publicKey.toString()}`);
  console.log(`The program would need sufficient SOL to cover storage rent.`);
  
  // Print info about the program's instructions
  console.log(`\nProgram Instructions:`);
  console.log(`1. check_in - Creates a session PDA for a user and camera`);
  console.log(`2. check_out - Closes a session PDA and returns the rent`);
  
  console.log(`\nPDA Seeds:`);
  console.log(`Session PDA: ["session", userPubkey, cameraPubkey]`);
  
  console.log(`\nFor actual deployment, you would need to:`);
  console.log(`1. Ensure the deployer wallet has enough SOL (at least 5 SOL recommended)`);
  console.log(`2. Run 'solana program deploy ${PROGRAM_PATH} --program-id ${PROGRAM_KEYPAIR_PATH}'`);
  console.log(`3. Due to M1 Mac limitations, you might need to deploy on a different machine or use a service like Anchor Cloud`);
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  }
); 