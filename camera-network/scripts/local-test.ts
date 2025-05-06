import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { 
  PublicKey, 
  Keypair, 
  SystemProgram, 
  Connection
} from '@solana/web3.js';
import { IDL } from '../target/types/simple_checkin';

// This script uses the IDL to simulate interactions with the program
// without requiring deployment to a validator

async function main() {
  // Create a connection to devnet
  const connection = new Connection("https://api.devnet.solana.com", 'confirmed');
  
  // Create a wallet for testing
  const walletKeypair = Keypair.generate();
  const wallet = new anchor.Wallet(walletKeypair);
  console.log("Test wallet pubkey:", wallet.publicKey.toString());
  
  // Create a provider
  const provider = new anchor.AnchorProvider(
    connection, 
    wallet,
    { commitment: 'confirmed' }
  );
  
  // Program ID from the build
  const programId = new PublicKey("GdAxahKyNHAhkEMWho693aNjTDefV1fhPqqFfEgdRFsf");
  
  // Initialize the program client
  const program = new anchor.Program(IDL, programId, provider);
  
  // Generate a fake camera for testing
  const cameraKeypair = Keypair.generate();
  console.log("Camera pubkey:", cameraKeypair.publicKey.toString());
  
  // Derive the session PDA
  const [sessionPDA, sessionBump] = PublicKey.findProgramAddressSync(
    [
      Buffer.from("session"),
      wallet.publicKey.toBuffer(),
      cameraKeypair.publicKey.toBuffer()
    ],
    programId
  );
  console.log("Session PDA:", sessionPDA.toString());
  
  // Print information about what would happen in a real deployment
  console.log("\nIn a real deployment, you would:");
  console.log("1. Execute check-in by calling program.methods.checkIn()");
  console.log("2. Which would create a session account at:", sessionPDA.toString());
  console.log("3. The session would store user and camera pubkeys plus a timestamp");
  console.log("4. To check out, call program.methods.checkOut()");
  console.log("5. This would close the session account and return the rent");
  
  console.log("\nProgram Structure:");
  console.log("- The program has two instructions: check_in and check_out");
  console.log("- check_in creates a new user session PDA account");
  console.log("- check_out closes the session account and returns the rent");
  console.log("- The session PDA is derived from: ['session', userPubkey, cameraPubkey]");
  
  console.log("\nExample Transaction for check-in:");
  const ixData = {
    accounts: {
      user: wallet.publicKey.toString(),
      camera: cameraKeypair.publicKey.toString(),
      session: sessionPDA.toString(),
      systemProgram: SystemProgram.programId.toString()
    },
    signers: ["<Your Wallet>"]
  };
  console.log(JSON.stringify(ixData, null, 2));
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  }
); 