import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { 
  PublicKey, 
  Keypair, 
  SystemProgram, 
  Connection,
  TransactionInstruction,
  Transaction,
  sendAndConfirmTransaction
} from '@solana/web3.js';
import { IDL, SimpleCheckin } from '../target/types/simple_checkin';

/**
 * This is an example client that shows how to interact with the Simple Check-in
 * Solana program. In a real application, you would:
 * 
 * 1. Replace the test keypair with a real wallet (like Phantom)
 * 2. Deploy the program and use the actual program ID
 * 3. Use a real camera account instead of a generated one
 * 
 * This client simply demonstrates the structure and flow.
 */
 
// Program ID would come from deployment
const PROGRAM_ID = "GdAxahKyNHAhkEMWho693aNjTDefV1fhPqqFfEgdRFsf";

class SimpleCheckinClient {
  connection: Connection;
  programId: PublicKey;
  program: Program<SimpleCheckin>;
  userWallet: anchor.Wallet;
  
  constructor(connection: Connection, wallet: anchor.Wallet) {
    this.connection = connection;
    this.programId = new PublicKey(PROGRAM_ID);
    this.userWallet = wallet;
    
    // Create provider
    const provider = new anchor.AnchorProvider(
      connection,
      wallet,
      { commitment: 'confirmed' }
    );
    
    // Create program interface using the IDL
    this.program = new anchor.Program(IDL, this.programId, provider) as Program<SimpleCheckin>;
  }
  
  /**
   * Derive the session PDA for a given user and camera
   */
  deriveSessionPDA(cameraPubkey: PublicKey): [PublicKey, number] {
    return PublicKey.findProgramAddressSync(
      [
        Buffer.from("session"),
        this.userWallet.publicKey.toBuffer(),
        cameraPubkey.toBuffer()
      ],
      this.programId
    );
  }
  
  /**
   * Check in to a camera
   */
  async checkIn(cameraPubkey: PublicKey): Promise<string> {
    const [sessionPDA, _] = this.deriveSessionPDA(cameraPubkey);
    
    console.log("Checking in user", this.userWallet.publicKey.toString());
    console.log("To camera", cameraPubkey.toString());
    console.log("Session will be created at", sessionPDA.toString());
    
    try {
      // On a real network, this would execute the transaction
      // In this example, we'll just print what would happen
      console.log("In a real deployment, this would execute:");
      console.log(`program.methods.checkIn()
        .accounts({
          user: ${this.userWallet.publicKey.toString()},
          camera: ${cameraPubkey.toString()},
          session: ${sessionPDA.toString()},
          systemProgram: ${SystemProgram.programId.toString()}
        })
        .rpc()`);
      
      // This is how you'd get data about the created session
      console.log(`
        After check-in succeeds, you could fetch the session data:
        const sessionData = await program.account.userSession.fetch("${sessionPDA.toString()}");
        // sessionData would contain: user, camera, timestamp, bump
      `);
      
      return "check-in-simulated";
    } catch (error) {
      console.error("Check-in failed:", error);
      throw error;
    }
  }
  
  /**
   * Check out from a camera
   */
  async checkOut(cameraPubkey: PublicKey): Promise<string> {
    const [sessionPDA, _] = this.deriveSessionPDA(cameraPubkey);
    
    console.log("Checking out user", this.userWallet.publicKey.toString());
    console.log("From camera", cameraPubkey.toString());
    console.log("Session at", sessionPDA.toString(), "will be closed");
    
    try {
      // On a real network, this would execute the transaction
      // In this example, we'll just print what would happen
      console.log("In a real deployment, this would execute:");
      console.log(`program.methods.checkOut()
        .accounts({
          user: ${this.userWallet.publicKey.toString()},
          camera: ${cameraPubkey.toString()},
          session: ${sessionPDA.toString()}
        })
        .rpc()`);
      
      return "check-out-simulated";
    } catch (error) {
      console.error("Check-out failed:", error);
      throw error;
    }
  }
  
  /**
   * Get all active sessions for the current user
   */
  async getUserSessions(): Promise<void> {
    console.log("To find all sessions for a user in a real deployment:");
    console.log(`program.account.userSession.all([
      {
        memcmp: {
          offset: 8, // skip discriminator
          bytes: "${this.userWallet.publicKey.toString()}"
        }
      }
    ])`);
  }
}

async function main() {
  // For demo purposes, use testnet/devnet
  const connection = new Connection("https://api.devnet.solana.com", 'confirmed');
  
  // Create a test wallet
  const walletKeypair = Keypair.generate();
  const wallet = new anchor.Wallet(walletKeypair);
  console.log("Demo wallet pubkey:", wallet.publicKey.toString());
  
  // Create our client
  const client = new SimpleCheckinClient(connection, wallet);
  
  // Generate a fake camera for testing
  const cameraKeypair = Keypair.generate();
  console.log("Demo camera pubkey:", cameraKeypair.publicKey.toString());
  
  // Simulate check-in
  console.log("\n=== CHECK-IN SIMULATION ===");
  await client.checkIn(cameraKeypair.publicKey);
  
  // Simulate check-out
  console.log("\n=== CHECK-OUT SIMULATION ===");
  await client.checkOut(cameraKeypair.publicKey);
  
  // Simulate fetching user sessions
  console.log("\n=== QUERY SESSIONS SIMULATION ===");
  await client.getUserSessions();
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  }
); 