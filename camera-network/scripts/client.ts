import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { SimpleCheckin } from '../target/types/simple_checkin';
import { 
  PublicKey, 
  Keypair, 
  SystemProgram, 
  Connection,
  clusterApiUrl
} from '@solana/web3.js';

async function main() {
  // Connect to devnet
  const connection = new Connection(clusterApiUrl('devnet'), 'confirmed');
  
  // Create a keypair for testing (in real app, this would be loaded from wallet)
  const userKeypair = Keypair.generate();
  console.log("Test user wallet:", userKeypair.publicKey.toString());
  
  // Fund the wallet with SOL
  const airdropSignature = await connection.requestAirdrop(
    userKeypair.publicKey,
    2 * 10 ** 9 // 2 SOL in lamports
  );
  await connection.confirmTransaction(airdropSignature);
  console.log("Airdropped 2 SOL to test wallet");
  
  // Setup provider with the generated wallet
  const wallet = new anchor.Wallet(userKeypair);
  const provider = new anchor.AnchorProvider(
    connection,
    wallet,
    { commitment: 'confirmed' }
  );
  
  // Create program interface from IDL
  const programId = new PublicKey("GdAxahKyNHAhkEMWho693aNjTDefV1fhPqqFfEgdRFsf");
  const program = new anchor.Program(
    require('../target/idl/simple_checkin.json'),
    programId,
    provider
  ) as Program<SimpleCheckin>;
  
  // Generate a fake camera keypair for testing
  const cameraKeypair = Keypair.generate();
  console.log("Camera pubkey:", cameraKeypair.publicKey.toString());
  
  // Derive the session PDA
  const [sessionPDA, sessionBump] = PublicKey.findProgramAddressSync(
    [
      Buffer.from('session'),
      wallet.publicKey.toBuffer(),
      cameraKeypair.publicKey.toBuffer()
    ],
    program.programId
  );
  console.log("Session PDA:", sessionPDA.toString());
  
  try {
    // Check-in transaction
    console.log("Executing check-in...");
    const checkInTx = await program.methods
      .checkIn()
      .accounts({
        user: wallet.publicKey,
        camera: cameraKeypair.publicKey,
        session: sessionPDA,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
    console.log("Check-in transaction signature:", checkInTx);
    
    // Fetch session account data
    const sessionAccount = await program.account.userSession.fetch(sessionPDA);
    console.log("Session account data:", {
      user: sessionAccount.user.toString(),
      camera: sessionAccount.camera.toString(),
      timestamp: sessionAccount.timestamp.toString(),
      bump: sessionAccount.bump
    });
    
    // Check-out transaction
    console.log("Executing check-out...");
    const checkOutTx = await program.methods
      .checkOut()
      .accounts({
        user: wallet.publicKey,
        camera: cameraKeypair.publicKey,
        session: sessionPDA,
      })
      .rpc();
    console.log("Check-out transaction signature:", checkOutTx);
    
    // Try to fetch session account (should fail)
    try {
      await program.account.userSession.fetch(sessionPDA);
      console.log("ERROR: Session account still exists");
    } catch (error) {
      console.log("Success: Session account closed");
    }
  } catch (error) {
    console.error("Error:", error);
  }
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  }
); 