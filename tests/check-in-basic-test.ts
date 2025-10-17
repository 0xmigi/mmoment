import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram } from '@solana/web3.js';
import { assert } from "chai";

describe("my-solana-project check-in-basic", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  // Use the correct program name
  const program = anchor.workspace.MySolanaProject;
  console.log("Program ID:", program.programId.toString());
  
  // Create a keypair to represent a camera
  const cameraKeypair = Keypair.generate();
  console.log("Camera public key:", cameraKeypair.publicKey.toString());
  
  // Find the session PDA
  const [sessionPDA] = PublicKey.findProgramAddressSync(
    [
      Buffer.from("session"), 
      provider.wallet.publicKey.toBuffer(), 
      cameraKeypair.publicKey.toBuffer()
    ],
    program.programId
  );
  console.log("Session PDA:", sessionPDA.toString());

  it("Can perform a basic check in", async () => {
    // Call the check_in_basic instruction
    const tx = await program.methods
      .checkInBasic()
      .accounts({
        user: provider.wallet.publicKey,
        camera: cameraKeypair.publicKey,
        session: sessionPDA,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
    
    console.log("Check-in transaction signature:", tx);

    // Fetch the session account to verify it exists
    const sessionAccount = await program.account.userSession.fetch(sessionPDA);
    
    // Verify the session data is correct
    assert.ok(sessionAccount.user.equals(provider.wallet.publicKey), "User doesn't match");
    assert.ok(sessionAccount.camera.equals(cameraKeypair.publicKey), "Camera doesn't match");
    assert.isAbove(sessionAccount.sessionStart, 0, "Session start timestamp should be positive");
    
    console.log("Session data:", {
      user: sessionAccount.user.toString(),
      camera: sessionAccount.camera.toString(),
      sessionStart: new Date(sessionAccount.sessionStart * 1000).toISOString(),
      features: {
        faceRecognition: sessionAccount.featuresEnabled.faceRecognition,
        gestureDetection: sessionAccount.featuresEnabled.gestureDetection
      }
    });
  });
}); 