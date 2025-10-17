import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { SimpleCheckin } from "../target/types/simple_checkin";
import { PublicKey, Keypair, SystemProgram } from '@solana/web3.js';
import { assert } from "chai";

describe("simple-checkin", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.SimpleCheckin as Program<SimpleCheckin>;
  
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

  it("Can check in a user to a camera", async () => {
    // Call the check_in instruction
    const tx = await program.methods
      .checkIn()
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
    assert.isAbove(sessionAccount.timestamp, 0, "Timestamp should be positive");
    
    console.log("Session data:", {
      user: sessionAccount.user.toString(),
      camera: sessionAccount.camera.toString(),
      timestamp: new Date(sessionAccount.timestamp * 1000).toISOString(),
    });
  });

  it("Can check out a user from a camera", async () => {
    // Call the check_out instruction
    const tx = await program.methods
      .checkOut()
      .accounts({
        user: provider.wallet.publicKey,
        camera: cameraKeypair.publicKey,
        session: sessionPDA,
      })
      .rpc();
    
    console.log("Check-out transaction signature:", tx);

    // Try to fetch the session account, which should fail because it's been closed
    try {
      await program.account.userSession.fetch(sessionPDA);
      assert.fail("Session account still exists after check-out");
    } catch (error) {
      // This is expected - account should no longer exist
      assert.include(
        error.toString(), 
        "Account does not exist", 
        "Expected 'Account does not exist' error"
      );
      console.log("Session account successfully closed");
    }
  });
}); 