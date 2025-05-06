import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram } from '@solana/web3.js';
import { assert } from "chai";
import { MySolanaProject } from "../target/types/my_solana_project";

describe("Check-in/Check-out Tests", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.MySolanaProject as Program<MySolanaProject>;
  
  // We'll use this keypair to represent a camera
  const cameraKeypair = Keypair.generate();
  
  // Find the session PDA
  const [sessionPDA] = PublicKey.findProgramAddressSync(
    [
      Buffer.from("session"),
      provider.wallet.publicKey.toBuffer(),
      cameraKeypair.publicKey.toBuffer()
    ],
    program.programId
  );
  
  it("Can check in using checkIn instruction", async () => {
    console.log("Testing checkIn instruction...");
    console.log("User:", provider.wallet.publicKey.toString());
    console.log("Camera:", cameraKeypair.publicKey.toString());
    console.log("Session PDA:", sessionPDA.toString());
    
    try {
      // Call the checkIn instruction
      const tx = await program.methods
        .checkIn()
        .accounts({
          user: provider.wallet.publicKey,
          camera: cameraKeypair.publicKey,
          session: sessionPDA,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
      
      console.log("Transaction signature:", tx);
      
      // Verify session exists
      const sessionAccount = await program.account.userSession.fetch(sessionPDA);
      assert.ok(sessionAccount, "Session account not found");
      assert.ok(sessionAccount.user.equals(provider.wallet.publicKey), "User doesn't match");
      assert.ok(sessionAccount.camera.equals(cameraKeypair.publicKey), "Camera doesn't match");
      
      console.log("Session data:", sessionAccount);
    } catch (error) {
      console.error("Transaction failed with error:", error);
      if (error.logs) {
        console.error("Transaction logs:");
        error.logs.forEach(log => console.error(`  ${log}`));
      }
      throw error; // Rethrow to fail the test
    }
  });
  
  it("Can check out using checkOut instruction", async () => {
    console.log("Testing checkOut instruction...");
    
    try {
      // Call the checkOut instruction
      const tx = await program.methods
        .checkOut()
        .accounts({
          user: provider.wallet.publicKey,
          camera: cameraKeypair.publicKey,
          session: sessionPDA,
        })
        .rpc();
      
      console.log("Transaction signature:", tx);
      
      // Verify session is closed
      try {
        await program.account.userSession.fetch(sessionPDA);
        assert.fail("Session account still exists after check-out");
      } catch (error) {
        // This error is expected since the account should be closed
        if (error.message.includes("Account does not exist")) {
          console.log("Session successfully closed");
        } else {
          throw error; // Re-throw if it's a different error
        }
      }
    } catch (error) {
      console.error("Transaction failed with error:", error);
      if (error.logs) {
        console.error("Transaction logs:");
        error.logs.forEach(log => console.error(`  ${log}`));
      }
      throw error; // Rethrow to fail the test
    }
  });
}); 