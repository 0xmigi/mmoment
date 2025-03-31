import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { BN } from "bn.js";
import { expect } from "chai";

describe("camera-activation", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  // Get the program from the IDL
  const programId = new PublicKey("77HrUp2XLQGe4tN6pMmHxLLkZnERhVgjJBRxujaVBF2");
  const idl = require("../target/idl/camera_activation.json");
  const program = new anchor.Program(idl, programId);
  
  // Test camera data
  const cameraId = "cam_" + Math.floor(Math.random() * 1000000).toString();
  const cameraName = "Test Camera";
  const cameraModel = "Raspberry Pi Camera Module";
  const cameraLocation = [37.7749, -122.4194]; // San Francisco

  it("Initializes the camera registry", async () => {
    // Find the registry PDA
    const [registryAddress, _] = await PublicKey.findProgramAddress(
      [Buffer.from("camera-registry")],
      program.programId
    );

    try {
      const tx = await program.methods
        .initialize()
        .accounts({
          authority: provider.wallet.publicKey,
          registry: registryAddress,
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      console.log("Transaction signature", tx);

      // Fetch and verify the registry
      const registry = await program.account.cameraRegistry.fetch(registryAddress);
      expect(registry.authority.toString()).to.equal(provider.wallet.publicKey.toString());
      expect(registry.cameraCount.toNumber()).to.equal(0);
    } catch (error) {
      console.error("Error in initializing camera registry:", error);
      throw error;
    }
  });

  it("Registers a new camera", async () => {
    try {
      // Find PDAs
      const [registryAddress, _] = await PublicKey.findProgramAddress(
        [Buffer.from("camera-registry")],
        program.programId
      );
      
      const [cameraAddress, __] = await PublicKey.findProgramAddress(
        [
          Buffer.from("camera"),
          Buffer.from(cameraId),
          provider.wallet.publicKey.toBuffer()
        ],
        program.programId
      );

      const tx = await program.methods
        .registerCamera({
          cameraId,
          name: cameraName,
          model: cameraModel,
          location: cameraLocation.map(coord => new BN(coord)),
          fee: new BN(100)
        })
        .accounts({
          owner: provider.wallet.publicKey,
          registry: registryAddress,
          camera: cameraAddress,
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      console.log("Transaction signature", tx);

      // Fetch the camera account
      const camera = await program.account.cameraAccount.fetch(cameraAddress);
      console.log("Camera data:", camera);
      expect(camera.owner.toString()).to.equal(provider.wallet.publicKey.toString());
      expect(camera.cameraId).to.equal(cameraId);
    } catch (error) {
      console.error("Error in registering camera:", error);
      throw error;
    }
  });
}); 