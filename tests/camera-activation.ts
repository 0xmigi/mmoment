import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorProvider } from "@coral-xyz/anchor";
import { CameraActivation } from "../target/types/camera_activation";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";
import { BN } from "bn.js";

declare global {
  namespace anchor {
    interface Workspace {
      CameraActivation: Program<CameraActivation>;
    }
  }
}

describe("camera-activation", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.CameraActivation as Program<CameraActivation>;
  
  // Test camera data
  const cameraId = "cam_" + Math.floor(Math.random() * 1000000).toString();
  const cameraName = "Test Camera";
  const cameraModel = "Raspberry Pi Camera Module";
  const cameraLocation: [number, number] = [37.7749, -122.4194]; // San Francisco

  // Helper function to get the camera registry seeds
  function cameraRegistrySeeds(): Buffer {
    return Buffer.from("camera-registry");
  }
  
  it("Initializes the camera registry", async () => {
    // Find the registry PDA
    const [registryAddress, _] = await PublicKey.findProgramAddress(
      [cameraRegistrySeeds()],
      program.programId
    );

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
  });

  it("Registers a new camera", async () => {
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

    // Fetch and verify the camera
    const camera = await program.account.cameraAccount.fetch(cameraAddress);
    expect(camera.owner.toString()).to.equal(provider.wallet.publicKey.toString());
    expect(camera.cameraId).to.equal(cameraId);
    expect(camera.isActive).to.be.true;
    expect(camera.metadata.name).to.equal(cameraName);
    expect(camera.metadata.model).to.equal(cameraModel);
    
    // Fetch and verify the registry
    const registry = await program.account.cameraRegistry.fetch(registryAddress);
    expect(registry.cameraCount.toNumber()).to.equal(1);
  });

  it("Updates camera information", async () => {
    const [cameraAddress, _] = await PublicKey.findProgramAddress(
      [
        Buffer.from("camera"),
        Buffer.from(cameraId),
        provider.wallet.publicKey.toBuffer()
      ],
      program.programId
    );

    const newName = "Updated Camera";
    const newLocation: [number, number] = [34.0522, -118.2437]; // Los Angeles
    
    const tx = await program.methods
      .updateCamera({
        name: newName,
        location: newLocation.map(coord => new BN(coord)),
        model: null // Don't update the model
      })
      .accounts({
        owner: provider.wallet.publicKey,
        camera: cameraAddress,
      })
      .rpc();

    console.log("Transaction signature", tx);

    // Fetch and verify the updated camera
    const camera = await program.account.cameraAccount.fetch(cameraAddress);
    expect(camera.metadata.name).to.equal(newName);
    expect(camera.metadata.model).to.equal(cameraModel); // Should remain unchanged
  });

  it("Sets camera inactive", async () => {
    const [cameraAddress, _] = await PublicKey.findProgramAddress(
      [
        Buffer.from("camera"),
        Buffer.from(cameraId),
        provider.wallet.publicKey.toBuffer()
      ],
      program.programId
    );

    const tx = await program.methods
      .setCameraActive({
        isActive: false
      })
      .accounts({
        owner: provider.wallet.publicKey,
        camera: cameraAddress,
      })
      .rpc();

    console.log("Transaction signature", tx);

    // Fetch and verify camera is inactive
    const camera = await program.account.cameraAccount.fetch(cameraAddress);
    expect(camera.isActive).to.be.false;
  });

  it("Records an activity", async () => {
    // First, reactivate the camera
    const [cameraAddress, _] = await PublicKey.findProgramAddress(
      [
        Buffer.from("camera"),
        Buffer.from(cameraId),
        provider.wallet.publicKey.toBuffer()
      ],
      program.programId
    );

    await program.methods
      .setCameraActive({
        isActive: true
      })
      .accounts({
        owner: provider.wallet.publicKey,
        camera: cameraAddress,
      })
      .rpc();

    // Now record the activity
    // Current timestamp in seconds as bytes for the PDA seed
    const timestamp = new BN(Math.floor(Date.now() / 1000));
    const timestampBytes = timestamp.toArrayLike(Buffer, 'le', 8);
    
    const [activityAddress, __] = await PublicKey.findProgramAddress(
      [
        Buffer.from("activity"),
        cameraAddress.toBuffer(),
        timestampBytes
      ],
      program.programId
    );

    const activityMetadata = "Test photo capture metadata";
    
    const tx = await program.methods
      .recordActivity({
        activityType: { photoCapture: {} },
        metadata: activityMetadata
      })
      .accounts({
        owner: provider.wallet.publicKey,
        camera: cameraAddress,
        activity: activityAddress,
        systemProgram: SystemProgram.programId
      })
      .rpc();

    console.log("Transaction signature", tx);

    // Fetch and verify activity
    const activity = await program.account.activity.fetch(activityAddress);
    expect(activity.camera.toString()).to.equal(cameraAddress.toString());
    expect(activity.metadata).to.equal(activityMetadata);
  });
});