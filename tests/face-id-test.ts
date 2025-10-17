import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { MysolanaProject } from "../target/types/my_solana_project";
import { expect } from "chai";

describe("Face ID Tests", () => {
  // Configure the client
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.MysolanaProject as Program<MysolanaProject>;
  const user = anchor.web3.Keypair.generate();
  const cameraOwner = anchor.web3.Keypair.generate();
  
  // Mock encrypted facial embedding (in a real scenario, this would be encrypted client-side)
  const mockEncryptedEmbedding = Buffer.from(Array(128).fill(0).map(() => Math.floor(Math.random() * 256)));
  
  // Camera details
  const cameraName = "TestCamera";
  const cameraModel = "Pi5 Camera";
  
  // PDA addresses
  let registryAddress: anchor.web3.PublicKey;
  let cameraAddress: anchor.web3.PublicKey;
  let faceIdAddress: anchor.web3.PublicKey;
  let sessionAddress: anchor.web3.PublicKey;

  before(async () => {
    // Airdrop SOL to user and camera owner for transaction fees
    const userAirdropSig = await provider.connection.requestAirdrop(
      user.publicKey,
      2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(userAirdropSig);
    
    const ownerAirdropSig = await provider.connection.requestAirdrop(
      cameraOwner.publicKey,
      2 * anchor.web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(ownerAirdropSig);
    
    // Derive PDA addresses
    [registryAddress] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("camera-registry")],
      program.programId
    );
    
    [cameraAddress] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("camera"), Buffer.from(cameraName), cameraOwner.publicKey.toBuffer()],
      program.programId
    );
    
    [faceIdAddress] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("face-id"), user.publicKey.toBuffer()],
      program.programId
    );
    
    [sessionAddress] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("session"), user.publicKey.toBuffer(), cameraAddress.toBuffer()],
      program.programId
    );
    
    // Initialize registry
    await program.methods.initialize()
      .accounts({
        authority: provider.wallet.publicKey,
        registry: registryAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .rpc();
      
    // Register camera
    await program.methods.registerCamera({
        name: cameraName,
        model: cameraModel,
        location: null,
        fee: new anchor.BN(100),
      })
      .accounts({
        owner: cameraOwner.publicKey,
        registry: registryAddress,
        camera: cameraAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .signers([cameraOwner])
      .rpc();
  });

  it("Enrolls a face ID", async () => {
    await program.methods.enrollFace({
        encrypted_embedding: Array.from(mockEncryptedEmbedding),
      })
      .accounts({
        user: user.publicKey,
        faceId: faceIdAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .signers([user])
      .rpc();
      
    // Fetch and validate the face ID account
    const faceIdAccount = await program.account.faceIdNFT.fetch(faceIdAddress);
    expect(faceIdAccount.owner.toString()).to.equal(user.publicKey.toString());
    expect(faceIdAccount.isSoulbound).to.be.true;
    expect(faceIdAccount.authorizedCameras.length).to.equal(0);
  });

  it("Checks in a user with face recognition", async () => {
    await program.methods.userCheckIn({
        face_recognition_enabled: true,
        gesture_detection_enabled: false,
      })
      .accounts({
        user: user.publicKey,
        camera: cameraAddress,
        session: sessionAddress,
        faceId: faceIdAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .signers([user])
      .rpc();
      
    // Fetch and validate the session
    const sessionAccount = await program.account.userSession.fetch(sessionAddress);
    expect(sessionAccount.user.toString()).to.equal(user.publicKey.toString());
    expect(sessionAccount.camera.toString()).to.equal(cameraAddress.toString());
    expect(sessionAccount.featuresEnabled.faceRecognition).to.be.true;
    expect(sessionAccount.featuresEnabled.gestureDetection).to.be.false;
    
    // Fetch and validate the face ID has authorized the camera
    const faceIdAccount = await program.account.faceIdNFT.fetch(faceIdAddress);
    expect(faceIdAccount.authorizedCameras.length).to.equal(1);
    expect(faceIdAccount.authorizedCameras[0].toString()).to.equal(cameraAddress.toString());
  });

  it("Checks out a user", async () => {
    await program.methods.userCheckOut()
      .accounts({
        user: user.publicKey,
        camera: cameraAddress,
        session: sessionAddress,
      })
      .signers([user])
      .rpc();
      
    // Verify the session is closed by trying to fetch it
    try {
      await program.account.userSession.fetch(sessionAddress);
      expect.fail("Session should be closed");
    } catch (e) {
      // Session account should not exist anymore
      expect(e).to.be.an("error");
    }
  });
}); 