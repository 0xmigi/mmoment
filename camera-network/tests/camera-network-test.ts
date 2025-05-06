import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { PublicKey, Keypair, SystemProgram } from '@solana/web3.js';
import { expect } from 'chai';

describe('camera-network', () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  // Use the program ID for camera-network
  const PROGRAM_ID = new PublicKey("Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S");
  
  // Create program interface from IDL
  const program = new anchor.Program(
    require('../target/idl/camera_network.json'),
    PROGRAM_ID,
    provider
  );
  
  // Find the registry PDA
  const [registryPDA, registryBump] = PublicKey.findProgramAddressSync(
    [Buffer.from('camera-registry')],
    program.programId
  );
  
  // Generate a unique camera name for testing
  const cameraName = `TestCam-${Math.floor(Math.random() * 10000)}`;
  
  // Find the camera PDA
  const [cameraPDA, cameraBump] = PublicKey.findProgramAddressSync(
    [
      Buffer.from('camera'),
      Buffer.from(cameraName),
      provider.wallet.publicKey.toBuffer()
    ],
    program.programId
  );
  
  // Find the session PDA
  const [sessionPDA, sessionBump] = PublicKey.findProgramAddressSync(
    [
      Buffer.from('session'),
      provider.wallet.publicKey.toBuffer(),
      cameraPDA.toBuffer()
    ],
    program.programId
  );
  
  // Calculate PDA for face data
  const [faceDataPDA, faceDataBump] = PublicKey.findProgramAddressSync(
    [Buffer.from('face-nft'), provider.wallet.publicKey.toBuffer()],
    program.programId
  );

  it('Initialize camera registry', async () => {
    try {
      // Check if registry already exists
      try {
        await program.account.cameraRegistry.fetch(registryPDA);
        console.log('Registry already exists, skipping initialization');
      } catch (error) {
        // Initialize the registry
        const tx = await program.methods
          .initialize()
          .accounts({
            authority: provider.wallet.publicKey,
            cameraRegistry: registryPDA,
            systemProgram: SystemProgram.programId
          })
          .rpc();
        
        console.log("Registry initialization transaction signature:", tx);
        
        // Verify the registry account was created
        const registryAccount = await program.account.cameraRegistry.fetch(registryPDA);
        expect(registryAccount.authority.toString()).to.equal(provider.wallet.publicKey.toString());
      }
    } catch (error) {
      console.error('Error in registry initialization test:', error);
      throw error;
    }
  });

  it('Register a camera with face recognition', async () => {
    try {
      const tx = await program.methods
        .registerCamera({
          name: cameraName,
          model: "TestModel-X1",
          location: null,
          description: "Test camera for automated tests",
          features: {
            faceRecognition: true,
            gestureControl: true,
            videoRecording: true,
            liveStreaming: true,
            messaging: false
          }
        })
        .accounts({
          owner: provider.wallet.publicKey,
          cameraRegistry: registryPDA,
          camera: cameraPDA,
          systemProgram: SystemProgram.programId
        })
        .rpc();
      
      console.log("Camera registration transaction signature:", tx);
      
      // Verify the camera account was created
      const cameraAccount = await program.account.cameraAccount.fetch(cameraPDA);
      expect(cameraAccount.owner.toString()).to.equal(provider.wallet.publicKey.toString());
      expect(cameraAccount.metadata.name).to.equal(cameraName);
      expect(cameraAccount.features.faceRecognition).to.equal(true);
    } catch (error) {
      console.error('Error in camera registration test:', error);
      throw error;
    }
  });

  it('Enroll face data', async () => {
    try {
      // Simple test face data
      const faceData = Buffer.from([1, 2, 3, 4, 5, 6, 7, 8]);
      
      try {
        const tx = await program.methods
          .enrollFace(faceData)
          .accounts({
            user: provider.wallet.publicKey,
            faceNft: faceDataPDA,
            systemProgram: SystemProgram.programId
          })
          .rpc();
        
        console.log("Face enrollment transaction signature:", tx);
      } catch (error) {
        // If account already exists, we can continue
        if (!error.message.includes('already in use')) {
          throw error;
        }
        console.log('Face data already enrolled, continuing with test');
      }
      
      // Verify the face data account exists
      const faceAccount = await program.account.faceData.fetch(faceDataPDA);
      expect(faceAccount.user.toString()).to.equal(provider.wallet.publicKey.toString());
      expect(faceAccount.dataHash.some(byte => byte !== 0)).to.equal(true);
    } catch (error) {
      console.error('Error in face enrollment test:', error);
      throw error;
    }
  });

  it('Check in with face recognition', async () => {
    try {
      const tx = await program.methods
        .checkIn(true) // Use face recognition
        .accounts({
          user: provider.wallet.publicKey,
          camera: cameraPDA,
          session: sessionPDA,
          systemProgram: SystemProgram.programId
        })
        .rpc();
      
      console.log("Check-in transaction signature:", tx);
      
      // Verify the session account was created
      const sessionAccount = await program.account.userSession.fetch(sessionPDA);
      expect(sessionAccount.user.toString()).to.equal(provider.wallet.publicKey.toString());
      expect(sessionAccount.camera.toString()).to.equal(cameraPDA.toString());
      expect(sessionAccount.enabledFeatures.faceRecognition).to.equal(true);
      expect(sessionAccount.checkInTime).to.be.greaterThan(0);
    } catch (error) {
      console.error('Error in check-in test:', error);
      throw error;
    }
  });

  it('Check out from camera', async () => {
    try {
      const tx = await program.methods
        .checkOut()
        .accounts({
          user: provider.wallet.publicKey,
          camera: cameraPDA,
          session: sessionPDA
        })
        .rpc();
      
      console.log("Check-out transaction signature:", tx);
      
      // Verify session account is closed
      try {
        await program.account.userSession.fetch(sessionPDA);
        expect.fail("Session account should be closed");
      } catch (error) {
        expect(error.toString()).to.include("Account does not exist");
      }
    } catch (error) {
      console.error('Error in check-out test:', error);
      throw error;
    }
  });
}); 