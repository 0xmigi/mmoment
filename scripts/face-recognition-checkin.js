const { Connection, Keypair, PublicKey, SystemProgram } = require('@solana/web3.js');
const anchor = require('@coral-xyz/anchor');
const fs = require('fs');
const path = require('path');

// Camera Network Program ID 
const PROGRAM_ID_STRING = 'Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S';
// Path to the program IDL
const IDL_PATH = path.join(__dirname, '../target/idl/camera_network.json');
// Path to the wallet file
const WALLET_PATH = path.join(__dirname, '../test-wallet.json');

// Function to load the wallet
function loadWallet() {
  try {
    console.log('Loading wallet from', WALLET_PATH);
    
    if (!fs.existsSync(WALLET_PATH)) {
      throw new Error(`Wallet file not found at: ${WALLET_PATH}`);
    }
    
    const walletData = fs.readFileSync(WALLET_PATH, 'utf8');
    let secretKeyArray;
    
    try {
      const parsed = JSON.parse(walletData);
      secretKeyArray = Array.isArray(parsed) ? parsed : parsed.secretKey;
    } catch (error) {
      console.error('Error parsing wallet data:', error.message);
      process.exit(1);
    }
    
    return anchor.web3.Keypair.fromSecretKey(
      new Uint8Array(secretKeyArray)
    );
  } catch (error) {
    console.error('Error loading wallet:', error.message);
    process.exit(1);
  }
}

// Function to load the IDL
function loadIDL() {
  try {
    console.log('Loading IDL from', IDL_PATH);
    
    if (!fs.existsSync(IDL_PATH)) {
      throw new Error(`IDL file not found at: ${IDL_PATH}`);
    }
    
    const idlData = fs.readFileSync(IDL_PATH, 'utf8');
    return JSON.parse(idlData);
  } catch (error) {
    console.error('Error loading IDL:', error.message);
    process.exit(1);
  }
}

// Function to confirm a transaction
async function confirmTransaction(connection, signature) {
  const latestBlockhash = await connection.getLatestBlockhash();
  return connection.confirmTransaction({
    signature,
    ...latestBlockhash,
  });
}

async function main() {
  try {
    console.log('=== FACE RECOGNITION CHECK-IN SCRIPT ===');
    
    // 1. Load wallet
    const wallet = loadWallet();
    console.log('Using wallet:', wallet.publicKey.toString());
    
    // 2. Load IDL
    const idl = loadIDL();
    console.log('IDL loaded successfully');
    
    // 3. Set up connection to Solana devnet
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    console.log('Connected to Solana devnet');
    
    // 4. Get balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`Wallet balance: ${balance / anchor.web3.LAMPORTS_PER_SOL} SOL`);
    
    // 5. Create program ID
    const programId = new PublicKey(PROGRAM_ID_STRING);
    
    // 6. Create Anchor provider
    const provider = new anchor.AnchorProvider(
      connection,
      new anchor.Wallet(wallet),
      { commitment: "confirmed" }
    );
    
    // 7. Initialize the program with properly constructed parameters
    const program = new anchor.Program(
      idl,
      programId,
      provider
    );
    console.log('Program initialized:', program.programId.toString());
    
    // 8. Find registry PDA - using the correct seed from the Solana program
    const [registryPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("camera-registry")],
      programId
    );
    console.log('Registry PDA:', registryPDA.toString());
    
    // 9. Fetch registry data
    console.log('\n=== FETCHING REGISTRY ===');
    let registry;
    try {
      registry = await program.account.cameraRegistry.fetch(registryPDA);
      console.log(`Registry found with ${registry.cameraCount} cameras`);
      console.log(`Registry authority: ${registry.authority.toString()}`);
    } catch (error) {
      console.error('\n❌ Error fetching registry:', error.message);
      console.log('Registry may not be initialized yet. Please run camera-network-client.js first.');
      process.exit(1);
    }
    
    // 10. Fetch all camera accounts
    console.log('\n=== FETCHING CAMERAS ===');
    let cameraAccounts;
    try {
      cameraAccounts = await program.account.cameraAccount.all();
      console.log(`Found ${cameraAccounts.length} cameras`);
      
      if (cameraAccounts.length === 0) {
        console.error('\n❌ No cameras found. Please register a camera first.');
        process.exit(1);
      }
      
      // Display camera details
      cameraAccounts.forEach((cam, index) => {
        console.log(`\nCamera ${index + 1}:`);
        console.log(`  PDA: ${cam.publicKey.toString()}`);
        console.log(`  Owner: ${cam.account.owner.toString()}`);
        console.log(`  Name: ${cam.account.metadata.name}`);
        console.log(`  Model: ${cam.account.metadata.model}`);
        console.log(`  Active: ${cam.account.isActive ? 'Yes' : 'No'}`);
        
        // Check if this camera belongs to the user
        if (cam.account.owner.toString() === wallet.publicKey.toString()) {
          console.log('  ✅ You own this camera');
        }
      });
    } catch (error) {
      console.error('\n❌ Error fetching cameras:', error.message);
      process.exit(1);
    }
    
    // 11. Select the first active camera
    const activeCamera = cameraAccounts.find(cam => cam.account.isActive);
    if (!activeCamera) {
      console.error('\n❌ No active cameras found.');
      console.log('Please activate a camera using the set_camera_active instruction.');
      process.exit(1);
    }
    
    const selectedCamera = activeCamera;
    console.log('\n=== SELECTED CAMERA ===');
    console.log(`Camera: ${selectedCamera.publicKey.toString()}`);
    console.log(`Name: ${selectedCamera.account.metadata.name}`);
    console.log(`Model: ${selectedCamera.account.metadata.model}`);
    
    // 12. Find face data PDA - using the exact same seed as in the Solana program
    const [faceDataPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("face-nft"),
        wallet.publicKey.toBuffer()
      ],
      programId
    );
    console.log('\n=== FACE DATA ===');
    console.log(`Face Data PDA: ${faceDataPDA.toString()}`);
    
    // 13. Check if face data exists
    let faceData;
    let hasFaceData = false;
    try {
      faceData = await program.account.faceData.fetch(faceDataPDA);
      console.log('Face data found:');
      console.log(`  Owner: ${faceData.user.toString()}`);
      console.log(`  Created: ${new Date(faceData.creationDate * 1000).toISOString()}`);
      console.log(`  Last used: ${new Date(faceData.lastUsed * 1000).toISOString()}`);
      hasFaceData = true;
    } catch (error) {
      console.log('No existing face data found. Creating new face enrollment...');
      
      // 14. Create face enrollment if it doesn't exist
      const mockEmbedding = Buffer.from(Array(128).fill(0).map(() => Math.floor(Math.random() * 256)));
      
      try {
        const tx = await program.methods
          .enrollFace(mockEmbedding)
          .accounts({
            user: wallet.publicKey,
            faceNft: faceDataPDA,
            systemProgram: SystemProgram.programId
          })
          .signers([wallet]) // Explicitly add signers
          .rpc();
          
        console.log('Face enrollment successful. Transaction signature:', tx);
        console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
        
        await confirmTransaction(connection, tx);
        console.log('✅ Face enrollment confirmed');
        hasFaceData = true;
      } catch (enrollError) {
        console.error('\n❌ Error enrolling face:', enrollError.message);
        if (enrollError.logs) {
          console.error('Transaction logs:');
          enrollError.logs.forEach(log => console.error(`  ${log}`));
        }
        process.exit(1);
      }
    }
    
    if (!hasFaceData) {
      console.error('\n❌ Failed to create or find face data.');
      process.exit(1);
    }
    
    // 15. Create session PDA - using exact seeds from the program
    const [sessionPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("session"),
        wallet.publicKey.toBuffer(),
        selectedCamera.publicKey.toBuffer()
      ],
      programId
    );
    console.log('\n=== SESSION ===');
    console.log(`Session PDA: ${sessionPDA.toString()}`);
    
    // 16. Check if session already exists
    let sessionExists = false;
    try {
      await program.account.userSession.fetch(sessionPDA);
      console.log('⚠️ You are already checked in to this camera.');
      console.log('Please check out first before checking in again.');
      sessionExists = true;
    } catch (error) {
      // Session doesn't exist, which is what we want
      console.log('No existing session found. Ready to check in.');
    }
    
    if (sessionExists) {
      // If already checked in, offer to check out
      console.log('\n=== CHECKING OUT FIRST ===');
      try {
        const tx = await program.methods
          .checkOut()
          .accounts({
            user: wallet.publicKey,
            camera: selectedCamera.publicKey,
            session: sessionPDA
          })
          .signers([wallet]) // Explicitly add signers
          .rpc();
          
        console.log('Check-out successful. Transaction signature:', tx);
        console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
        
        await confirmTransaction(connection, tx);
        console.log('✅ Check-out confirmed');
      } catch (error) {
        console.error('\n❌ Error checking out:', error.message);
        if (error.logs) {
          console.error('Transaction logs:');
          error.logs.forEach(log => console.error(`  ${log}`));
        }
        process.exit(1);
      }
    }
    
    // 17. Now perform check-in with face recognition
    console.log('\n=== PERFORMING FACE RECOGNITION CHECK-IN ===');
    try {
      const tx = await program.methods
        .checkIn(true) // Use face recognition = true
        .accounts({
          user: wallet.publicKey,
          camera: selectedCamera.publicKey,
          session: sessionPDA,
          systemProgram: SystemProgram.programId
        })
        .signers([wallet]) // Explicitly add signers
        .rpc();
        
      console.log('Check-in successful. Transaction signature:', tx);
      console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
      
      await confirmTransaction(connection, tx);
      console.log('✅ Face recognition check-in confirmed');
      
      // 18. Fetch session data to verify
      const sessionAccount = await program.account.userSession.fetch(sessionPDA);
      console.log('\n=== SESSION DATA ===');
      console.log(`User: ${sessionAccount.user.toString()}`);
      console.log(`Camera: ${sessionAccount.camera.toString()}`);
      console.log(`Check-in time: ${new Date(sessionAccount.checkInTime * 1000).toISOString()}`);
      console.log('Enabled features:');
      console.log(`  Face recognition: ${sessionAccount.enabledFeatures.faceRecognition}`);
      console.log(`  Gesture control: ${sessionAccount.enabledFeatures.gestureControl}`);
      console.log(`  Video recording: ${sessionAccount.enabledFeatures.videoRecording}`);
      console.log(`  Live streaming: ${sessionAccount.enabledFeatures.liveStreaming}`);
    } catch (error) {
      console.error('\n❌ Error checking in with face recognition:', error.message);
      if (error.logs) {
        console.error('Transaction logs:');
        error.logs.forEach(log => console.error(`  ${log}`));
      }
      process.exit(1);
    }
  } catch (error) {
    console.error('Error in face recognition check-in:', error.message);
    process.exit(1);
  }
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  }
); 