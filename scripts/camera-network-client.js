const { Connection, Keypair, PublicKey, SystemProgram, Transaction, sendAndConfirmTransaction } = require('@solana/web3.js');
const anchor = require('@coral-xyz/anchor');
const fs = require('fs');
const path = require('path');

// Camera Network Program ID - Using the actual program ID from the keypair
const PROGRAM_ID = new PublicKey('Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S'); // Updated to match the deployed program
// Path to the program IDL
const IDL_PATH = path.join(__dirname, '../target/idl/camera_network.json');
// Path to the wallet file
const WALLET_PATH = path.join(__dirname, '../test-wallet.json');

// Helper function to confirm transactions
async function confirmTransaction(connection, signature) {
  const latestBlockhash = await connection.getLatestBlockhash();
  return connection.confirmTransaction({
    signature,
    ...latestBlockhash
  });
}

// Function to ensure the IDL file exists
async function ensureIdlExists() {
  try {
    fs.accessSync(IDL_PATH, fs.constants.F_OK);
    console.log('IDL file found at:', IDL_PATH);
    return true;
  } catch (error) {
    console.error(`\n❌ IDL file not found at: ${IDL_PATH}`);
    console.error('Please build the program first with: yarn build-camera');
    return false;
  }
}

// Function to load wallet with robust error handling
function loadWallet() {
  try {
    console.log('Loading wallet from', WALLET_PATH);
    
    if (!fs.existsSync(WALLET_PATH)) {
      throw new Error(`Wallet file not found at: ${WALLET_PATH}`);
    }
    
    const walletData = fs.readFileSync(WALLET_PATH, 'utf8');
    
    let secretKeyArray;
    try {
      // Try parsing as JSON
      const parsed = JSON.parse(walletData);
      // Check if it's already an array
      secretKeyArray = Array.isArray(parsed) ? parsed : parsed.secretKey;
      
      // If still not an array, try more parsing
      if (!Array.isArray(secretKeyArray)) {
        try {
          secretKeyArray = JSON.parse(secretKeyArray);
        } catch (e) {
          throw new Error('Wallet file content is not in a recognized format');
        }
      }
    } catch (e) {
      // If not JSON, try as a Uint8Array string representation
      try {
        // Handle possible [1,2,3,...] format as string
        if (walletData.trim().startsWith('[') && walletData.trim().endsWith(']')) {
          secretKeyArray = JSON.parse(walletData);
        } else {
          throw new Error('Unable to parse wallet file');
        }
      } catch (err) {
        throw new Error(`Failed to parse wallet data: ${err.message}`);
      }
    }
    
    if (!Array.isArray(secretKeyArray) || secretKeyArray.length !== 64) {
      throw new Error(`Invalid wallet format: Expected an array of 64 numbers, got ${secretKeyArray ? secretKeyArray.length : 'undefined'}`);
    }
    
    return Keypair.fromSecretKey(Uint8Array.from(secretKeyArray));
  } catch (error) {
    console.error('Error loading wallet:', error.message);
    console.error('\nPlease ensure test-wallet.json contains a valid Solana keypair in one of these formats:');
    console.error('1. A JSON array of 64 numbers');
    console.error('2. An object with a secretKey field');
    throw error;
  }
}

// Main function to demonstrate camera network functionality
async function main() {
  try {
    console.log('=== CAMERA NETWORK CLIENT ===');
    
    // Connect to Devnet
    console.log('Connecting to Solana devnet...');
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    
    // Ensure IDL exists before proceeding
    const idlExists = await ensureIdlExists();
    if (!idlExists) {
      process.exit(1);
    }
    
    // Load the test wallet from file with better error handling
    let wallet;
    try {
      wallet = loadWallet();
      console.log('Wallet public key:', wallet.publicKey.toString());
    } catch (error) {
      console.error('\n❌ Failed to load wallet. Please fix the wallet file and try again.');
      process.exit(1);
    }
    
    // Check the wallet balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`Wallet balance: ${balance / 1000000000} SOL`);
    
    if (balance < 0.1 * 1000000000) {
      console.log('\n⚠️ Warning: Wallet balance is low. You may need to fund this wallet.');
      console.log('To fund your wallet, visit: https://solfaucet.com/');
      console.log('Use this address:', wallet.publicKey.toString());
      const userResponse = await new Promise(resolve => {
        console.log('\nDo you want to continue despite low balance? (y/N): ');
        process.stdin.once('data', data => {
          resolve(data.toString().trim().toLowerCase());
        });
      });
      
      if (userResponse !== 'y') {
        console.log('Exiting due to low balance. Please fund your wallet and try again.');
        process.exit(0);
      }
    }
    
    // Create Anchor provider
    const walletProvider = new anchor.Wallet(wallet);
    const provider = new anchor.AnchorProvider(
      connection,
      walletProvider,
      { commitment: 'confirmed' }
    );
    
    // Read the IDL file
    let idl;
    try {
      const idlFile = fs.readFileSync(IDL_PATH, 'utf8');
      idl = JSON.parse(idlFile);
      console.log('Successfully loaded IDL from:', IDL_PATH);
    } catch (error) {
      console.error(`\n❌ Error reading IDL file: ${error.message}`);
      console.error('Make sure you have built the program with: yarn build-camera');
      process.exit(1);
    }
    
    // Create a program instance
    const program = new anchor.Program(idl, PROGRAM_ID, provider);
    
    // Find registry PDA
    const [registryPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("camera-registry")],
      program.programId
    );
    
    console.log('\n=== INITIALIZING REGISTRY ===');
    
    // Check if registry already exists
    let registryExists = false;
    try {
      const registryAccount = await program.account.cameraRegistry.fetch(registryPDA);
      console.log('Registry already exists with authority:', registryAccount.authority.toString());
      console.log('Camera count:', registryAccount.cameraCount.toString());
      registryExists = true;
    } catch (error) {
      console.log('Registry does not exist yet, will initialize...');
    }
    
    // Initialize registry if it doesn't exist
    if (!registryExists) {
      try {
        const tx = await program.methods
          .initialize()
          .accounts({
            authority: wallet.publicKey,
            cameraRegistry: registryPDA,
            systemProgram: SystemProgram.programId
          })
          .rpc();
          
        console.log('Registry initialized successfully. Transaction signature:', tx);
        console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
        
        await confirmTransaction(connection, tx);
        console.log('✅ Registry initialization confirmed');
      } catch (error) {
        console.error('\n❌ Error initializing registry:', error.message);
        if (error.logs) {
          console.error('Transaction logs:');
          error.logs.forEach(log => console.error(`  ${log}`));
        }
        process.exit(1);
      }
    }
    
    // Create a camera
    console.log('\n=== REGISTERING CAMERA ===');
    
    // Generate a unique camera name
    const cameraName = `Camera-${Math.floor(Math.random() * 10000)}`;
    console.log('Camera name:', cameraName);
    
    // Find camera PDA
    const [cameraPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("camera"),
        Buffer.from(cameraName),
        wallet.publicKey.toBuffer()
      ],
      program.programId
    );
    
    console.log('Camera PDA:', cameraPDA.toString());
    
    // Register the camera
    try {
      const tx = await program.methods
        .registerCamera({
          name: cameraName,
          model: "Test-Model-X1",
          location: null,
          description: "Test camera for demonstration",
          features: {
            faceRecognition: true,
            gestureControl: true, 
            videoRecording: true,
            liveStreaming: true,
            messaging: true
          }
        })
        .accounts({
          owner: wallet.publicKey,
          cameraRegistry: registryPDA,
          camera: cameraPDA,
          systemProgram: SystemProgram.programId
        })
        .rpc();
        
      console.log('Camera registered successfully. Transaction signature:', tx);
      console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
      
      await confirmTransaction(connection, tx);
      console.log('✅ Camera registration confirmed');
    } catch (error) {
      console.error('\n❌ Error registering camera:', error.message);
      if (error.logs) {
        console.error('Transaction logs:');
        error.logs.forEach(log => console.error(`  ${log}`));
      }
      process.exit(1);
    }
    
    // Make a second keypair to simulate another user
    const otherUser = Keypair.generate();
    console.log('\n=== CREATED TEST USER ===');
    console.log('Test user public key:', otherUser.publicKey.toString());
    
    // Fund the test user with a small amount
    console.log('\n=== FUNDING TEST USER ===');
    try {
      const fundTx = new Transaction().add(
        SystemProgram.transfer({
          fromPubkey: wallet.publicKey,
          toPubkey: otherUser.publicKey,
          lamports: 0.01 * 1000000000 // 0.01 SOL
        })
      );
      
      const fundSig = await sendAndConfirmTransaction(connection, fundTx, [wallet]);
      console.log('Funding transaction signature:', fundSig);
      console.log('✅ Test user funded with 0.01 SOL');
    } catch (error) {
      console.error('\n❌ Error funding test user:', error.message);
      process.exit(1);
    }
    
    // Perform check-in with the second user
    console.log('\n=== CHECKING IN TEST USER ===');
    
    // Find session PDA
    const [sessionPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("session"),
        otherUser.publicKey.toBuffer(),
        cameraPDA.toBuffer()
      ],
      program.programId
    );
    
    console.log('Session PDA:', sessionPDA.toString());
    
    // Create check-in transaction
    try {
      // To make the second user the signer, we need to create a new provider
      const otherUserProvider = new anchor.AnchorProvider(
        connection,
        new anchor.Wallet(otherUser),
        { commitment: 'confirmed' }
      );
      
      const otherUserProgram = new anchor.Program(idl, PROGRAM_ID, otherUserProvider);
      
      const tx = await otherUserProgram.methods
        .checkIn(false)
        .accounts({
          user: otherUser.publicKey,
          camera: cameraPDA,
          session: sessionPDA,
          systemProgram: SystemProgram.programId
        })
        .signers([otherUser])
        .rpc();
        
      console.log('Check-in successful. Transaction signature:', tx);
      console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
      
      await confirmTransaction(connection, tx);
      console.log('✅ Check-in confirmed');
      
      // Fetch session data
      const sessionAccount = await program.account.userSession.fetch(sessionPDA);
      console.log('\n=== SESSION DATA ===');
      console.log('User:', sessionAccount.user.toString());
      console.log('Camera:', sessionAccount.camera.toString());
      console.log('Check-in time:', new Date(sessionAccount.checkInTime * 1000).toISOString());
      console.log('Enabled features:');
      console.log('- Face recognition:', sessionAccount.enabledFeatures.faceRecognition);
      console.log('- Gesture control:', sessionAccount.enabledFeatures.gestureControl);
      console.log('- Video recording:', sessionAccount.enabledFeatures.videoRecording);
      console.log('- Live streaming:', sessionAccount.enabledFeatures.liveStreaming);
    } catch (error) {
      console.error('\n❌ Error checking in:', error.message);
      if (error.logs) {
        console.error('Transaction logs:');
        error.logs.forEach(log => console.error(`  ${log}`));
      }
      process.exit(1);
    }
    
    // Check out the user
    console.log('\n=== CHECKING OUT TEST USER ===');
    
    try {
      const otherUserProvider = new anchor.AnchorProvider(
        connection,
        new anchor.Wallet(otherUser),
        { commitment: 'confirmed' }
      );
      
      const otherUserProgram = new anchor.Program(idl, PROGRAM_ID, otherUserProvider);
      
      const tx = await otherUserProgram.methods
        .checkOut()
        .accounts({
          user: otherUser.publicKey,
          camera: cameraPDA,
          session: sessionPDA
        })
        .signers([otherUser])
        .rpc();
        
      console.log('Check-out successful. Transaction signature:', tx);
      console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
      
      await confirmTransaction(connection, tx);
      console.log('✅ Check-out confirmed');
      
      // Verify session account is closed
      try {
        await program.account.userSession.fetch(sessionPDA);
        console.log('❌ ERROR: Session still exists after check-out!');
      } catch (error) {
        if (error.message.includes('Account does not exist')) {
          console.log('✅ Session successfully closed');
        } else {
          console.error('Unexpected error checking session:', error.message);
        }
      }
    } catch (error) {
      console.error('\n❌ Error checking out:', error.message);
      if (error.logs) {
        console.error('Transaction logs:');
        error.logs.forEach(log => console.error(`  ${log}`));
      }
    }
    
    console.log('\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉');
    console.log('You have successfully:');
    console.log('1. Initialized the camera registry');
    console.log('2. Registered a camera with full features');
    console.log('3. Created a test user and funded it');
    console.log('4. Checked in the test user to the camera');
    console.log('5. Checked out the test user');
    console.log('\nTo run this demonstration again, use: yarn camera-client');
    
  } catch (error) {
    console.error('\n❌ Unexpected error:', error);
    console.log('\nPlease try:');
    console.log('1. Ensuring the program is built and deployed: yarn build-camera && yarn deploy-camera');
    console.log('2. Ensuring you have enough SOL in your wallet');
    console.log('3. Checking program ID matches declaration in code');
    process.exit(1);
  }
}

main(); 