const { Connection, Keypair, PublicKey, SystemProgram } = require('@solana/web3.js');
const anchor = require('@coral-xyz/anchor');
const fs = require('fs');
const path = require('path');

// Debug settings
const DEBUG = true;

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
      
      if (DEBUG) {
        console.log('Wallet data parsed successfully');
        console.log('Secret key type:', typeof secretKeyArray);
        console.log('Secret key is array:', Array.isArray(secretKeyArray));
        console.log('Secret key length:', secretKeyArray.length);
      }
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
    const idl = JSON.parse(idlData);
    
    if (DEBUG) {
      console.log('IDL loaded successfully');
      console.log('IDL contains these fields:', Object.keys(idl).join(', '));
      console.log('Program name:', idl.name);
      console.log('Instructions:', idl.instructions.map(i => i.name).join(', '));
      console.log('Accounts:', idl.accounts.map(a => a.name).join(', '));
    }
    
    return idl;
  } catch (error) {
    console.error('Error loading IDL:', error.message);
    process.exit(1);
  }
}

async function main() {
  try {
    console.log('=== DEBUG FACE RECOGNITION SCRIPT ===');
    
    // 1. Load wallet
    const wallet = loadWallet();
    console.log('Using wallet:', wallet.publicKey.toString());
    
    // 2. Load IDL
    const idl = loadIDL();
    
    // 3. Set up connection to Solana devnet
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    console.log('Connected to Solana devnet');
    
    // 4. Get balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`Wallet balance: ${balance / anchor.web3.LAMPORTS_PER_SOL} SOL`);
    
    // 5. Create program ID
    console.log('Creating program ID from:', PROGRAM_ID_STRING);
    const programId = new PublicKey(PROGRAM_ID_STRING);
    console.log('Program ID created successfully:', programId.toString());
    
    // 6. Debug the Program creation
    console.log('\n=== DEBUG PROGRAM CREATION ===');
    
    // First, check the IDL metadata for programID
    if (idl.metadata && idl.metadata.address) {
      console.log('IDL contains programID:', idl.metadata.address);
      
      // Check if the IDL programID matches our expected program ID
      if (idl.metadata.address !== PROGRAM_ID_STRING) {
        console.log('⚠️ WARNING: IDL programID does not match expected program ID!');
        console.log('Expected:', PROGRAM_ID_STRING);
        console.log('Found in IDL:', idl.metadata.address);
        
        // Try to fix IDL programID
        console.log('Fixing IDL programID to match expected value');
        idl.metadata.address = PROGRAM_ID_STRING;
      }
    } else {
      console.log('⚠️ WARNING: IDL does not contain programID metadata!');
      console.log('Adding programID metadata to IDL');
      
      // Add metadata if missing
      if (!idl.metadata) {
        idl.metadata = {};
      }
      idl.metadata.address = PROGRAM_ID_STRING;
    }
    
    // Create provider
    console.log('Creating Anchor provider...');
    const provider = new anchor.AnchorProvider(
      connection,
      new anchor.Wallet(wallet),
      { commitment: "confirmed" }
    );
    
    // Attempt to create program with manual approach
    try {
      console.log('Creating program with full parameter details...');
      console.log('IDL name:', idl.name);
      console.log('Program ID:', programId.toString());
      
      // Create the program instance with explicit coder
      const program = new anchor.Program(
        idl,
        programId.toString(), // Use string version as object may have issues
        provider,
        new anchor.BorshCoder(idl) // Explicitly create the coder
      );
      
      console.log('Program created successfully!');
      console.log('Program ID:', program.programId.toString());
      console.log('Program namespace contains:', Object.keys(program.methods).join(', '));
      
      // Find face data PDA
      console.log('\n=== FINDING FACE DATA PDA ===');
      const [faceDataPDA] = PublicKey.findProgramAddressSync(
        [
          Buffer.from("face-nft"),
          wallet.publicKey.toBuffer()
        ],
        programId
      );
      console.log('Face data PDA found:', faceDataPDA.toString());
      
      // Success - we've gotten past the error point
      console.log('\n✅ DEBUG SUCCESSFUL: Program instance created and PDA derived correctly!');
      
    } catch (error) {
      console.error('\n❌ ERROR CREATING PROGRAM:', error);
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
    }
    
  } catch (error) {
    console.error('Top-level error:', error);
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    process.exit(1);
  }
}

main().then(
  () => process.exit(0),
  err => {
    console.error('Final promise rejection:', err);
    console.error('Error message:', err.message);
    console.error('Error stack:', err.stack);
    process.exit(1);
  }
); 