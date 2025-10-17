const { Connection, Keypair, PublicKey } = require('@solana/web3.js');
const anchor = require('@coral-xyz/anchor');
const fs = require('fs');
const path = require('path');

// Camera Network Program ID
const PROGRAM_ID = new PublicKey('Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S');
// Path to the program IDL
const IDL_PATH = path.join(__dirname, '../target/idl/camera_network.json');
// Path to the wallet file
const WALLET_PATH = path.join(__dirname, '../test-wallet.json');

function loadWallet() {
  try {
    console.log('Loading wallet from', WALLET_PATH);
    
    if (!fs.existsSync(WALLET_PATH)) {
      throw new Error(`Wallet file not found at: ${WALLET_PATH}`);
    }
    
    const walletData = fs.readFileSync(WALLET_PATH, 'utf8');
    const secretKeyArray = JSON.parse(walletData);
    
    return Keypair.fromSecretKey(Uint8Array.from(secretKeyArray));
  } catch (error) {
    console.error('Error loading wallet:', error.message);
    process.exit(1);
  }
}

async function main() {
  try {
    console.log('=== CAMERA NETWORK VERIFICATION ===');
    
    // Connect to Devnet
    console.log('Connecting to Solana devnet...');
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    
    // Load wallet
    const wallet = loadWallet();
    console.log('Wallet public key:', wallet.publicKey.toString());
    
    // Check wallet balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`Wallet balance: ${balance / 1000000000} SOL`);
    
    // Read the IDL file
    let idl;
    try {
      console.log('Reading IDL from:', IDL_PATH);
      const idlFile = fs.readFileSync(IDL_PATH, 'utf8');
      idl = JSON.parse(idlFile);
      console.log('Successfully loaded IDL');
    } catch (error) {
      console.error(`Error reading IDL file: ${error.message}`);
      process.exit(1);
    }

    // Create Anchor provider
    const walletProvider = new anchor.Wallet(wallet);
    const provider = new anchor.AnchorProvider(
      connection,
      walletProvider,
      { commitment: 'confirmed' }
    );
    
    // Create a program instance
    console.log('Creating program instance with ID:', PROGRAM_ID.toString());
    const program = new anchor.Program(idl, PROGRAM_ID, provider);
    console.log('Program instance created successfully');
    
    // Find registry PDA
    const [registryPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("camera-registry")],
      program.programId
    );
    
    console.log('Registry PDA:', registryPDA.toString());
    
    // Try to fetch the registry account
    try {
      console.log('Fetching registry account...');
      const registryAccount = await program.account.cameraRegistry.fetch(registryPDA);
      console.log('Registry account found!');
      console.log('Authority:', registryAccount.authority.toString());
      console.log('Camera count:', registryAccount.cameraCount.toString());
    } catch (error) {
      console.log('Registry account not found, may need to be initialized:',  
                  error.message.substring(0, 100) + '...');
    }
    
    console.log('\n✅ Camera network program connection verified successfully!');
    
  } catch (error) {
    console.error('Error connecting to camera network program:', error);
    process.exit(1);
  }
}

main(); 