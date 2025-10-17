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
    console.log('=== FACE RECOGNITION TEST SCRIPT ===');
    
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
    
    // 5. Create program ID - exactly as in the program's declare_id!()
    const programId = new PublicKey(PROGRAM_ID_STRING);
    
    // 6. Create Anchor provider directly with the wallet keypair
    const provider = new anchor.AnchorProvider(
      connection,
      new anchor.Wallet(wallet),
      { commitment: "confirmed" }
    );
    
    // 7. Create program with properly initialized parameters
    // The issue is fixed by ensuring all properties are properly provided
    const program = new anchor.Program(
      idl,
      programId,
      provider
    );
    console.log('Program initialized:', program.programId.toString());
    
    // 8. Generate mock face embedding data
    console.log('\n=== GENERATING MOCK FACE EMBEDDING ===');
    const mockEmbedding = Buffer.from(Array(128).fill(0).map(() => Math.floor(Math.random() * 256)));
    console.log('Mock embedding generated');
    
    // 9. Find face data PDA - using the exact seeds from the program: "face-nft"
    const [faceDataPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("face-nft"), // EXACTLY as specified in the program
        wallet.publicKey.toBuffer()
      ],
      programId
    );
    console.log('Face data PDA:', faceDataPDA.toString());
    
    // 10. Enroll face by calling the program's enrollFace method
    console.log('\n=== ENROLLING FACE ===');
    try {
      // Get the transaction instruction with explicit accounts
      const tx = await program.methods
        .enrollFace(mockEmbedding)
        .accounts({
          user: wallet.publicKey,
          faceNft: faceDataPDA, // Matches the account name in the program
          systemProgram: SystemProgram.programId
        })
        .signers([wallet]) // Explicitly add signers
        .rpc();
        
      console.log('Face enrollment transaction sent. Signature:', tx);
      console.log('Transaction explorer URL:', `https://explorer.solana.com/tx/${tx}?cluster=devnet`);
      
      // Wait for confirmation
      await confirmTransaction(connection, tx);
      console.log('✅ Face enrollment confirmed');
      
      // Fetch the created face data account
      try {
        // NOTE: Use FaceData, NOT FaceNft - this matches the program account type
        const faceData = await program.account.faceData.fetch(faceDataPDA);
        console.log('\n=== FACE DATA ACCOUNT ===');
        console.log('Owner:', faceData.user.toString());
        console.log('Data hash:', Buffer.from(faceData.dataHash).toString('hex').substring(0, 16) + '...');
        console.log('Created at:', new Date(faceData.creationDate * 1000).toISOString());
        console.log('Last used:', new Date(faceData.lastUsed * 1000).toISOString());
        console.log('Authorized cameras:', faceData.authorizedCameras.length);
      } catch (error) {
        console.error('\n❌ Error fetching face data:', error.message);
      }
    } catch (error) {
      console.error('\n❌ Error enrolling face:', error.message);
      if (error.logs) {
        console.error('Transaction logs:');
        error.logs.forEach(log => console.error(`  ${log}`));
      }
    }
  } catch (error) {
    console.error('Error in face recognition test:', error.message);
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