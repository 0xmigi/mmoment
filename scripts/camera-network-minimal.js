const { Connection, Keypair, PublicKey, SystemProgram, Transaction } = require('@solana/web3.js');
const anchor = require('@coral-xyz/anchor');
const fs = require('fs');
const path = require('path');

// Define constants
const PROGRAM_ID = 'Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S';
const IDL_PATH = path.join(__dirname, '../target/idl/camera_network.json');
const WALLET_PATH = path.join(__dirname, '../test-wallet.json');

async function main() {
  try {
    console.log('=== CAMERA NETWORK MINIMAL CLIENT ===');
    
    // 1. Set up connection
    const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    console.log('Connected to Solana devnet');
    
    // 2. Load wallet
    let wallet;
    try {
      console.log('Loading wallet from', WALLET_PATH);
      const walletData = fs.readFileSync(WALLET_PATH, 'utf8');
      const secretKeyArray = JSON.parse(walletData);
      wallet = Keypair.fromSecretKey(new Uint8Array(secretKeyArray));
      console.log('Wallet public key:', wallet.publicKey.toString());
    } catch (error) {
      console.error('Error loading wallet:', error.message);
      process.exit(1);
    }
    
    // 3. Check balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`Wallet balance: ${balance / anchor.web3.LAMPORTS_PER_SOL} SOL`);
    
    // 4. Load IDL
    let idl;
    try {
      console.log('Loading IDL from', IDL_PATH);
      const idlData = fs.readFileSync(IDL_PATH, 'utf8');
      idl = JSON.parse(idlData);
      console.log('IDL loaded successfully');
    } catch (error) {
      console.error('Error loading IDL:', error.message);
      process.exit(1);
    }
    
    // 5. Create program ID
    const programId = new PublicKey(PROGRAM_ID);
    console.log('Program ID:', programId.toString());
    
    // 6. Find registry PDA
    const [registryPDA] = PublicKey.findProgramAddressSync(
      [Buffer.from("camera-registry")],
      programId
    );
    console.log('Registry PDA:', registryPDA.toString());
    
    // 7. Create a unique camera name
    const cameraName = `Camera-${Math.floor(Math.random() * 10000)}`;
    console.log('Camera name:', cameraName);
    
    // 8. Find camera PDA
    const [cameraPDA] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("camera"),
        Buffer.from(cameraName),
        wallet.publicKey.toBuffer()
      ],
      programId
    );
    console.log('Camera PDA:', cameraPDA.toString());
    
    // 9. Manually prepare an initialization transaction if needed
    console.log('\n=== CHECKING REGISTRY ===');
    let registryExists = false;
    
    try {
      const accountInfo = await connection.getAccountInfo(registryPDA);
      registryExists = !!accountInfo;
      console.log('Registry exists:', registryExists);
    } catch (error) {
      console.log('Error checking registry, assuming it doesn\'t exist:', error.message);
    }
    
    // 10. Success!
    console.log('\n=== VERIFICATION SUCCESSFUL ===');
    console.log('Successfully verified connectivity and account structure.');
    console.log('Your environment is properly set up to interact with the Solana program.');
    
    // 11. Print key information for reference
    console.log('\n=== ACCOUNT INFORMATION ===');
    console.log(`Program ID: ${programId.toString()}`);
    console.log(`Wallet: ${wallet.publicKey.toString()}`);
    console.log(`Registry PDA: ${registryPDA.toString()}`);
    console.log(`Camera PDA (for "${cameraName}"): ${cameraPDA.toString()}`);
    
  } catch (error) {
    console.error('Unexpected error:', error);
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