import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { PublicKey, Keypair, Connection } from '@solana/web3.js';
import fs from 'fs';
import path from 'path';

// Use the updated program ID
const PROGRAM_ID = new PublicKey('7kRohTiv527zqCZ8CTNUgh82R6HWLBg1RwfwHXw4qTD1');

async function main() {
  // Set up connection to devnet
  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  
  // Load your keypair - modify this path as needed
  let keypairPath;
  try {
    keypairPath = path.resolve(process.env.HOME || '', '.config/solana/id.json');
    console.log(`Looking for keypair at: ${keypairPath}`);
  } catch (err) {
    console.error('Failed to resolve keypair path:', err);
    process.exit(1);
  }
  
  // Load the keypair from the file
  let keypairData;
  try {
    keypairData = JSON.parse(fs.readFileSync(keypairPath, 'utf-8'));
  } catch (err) {
    console.error(`Failed to read keypair from ${keypairPath}:`, err);
    process.exit(1);
  }
  
  // Create keypair from the loaded data
  const wallet = Keypair.fromSecretKey(new Uint8Array(keypairData));
  console.log(`Using wallet address: ${wallet.publicKey.toString()}`);
  
  // Create provider
  const provider = new anchor.AnchorProvider(
    connection,
    {
      publicKey: wallet.publicKey,
      signTransaction: async (tx) => {
        tx.partialSign(wallet);
        return tx;
      },
      signAllTransactions: async (txs) => {
        return txs.map((tx) => {
          tx.partialSign(wallet);
          return tx;
        });
      },
    },
    { commitment: 'confirmed' }
  );
  
  // Load the IDL from the deployment
  let idl;
  try {
    idl = await anchor.Program.fetchIdl(PROGRAM_ID, provider);
    console.log('Successfully fetched IDL');
  } catch (err) {
    console.error('Failed to fetch IDL:', err);
    process.exit(1);
  }
  
  if (!idl) {
    console.error('IDL not found');
    process.exit(1);
  }
  
  // Create program instance
  const program = new anchor.Program(idl, PROGRAM_ID, provider);
  console.log('Program initialized');
  
  try {
    // Find the registry PDA
    const [registryAddress, _] = await PublicKey.findProgramAddress(
      [Buffer.from('camera-registry')],
      PROGRAM_ID
    );
    console.log(`Registry address: ${registryAddress.toString()}`);
    
    // Check if registry exists
    try {
      const registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
      console.log('Registry already initialized:', registryAccount);
    } catch (err) {
      console.log('Registry does not exist yet, will try to initialize');
      
      // Initialize the registry
      try {
        const tx = await program.methods
          .initialize()
          .accounts({
            authority: wallet.publicKey,
            registry: registryAddress,
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([wallet])
          .rpc();
        
        console.log('Registry initialized with tx:', tx);
      } catch (initErr) {
        console.error('Failed to initialize registry:', initErr);
      }
    }
    
    // Register a camera
    const cameraId = `cam_${Date.now().toString().slice(-6)}`;
    
    const [cameraAddress] = await PublicKey.findProgramAddress(
      [
        Buffer.from('camera'),
        Buffer.from(cameraId),
        wallet.publicKey.toBuffer()
      ],
      PROGRAM_ID
    );
    
    console.log(`Registering camera with ID ${cameraId} at address ${cameraAddress.toString()}`);
    
    const tx = await program.methods
      .registerCamera({
        name: 'Test Camera',
        model: 'Test Model',
        location: null,
        fee: new anchor.BN(100),
      })
      .accounts({
        owner: wallet.publicKey,
        registry: registryAddress,
        camera: cameraAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .signers([wallet])
      .rpc();
    
    console.log('Camera registered with tx:', tx);
    
    // Fetch the camera details to verify
    const cameraAccount = await program.account.cameraAccount.fetch(cameraAddress);
    console.log('Camera details:', {
      owner: cameraAccount.owner.toString(),
      isActive: cameraAccount.isActive,
      metadata: {
        name: cameraAccount.metadata.name,
        model: cameraAccount.metadata.model,
        registrationDate: cameraAccount.metadata.registrationDate.toString(),
        lastActivity: cameraAccount.metadata.lastActivity.toString(),
      }
    });
    
    // Record an activity
    const activity = {
      photoCapture: {},
    };
    
    const activityTx = await program.methods
      .recordActivity({
        activityType: activity,
        metadata: 'Test activity',
      })
      .accounts({
        owner: wallet.publicKey,
        camera: cameraAddress,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .signers([wallet])
      .rpc();
    
    console.log('Activity recorded with tx:', activityTx);
    
    // Fetch updated camera to check activity counter
    const updatedCamera = await program.account.cameraAccount.fetch(cameraAddress);
    console.log('Updated camera activity counter:', updatedCamera.activityCounter.toString());
    
  } catch (err) {
    console.error('Error during execution:', err);
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  }
); 