import * as anchor from '@coral-xyz/anchor';
import { Program, Idl, AnchorProvider } from '@coral-xyz/anchor';
import { PublicKey, Keypair, Connection, Transaction } from '@solana/web3.js';
import fs from 'fs';
import path from 'path';

// Use the updated program ID
const PROGRAM_ID = new PublicKey('7kRohTiv527zqCZ8CTNUgh82R6HWLBg1RwfwHXw4qTD1');

// Define a simple type for our program
interface ProgramAccounts {
  camera: any;
  session: any;
}

async function main() {
  // Set up connection to devnet
  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  
  // Load keypair
  let keypairPath;
  try {
    keypairPath = path.resolve(process.env.HOME || '', '.config/solana/id.json');
    console.log(`Looking for keypair at: ${keypairPath}`);
  } catch (err: any) {
    console.error('Failed to resolve keypair path:', err.message);
    process.exit(1);
  }
  
  let keypairData;
  try {
    keypairData = JSON.parse(fs.readFileSync(keypairPath, 'utf-8'));
  } catch (err: any) {
    console.error(`Failed to read keypair from ${keypairPath}:`, err.message);
    process.exit(1);
  }
  
  const wallet = Keypair.fromSecretKey(new Uint8Array(keypairData));
  console.log(`Using wallet address: ${wallet.publicKey.toString()}`);
  
  // Create provider with proper typing for transactions
  const provider = new anchor.AnchorProvider(
    connection,
    {
      publicKey: wallet.publicKey,
      signTransaction: async (tx) => {
        if (tx instanceof Transaction) {
          tx.partialSign(wallet);
        }
        return tx;
      },
      signAllTransactions: async (txs) => {
        return txs.map((tx) => {
          if (tx instanceof Transaction) {
            tx.partialSign(wallet);
          }
          return tx;
        });
      },
    },
    { commitment: 'confirmed' }
  );
  
  // Set provider as the default provider
  anchor.setProvider(provider);
  
  // Load the IDL from the deployment
  let idl;
  try {
    console.log('Attempting to fetch IDL from the chain for program:', PROGRAM_ID.toString());
    idl = await anchor.Program.fetchIdl(
      PROGRAM_ID, 
      provider
    );
    
    if (!idl) {
      throw new Error('IDL is null');
    }
    console.log('Successfully fetched IDL');
  } catch (err: any) {
    console.log('Failed to fetch IDL from chain:', err.message);
    
    // Fallback to local IDL if available
    try {
      const localIdlPath = path.resolve('./target/idl/simple_checkin.json');
      console.log(`Trying to load local IDL from ${localIdlPath}`);
      idl = JSON.parse(fs.readFileSync(localIdlPath, 'utf-8'));
      console.log('Successfully loaded local IDL');
    } catch (localErr: any) {
      console.error('Failed to load local IDL:', localErr.message);
      process.exit(1);
    }
  }
  
  if (!idl) {
    console.error('IDL not found');
    process.exit(1);
  }
  
  // Create program instance with the correct constructor for Anchor 0.31.x
  // The signature is: new Program<T>(idl, programId, provider)
  const program = new anchor.Program(
    idl,
    PROGRAM_ID,
    provider
  );
  
  console.log('Program initialized');
  
  try {
    // Use an existing camera or register a new one
    const cameraId = process.argv[2] || `camera_${Date.now().toString().slice(-6)}`;
    console.log('Getting or creating camera with ID:', cameraId);
    
    // Find camera PDA
    const [cameraAddress] = await PublicKey.findProgramAddress(
      [
        Buffer.from('camera'),
        Buffer.from(cameraId),
        wallet.publicKey.toBuffer()
      ],
      PROGRAM_ID
    );
    console.log(`Camera address: ${cameraAddress.toString()}`);
    
    // Find registry address
    const [registryAddress] = await PublicKey.findProgramAddress(
      [Buffer.from('camera-registry')],
      PROGRAM_ID
    );
    console.log(`Registry address: ${registryAddress.toString()}`);
    
    // Check if camera exists or register it
    let cameraExists = false;
    try {
      // Use generic fetch with the account name based on IDL
      await program.account.camera.fetch(cameraAddress);
      cameraExists = true;
      console.log('Camera already exists');
    } catch (err) {
      console.log('Camera does not exist, registering new camera');
      
      // Register camera
      try {
        const tx = await program.methods
          .registerCamera({
            name: 'Check-In Test Camera',
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
        cameraExists = true;
      } catch (regErr: any) {
        console.error('Failed to register camera:', regErr.message);
      }
    }
    
    if (!cameraExists) {
      console.error('Failed to get or create camera');
      process.exit(1);
    }
    
    // Now let's try to check in
    // First find the session PDA
    const [sessionAddress] = await PublicKey.findProgramAddress(
      [
        Buffer.from('session'),
        wallet.publicKey.toBuffer(),
        cameraAddress.toBuffer()
      ],
      PROGRAM_ID
    );
    console.log(`Session address: ${sessionAddress.toString()}`);
    
    // Check if we're already checked in
    let alreadyCheckedIn = false;
    try {
      // Use generic fetch with the account name based on IDL
      await program.account.session.fetch(sessionAddress);
      alreadyCheckedIn = true;
      console.log('Already checked in, will try to check out first');
      
      // Check out first
      try {
        const checkOutTx = await program.methods
          .checkOut()
          .accounts({
            user: wallet.publicKey,
            camera: cameraAddress,
            session: sessionAddress,
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([wallet])
          .rpc();
        
        console.log('Successfully checked out with tx:', checkOutTx);
        alreadyCheckedIn = false;
      } catch (checkOutErr: any) {
        console.error('Failed to check out:', checkOutErr.message);
      }
    } catch (err) {
      console.log('Not currently checked in, proceeding with check-in');
    }
    
    if (alreadyCheckedIn) {
      console.error('Still checked in, cannot proceed with new check-in');
      process.exit(1);
    }
    
    // Now check in using the new simplified check-in endpoint
    console.log('Attempting to check in using the simplified check-in method');
    try {
      const checkInTx = await program.methods
        .checkIn()
        .accounts({
          user: wallet.publicKey,
          camera: cameraAddress,
          session: sessionAddress,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([wallet])
        .rpc();
      
      console.log('Successfully checked in with tx:', checkInTx);
      
      // Verify session data
      const sessionData = await program.account.session.fetch(sessionAddress);
      console.log('Session data:', {
        user: sessionData.user.toString(),
        camera: sessionData.camera.toString(),
        sessionStart: sessionData.sessionStart.toString(),
        features: sessionData.featuresEnabled ? {
          faceRecognition: sessionData.featuresEnabled.faceRecognition,
          gestureDetection: sessionData.featuresEnabled.gestureDetection,
        } : 'No features data'
      });
      
      // Optionally try the most basic check-in method too
      console.log('Now trying check-in-basic method as another option');
      try {
        // Check out first
        const checkOutTx = await program.methods
          .checkOut()
          .accounts({
            user: wallet.publicKey,
            camera: cameraAddress,
            session: sessionAddress,
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([wallet])
          .rpc();
        
        console.log('Successfully checked out with tx:', checkOutTx);
        
        // Now check in using check-in-basic
        const basicCheckInTx = await program.methods
          .checkInBasic()
          .accounts({
            user: wallet.publicKey,
            camera: cameraAddress,
            session: sessionAddress,
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([wallet])
          .rpc();
        
        console.log('Successfully checked in using basic method with tx:', basicCheckInTx);
      } catch (basicErr: any) {
        console.log('Note: Basic check-in alternative test failed:', basicErr.message);
      }
      
    } catch (checkInErr: any) {
      console.error('Failed to check in:', checkInErr.message);
    }
    
  } catch (err: any) {
    console.error('Error during execution:', err.message);
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  }
); 