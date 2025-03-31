const { Connection, PublicKey, Keypair, SystemProgram } = require('@solana/web3.js');
const { Program, AnchorProvider, web3, BN } = require('@coral-xyz/anchor');
const fs = require('fs');

// Load the IDL
const idl = JSON.parse(fs.readFileSync('./target/idl/camera_activation.json', 'utf8'));

// Set up the connection to the local Solana cluster
const connection = new Connection('http://localhost:8899', 'confirmed');

// Use default keypair
const keypairFile = process.env.HOME + '/.config/solana/id.json';
const secretKeyString = fs.readFileSync(keypairFile, 'utf8');
const secretKey = Uint8Array.from(JSON.parse(secretKeyString));
const wallet = Keypair.fromSecretKey(secretKey);

// Create the provider
const provider = new AnchorProvider(
  connection,
  { publicKey: wallet.publicKey, signTransaction: async (tx) => { 
    tx.partialSign(wallet); 
    return tx; 
  }},
  { commitment: 'confirmed' }
);

// Create the program interface
const programId = new PublicKey('77HrUp2XLQGe4tN6pMmHxLLkZnERhVgjJBRxujaVBF2');
const program = new Program(idl, programId, provider);

async function main() {
  try {
    console.log('Using wallet public key:', wallet.publicKey.toString());
    
    // Find the registry PDA
    const [registryAddress, _] = await PublicKey.findProgramAddress(
      [Buffer.from('camera-registry')], 
      programId
    );
    
    console.log('Registry PDA:', registryAddress.toString());
    
    // Initialize the registry
    try {
      const tx = await program.methods
        .initialize()
        .accounts({
          authority: wallet.publicKey,
          registry: registryAddress,
          systemProgram: SystemProgram.programId,
        })
        .signers([wallet])
        .rpc();
      
      console.log('Successfully initialized camera registry, transaction signature:', tx);
    } catch (error) {
      console.log('Error initializing registry (might already be initialized):', error.message);
    }
    
    // Fetch registry data
    try {
      const registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
      console.log('Registry data:', {
        authority: registryAccount.authority.toString(),
        cameraCount: registryAccount.cameraCount.toString(),
        bump: registryAccount.bump
      });
    } catch (error) {
      console.log('Error fetching registry:', error.message);
    }
    
    // Register a camera
    const cameraId = 'cam_' + Math.floor(Math.random() * 1000000).toString();
    const [cameraAddress, __] = await PublicKey.findProgramAddress(
      [
        Buffer.from('camera'),
        Buffer.from(cameraId),
        wallet.publicKey.toBuffer()
      ],
      programId
    );
    
    console.log('Camera PDA:', cameraAddress.toString());
    
    try {
      const tx = await program.methods
        .registerCamera({
          cameraId: cameraId,
          name: 'Test Camera',
          model: 'Raspberry Pi Camera Module',
          location: [new BN(37.7749), new BN(-122.4194)],
          fee: new BN(100)
        })
        .accounts({
          owner: wallet.publicKey,
          registry: registryAddress,
          camera: cameraAddress,
          systemProgram: SystemProgram.programId,
        })
        .signers([wallet])
        .rpc();
      
      console.log('Successfully registered camera, transaction signature:', tx);
    } catch (error) {
      console.log('Error registering camera:', error.message);
    }
    
    // Fetch camera data
    try {
      const cameraAccount = await program.account.cameraAccount.fetch(cameraAddress);
      console.log('Camera data:', {
        owner: cameraAccount.owner.toString(),
        cameraId: cameraAccount.cameraId,
        isActive: cameraAccount.isActive,
        model: cameraAccount.metadata.model,
        name: cameraAccount.metadata.name
      });
    } catch (error) {
      console.log('Error fetching camera:', error.message);
    }
    
    // Fetch updated registry data
    try {
      const registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
      console.log('Updated registry data:', {
        authority: registryAccount.authority.toString(),
        cameraCount: registryAccount.cameraCount.toString(),
        bump: registryAccount.bump
      });
    } catch (error) {
      console.log('Error fetching updated registry:', error.message);
    }
    
  } catch (error) {
    console.error('Unexpected error:', error);
  }
}

main().then(() => console.log('Done')).catch(console.error); 