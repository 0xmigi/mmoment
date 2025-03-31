const { Connection, PublicKey } = require('@solana/web3.js');

// Connect to the local Solana cluster
const connection = new Connection('http://localhost:8899', 'confirmed');

// Program IDs
const programIds = [
  new PublicKey('8ik6yJHxg3KSmfGPXsCRfancPFuA5mWJbZQTvfpncJVm'), // my_solana_project
  new PublicKey('77HrUp2XLQGe4tN6pMmHxLLkZnERhVgjJBRxujaVBF2'),  // camera_activation
];

async function main() {
  try {
    console.log('Verifying program deployment on local testnet...');
    
    for (const programId of programIds) {
      try {
        const programInfo = await connection.getAccountInfo(programId);
        
        if (programInfo) {
          console.log(`Program ${programId.toString()} is deployed!`);
          console.log(`- Owner: ${programInfo.owner.toString()}`);
          console.log(`- Executable: ${programInfo.executable}`);
          console.log(`- Data Length: ${programInfo.data.length} bytes`);
          console.log(`- Lamports (balance): ${programInfo.lamports / 1000000000} SOL`);
        } else {
          console.log(`Program ${programId.toString()} NOT FOUND!`);
        }
      } catch (err) {
        console.error(`Error checking program ${programId.toString()}:`, err);
      }
      
      console.log('---');
    }
    
  } catch (error) {
    console.error('Unexpected error:', error);
  }
}

main().then(() => console.log('Verification complete!')).catch(console.error); 