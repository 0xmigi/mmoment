const { PublicKey } = require('@solana/web3.js');

// Self-executing async function
(async () => {
  const programId = new PublicKey('7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4');
  const [pda, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("camera-registry")], // Matches your Rust code's `camera_registry_seeds()`
    programId
  );
  console.log('CameraRegistry PDA:', pda.toString());
  console.log('Bump:', bump);
})().catch(err => {
  console.error('Error:', err);
}); 