import * as anchor from "@coral-xyz/anchor";
import { PublicKey, Keypair, Connection } from "@solana/web3.js";
import * as fs from "fs";
import * as path from "path";

const PROGRAM_ID = new PublicKey("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL");
const CAMERA_PDA = new PublicKey("ArQxL9kzhZ8QhJtNodnuMvkd3HGdkwSsTzbD4qD9QqKv");
const NEW_DEVICE_KEY = new PublicKey("3ceanM1MHQi79hBERfv3Ur1TYbVDbBCs2m9aq4PGRK2X");

async function main() {
  // Load owner keypair
  const ownerKeypath = path.join(
    process.env.HOME!,
    ".config/solana/program-keypairs/camera-owner-keypair.json"
  );
  const ownerKey = Uint8Array.from(JSON.parse(fs.readFileSync(ownerKeypath, "utf-8")));
  const ownerKeypair = Keypair.fromSecretKey(ownerKey);
  console.log("Owner:", ownerKeypair.publicKey.toBase58());

  // Setup connection and provider
  const connection = new Connection("https://api.devnet.solana.com", "confirmed");
  const wallet = new anchor.Wallet(ownerKeypair);
  const provider = new anchor.AnchorProvider(connection, wallet, {
    commitment: "confirmed",
  });

  // Load IDL
  const idl = JSON.parse(
    fs.readFileSync(path.join(__dirname, "../app/backend/src/idl.json"), "utf-8")
  );
  const program = new anchor.Program(idl, PROGRAM_ID, provider);

  console.log("Calling updateCamera...");
  console.log("  Camera PDA:", CAMERA_PDA.toBase58());
  console.log("  New device key:", NEW_DEVICE_KEY.toBase58());

  const tx = await program.methods
    .updateCamera({
      name: null,
      location: null,
      description: null,
      features: null,
      devicePubkey: NEW_DEVICE_KEY,
    })
    .accounts({
      owner: ownerKeypair.publicKey,
      camera: CAMERA_PDA,
    })
    .rpc();

  console.log("Success! Tx:", tx);
}

main().catch(console.error);
