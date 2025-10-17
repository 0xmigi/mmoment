import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { BN } from "bn.js";
import { expect } from "chai";

describe("camera-activation-simple", () => {
  // Configure the client to use the local cluster
  const provider = new anchor.AnchorProvider(
    new anchor.web3.Connection("http://localhost:8899", "confirmed"),
    new anchor.Wallet(anchor.web3.Keypair.generate()),
    { commitment: "confirmed" }
  );

  // Load the program directly from the IDL file
  const cameraActivationIdl = require("../target/idl/camera_activation.json");
  const cameraActivationProgramId = new PublicKey("77HrUp2XLQGe4tN6pMmHxLLkZnERhVgjJBRxujaVBF2");
  const cameraActivationProgram = new Program(cameraActivationIdl, cameraActivationProgramId, provider);

  it("Connects to the program", async () => {
    console.log("Program ID:", cameraActivationProgram.programId.toString());
    console.log("Provider connection status:", provider.connection.rpcEndpoint);
    expect(cameraActivationProgram.programId.toString()).to.equal("77HrUp2XLQGe4tN6pMmHxLLkZnERhVgjJBRxujaVBF2");
  });
}); 