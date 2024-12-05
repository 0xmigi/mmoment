import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { Connection, PublicKey, clusterApiUrl, Transaction } from "@solana/web3.js";
import { IDL } from "./idl";

const programID = new PublicKey("5HFxUPQ7aZPZF8UqRSSHsoamq3X4VU1K9euMXuuGcUfj");
const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

// Define a minimal type for our signer
type DynamicSigner = {
  signTransaction(tx: Transaction): Promise<Transaction>;
  signAllTransactions(txs: Transaction[]): Promise<Transaction[]>;
};

export const useProgram = () => {
  const { primaryWallet } = useDynamicContext();
  
  if (!primaryWallet?.address) return null;

  // @ts-ignore - Dynamic SDK types don't match runtime behavior
  if (!primaryWallet?.connector?.getSigner) {
    console.error('Wallet not properly initialized');
    return null;
  }

  const walletAdapter = {
    publicKey: new PublicKey(primaryWallet.address),
    async signTransaction(tx: Transaction) {
      try {
        // @ts-ignore - We know getSigner exists at runtime
        const signer = await primaryWallet.connector.getSigner() as DynamicSigner;
        return signer.signTransaction(tx);
      } catch (err) {
        console.error('Failed to sign transaction:', err);
        throw new Error('Failed to sign transaction');
      }
    },
    async signAllTransactions(txs: Transaction[]) {
      try {
        // @ts-ignore - We know getSigner exists at runtime
        const signer = await primaryWallet.connector.getSigner() as DynamicSigner;
        return signer.signAllTransactions(txs);
      } catch (err) {
        console.error('Failed to sign transactions:', err);
        throw new Error('Failed to sign transactions');
      }
    }
  };

  try {
    const provider = new AnchorProvider(connection, walletAdapter as any, {
      commitment: 'confirmed',
      preflightCommitment: 'processed'
    });

    return new Program(IDL, programID, provider);
  } catch (err) {
    console.error('Failed to create program:', err);
    return null;
  }
};