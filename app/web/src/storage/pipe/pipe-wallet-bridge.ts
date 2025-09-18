/**
 * Dynamic Wallet → Pipe Wallet Bridge
 *
 * Handles the integration between user's Dynamic wallet and their Pipe storage account.
 * Enables users to fund their Pipe storage directly from their main wallet.
 */

// Import removed - not used in this service file

import { pipeService } from "./pipe-service";
import {
  Connection,
  PublicKey,
  Transaction,
  SystemProgram,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";

export interface PipeWalletInfo {
  pipeWalletAddress: string;
  solBalance: number;
  pipeBalance: number;
  storageUsed: number; // in MB
  storageLimit: number; // in MB
}

export interface StoragePurchaseOption {
  label: string;
  sizeGB: number;
  priceSol: number;
  recommended?: boolean;
}

export class PipeWalletBridge {
  private connection: Connection;

  constructor() {
    // Use devnet for now
    this.connection = new Connection("https://api.devnet.solana.com");
  }

  /**
   * Get user's Pipe wallet info (balance, storage, etc.)
   */
  async getUserPipeInfo(
    userWalletAddress: string
  ): Promise<PipeWalletInfo | null> {
    try {
      // Load credentials for this wallet first
      await pipeService.loadCredentialsForWallet(userWalletAddress);
      
      // Check if user has Pipe credentials
      if (!pipeService.isAvailable()) {
        // Try to create account
        console.log("🔄 No Pipe account found, creating one...");
        await pipeService.createOrGetAccount(userWalletAddress);
        
        // Load credentials again after creation
        await pipeService.loadCredentialsForWallet(userWalletAddress);
        
        if (!pipeService.isAvailable()) {
          console.log("❌ Failed to create Pipe account");
          return null;
        }
      }

      // Get balance from Pipe API
      const balance = await pipeService.checkBalance(userWalletAddress);

      // TODO: Get actual Pipe wallet address from user's main wallet
      // For now, we'll use a derived address or the one from credentials
      const pipeWalletAddress = await this.derivePipeWalletAddress(
        userWalletAddress
      );

      return {
        pipeWalletAddress,
        solBalance: balance.sol,
        pipeBalance: balance.pipe,
        storageUsed: 0, // TODO: Get from Pipe API
        storageLimit: balance.pipe * 1000, // 1 PIPE = 1GB = 1000MB
      };
    } catch (error) {
      console.error("Failed to get Pipe info:", error);
      return null;
    }
  }

  /**
   * Get the actual Pipe wallet address from the API
   */
  private async derivePipeWalletAddress(
    userWalletAddress: string
  ): Promise<string> {
    try {
      console.log(`🔍 Getting Pipe wallet address for: ${userWalletAddress}`);

      // The backend proxy uses the wallet address to look up stored JWT tokens
      // We only need to send the wallet address header
      const walletResponse = await fetch(`/api/pipe/proxy/checkWallet`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Wallet-Address": userWalletAddress,
        },
        body: JSON.stringify({}),
      });

      if (!walletResponse.ok) {
        const error = await walletResponse.text();
        console.error(`❌ checkWallet proxy failed (${walletResponse.status}):`, error);
        throw new Error(`checkWallet failed: ${error}`);
      }

      const data = await walletResponse.json();
      console.log(`📥 checkWallet response:`, data);

      if (!data.public_key) {
        throw new Error("No public_key in checkWallet response");
      }

      console.log(`🔑 Found Pipe wallet address: ${data.public_key}`);
      return data.public_key;

    } catch (error) {
      console.error("❌ Error getting Pipe wallet address:", error);
      throw error;
    }
  }

  /**
   * Send SOL from user's Dynamic wallet to their Pipe wallet
   */
  async fundPipeWallet(
    userWalletAddress: string,
    solAmount: number,
    signTransaction: (tx: Transaction) => Promise<Transaction>
  ): Promise<{ success: boolean; txHash?: string; error?: string }> {
    try {
      console.log(`💰 Funding Pipe wallet with ${solAmount} SOL...`);

      // Get user's Pipe wallet address
      const pipeWalletAddress = await this.derivePipeWalletAddress(
        userWalletAddress
      );

      // Create transaction to send SOL
      const fromPubkey = new PublicKey(userWalletAddress);
      const toPubkey = new PublicKey(pipeWalletAddress);

      const transaction = new Transaction().add(
        SystemProgram.transfer({
          fromPubkey,
          toPubkey,
          lamports: solAmount * LAMPORTS_PER_SOL,
        })
      );

      // Set recent blockhash and get lastValidBlockHeight for confirmation
      const { blockhash, lastValidBlockHeight } =
        await this.connection.getLatestBlockhash();
      transaction.recentBlockhash = blockhash;
      transaction.feePayer = fromPubkey;

      // Sign with Dynamic wallet
      const signedTransaction = await signTransaction(transaction);

      // Send transaction
      const txHash = await this.connection.sendRawTransaction(
        signedTransaction.serialize(),
        { skipPreflight: false }
      );

      console.log(`✅ SOL transfer sent: ${txHash}`);

      // Wait for confirmation with proper strategy
      await this.connection.confirmTransaction(
        {
          signature: txHash,
          blockhash,
          lastValidBlockHeight,
        },
        "confirmed"
      );

      console.log(`✅ SOL transfer confirmed!`);

      return { success: true, txHash };
    } catch (error) {
      console.error("Failed to fund Pipe wallet:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  /**
   * Auto-swap SOL to PIPE tokens after funding
   */
  async exchangeSolToPipe(
    solAmount: number,
    userWalletAddress: string
  ): Promise<{ success: boolean; pipeReceived?: number; error?: string }> {
    try {
      const result = await pipeService.exchangeSolForPipe(solAmount, userWalletAddress);

      if (result.success) {
        return {
          success: true,
          pipeReceived: result.tokensReceived,
        };
      } else {
        return {
          success: false,
          error: result.error,
        };
      }
    } catch (error) {
      console.error("Failed to swap SOL to PIPE:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Swap failed",
      };
    }
  }

  /**
   * Complete storage purchase flow
   */
  async purchaseStorage(
    userWalletAddress: string,
    sizeGB: number,
    priceSol: number,
    signTransaction: (tx: Transaction) => Promise<Transaction>
  ): Promise<{ success: boolean; error?: string }> {
    try {
      console.log(`🛒 Purchasing ${sizeGB}GB storage for ${priceSol} SOL`);

      // Step 1: Send SOL to Pipe wallet
      const fundResult = await this.fundPipeWallet(
        userWalletAddress,
        priceSol,
        signTransaction
      );

      if (!fundResult.success) {
        return { success: false, error: fundResult.error };
      }

      // Step 2: Try to exchange SOL for PIPE tokens (optional for now)
      console.log("✅ SOL transferred to Pipe wallet successfully!");
      console.log("🔄 Attempting to exchange SOL for PIPE tokens...");
      
      const exchangeResult = await this.exchangeSolToPipe(
        priceSol,
        userWalletAddress
      );

      if (exchangeResult.success) {
        console.log(`✅ Storage purchase complete!`);
        console.log(`   Size: ${sizeGB}GB`);
        console.log(`   SOL transferred: ${priceSol}`);
        console.log(`   PIPE tokens received: ${exchangeResult.pipeReceived}`);
      } else {
        console.log("⚠️ SOL transfer succeeded but token exchange failed:", exchangeResult.error);
        console.log("💡 Your SOL is in your Pipe wallet and can be manually exchanged later.");
        console.log(`✅ Storage purchase completed with SOL transfer!`);
        console.log(`   Size: ${sizeGB}GB`);
        console.log(`   SOL transferred: ${priceSol}`);
        console.log(`   Note: Exchange to PIPE tokens failed - SOL available in Pipe wallet`);
      }

      return { success: true };
    } catch (error) {
      console.error("Storage purchase failed:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Purchase failed",
      };
    }
  }

  /**
   * Get available storage purchase options
   */
  getStorageOptions(): StoragePurchaseOption[] {
    // Based on discovery: 0.1 SOL = 1 PIPE token
    // Assuming 1 PIPE token = 1GB storage (adjust based on actual Pipe docs)
    return [
      {
        label: "1GB Storage (0.1 SOL)",
        sizeGB: 1,
        priceSol: 0.1,
        recommended: true,
      },
      {
        label: "5GB Storage (0.5 SOL)",
        sizeGB: 5,
        priceSol: 0.5,
      },
      {
        label: "10GB Storage (1.0 SOL)",
        sizeGB: 10,
        priceSol: 1.0,
      },
    ];
  }
}

// Export singleton instance
export const pipeWalletBridge = new PipeWalletBridge();
