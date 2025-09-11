/**
 * Dynamic Wallet → Pipe Wallet Bridge
 * 
 * Handles the integration between user's Dynamic wallet and their Pipe storage account.
 * Enables users to fund their Pipe storage directly from their main wallet.
 */

// Import removed - not used in this service file
import { Connection, PublicKey, Transaction, SystemProgram, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { pipeService } from './pipe-service';

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
    this.connection = new Connection('https://api.devnet.solana.com');
  }

  /**
   * Get user's Pipe wallet info (balance, storage, etc.)
   */
  async getUserPipeInfo(userWalletAddress: string): Promise<PipeWalletInfo | null> {
    try {
      // Check if user has Pipe credentials
      if (!pipeService.isAvailable()) {
        return null;
      }

      // Get balance from Pipe API
      const balance = await pipeService.checkBalance();
      
      // TODO: Get actual Pipe wallet address from user's main wallet
      // For now, we'll use a derived address or the one from credentials
      const pipeWalletAddress = await this.derivePipeWalletAddress(userWalletAddress);

      return {
        pipeWalletAddress,
        solBalance: balance.sol,
        pipeBalance: balance.pipe,
        storageUsed: 0, // TODO: Get from Pipe API
        storageLimit: balance.pipe * 1000 // Assume 1 PIPE = 1GB for now
      };
    } catch (error) {
      console.error('Failed to get Pipe info:', error);
      return null;
    }
  }

  /**
   * Derive Pipe wallet address from user's main wallet
   */
  private async derivePipeWalletAddress(userWalletAddress: string): Promise<string> {
    // TODO: Get the actual Pipe wallet pubkey from the Pipe API for this user
    // For now, using the correct Pipe pubkey for your Dynamic wallet
    // In production, this should query the Pipe API to get the user's Pipe wallet address
    return '4k4rcjMtiz7DozHVirFpYTdyQ4gK1CunKqcaXKSZV5Ng';
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
      const pipeWalletAddress = await this.derivePipeWalletAddress(userWalletAddress);
      
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

      // Set recent blockhash
      const { blockhash } = await this.connection.getLatestBlockhash();
      transaction.recentBlockhash = blockhash;
      transaction.feePayer = fromPubkey;

      // Sign with Dynamic wallet
      const signedTransaction = await signTransaction(transaction);

      // Send transaction
      const txHash = await this.connection.sendRawTransaction(
        signedTransaction.serialize()
      );

      console.log(`✅ SOL transfer sent: ${txHash}`);

      // Wait for confirmation
      await this.connection.confirmTransaction(txHash);
      
      console.log(`✅ SOL transfer confirmed!`);
      
      return { success: true, txHash };

    } catch (error) {
      console.error('Failed to fund Pipe wallet:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  }

  /**
   * Auto-swap SOL to PIPE tokens after funding
   */
  async swapSolToPipe(solAmount: number): Promise<{ success: boolean; pipeReceived?: number; error?: string }> {
    try {
      console.log(`🔄 Swapping ${solAmount} SOL to PIPE...`);

      // Keep 10% SOL for fees, swap the rest
      const swapAmount = solAmount * 0.9;
      
      // Call actual Pipe API to swap
      const result = await pipeService.swapSolForPipe(swapAmount);
      
      if (result.success) {
        console.log(`✅ Swap successful: ${result.tokensReceived} PIPE received`);
        return { 
          success: true, 
          pipeReceived: result.tokensReceived 
        };
      } else {
        return {
          success: false,
          error: result.error
        };
      }

    } catch (error) {
      console.error('Failed to swap SOL to PIPE:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Swap failed' 
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
      const fundResult = await this.fundPipeWallet(userWalletAddress, priceSol, signTransaction);
      if (!fundResult.success) {
        return { success: false, error: fundResult.error };
      }

      // Step 2: Auto-swap SOL to PIPE
      const swapResult = await this.swapSolToPipe(priceSol);
      if (!swapResult.success) {
        return { success: false, error: swapResult.error };
      }

      console.log(`✅ Storage purchase complete!`);
      console.log(`   Size: ${sizeGB}GB`);
      console.log(`   PIPE received: ${swapResult.pipeReceived}`);

      return { success: true };

    } catch (error) {
      console.error('Storage purchase failed:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Purchase failed' 
      };
    }
  }

  /**
   * Get available storage purchase options
   */
  getStorageOptions(): StoragePurchaseOption[] {
    return [
      {
        label: '1GB Storage',
        sizeGB: 1,
        priceSol: 0.1,
      },
      {
        label: '10GB Storage',
        sizeGB: 10,
        priceSol: 0.8, // Slight discount
        recommended: true
      },
      {
        label: '100GB Storage',
        sizeGB: 100,
        priceSol: 7.0, // Better discount
      }
    ];
  }
}

// Export singleton
export const pipeWalletBridge = new PipeWalletBridge();