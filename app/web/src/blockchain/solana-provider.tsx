import { ReactNode } from 'react';
import { ConnectionProvider, WalletProvider as SolanaWalletProvider } from '@solana/wallet-adapter-react';
import { CONFIG } from '../core/config';

// Solana provider props
interface SolanaProviderProps {
  children: ReactNode;
  endpoint?: string;
}

// Wrapper for Solana Connection and Wallet providers
export function SolanaProvider({ 
  children, 
  endpoint = CONFIG.rpcEndpoint 
}: SolanaProviderProps) {
  return (
    <ConnectionProvider endpoint={endpoint}>
      <SolanaWalletProvider wallets={[]} autoConnect={true}>
        {children}
      </SolanaWalletProvider>
    </ConnectionProvider>
  );
} 