import React, { createContext, useContext, useState, useCallback } from 'react';

// Wallet provider types
export type EmbeddedWalletProvider = 'turnkey' | 'crossmint';

// Wallet context type
interface WalletContextType {
  provider: EmbeddedWalletProvider;
  setProvider: (provider: EmbeddedWalletProvider) => void;
  isConfiguring: boolean;
  error: string | null;
}

// Create wallet context
export const WalletContext = createContext<WalletContextType | null>(null);

// Wallet context provider component
export function WalletProvider({ 
  children,
  defaultProvider = 'turnkey'
}: { 
  children: React.ReactNode;
  defaultProvider?: EmbeddedWalletProvider;
}) {
  const [provider, setProviderInternal] = useState<EmbeddedWalletProvider>(
    () => {
      // Check localStorage for saved preference
      const saved = localStorage.getItem('preferredWalletProvider');
      if (saved === 'turnkey' || saved === 'crossmint') {
        return saved;
      }
      return defaultProvider;
    }
  );
  const [isConfiguring, setIsConfiguring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Switch wallet provider implementation
  const setProvider = useCallback(async (newProvider: EmbeddedWalletProvider) => {
    try {
      setIsConfiguring(true);
      setError(null);
      
      // In a real implementation, we would reconfigure Dynamic or other auth providers here
      console.log(`Switching to wallet provider: ${newProvider}`);
      
      // For now, just update state and localStorage
      setProviderInternal(newProvider);
      localStorage.setItem('preferredWalletProvider', newProvider);
      
    } catch (err) {
      console.error(`Failed to switch to wallet provider ${newProvider}:`, err);
      setError(err instanceof Error ? err.message : 'Failed to switch wallet provider');
    } finally {
      setIsConfiguring(false);
    }
  }, []);
  
  return (
    <WalletContext.Provider value={{ 
      provider, 
      setProvider,
      isConfiguring,
      error
    }}>
      {children}
    </WalletContext.Provider>
  );
}

// Hook to use wallet provider
export function useWalletProvider() {
  const context = useContext(WalletContext);
  if (!context) {
    throw new Error('useWalletProvider must be used within a WalletProvider');
  }
  return context;
} 