// src/providers/DynamicAuthProvider.tsx
import { ReactNode } from 'react';
import { DynamicContextProvider } from '@dynamic-labs/sdk-react-core';
import { SolanaWalletConnectors } from '@dynamic-labs/solana';

interface DynamicAuthProviderProps {
  children: ReactNode;
}

export function DynamicAuthProvider({ children }: DynamicAuthProviderProps) {
  return (
    <DynamicContextProvider
      settings={{
        environmentId: import.meta.env.VITE_DYNAMIC_ENVIRONMENT_ID || '',
        walletConnectors: [SolanaWalletConnectors],
        walletConnectorExtensions: [],
      }}
    >
      {children}
    </DynamicContextProvider>
  );
}