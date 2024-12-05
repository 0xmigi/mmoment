// App.tsx
import { useMemo } from "react";
import { BrowserRouter } from 'react-router-dom';
import MainContent from './MainContent';
import { DynamicContextProvider } from "@dynamic-labs/sdk-react-core";
import { SolanaWalletConnectors, SolanaWalletConnectorsWithConfig } from "@dynamic-labs/solana";
import { clusterApiUrl } from '@solana/web3.js';

function App() {

  const solanaDevnetConfig = useMemo(() => ({
    rpcUrl: clusterApiUrl('devnet'),
    network: 'devnet'
  }), []);

  return (
    <DynamicContextProvider
      settings={{
        environmentId: "93e6248c-4446-4f78-837d-fedf6391d174",
        walletConnectors: [
          SolanaWalletConnectors,
          SolanaWalletConnectorsWithConfig(solanaDevnetConfig as any)
        ],
        initialAuthenticationMode: "connect-only",
        eventsCallbacks: {
          onAuthSuccess: (args) => {
            console.log("Auth Success:", args);
          },
          onAuthFailure: (args) => {
            console.error("Auth Error:", args);
          },
          onLogout: () => {
            console.log("User logged out");
          },
          onAuthFlowClose: () => {
            console.log("Auth flow closed");
          }
        }
      }}
    >
      <BrowserRouter>
        <MainContent />
      </BrowserRouter>
    </DynamicContextProvider>
  );
}

export default App;