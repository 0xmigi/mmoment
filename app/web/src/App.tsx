import { useMemo } from "react";
import { BrowserRouter } from 'react-router-dom';
import MainContent from './MainContent';
import { DynamicContextProvider } from "@dynamic-labs/sdk-react-core";
import { SolanaWalletConnectors, SolanaWalletConnectorsWithConfig } from "@dynamic-labs/solana";
import { clusterApiUrl } from '@solana/web3.js';
import {
  LivepeerConfig,
  createReactClient,
  studioProvider,
} from "@livepeer/react";

function App() {
  const solanaDevnetConfig = useMemo(() => ({
    rpcUrl: clusterApiUrl('devnet'),
    network: 'devnet'
  }), []);

  // Create Livepeer client with studio provider
  const livepeerClient = createReactClient({
    provider: studioProvider({ 
      apiKey: '7019f80b-f416-45d4-9b90-1ea130039e97'
    }),
  });

  return (
    <LivepeerConfig client={livepeerClient}>
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
    </LivepeerConfig>
  );
}

export default App;