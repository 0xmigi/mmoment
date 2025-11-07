import React from 'react';
import ReactDOM from 'react-dom/client';
import { DynamicContextProvider, type WalletOption } from "@dynamic-labs/sdk-react-core";
import { SolanaWalletConnectors } from "@dynamic-labs/solana";
import { LivepeerConfig, createReactClient, studioProvider } from "@livepeer/react";
import { CONFIG } from './core/config';
import App from './core/App';
import './core/styles/index.css';
import { SolanaProvider } from './blockchain/solana-provider';
import { StorageProvider } from './storage/storage-provider';
import { WalletProvider } from './auth/WalletProvider';
import { PipeProvider } from './storage/pipe/PipeProvider';

// Create Livepeer client
const livepeerClient = createReactClient({
  provider: studioProvider({
    apiKey: '7019f80b-f416-45d4-9b90-1ea130039e97'
  }),
});

console.log('App initializing with config:', {
  baseUrl: CONFIG.baseUrl,
  rpcEndpoint: CONFIG.rpcEndpoint,
  isProduction: CONFIG.isProduction,
  timelineWsUrl: CONFIG.TIMELINE_WS_URL
});

// Render the application
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SolanaProvider>
      <LivepeerConfig client={livepeerClient}>
        <DynamicContextProvider
          settings={{
            // environmentId: "0a6c159-0b98-4ce9-a715-d30a29b09a43",
            environmentId: "0a64c159-0b98-4ce9-a715-d30a29b09a43",
            walletConnectors: [SolanaWalletConnectors],
            walletsFilter: (wallets: WalletOption[]) => {
              const phantomWallet = wallets.find(w => w.name.toLowerCase() === 'phantom');
              return phantomWallet ? [phantomWallet] : [];
            }
          }}
        >
          <PipeProvider>
            <StorageProvider>
              <WalletProvider>
                <App />
              </WalletProvider>
            </StorageProvider>
          </PipeProvider>
        </DynamicContextProvider>
      </LivepeerConfig>
    </SolanaProvider>
  </React.StrictMode>
); 