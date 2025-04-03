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
            environmentId: "93e6248c-4446-4f78-837d-fedf6391d174",
            walletConnectors: [SolanaWalletConnectors],
            walletsFilter: (wallets: WalletOption[]) => {
              const phantomWallet = wallets.find(w => w.name.toLowerCase() === 'phantom');
              return phantomWallet ? [phantomWallet] : [];
            }
          }}
        >
          <StorageProvider>
            <WalletProvider>
              <App />
            </WalletProvider>
          </StorageProvider>
        </DynamicContextProvider>
      </LivepeerConfig>
    </SolanaProvider>
  </React.StrictMode>
); 