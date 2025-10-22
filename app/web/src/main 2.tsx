import React from 'react';
import ReactDOM from 'react-dom/client';
import { PrivyProvider } from '@privy-io/react-auth';
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
    <PrivyProvider 
      appId={import.meta.env.VITE_PRIVY_APP_ID}
    >
      <SolanaProvider>
        <LivepeerConfig client={livepeerClient}>
          <StorageProvider>
            <WalletProvider>
              <App />
            </WalletProvider>
          </StorageProvider>
        </LivepeerConfig>
      </SolanaProvider>
    </PrivyProvider>
  </React.StrictMode>
); 