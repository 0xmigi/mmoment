import { BrowserRouter } from 'react-router-dom';
import MainContent from './MainContent';
import { DynamicContextProvider, type WalletOption } from "@dynamic-labs/sdk-react-core";
import { SolanaWalletConnectors } from "@dynamic-labs/solana";
import {
  LivepeerConfig,
  createReactClient,
  studioProvider,
} from "@livepeer/react";
import { CameraProvider } from './components/CameraProvider';

function App() {
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
          walletConnectors: [SolanaWalletConnectors],
          // Show both Phantom wallet and social logins
          walletsFilter: (wallets: WalletOption[]) => {
            const phantomWallet = wallets.find(w => w.name.toLowerCase() === 'phantom');
            return phantomWallet ? [phantomWallet] : [];
          }
        }}
      >
        <CameraProvider>
          <BrowserRouter>
            <MainContent />
          </BrowserRouter>
        </CameraProvider>
      </DynamicContextProvider>
    </LivepeerConfig>
  );
}
export default App;
