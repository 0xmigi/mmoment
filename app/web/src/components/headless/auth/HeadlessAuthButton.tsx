import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';

interface FarcasterCredential {
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

export function HeadlessAuthButton() {
  const { primaryWallet, user } = useDynamicContext();
  const navigate = useNavigate();

  if (!primaryWallet?.address) {
    return null;
  }

  // Find Farcaster credentials if they exist
  const farcasterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is FarcasterCredential => 
      cred?.oauthProvider?.toLowerCase() === 'farcaster'
  );

  // If user is connected with Farcaster, show their profile
  if (farcasterCred) {
    return (
      <button
        onClick={() => navigate('/account')}
        className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
      >
        {farcasterCred.oauthAccountPhotos?.[0] && (
          <img
            src={farcasterCred.oauthAccountPhotos[0]}
            alt="Profile"
            className="w-6 h-6 rounded-full"
          />
        )}
        <span className="font-medium">{farcasterCred.oauthDisplayName}</span>
      </button>
    );
  }

  // Fallback to showing wallet address
  return (
    <button
      onClick={() => navigate('/account')}
      className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
    >
      <span className="font-mono">{primaryWallet.address.slice(0, 4)}...{primaryWallet.address.slice(-4)}</span>
    </button>
  );
} 