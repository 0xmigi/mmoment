import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { User } from 'lucide-react';
import { AuthModal } from './AuthModal';

interface SocialCredential {
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

export function HeadlessAuthButton() {
  const { primaryWallet, user } = useDynamicContext();
  const navigate = useNavigate();
  const [showAuthModal, setShowAuthModal] = useState(false);

  if (!primaryWallet?.address) {
    return (
      <>
        <button
          onClick={() => setShowAuthModal(true)}
          className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors"
        >
          Log in
        </button>
        <AuthModal 
          isOpen={showAuthModal} 
          onClose={() => setShowAuthModal(false)} 
        />
      </>
    );
  }

  // Find Farcaster credentials if they exist
  const farcasterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is SocialCredential => 
      cred?.oauthProvider?.toLowerCase() === 'farcaster'
  );

  // Find Twitter credentials if they exist
  const twitterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is SocialCredential => 
      cred?.oauthProvider?.toLowerCase() === 'twitter'
  );

  // Prioritize Farcaster, but fall back to Twitter if available
  const socialCred = farcasterCred || twitterCred;

  // If user is connected with a social account, show their profile
  if (socialCred) {
    return (
      <button
        onClick={() => navigate('/account')}
        className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
      >
        {socialCred.oauthAccountPhotos?.[0] && (
          <img
            src={socialCred.oauthAccountPhotos[0]}
            alt="Profile"
            className="w-6 h-6 rounded-full"
          />
        )}
        <span className="font-medium">{socialCred.oauthDisplayName}</span>
      </button>
    );
  }

  // Fallback to showing wallet address
  return (
    <button
      onClick={() => navigate('/account')}
      className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
    >
      <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center">
        <User className="w-4 h-4 text-gray-500" />
      </div>
      <span className="font-mono text-sm">{primaryWallet.address.slice(0, 3)}..{primaryWallet.address.slice(-3)}</span>
    </button>
  );
} 