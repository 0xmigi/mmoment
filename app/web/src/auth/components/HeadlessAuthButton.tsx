import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { User } from 'lucide-react';
import { AuthModal } from './AuthModal';
import { useDisplayProfile } from '../useDisplayProfile';

export function HeadlessAuthButton() {
  const { primaryWallet } = useDynamicContext();
  const displayProfile = useDisplayProfile();
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

  return (
    <button
      onClick={() => navigate('/account')}
      className="px-4 py-2 bg-gray-100 text-black rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2"
    >
      {displayProfile?.profileImage ? (
        <img
          src={displayProfile.profileImage}
          alt="Profile"
          referrerPolicy="no-referrer"
          className="w-6 h-6 rounded-full"
        />
      ) : (
        <User className="w-5 h-5" />
      )}
      <span className="font-medium">{displayProfile?.name || 'Account'}</span>
    </button>
  );
}
