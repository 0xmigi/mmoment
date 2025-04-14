import { useSocialAccounts } from '@dynamic-labs/sdk-react-core';
import { ProviderEnum } from '@dynamic-labs/types';
import { FarcasterIcon, TwitterIcon } from '@dynamic-labs/iconic';
import { useNavigate } from 'react-router-dom';

interface HeadlessSocialLoginProps {
  onSuccess?: () => void;
}

export function HeadlessSocialLogin({ onSuccess }: HeadlessSocialLoginProps) {
  const { error, isProcessing, signInWithSocialAccount } = useSocialAccounts();
  const navigate = useNavigate();

  const handleFarcasterLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Farcaster);
      onSuccess?.();
      navigate('/app');
    } catch (err) {
      console.error('Failed to sign in with Farcaster:', err);
    }
  };

  const handleTwitterLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Twitter);
      onSuccess?.();
      navigate('/app');
    } catch (err) {
      console.error('Failed to sign in with Twitter:', err);
    }
  };

  return (
    <div className="space-y-4">
      <button
        onClick={handleFarcasterLogin}
        disabled={isProcessing}
        className="w-full flex items-center p-3 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="w-6 h-6 mr-3">
          <FarcasterIcon className="w-full h-full" />
        </div>
        <span className="flex-1 text-left text-sm font-medium">Farcaster</span>
      </button>

      <button
        onClick={handleTwitterLogin}
        disabled={isProcessing}
        className="w-full flex items-center p-3 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="w-6 h-6 mr-3">
          <TwitterIcon className="w-full h-full" />
        </div>
        <span className="flex-1 text-left text-sm font-medium">X / Twitter</span>
      </button>
      
      {error && (
        <p className="text-sm text-red-600 text-center">
          {error.message}
        </p>
      )}
    </div>
  );
} 