import { useSocialAccounts } from '@dynamic-labs/sdk-react-core';
import { ProviderEnum } from '@dynamic-labs/types';
import { FarcasterIcon } from '@dynamic-labs/iconic';

interface HeadlessSocialLoginProps {
  onSuccess?: () => void;
}

export function HeadlessSocialLogin({ onSuccess }: HeadlessSocialLoginProps) {
  const { error, isProcessing, signInWithSocialAccount } = useSocialAccounts();

  const handleFarcasterLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Farcaster);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to sign in with Farcaster:', err);
    }
  };

  return (
    <div className="space-y-4">
      <button
        onClick={handleFarcasterLogin}
        disabled={isProcessing}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[#855DCD] text-white rounded-lg hover:bg-[#6f4eb3] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <FarcasterIcon className="w-5 h-5" />
        <span>{isProcessing ? 'Connecting...' : 'Continue with Farcaster'}</span>
      </button>
      
      {error && (
        <p className="text-sm text-red-600 text-center">
          {error.message}
        </p>
      )}
    </div>
  );
} 