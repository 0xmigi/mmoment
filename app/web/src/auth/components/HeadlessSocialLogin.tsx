import { useSocialAccounts } from '@dynamic-labs/sdk-react-core';
import { ProviderEnum } from '@dynamic-labs/types';
import { GoogleIcon, FarcasterIcon, TwitterIcon } from '@dynamic-labs/iconic';
interface HeadlessSocialLoginProps {
  onSuccess?: () => void;
}

export function HeadlessSocialLogin({ onSuccess }: HeadlessSocialLoginProps) {
  const { error, isProcessing, signInWithSocialAccount } = useSocialAccounts();

  const handleGoogleLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Google);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to sign in with Google:', err);
    }
  };

  const handleFarcasterLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Farcaster);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to sign in with Farcaster:', err);
    }
  };

  const handleTwitterLogin = async () => {
    try {
      await signInWithSocialAccount(ProviderEnum.Twitter);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to sign in with Twitter:', err);
    }
  };

  return (
    <div className="space-y-4">
      <button
        onClick={handleGoogleLogin}
        disabled={isProcessing}
        className="w-full flex items-center p-3 rounded-lg bg-neutral-100 hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="w-7 h-7 mr-3 flex items-center justify-center">
          <GoogleIcon className="w-full h-full" />
        </div>
        <span className="flex-1 text-left text-sm font-medium">Google</span>
      </button>

      <button
        onClick={handleFarcasterLogin}
        disabled={isProcessing}
        className="w-full flex items-center p-3 rounded-lg bg-neutral-100 hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="w-7 h-7 mr-3 flex items-center justify-center">
          <FarcasterIcon className="w-full h-full" />
        </div>
        <span className="flex-1 text-left text-sm font-medium">Farcaster</span>
      </button>

      <button
        onClick={handleTwitterLogin}
        disabled={isProcessing}
        className="w-full flex items-center p-3 rounded-lg bg-neutral-100 hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="w-7 h-7 mr-3 flex items-center justify-center">
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
