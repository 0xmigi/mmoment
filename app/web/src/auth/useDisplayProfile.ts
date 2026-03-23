import { useMemo, useState, useEffect, useCallback } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { profileService } from '../services/profile-service';

/**
 * Resolved display profile — the SINGLE SOURCE OF TRUTH for how
 * the current user should be displayed throughout the entire app.
 *
 * Rules:
 * - Email addresses NEVER appear in any field
 * - Display name comes from social auth, backend profile, or is empty
 * - Username is only set for providers with real handles (Twitter, Farcaster)
 * - Google's oauthUsername is an email — always excluded
 * - Wallet address (truncated) is the ultimate fallback for display
 */
export interface DisplayProfile {
  walletAddress: string;
  fullWalletAddress: string;
  displayName: string | undefined;
  username: string | undefined;
  profileImage: string | undefined;
  provider: string;
  providerLabel: string;
  /** What to show as the user's name — displayName or truncated wallet. Never email. */
  name: string;
  hasSocialAuth: boolean;
  socials: {
    google?: { displayName: string; profileImage?: string };
    twitter?: { displayName?: string; username: string; profileImage?: string };
    farcaster?: { displayName?: string; username: string; profileImage?: string };
  };
  /** Force refresh from backend (call after manual profile edits) */
  refresh: () => void;
}

const PROVIDER_LABELS: Record<string, string> = {
  farcaster: 'Farcaster',
  google: 'Google',
  twitter: 'X / Twitter',
  email: 'Email',
  wallet: 'Wallet',
};

export function useDisplayProfile(): DisplayProfile | null {
  const { user, primaryWallet } = useDynamicContext();
  const [backendDisplayName, setBackendDisplayName] = useState<string | undefined>();
  const [backendProfileImage, setBackendProfileImage] = useState<string | undefined>();
  const [refreshCounter, setRefreshCounter] = useState(0);

  const walletAddress = primaryWallet?.address;

  // Fetch backend profile for fields not available from social auth
  useEffect(() => {
    if (!walletAddress) return;

    profileService.getProfile(walletAddress).then(profile => {
      if (profile) {
        setBackendDisplayName(profile.displayName || undefined);
        setBackendProfileImage(profile.profileImage || undefined);
      }
    }).catch(() => {});
  }, [walletAddress, refreshCounter]);

  const refresh = useCallback(() => {
    setRefreshCounter(c => c + 1);
  }, []);

  return useMemo(() => {
    if (!walletAddress) return null;

    const truncatedWallet = `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`;

    const creds = user?.verifiedCredentials || [];

    const farcasterCred = creds.find((c: any) => c.oauthProvider === 'farcaster');
    const googleCred = creds.find((c: any) => c.oauthProvider === 'google');
    const twitterCred = creds.find((c: any) => c.oauthProvider === 'twitter');

    const socials: DisplayProfile['socials'] = {};

    if (googleCred) {
      socials.google = {
        displayName: googleCred.oauthDisplayName || 'Connected',
        profileImage: googleCred.oauthAccountPhotos?.[0] || undefined,
      };
    }

    if (twitterCred) {
      socials.twitter = {
        displayName: twitterCred.oauthDisplayName || undefined,
        username: twitterCred.oauthUsername || '',
        profileImage: twitterCred.oauthAccountPhotos?.[0] || undefined,
      };
    }

    if (farcasterCred) {
      socials.farcaster = {
        displayName: farcasterCred.oauthDisplayName || undefined,
        username: farcasterCred.oauthUsername || '',
        profileImage: farcasterCred.oauthAccountPhotos?.[0] || undefined,
      };
    }

    const primaryCred = farcasterCred || googleCred || twitterCred;

    // Display name: social auth > backend profile > undefined
    const displayName = primaryCred?.oauthDisplayName || backendDisplayName || undefined;

    // Username: only real handles (Farcaster/Twitter), never Google
    const username = farcasterCred?.oauthUsername
      || twitterCred?.oauthUsername
      || undefined;

    // Profile image: social auth > backend profile > undefined
    const profileImage = primaryCred?.oauthAccountPhotos?.[0] || backendProfileImage || undefined;

    const provider = primaryCred?.oauthProvider
      || (user?.email ? 'email' : 'wallet');

    return {
      walletAddress: truncatedWallet,
      fullWalletAddress: walletAddress,
      displayName,
      username,
      profileImage,
      provider,
      providerLabel: PROVIDER_LABELS[provider] || provider,
      name: displayName || truncatedWallet,
      hasSocialAuth: !!primaryCred,
      socials,
      refresh,
    };
  }, [user, walletAddress, user?.verifiedCredentials, backendDisplayName, backendProfileImage, refresh]);
}
