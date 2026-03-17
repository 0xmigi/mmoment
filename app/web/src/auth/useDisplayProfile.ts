import { useMemo } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

/**
 * Resolved display profile — the SINGLE SOURCE OF TRUTH for how
 * the current user should be displayed throughout the entire app.
 *
 * Rules:
 * - Email addresses NEVER appear in any field
 * - Display name comes from social auth or is empty (user can set manually)
 * - Username is only set for providers with real handles (Twitter, Farcaster)
 * - Google's oauthUsername is an email — always excluded
 * - Wallet address (truncated) is the ultimate fallback for display
 */
export interface DisplayProfile {
  /** Truncated wallet address (always available) */
  walletAddress: string;
  /** Full wallet address */
  fullWalletAddress: string;
  /** Human-readable display name (from social or manual). Never an email. */
  displayName: string | undefined;
  /** Social handle — only for Twitter/Farcaster, never for Google/email */
  username: string | undefined;
  /** Profile picture URL from social provider */
  profileImage: string | undefined;
  /** Which auth provider: 'google' | 'twitter' | 'farcaster' | 'email' | 'wallet' */
  provider: string;
  /** Provider label for display: 'Google' | 'X / Twitter' | 'Farcaster' | 'Email' | 'Wallet' */
  providerLabel: string;
  /** What to show as the user's name — displayName or truncated wallet. Never email. */
  name: string;
  /** Whether user has a social credential attached */
  hasSocialAuth: boolean;
  /** Connected social accounts (safe — no emails) */
  socials: {
    google?: { displayName: string; profileImage?: string };
    twitter?: { displayName?: string; username: string; profileImage?: string };
    farcaster?: { displayName?: string; username: string; profileImage?: string };
  };
}

const PROVIDER_LABELS: Record<string, string> = {
  farcaster: 'Farcaster',
  google: 'Google',
  twitter: 'X / Twitter',
  email: 'Email',
  wallet: 'Wallet',
};

/**
 * Single hook that resolves the current user's display profile.
 * Use this everywhere instead of manually parsing verifiedCredentials.
 */
export function useDisplayProfile(): DisplayProfile | null {
  const { user, primaryWallet } = useDynamicContext();

  return useMemo(() => {
    if (!primaryWallet?.address) return null;

    const walletAddress = primaryWallet.address;
    const truncatedWallet = `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`;

    // Parse verified credentials once
    const creds = user?.verifiedCredentials || [];

    const farcasterCred = creds.find((c: any) => c.oauthProvider === 'farcaster');
    const googleCred = creds.find((c: any) => c.oauthProvider === 'google');
    const twitterCred = creds.find((c: any) => c.oauthProvider === 'twitter');

    // Build safe social profiles (no emails ever)
    const socials: DisplayProfile['socials'] = {};

    if (googleCred) {
      socials.google = {
        displayName: googleCred.oauthDisplayName || 'Connected',
        profileImage: googleCred.oauthAccountPhotos?.[0],
      };
      // NOTE: googleCred.oauthUsername is the EMAIL — intentionally excluded
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

    // Priority: Farcaster > Google > Twitter
    const primaryCred = farcasterCred || googleCred || twitterCred;

    // Resolve display name — NEVER use email
    const displayName = primaryCred?.oauthDisplayName || undefined;

    // Resolve username — only real handles, never Google (which is email)
    const username = farcasterCred?.oauthUsername
      || twitterCred?.oauthUsername
      || undefined;

    // Resolve profile image
    const profileImage = primaryCred?.oauthAccountPhotos?.[0] || undefined;

    // Resolve provider
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
    };
  }, [user, primaryWallet?.address, user?.verifiedCredentials]);
}
