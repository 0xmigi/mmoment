import { useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { profileService } from '../services/profile-service';
import { socialService } from './social/social-service';

/**
 * Hook that automatically syncs user profile to backend when wallet connects
 * or social auth changes.
 *
 * This ensures the backend always has the latest profile data for timeline display.
 */
export function useProfileSync() {
  const { user, primaryWallet } = useDynamicContext();

  useEffect(() => {
    async function syncProfile() {
      // Only sync if user is authenticated with a wallet
      if (!user || !primaryWallet?.address) {
        return;
      }

      try {
        // Get social profiles from verified credentials
        const socialProfiles = user.verifiedCredentials
          ? socialService.getProfileFromVerifiedCredentials(user.verifiedCredentials)
          : [];

        // Prioritize Farcaster over Twitter
        const farcasterProfile = socialProfiles.find(p => p.provider === 'farcaster');
        const twitterProfile = socialProfiles.find(p => p.provider === 'twitter');
        const primarySocialProfile = farcasterProfile || twitterProfile;

        // Build profile data
        const profileData = {
          walletAddress: primaryWallet.address,
          displayName: primarySocialProfile?.displayName,
          username: primarySocialProfile?.username,
          profileImage: primarySocialProfile?.pfpUrl,
          provider: primarySocialProfile?.provider,
        };

        // Save to backend (async, don't block on it)
        await profileService.saveProfile(profileData);

        console.log('📤 Profile synced to backend:', {
          wallet: primaryWallet.address.slice(0, 8) + '...',
          displayName: profileData.displayName,
          provider: profileData.provider,
        });
      } catch (error) {
        console.error('Failed to sync profile to backend:', error);
        // Non-critical error, don't throw
      }
    }

    syncProfile();
  }, [user, primaryWallet?.address, user?.verifiedCredentials]);
}
