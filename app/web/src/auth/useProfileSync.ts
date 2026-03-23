import { useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { profileService } from '../services/profile-service';
import { socialService } from './social/social-service';

/**
 * Hook that syncs user profile to backend using progressive enrichment.
 *
 * Rules:
 * - Only fill in BLANK fields, never overwrite existing data
 * - Never store email addresses as username (Google's oauthUsername = email)
 * - Display name and profile image are always safe to store
 * - Username is only stored for providers that have real handles (Twitter, Farcaster)
 */
export function useProfileSync() {
  const { user, primaryWallet } = useDynamicContext();

  useEffect(() => {
    async function syncProfile() {
      if (!user || !primaryWallet?.address) {
        return;
      }

      try {
        // Get social profiles from verified credentials
        const socialProfiles = user.verifiedCredentials
          ? socialService.getProfileFromVerifiedCredentials(user.verifiedCredentials)
          : [];

        // Prioritize Farcaster > Google > Twitter
        const farcasterProfile = socialProfiles.find(p => p.provider === 'farcaster');
        const googleProfile = socialProfiles.find(p => p.provider === 'google');
        const twitterProfile = socialProfiles.find(p => p.provider === 'twitter');
        const primarySocialProfile = farcasterProfile || googleProfile || twitterProfile;

        // Nothing to sync if no social profile
        if (!primarySocialProfile) return;

        // Fetch existing profile from backend
        const existing = await profileService.getProfile(primaryWallet.address);

        // Progressive enrichment: only fill blank fields
        const profileData = {
          walletAddress: primaryWallet.address,
          displayName: existing?.displayName || primarySocialProfile.displayName,
          username: existing?.username || primarySocialProfile.username,
          profileImage: existing?.profileImage || primarySocialProfile.pfpUrl,
          provider: existing?.provider || primarySocialProfile.provider,
        };

        // Check if anything actually changed
        const hasNewData =
          (!existing?.displayName && profileData.displayName) ||
          (!existing?.username && profileData.username) ||
          (!existing?.profileImage && profileData.profileImage) ||
          (!existing?.provider && profileData.provider);

        if (!hasNewData && existing) {
          // Nothing new to sync
          return;
        }

        await profileService.saveProfile(profileData);

        console.log('[ProfileSync] Profile synced:', {
          wallet: primaryWallet.address.slice(0, 8) + '...',
          displayName: profileData.displayName,
          provider: profileData.provider,
          enriched: !existing ? 'new' : 'filled gaps',
        });
      } catch (error) {
        console.error('Failed to sync profile to backend:', error);
      }
    }

    syncProfile();
  }, [user, primaryWallet?.address, user?.verifiedCredentials]);
}
