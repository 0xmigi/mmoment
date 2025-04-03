import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useEffect, useState } from 'react';
import { farcasterService, FarcasterProfile } from '../services/farcaster-service';

export function useFarcasterProfile() {
  const { user } = useDynamicContext();
  const [profile, setProfile] = useState<FarcasterProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchProfile() {
      if (!user?.verifiedCredentials) return;

      try {
        setLoading(true);
        setError(null);

        // First try to get profile from Dynamic's verified credentials
        let profile = farcasterService.getProfileFromVerifiedCredentials(user.verifiedCredentials);

        // If we need more data (like bio), fetch it from the Farcaster API
        if (profile?.fid) {
          const fullProfile = await farcasterService.getProfileByFid(profile.fid);
          profile = { ...profile, ...fullProfile };
        }

        setProfile(profile);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to fetch Farcaster profile'));
      } finally {
        setLoading(false);
      }
    }

    fetchProfile();
  }, [user?.verifiedCredentials]);

  return { profile, loading, error };
} 