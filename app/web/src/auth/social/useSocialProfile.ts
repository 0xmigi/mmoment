import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useEffect, useState } from 'react';
import { socialService } from './social-service';
import { CONFIG } from '../../core/config';

export interface SocialProfile {
  id: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
  bio?: string;
  provider: string;
  isVerified?: boolean;
}

export function useSocialProfile() {
  const { user } = useDynamicContext();
  const [profiles, setProfiles] = useState<SocialProfile[]>([]);
  const [primaryProfile, setPrimaryProfile] = useState<SocialProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchProfiles() {
      if (!user?.verifiedCredentials) return;

      try {
        setLoading(true);
        setError(null);

        // Get profiles from social service
        const socialProfiles = socialService.getProfileFromVerifiedCredentials(user.verifiedCredentials);

        // Enhance profiles with additional data if needed
        const enhancedProfiles = await Promise.all(
          socialProfiles.map(async (profile) => {
            // Add Farcaster additional data like bio
            if (profile.provider === 'farcaster' && profile.id) {
              try {
                const fid = parseInt(profile.id);
                if (!isNaN(fid)) {
                  const fullProfile = await socialService.getFarcasterProfileByFid(fid);
                  return { ...profile, ...fullProfile };
                }
              } catch (err) {
                console.error('Failed to fetch detailed Farcaster profile:', err);
              }
            }
            return profile;
          })
        );

        setProfiles(enhancedProfiles);

        // Set first profile as primary by default, or maintain existing if it still exists
        if (enhancedProfiles.length > 0) {
          if (!primaryProfile || !enhancedProfiles.find(p => p.id === primaryProfile.id)) {
            setPrimaryProfile(enhancedProfiles[0]);
          }
          
          // Save the profiles to the backend API for other clients
          for (const profile of enhancedProfiles) {
            // Get the address from the verified credential
            const credential = user.verifiedCredentials.find(
              cred => cred.oauthProvider === profile.provider
            );
            
            if (credential?.address) {
              try {
                // Get the proper API URL from CONFIG or use the correct port
                const apiBaseUrl = CONFIG.BACKEND_URL;
                
                // Save to backend API
                await fetch(`${apiBaseUrl}/api/profiles/${credential.address}`, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify(profile)
                });
                
                console.log(`Saved ${profile.provider} profile to backend for ${credential.address}`);
                
                // Also update the profile in the timeline service to broadcast to other clients
                try {
                  // Import here to avoid circular dependency
                  const { timelineService } = await import('../../timeline/timeline-service');
                  timelineService.updateUserProfile({
                    address: credential.address,
                    profile: {
                      ...profile,
                      address: credential.address
                    }
                  });
                } catch (err) {
                  console.error('Failed to broadcast profile via timeline service:', err);
                }
              } catch (err) {
                console.error(`Failed to save profile to backend for ${credential.address}:`, err);
              }
            }
          }
        } else {
          setPrimaryProfile(null);
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Failed to fetch social profiles'));
      } finally {
        setLoading(false);
      }
    }

    fetchProfiles();
  }, [user?.verifiedCredentials]);

  const setPrimaryProfileById = (id: string) => {
    const profile = profiles.find(p => p.id === id);
    if (profile) {
      setPrimaryProfile(profile);
    }
  };

  return { 
    profiles, 
    primaryProfile, 
    setPrimaryProfile: setPrimaryProfileById, 
    loading, 
    error 
  };
} 