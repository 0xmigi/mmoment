import axios from 'axios';
import { SocialProfile } from './useSocialProfile';
import { CONFIG } from '../../core/config';

interface FarcasterProfileResponse {
  messages: Array<{
    data: {
      userDataBody: {
        type: string | number;
        value: string;
      }
    }
  }>;
}

export class SocialService {
  // Use dynamic base URL that works in both dev and production
  private readonly FARCASTER_API_BASE = `${window.location.protocol}//${window.location.hostname}:2281/v1`;
  
  // For accessing our backend API
  private readonly API_BASE = CONFIG.BACKEND_URL;

  // Farcaster Methods
  async getFarcasterProfileByFid(fid: number): Promise<Partial<SocialProfile>> {
    try {
      const response = await axios.get<FarcasterProfileResponse>(
        `${this.FARCASTER_API_BASE}/userDataByFid?fid=${fid}`
      );
      const messages = response.data.messages || [];
      
      const profile: Partial<SocialProfile> = { 
        id: fid.toString(),
        provider: 'farcaster'
      };
      
      messages.forEach((message) => {
        const { type, value } = message.data.userDataBody;
        
        switch (type) {
          case 'USER_DATA_TYPE_USERNAME':
          case 6: // Numeric type from console output
            profile.username = value;
            break;
          case 'USER_DATA_TYPE_DISPLAY':
          case 2: // Numeric type from console output
            profile.displayName = value;
            break;
          case 'USER_DATA_TYPE_BIO':
          case 3: // Numeric type from console output
            profile.bio = value;
            break;
          case 'USER_DATA_TYPE_PFP':
          case 1: // Numeric type from console output
            profile.pfpUrl = value;
            break;
        }
      });

      return profile;
    } catch (error) {
      console.error('Failed to fetch Farcaster profile:', error);
      throw error;
    }
  }

  async getFarcasterFidByAddress(address: string): Promise<string | null> {
    try {
      const response = await axios.get(`${this.FARCASTER_API_BASE}/fidByAddress?address=${address}`);
      return response.data.fid?.toString() || null;
    } catch (error) {
      console.error('Failed to fetch FID for address:', error);
      return null;
    }
  }

  // Get social profiles for any wallet address
  async getProfilesByAddress(address: string): Promise<SocialProfile[]> {
    try {
      console.log(`[SocialService] Fetching profiles for address ${address} from ${this.API_BASE}/api/profiles/${address}`);
      
      // Try to fetch from our API endpoint
      const response = await axios.get(`${this.API_BASE}/api/profiles/${address}`);
      
      if (response.data && Array.isArray(response.data.profiles)) {
        console.log(`[SocialService] Found ${response.data.profiles.length} profiles for ${address}:`, response.data.profiles);
        return response.data.profiles;
      }
      
      // If API endpoint doesn't exist yet or returns no profiles, try Farcaster directly
      try {
        console.log(`[SocialService] No profiles found in API, trying Farcaster lookup for ${address}`);
        const fid = await this.getFarcasterFidByAddress(address);
        if (fid) {
          console.log(`[SocialService] Found FID ${fid} for address ${address}, fetching profile`);
          const profile = await this.getFarcasterProfileByFid(parseInt(fid));
          if (profile.displayName || profile.username) {
            console.log(`[SocialService] Found Farcaster profile for ${address}:`, profile);
            return [{
              id: fid,
              username: profile.username,
              displayName: profile.displayName,
              pfpUrl: profile.pfpUrl,
              bio: profile.bio,
              provider: 'farcaster',
              isVerified: true
            }];
          }
        }
      } catch (err) {
        console.error('[SocialService] Failed to get Farcaster profile directly:', err);
      }
      
      console.log(`[SocialService] No profiles found for ${address}`);
      return [];
    } catch (error) {
      console.error('[SocialService] Failed to fetch profiles for address:', error);
      return [];
    }
  }

  // Twitter methods can be added here in the future
  
  // Generic methods
  getProfileFromVerifiedCredentials(verifiedCredentials: any[]): SocialProfile[] {
    const profiles: SocialProfile[] = [];
    
    // Process Farcaster credentials
    const farcasterCred = verifiedCredentials?.find(
      cred => cred.oauthProvider === 'farcaster'
    );

    if (farcasterCred) {
      profiles.push({
        id: farcasterCred.oauthAccountId || farcasterCred.id,
        username: farcasterCred.oauthUsername,
        displayName: farcasterCred.oauthDisplayName,
        pfpUrl: farcasterCred.oauthAccountPhotos?.[0],
        provider: 'farcaster',
        isVerified: farcasterCred.isVerified
      });
    }
    
    // Process Twitter credentials
    const twitterCred = verifiedCredentials?.find(
      cred => cred.oauthProvider === 'twitter'
    );

    if (twitterCred) {
      profiles.push({
        id: twitterCred.oauthAccountId || twitterCred.id,
        username: twitterCred.oauthUsername,
        displayName: twitterCred.oauthDisplayName,
        pfpUrl: twitterCred.oauthAccountPhotos?.[0],
        provider: 'twitter',
        isVerified: twitterCred.isVerified
      });
    }
    
    return profiles;
  }
}

export const socialService = new SocialService(); 