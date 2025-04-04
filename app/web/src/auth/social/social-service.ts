import axios from 'axios';
import { SocialProfile } from './useSocialProfile';

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
  private readonly FARCASTER_API_BASE = 'http://localhost:2281/v1'; // Update with your actual API endpoint

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