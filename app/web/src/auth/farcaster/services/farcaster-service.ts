// src/services/farcaster-service.ts
import axios from 'axios';

export interface FarcasterProfile {
  username?: string;
  displayName?: string;
  bio?: string;
  pfpUrl?: string;
  fid?: number;
}

export class FarcasterService {
  private readonly API_BASE = 'http://localhost:2281/v1'; // Update this with your actual API endpoint

  async getProfileByFid(fid: number): Promise<FarcasterProfile> {
    try {
      const response = await axios.get(`${this.API_BASE}/userDataByFid?fid=${fid}`);
      const messages = response.data.messages || [];
      
      const profile: FarcasterProfile = { fid };
      
      messages.forEach((message: any) => {
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

  async getFidByAddress(address: string): Promise<number | null> {
    // This is a placeholder - you'll need to implement the actual logic
    // to get FID from an Ethereum address based on your requirements
    try {
      const response = await axios.get(`${this.API_BASE}/fidByAddress?address=${address}`);
      return response.data.fid;
    } catch (error) {
      console.error('Failed to fetch FID for address:', error);
      return null;
    }
  }

  getProfileFromVerifiedCredentials(verifiedCredentials: any[]): FarcasterProfile | null {
    const farcasterCred = verifiedCredentials?.find(
      cred => cred.oauthProvider === 'farcaster'
    );

    if (!farcasterCred) return null;

    return {
      username: farcasterCred.oauthUsername,
      displayName: farcasterCred.oauthDisplayName,
      pfpUrl: farcasterCred.oauthAccountPhotos?.[0],
      fid: parseInt(farcasterCred.oauthAccountId)
    };
  }
}

export const farcasterService = new FarcasterService();