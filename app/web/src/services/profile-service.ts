import { CONFIG } from '../core/config';

export interface UserProfile {
  walletAddress: string;
  displayName?: string;
  username?: string;
  profileImage?: string;
  provider?: string;
}

/**
 * Service for managing user profiles in the backend
 *
 * This service saves user profile data directly to the backend database,
 * bypassing the camera service which doesn't need profile images or provider info.
 */
class ProfileService {
  private readonly backendUrl: string;

  constructor() {
    this.backendUrl = CONFIG.BACKEND_URL;
  }

  /**
   * Save or update user profile in the backend database
   * Called when user connects wallet or updates social auth
   */
  async saveProfile(profile: UserProfile): Promise<void> {
    try {
      const response = await fetch(`${this.backendUrl}/api/profile/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to save profile');
      }

      console.log(`✅ Saved profile to backend for ${profile.walletAddress.slice(0, 8)}...`);
    } catch (error) {
      console.error('Failed to save profile to backend:', error);
      // Don't throw - profile saving is not critical for app functionality
    }
  }

  /**
   * Get user profile from backend
   */
  async getProfile(walletAddress: string): Promise<UserProfile | null> {
    try {
      const response = await fetch(`${this.backendUrl}/api/profile/${walletAddress}`);

      if (response.status === 404) {
        return null;
      }

      if (!response.ok) {
        throw new Error('Failed to fetch profile');
      }

      const data = await response.json();
      return data.profile;
    } catch (error) {
      console.error('Failed to fetch profile from backend:', error);
      return null;
    }
  }

  /**
   * Batch get multiple user profiles
   */
  async getProfiles(walletAddresses: string[]): Promise<Record<string, UserProfile>> {
    try {
      const response = await fetch(`${this.backendUrl}/api/profile/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ walletAddresses }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch profiles');
      }

      const data = await response.json();
      return data.profiles || {};
    } catch (error) {
      console.error('Failed to fetch profiles from backend:', error);
      return {};
    }
  }
}

export const profileService = new ProfileService();
