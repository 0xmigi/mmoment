import { useFarcasterProfile } from './farcaster/services/useFarcasterProfile';

export function FarcasterProfile() {
  const { profile, loading, error } = useFarcasterProfile();

  if (loading) {
    return (
      <div className="flex items-center space-x-2">
        <div className="w-4 h-4 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-gray-600">Loading Farcaster profile...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-600">
        Failed to load Farcaster profile
      </div>
    );
  }

  if (!profile) {
    return null;
  }

  return (
    <div className="flex items-center space-x-4 p-4 bg-white rounded-lg shadow-sm">
      {profile.pfpUrl && (
        <img
          src={profile.pfpUrl}
          alt={profile.displayName || profile.username || 'Profile'}
          className="w-10 h-10 rounded-full"
        />
      )}
      <div>
        {profile.displayName && (
          <h3 className="font-medium text-gray-900">{profile.displayName}</h3>
        )}
        {profile.username && (
          <p className="text-sm text-gray-500">@{profile.username}</p>
        )}
        {profile.bio && (
          <p className="mt-1 text-sm text-gray-600">{profile.bio}</p>
        )}
      </div>
    </div>
  );
} 