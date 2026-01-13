import { useSocialProfile } from './useSocialProfile';
import { FarcasterIcon, TwitterIcon } from '@dynamic-labs/iconic';

export function SocialProfile() {
  const { profiles, primaryProfile, setPrimaryProfile, loading, error } = useSocialProfile();

  if (loading) {
    return (
      <div className="flex items-center space-x-2">
        <div className="w-4 h-4 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-gray-600">Loading profile...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-600">
        Failed to load profile
      </div>
    );
  }

  if (!primaryProfile) {
    return null;
  }

  // Function to render the appropriate provider icon
  const renderProviderIcon = (provider: string) => {
    switch (provider) {
      case 'farcaster':
        return <FarcasterIcon className="w-5 h-5" />;
      case 'twitter':
        return <TwitterIcon className="w-5 h-5" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Primary profile display */}
      <div className="flex items-center space-x-4 p-4 bg-white rounded-lg shadow-sm">
        <div className="relative">
          {primaryProfile.pfpUrl ? (
            <img
              src={primaryProfile.pfpUrl}
              alt={primaryProfile.displayName || primaryProfile.username || 'Profile'}
              className="w-10 h-10 rounded-full"
            />
          ) : (
            <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center">
              <span className="text-gray-500">
                {primaryProfile.displayName?.[0] || primaryProfile.username?.[0] || '?'}
              </span>
            </div>
          )}
          <div className="absolute -bottom-1 -right-1 bg-white p-0.5 rounded-full">
            {renderProviderIcon(primaryProfile.provider)}
          </div>
        </div>
        <div>
          {primaryProfile.displayName && (
            <h3 className="font-medium text-gray-900">{primaryProfile.displayName}</h3>
          )}
          {primaryProfile.username && (
            <p className="text-sm text-gray-500">@{primaryProfile.username}</p>
          )}
        </div>
      </div>

      {/* Profile selector (only shown if multiple profiles) */}
      {profiles.length > 1 && (
        <div className="mt-4 p-4 bg-white rounded-lg shadow-sm">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Connected accounts</h4>
          <div className="space-y-2">
            {profiles.map((profile) => (
              <div 
                key={`${profile.provider}-${profile.id}`}
                className={`flex items-center justify-between p-2 rounded-lg cursor-pointer hover:bg-gray-50 ${
                  primaryProfile.id === profile.id ? 'bg-primary-light border border-primary-light' : ''
                }`}
                onClick={() => setPrimaryProfile(profile.id)}
              >
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    {profile.pfpUrl ? (
                      <img 
                        src={profile.pfpUrl} 
                        alt={profile.displayName || profile.username || 'Profile'}
                        className="w-8 h-8 rounded-full" 
                      />
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <span className="text-gray-500">
                          {profile.displayName?.[0] || profile.username?.[0] || '?'}
                        </span>
                      </div>
                    )}
                    <div className="absolute -bottom-1 -right-1 bg-white p-0.5 rounded-full">
                      {renderProviderIcon(profile.provider)}
                    </div>
                  </div>
                  <div>
                    {profile.displayName && (
                      <p className="text-sm font-medium">{profile.displayName}</p>
                    )}
                    {profile.username && (
                      <p className="text-xs text-gray-500">@{profile.username}</p>
                    )}
                  </div>
                </div>
                {primaryProfile.id === profile.id && (
                  <span className="text-xs font-medium text-primary">Primary</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 