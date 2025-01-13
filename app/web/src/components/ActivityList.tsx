import { useEffect, useState } from 'react';
import { Camera, Power, User } from 'lucide-react';

export type ActivityType =
  | 'initialization'
  | 'user_connected'
  | 'photo_captured'
  | 'video_recorded';

interface ActivityUser {
  address: string;
  username?: string;
}

interface Activity {
  id: string;
  type: ActivityType;
  user: ActivityUser;
  timestamp: number;
}

interface ActivityListProps {
  filter?: 'all' | 'camera' | 'my';
  userAddress?: string;
  timelineRef?: any;
}

export function ActivityList({ filter = 'all', userAddress, timelineRef }: ActivityListProps) {
  const [activities, setActivities] = useState<Activity[]>([]);

  useEffect(() => {
    if (!timelineRef?.current) return;

    // Get initial state from Timeline component
    const timelineState = timelineRef.current.getState();
    setActivities(timelineState.events || []);

    // Subscribe to Timeline events
    const unsubscribe = timelineRef.current.subscribe((event: Activity) => {
      setActivities(prev => [event, ...prev]);
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [timelineRef]);

  // Filter activities based on selected filter
  const filteredActivities = activities.filter(activity => {
    if (filter === 'all') return true;
    if (filter === 'camera') return activity.type === 'photo_captured' || activity.type === 'video_recorded';
    if (filter === 'my' && userAddress) return activity.user.address === userAddress;
    return true;
  });

  const getActivityIcon = (type: ActivityType) => {
    switch (type) {
      case 'initialization':
        return <Power className="w-4 h-4 text-blue-500" />;
      case 'photo_captured':
        return <Camera className="w-4 h-4 text-green-500" />;
      case 'video_recorded':
        return <Camera className="w-4 h-4 text-purple-500" />;
      default:
        return <User className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return 'less than a minute ago';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} minutes ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  return (
    <div className="space-y-4">
      {filteredActivities.length === 0 ? (
        <p className="text-gray-500 text-sm">No activities to display</p>
      ) : (
        filteredActivities.map((activity) => (
          <div key={activity.id} className="flex items-center gap-4 p-4 bg-white rounded-lg shadow-sm">
            <div className="w-8 h-8 rounded-full bg-gray-50 flex items-center justify-center">
              {getActivityIcon(activity.type)}
            </div>
            <div className="flex-1">
              <p className="text-sm text-gray-900">
                <span className="font-medium">
                  {activity.user.address.slice(0, 6)}...{activity.user.address.slice(-4)}
                </span>
                {' '}
                {activity.type.replace(/_/g, ' ')}
              </p>
              <p className="text-xs text-gray-500">
                {formatTimestamp(activity.timestamp)}
              </p>
            </div>
          </div>
        ))
      )}
    </div>
  );
}