/**
 * SessionCard Component
 *
 * Collapsible card showing a session summary with activities.
 * Used in the Activities view to show historical sessions.
 */

import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Camera, Video, Radio, Clock, MapPin } from 'lucide-react';
import { SessionSummary, formatSessionDuration } from '../hooks/useUserSessions';
import { useEncryptedActivities, DecryptedActivityWithMeta } from '../hooks/useEncryptedActivities';
import { ACTIVITY_TYPE, getActivityTypeName } from '../utils/activity-crypto';

interface SessionCardProps {
  session: SessionSummary;
  defaultExpanded?: boolean;
}

/**
 * Format timestamp to readable date/time
 */
function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  const isYesterday = new Date(now.getTime() - 86400000).toDateString() === date.toDateString();

  const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  if (isToday) {
    return `Today at ${timeStr}`;
  } else if (isYesterday) {
    return `Yesterday at ${timeStr}`;
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ` at ${timeStr}`;
  }
}

/**
 * Get icon for activity type
 */
function ActivityIcon({ type }: { type: number }) {
  const className = "w-4 h-4";

  switch (type) {
    case ACTIVITY_TYPE.PHOTO:
      return <Camera className={className} />;
    case ACTIVITY_TYPE.VIDEO:
      return <Video className={className} />;
    case ACTIVITY_TYPE.STREAM_START:
    case ACTIVITY_TYPE.STREAM_END:
      return <Radio className={className} />;
    default:
      return <Camera className={className} />;
  }
}

/**
 * Activity item within a session
 */
function ActivityItem({ activity }: { activity: DecryptedActivityWithMeta }) {
  const content = activity.content;

  // Get display info based on content type
  let description = '';
  let fileName = '';

  if ('type' in content) {
    if (content.type === 'photo') {
      description = 'Took a photo';
      fileName = content.filename || content.pipe_file_name;
    } else if (content.type === 'video') {
      const duration = 'duration_seconds' in content ? content.duration_seconds : null;
      description = duration ? `Recorded ${Math.round(duration)}s video` : 'Recorded a video';
      fileName = content.filename || content.pipe_file_name;
    } else if (content.type === 'stream_start') {
      description = 'Started streaming';
    } else if (content.type === 'stream_end') {
      const duration = 'duration_seconds' in content ? content.duration_seconds : null;
      description = duration ? `Streamed for ${Math.round(duration)}s` : 'Ended stream';
    }
  }

  return (
    <div className="flex items-center gap-3 py-2 px-3 hover:bg-gray-50 rounded-lg transition-colors">
      <div className="flex-shrink-0 w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
        <ActivityIcon type={activity.activityType} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {description}
        </p>
        {fileName && (
          <p className="text-xs text-gray-500 truncate">
            {fileName}
          </p>
        )}
      </div>
      <div className="flex-shrink-0 text-xs text-gray-400">
        {new Date(activity.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  );
}

/**
 * Main SessionCard component
 */
export function SessionCard({ session, defaultExpanded = false }: SessionCardProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const { activities, isLoading, error, fetchActivities } = useEncryptedActivities();

  // Fetch activities when expanded
  useEffect(() => {
    if (isExpanded && activities.length === 0 && !isLoading) {
      fetchActivities(session.sessionId);
    }
  }, [isExpanded, activities.length, isLoading, fetchActivities, session.sessionId]);

  // Count unique activity types
  const uniqueTypes = new Set(session.activityTypes);
  const hasPhotos = uniqueTypes.has(ACTIVITY_TYPE.PHOTO);
  const hasVideos = uniqueTypes.has(ACTIVITY_TYPE.VIDEO);
  const hasStreams = uniqueTypes.has(ACTIVITY_TYPE.STREAM_START);

  // Format camera ID for display
  const shortCameraId = session.cameraId.slice(0, 8) + '...' + session.cameraId.slice(-4);

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      {/* Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {/* Activity type icons */}
          <div className="flex items-center gap-1">
            {hasPhotos && (
              <div className="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                <Camera className="w-4 h-4" />
              </div>
            )}
            {hasVideos && (
              <div className="w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center">
                <Video className="w-4 h-4" />
              </div>
            )}
            {hasStreams && (
              <div className="w-8 h-8 bg-red-100 text-red-600 rounded-full flex items-center justify-center">
                <Radio className="w-4 h-4" />
              </div>
            )}
          </div>

          {/* Session info */}
          <div className="text-left">
            <p className="text-sm font-medium text-gray-900">
              {formatTimestamp(session.startTime)}
            </p>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatSessionDuration(session.startTime, session.endTime)}
              </span>
              <span className="flex items-center gap-1">
                <MapPin className="w-3 h-3" />
                {shortCameraId}
              </span>
            </div>
          </div>
        </div>

        {/* Activity count & expand icon */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">
            {session.activityCount} {session.activityCount === 1 ? 'activity' : 'activities'}
          </span>
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-gray-100 px-4 py-2">
          {isLoading ? (
            <div className="py-4 text-center text-sm text-gray-500">
              <div className="animate-spin w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full mx-auto mb-2" />
              Decrypting activities...
            </div>
          ) : error ? (
            <div className="py-4 text-center text-sm text-red-500">
              {error}
            </div>
          ) : activities.length === 0 ? (
            <div className="py-4 text-center text-sm text-gray-500">
              No activities found
            </div>
          ) : (
            <div className="divide-y divide-gray-50">
              {activities.map((activity, index) => (
                <ActivityItem key={`${activity.timestamp}-${index}`} activity={activity} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
