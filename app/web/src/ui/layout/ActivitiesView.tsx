/**
 * ActivitiesView - Historical timeline of user sessions
 *
 * Shows all sessions where the user had access (was checked in).
 * Each session is collapsible and shows decrypted activities.
 */

import { useEffect, useState } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Calendar, Filter, Camera, Video, Radio } from 'lucide-react';
import { useUserSessions } from '../../hooks/useUserSessions';
import { SessionCard } from '../../timeline/SessionCard';
import { useCamera } from '../../camera/CameraProvider';
import { ACTIVITY_TYPE } from '../../utils/activity-crypto';

export function ActivitiesView() {
  const { primaryWallet } = useDynamicContext();
  const { selectedCamera } = useCamera();
  const [cameraFilter, setCameraFilter] = useState<string | null>(null);
  const [activityTypeFilter, setActivityTypeFilter] = useState<number | null>(null);
  const [withMediaOnly, setWithMediaOnly] = useState(false);

  const { sessions, isLoading, error, fetchSessions } = useUserSessions();

  // Get connected camera ID from context
  const connectedCameraId = selectedCamera?.publicKey || localStorage.getItem('directCameraId');

  // Fetch sessions on mount and when wallet changes
  useEffect(() => {
    if (primaryWallet?.address) {
      fetchSessions();
    }
  }, [primaryWallet?.address, fetchSessions]);

  // Filter sessions
  const filteredSessions = sessions.filter(session => {
    // Camera filter
    if (cameraFilter === 'connected' && connectedCameraId && session.cameraId !== connectedCameraId) {
      return false;
    }
    // Media filter
    if (withMediaOnly && session.activityCount === 0) {
      return false;
    }
    // Activity type filter
    if (activityTypeFilter !== null && !session.activityTypes.includes(activityTypeFilter)) {
      return false;
    }
    return true;
  });

  return (
    <div className="pb-20">
      <div className="max-w-3xl mx-auto pt-8 px-4">
        {/* Title */}
        <h2 className="text-xl font-semibold mb-4">Activities</h2>

        {/* Filters - clean single row */}
        <div className="flex items-center gap-2 overflow-x-auto mb-6 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
          <Filter className="w-4 h-4 text-gray-400 flex-shrink-0" />

          {/* Camera filter */}
          {connectedCameraId && (
            <button
              onClick={() => setCameraFilter(cameraFilter === 'connected' ? null : 'connected')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap
                ${cameraFilter === 'connected'
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              This Camera
            </button>
          )}

          {/* Media filter */}
          <button
            onClick={() => setWithMediaOnly(!withMediaOnly)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap
              ${withMediaOnly
                ? 'bg-gray-900 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            With Media
          </button>

          {/* Activity type filters */}
          <button
            onClick={() => setActivityTypeFilter(activityTypeFilter === ACTIVITY_TYPE.PHOTO ? null : ACTIVITY_TYPE.PHOTO)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center gap-1
              ${activityTypeFilter === ACTIVITY_TYPE.PHOTO
                ? 'bg-gray-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            <Camera className="w-3 h-3" />
            Photos
          </button>

          <button
            onClick={() => setActivityTypeFilter(activityTypeFilter === ACTIVITY_TYPE.VIDEO ? null : ACTIVITY_TYPE.VIDEO)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center gap-1
              ${activityTypeFilter === ACTIVITY_TYPE.VIDEO
                ? 'bg-gray-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            <Video className="w-3 h-3" />
            Videos
          </button>

          <button
            onClick={() => setActivityTypeFilter(activityTypeFilter === ACTIVITY_TYPE.STREAM ? null : ACTIVITY_TYPE.STREAM)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center gap-1
              ${activityTypeFilter === ACTIVITY_TYPE.STREAM
                ? 'bg-gray-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            <Radio className="w-3 h-3" />
            Streams
          </button>
        </div>

        {/* Content */}
      {!primaryWallet?.address ? (
        <div className="bg-gray-50 rounded-xl p-8 text-center">
          <Calendar className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Connect your wallet
          </h3>
          <p className="text-sm text-gray-500">
            Connect your wallet to see your activity history
          </p>
        </div>
      ) : isLoading && sessions.length === 0 ? (
        <div className="bg-gray-50 rounded-xl p-8 text-center">
          <div className="animate-spin w-8 h-8 border-2 border-gray-300 border-t-gray-600 rounded-full mx-auto mb-4" />
          <p className="text-sm text-gray-500">Loading your sessions...</p>
        </div>
      ) : error ? (
        <div className="bg-red-50 rounded-xl p-8 text-center">
          <p className="text-sm text-red-600">{error}</p>
          <button
            onClick={() => fetchSessions()}
            className="mt-4 px-4 py-2 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 transition-colors"
          >
            Try again
          </button>
        </div>
      ) : filteredSessions.length === 0 ? (
        <div className="bg-gray-50 rounded-xl p-8 text-center">
          <Calendar className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No completed sessions
          </h3>
          <p className="text-sm text-gray-500">
            {(cameraFilter || withMediaOnly || activityTypeFilter !== null)
              ? 'No sessions match your filters. Try adjusting your filter selection.'
              : 'Sessions appear here after you check out from a camera. Complete a session by checking out to see it here.'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredSessions.map((session, index) => (
            <SessionCard
              key={session.sessionId}
              session={session}
              defaultExpanded={index === 0}
            />
          ))}
        </div>
      )}

      {/* Session count */}
      {filteredSessions.length > 0 && (
        <p className="text-center text-sm text-gray-400 mt-6">
          Showing {filteredSessions.length} session{filteredSessions.length !== 1 ? 's' : ''}
        </p>
      )}
      </div>
    </div>
  );
}
