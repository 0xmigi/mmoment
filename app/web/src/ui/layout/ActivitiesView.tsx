/**
 * ActivitiesView - Historical timeline of user sessions
 *
 * Shows all sessions where the user had access (was checked in).
 * Each session is collapsible and shows decrypted activities.
 */

import { useEffect, useState } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useParams } from 'react-router-dom';
import { RefreshCw, Calendar, Filter } from 'lucide-react';
import { useUserSessions } from '../../hooks/useUserSessions';
import { SessionCard } from '../../timeline/SessionCard';

type FilterType = 'all' | 'with_media' | 'this_camera';

export function ActivitiesView() {
  const { primaryWallet } = useDynamicContext();
  const { cameraId: routeCameraId } = useParams<{ cameraId?: string }>();
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');

  const { sessions, isLoading, error, fetchSessions } = useUserSessions();

  // Fetch sessions on mount and when wallet changes
  useEffect(() => {
    if (primaryWallet?.address) {
      fetchSessions();
    }
  }, [primaryWallet?.address, fetchSessions]);

  // Debug: Log sessions and route camera ID
  console.log(`[ActivitiesView] routeCameraId: ${routeCameraId || 'none'}`);
  console.log(`[ActivitiesView] Total sessions fetched: ${sessions.length}`);
  if (sessions.length > 0) {
    console.log(`[ActivitiesView] Session cameraIds:`, sessions.map(s => s.cameraId));
  }

  // Filter sessions based on selected filter
  const filteredSessions = sessions.filter(session => {
    // If viewing from a specific camera route AND filter is 'this_camera', show only that camera
    // Otherwise show all sessions
    if (activeFilter === 'this_camera' && routeCameraId && session.cameraId !== routeCameraId) {
      return false;
    }

    // Apply user filter
    if (activeFilter === 'all') return true;
    if (activeFilter === 'with_media') return session.activityCount > 0;
    if (activeFilter === 'this_camera') return routeCameraId ? session.cameraId === routeCameraId : true;
    return true;
  });

  console.log(`[ActivitiesView] Filtered sessions: ${filteredSessions.length}`);

  // Dynamic filters based on context
  const filters: Array<{ id: FilterType; label: string }> = [
    { id: 'all', label: 'All Sessions' },
    { id: 'with_media', label: 'With Media' },
    // Add "This Camera" filter when viewing from a camera route
    ...(routeCameraId ? [{ id: 'this_camera' as FilterType, label: 'This Camera' }] : []),
  ];

  return (
    <div className="pb-20">
      <div className="max-w-3xl mx-auto pt-8 px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Sessions</h1>
            <p className="text-sm text-gray-500 mt-1">
              Camera sessions you've participated in
            </p>
          </div>
          <button
            onClick={() => fetchSessions()}
            disabled={isLoading}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 mb-6 overflow-x-auto pb-2">
          <Filter className="w-4 h-4 text-gray-400 flex-shrink-0" />
          {filters.map((filter) => (
            <button
              key={filter.id}
              onClick={() => setActiveFilter(filter.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors whitespace-nowrap
                ${activeFilter === filter.id
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              {filter.label}
            </button>
          ))}
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
              {activeFilter !== 'all'
                ? 'No sessions match this filter. Try a different filter.'
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
