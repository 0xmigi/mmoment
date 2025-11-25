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

type FilterType = 'all' | 'photos' | 'videos' | 'streams';

export function ActivitiesView() {
  const { primaryWallet } = useDynamicContext();
  const { cameraId } = useParams<{ cameraId?: string }>();
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');

  const { sessions, isLoading, error, fetchSessions } = useUserSessions();

  // Fetch sessions on mount and when wallet changes
  useEffect(() => {
    if (primaryWallet?.address) {
      fetchSessions();
    }
  }, [primaryWallet?.address, fetchSessions]);

  // Filter sessions based on selected filter
  const filteredSessions = sessions.filter(session => {
    // If cameraId is provided, only show sessions for that camera
    if (cameraId && session.cameraId !== cameraId) {
      return false;
    }

    // Filter by activity type
    if (activeFilter === 'all') return true;
    if (activeFilter === 'photos') return session.activityTypes.includes(10); // PHOTO
    if (activeFilter === 'videos') return session.activityTypes.includes(20); // VIDEO
    if (activeFilter === 'streams') return session.activityTypes.includes(30); // STREAM_START
    return true;
  });

  const filters: Array<{ id: FilterType; label: string }> = [
    { id: 'all', label: 'All' },
    { id: 'photos', label: 'Photos' },
    { id: 'videos', label: 'Videos' },
    { id: 'streams', label: 'Streams' },
  ];

  return (
    <div className="pb-20">
      <div className="max-w-3xl mx-auto pt-8 px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Activities</h1>
            <p className="text-sm text-gray-500 mt-1">
              Your session history and captured moments
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
              No sessions yet
            </h3>
            <p className="text-sm text-gray-500">
              {activeFilter !== 'all'
                ? `No ${activeFilter} found. Try a different filter.`
                : 'Check in to a camera and capture some moments to see them here.'}
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
