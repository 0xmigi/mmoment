import { useEffect, useState, forwardRef, useRef, useCallback, useMemo } from 'react';
import { Camera, Video, Power, User, Radio, Signal } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { ProfileModal } from './ProfileModal';
import { IPFSMedia } from '../services/ipfs-service';
import MediaViewer from './MediaViewer';
import { timelineService } from '../services/timeline-service';

export type TimelineEventType =
  | 'initialization'
  | 'user_connected'
  | 'photo_captured'
  | 'video_recorded'
  | 'stream_started'
  | 'stream_ended';

interface TimelineUser {
  address: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
}

interface TimelineEvent {
  id: string;
  type: TimelineEventType;
  user: TimelineUser;
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
}

interface TimelineProps {
  filter?: 'all' | 'camera' | 'my';
  userAddress?: string;
  variant?: 'camera' | 'full';
  cameraId?: string;
}

// Get the display count based on screen width
const getDisplayCount = () => {
  if (typeof window === 'undefined') return 13;
  return window.innerWidth < 640 ? 23 : 13;
};

// Add this function before the Timeline component
const getEventText = (type: TimelineEventType): string => {
  switch (type) {
    case 'photo_captured':
      return 'took a photo';
    case 'video_recorded':
      return 'recorded a video';
    case 'initialization':
      return 'initialized';
    case 'user_connected':
      return 'connected';
    case 'stream_started':
      return 'started the stream';
    case 'stream_ended':
      return 'ended the stream';
  }
};

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full', cameraId }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [selectedUser, setSelectedUser] = useState<TimelineUser | null>(null);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [displayCount, setDisplayCount] = useState(getDisplayCount());
  const { user } = useDynamicContext();
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [selectedMedia, setSelectedMedia] = useState<IPFSMedia | null>(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);

  // Enrich event with Farcaster info if available
  const enrichEventWithUserInfo = useCallback((event: TimelineEvent): TimelineEvent => {
    if (!user?.verifiedCredentials) return event;
    
    // Find matching Farcaster credential
    const farcasterCred = user.verifiedCredentials.find(
      cred => cred.oauthProvider === 'farcaster' && 
              cred.address === event.user.address
    );
    
    if (!farcasterCred) return event;
    
    return {
      ...event,
      user: {
        ...event.user,
        displayName: farcasterCred.oauthDisplayName || event.user.displayName,
        username: farcasterCred.oauthUsername || event.user.username,
        pfpUrl: farcasterCred.oauthAccountPhotos?.[0] || event.user.pfpUrl
      }
    };
  }, [user?.verifiedCredentials]);

  // Update display count on window resize
  useEffect(() => {
    if (variant !== 'camera') return;

    const handleResize = () => {
      setDisplayCount(getDisplayCount());
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [variant]);

  // Timeline service integration
  useEffect(() => {
    if (!cameraId) return;

    // Join the camera room
    timelineService.joinCamera(cameraId);

    // Subscribe to timeline events
    const unsubscribe = timelineService.subscribe((event) => {
      console.log('Timeline event received:', event);
      const enrichedEvent = enrichEventWithUserInfo(event);
      setEvents(prev => {
        // Find the correct position to insert the event (newest first)
        const index = prev.findIndex(e => e.timestamp < event.timestamp);
        const newEvents = [...prev];
        
        // Remove any existing event with the same ID
        const existingIndex = newEvents.findIndex(e => e.id === event.id);
        if (existingIndex !== -1) {
          newEvents.splice(existingIndex, 1);
        }
        
        // Insert the event at the correct position
        if (index === -1) {
          newEvents.push(enrichedEvent);
        } else {
          newEvents.splice(index, 0, enrichedEvent);
        }
        
        // Keep only the last 100 events
        return newEvents.slice(0, 100);
      });
    });

    // Expose addEvent to ref
    if (ref && 'current' in ref) {
      ref.current = {
        addEvent: (event: Omit<TimelineEvent, 'id'>) => {
          timelineService.emitEvent(event);
        }
      };
    }

    return () => {
      unsubscribe();
    };
  }, [cameraId, enrichEventWithUserInfo]);

  // Add polling mechanism for mobile browsers
  useEffect(() => {
    // Check if this is a mobile browser
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (!isMobile) return;

    // Add polling for mobile browsers to refresh timeline events
    const pollInterval = setInterval(() => {
      if (cameraId) {
        console.log('Mobile polling: Refreshing timeline events for camera', cameraId);
        // Re-join the camera to refresh events (this will fetch recent events)
        timelineService.joinCamera(cameraId);
      }
    }, 10000); // Poll every 10 seconds

    return () => {
      clearInterval(pollInterval);
    };
  }, [cameraId]);

  // Filter events based on selected filter
  const filteredEvents = useMemo(() => {
    // Sort events by timestamp (newest first) before filtering
    const sortedEvents = [...events].sort((a, b) => b.timestamp - a.timestamp);
    return sortedEvents.filter(event => {
      if (filter === 'my' && userAddress) {
        return event.user.address === userAddress;
      }
      // Add camera filter - this makes mobile browsers properly filter events
      if (filter === 'camera' && cameraId && event.cameraId) {
        return event.cameraId === cameraId;
      }
      return true;
    });
  }, [events, filter, userAddress, cameraId]);

  // Get display events based on variant and display count
  const displayEvents = useMemo(() => {
    return variant === 'camera' 
      ? filteredEvents.slice(0, displayCount)
      : filteredEvents;
  }, [filteredEvents, variant, displayCount]);

  const getEventIcon = (type: TimelineEventType) => {
    switch (type) {
      case 'initialization':
        return <Power className="w-4 h-4" />;
      case 'photo_captured':
        return <Camera className="w-4 h-4" />;
      case 'video_recorded':
        return <Video className="w-4 h-4" />;
      case 'stream_started':
        return <Radio className="w-4 h-4 text-red-500" />;
      case 'stream_ended':
        return <Signal className="w-4 h-4 text-gray-400" />;
      default:
        return <User className="w-4 h-4" />;
    }
  };

  const handleProfileClick = (event: TimelineEvent) => {
    setSelectedUser(event.user);
    setIsProfileModalOpen(true);
    setSelectedEvent(event);
  };

  const handleMediaClick = (event: TimelineEvent) => {
    if (event.mediaUrl) {
      const media: IPFSMedia = {
        id: event.mediaUrl.split('/').pop() || '',
        url: event.mediaUrl,
        type: event.type === 'video_recorded' ? 'video' : 'image',
        mimeType: event.type === 'video_recorded' ? 'video/mp4' : 'image/jpeg',
        walletAddress: event.user.address,
        timestamp: new Date(event.timestamp).toISOString(),
        backupUrls: [],
        provider: 'IPFS'
      };
      setSelectedMedia(media);
      setSelectedEvent(event);
      setIsViewerOpen(true);
    }
  };

  return (
    <div className="w-full relative" ref={timelineRef}>
      {/* Container with fixed height based on display count */}
      <div 
        className="relative"
        style={{ 
          height: variant === 'camera' 
            ? `${(displayCount + 1) * 3.5}rem`
            : 'auto'
        }}
      >
        {/* Vertical timeline line */}
        <div className="absolute left-[4px] sm:left-[6px] top-0 h-full w-px bg-gray-200" />
        
        <div className="space-y-4 sm:space-y-6 w-full">
          {displayEvents.length === 0 ? (
            <p className="text-gray-500 text-sm pl-16">No activity yet</p>
          ) : (
            <>
              {displayEvents.map((event, index) => (
                <div key={event.id} className="flex items-center">
                  {/* User avatar and action icon - always visible */}
                  <div className="relative flex items-center mr-3 sm:mr-5 z-10">
                    <div className="w-4 h-4 sm:w-5 sm:h-5 rounded-full bg-white border border-gray-200 -ml-[4px] sm:-ml-[4px] mr-2 flex items-center justify-center">
                      {getEventIcon(event.type)}
                    </div>
                    <div 
                      onClick={(e) => {
                        e.stopPropagation();
                        handleProfileClick(event);
                      }}
                      className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gray-100 flex items-center justify-center overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all"
                    >
                      {event.user.pfpUrl ? (
                        <img 
                          src={event.user.pfpUrl} 
                          alt={event.user.displayName || event.user.username || 'User'} 
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <User className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" />
                      )}
                    </div>
                  </div>

                  {/* Event details - fade out only in camera view */}
                  <div 
                    className={`ml-2 sm:ml-3 flex-left bg-white transition-opacity cursor-pointer ${
                      variant === 'camera' && index > 1 ? 'opacity-0' : ''
                    }`}
                    onClick={() => event.mediaUrl && handleMediaClick(event)}
                  >
                    <p className="text-xs sm:text-sm text-gray-800">
                      <span 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleProfileClick(event);
                        }}
                        className="font-medium cursor-pointer hover:text-blue-600 transition-colors"
                      >
                        {event.user.displayName || event.user.username || 
                         `${event.user.address.slice(0, 6)}...${event.user.address.slice(-4)}`}
                      </span>
                      {' '}
                      {getEventText(event.type)}
                    </p>
                    <p className="text-xs text-gray-500">
                      {event.timestamp > Date.now() - 60000 
                        ? 'less than a minute ago'
                        : `${Math.floor((Date.now() - event.timestamp) / 60000)} minutes ago`
                      }
                    </p>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Profile Stack with connected timeline */}
      {variant === 'camera' && (
        <div className="relative">
          {/* Corner and horizontal line container */}
          <div className="absolute left-0 top-0 w-full">
            {/* L-shaped corner with rounded curve using CSS */}
            <div className="absolute left-[4px] sm:left-[6px] w-[12px] h-[12px]">
              {/* Curved corner */}
              <div 
                className="absolute left-0 bottom-0 w-[12px] h-[12px] border-b border-l border-gray-200 rounded-bl-[12px]"
                style={{ borderBottomLeftRadius: '12px' }}
              />
              {/* Horizontal part of the L - made longer */}
              <div className="absolute left-[11px] bottom-0 w-[32px] h-px bg-gray-200" />
            </div>
          </div>

          {/* Profile stack - adjusted padding to align with curve */}
          <div className="pl-10 sm:pl-14">
            <div className="flex items-center">
              <div className="flex -space-x-1.5 sm:-space-x-2">
                {Array.from(new Set(events.map(e => e.user.address)))
                  .slice(0, 6)
                  .map((address, i) => {
                    const event = events.find(e => e.user.address === address);
                    return (
                      <div
                        key={address}
                        className="relative w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center overflow-hidden"
                        style={{ zIndex: 6 - i }}
                      >
                        {event?.user.pfpUrl ? (
                          <img 
                            src={event.user.pfpUrl} 
                            alt={event.user.displayName || event.user.username || 'User'} 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <User className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" />
                        )}
                      </div>
                    );
                  })}
              </div>
              <span className="ml-3 text-xs sm:text-sm text-gray-600 font-medium">
                {new Set(events.map(e => e.user.address)).size || 0} Recently active
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Profile Modal */}
      {selectedUser && selectedEvent && (
        <ProfileModal
          isOpen={isProfileModalOpen}
          onClose={() => {
            setIsProfileModalOpen(false);
            setSelectedUser(null);
            setSelectedEvent(null);
          }}
          user={{
            address: selectedUser.address,
            username: selectedUser.username,
            displayName: selectedUser.displayName,
            pfpUrl: selectedUser.pfpUrl,
            verifiedCredentials: user?.verifiedCredentials
              ?.filter(cred => 
                cred.oauthProvider === 'farcaster' && 
                selectedUser.address === cred.address
              )
              ?.map(cred => ({
                oauthProvider: 'farcaster',
                oauthDisplayName: cred.oauthDisplayName || undefined,
                oauthUsername: cred.oauthUsername,
                oauthAccountPhotos: cred.oauthAccountPhotos
              }))
          }}
          action={{
            type: selectedEvent.type,
            timestamp: selectedEvent.timestamp,
            transactionId: selectedEvent.transactionId,
            mediaUrl: selectedEvent.mediaUrl
          }}
        />
      )}

      {/* Media Viewer */}
      <MediaViewer
        isOpen={isViewerOpen}
        onClose={() => {
          setIsViewerOpen(false);
          setSelectedMedia(null);
          setSelectedEvent(null);
        }}
        media={selectedMedia}
        event={selectedEvent || undefined}
      />
    </div>
  );
});

Timeline.displayName = 'Timeline';