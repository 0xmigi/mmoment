import { useEffect, useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { Camera, Video, Power, User, Radio, Signal } from 'lucide-react';
import { io } from 'socket.io-client';
import { CONFIG, timelineConfig } from '../config';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

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
}

interface TimelineProps {
  filter?: 'all' | 'camera' | 'my';
  userAddress?: string;
  variant?: 'camera' | 'full';
}

// Connect to your backend
const socket = io(CONFIG.BACKEND_URL, timelineConfig.wsOptions);

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

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full' }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const timelineRef = useRef<HTMLDivElement>(null);
  const { user } = useDynamicContext();

  // Get the display count based on screen width
  const getDisplayCount = () => {
    if (variant !== 'camera') return Infinity;
    if (typeof window === 'undefined') return 14;
    
    return window.innerWidth < 640 ? 23 : 14;
  };

  const [displayCount, setDisplayCount] = useState(getDisplayCount());

  // Update display count on window resize
  useEffect(() => {
    if (variant !== 'camera') return;

    const handleResize = () => {
      setDisplayCount(getDisplayCount());
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [variant]);

  useEffect(() => {
    // Load stored events on mount
    const storedEvents = localStorage.getItem('timelineEvents');
    if (storedEvents) {
      const parsedEvents = JSON.parse(storedEvents);
      // Sort stored events by timestamp, newest first
      const sortedEvents = [...parsedEvents].sort((a, b) => b.timestamp - a.timestamp);
      setEvents(sortedEvents);
    }

    socket.on('timelineEvent', (event: TimelineEvent) => {
      // Enrich the event with Farcaster profile if available
      const farcasterCred = user?.verifiedCredentials?.find(
        cred => cred.oauthProvider === 'farcaster'
      );
      
      if (farcasterCred && event.user.address === user?.verifiedCredentials?.[0]?.address) {
        event.user = {
          ...event.user,
          displayName: farcasterCred.oauthDisplayName || undefined,
          username: farcasterCred.oauthUsername,
          pfpUrl: farcasterCred.oauthAccountPhotos?.[0] || undefined
        };
      }

      setEvents(prev => {
        const newEvents = [event, ...prev];
        localStorage.setItem('timelineEvents', JSON.stringify(newEvents));
        return newEvents;
      });
    });

    socket.on('recentEvents', (events: TimelineEvent[]) => {
      // Enrich events with Farcaster profiles if available
      const enrichedEvents = events.map(event => {
        const farcasterCred = user?.verifiedCredentials?.find(
          cred => cred.oauthProvider === 'farcaster' && 
          event.user.address === user?.verifiedCredentials?.[0]?.address
        );
        
        if (farcasterCred) {
          return {
            ...event,
            user: {
              ...event.user,
              displayName: farcasterCred.oauthDisplayName || undefined,
              username: farcasterCred.oauthUsername,
              pfpUrl: farcasterCred.oauthAccountPhotos?.[0] || undefined
            }
          };
        }
        return event;
      });

      // Sort events by timestamp, newest first
      const sortedEvents = [...enrichedEvents].sort((a, b) => b.timestamp - a.timestamp);
      setEvents(sortedEvents);
      localStorage.setItem('timelineEvents', JSON.stringify(sortedEvents));
    });

    return () => {
      socket.off('recentEvents');
      socket.off('timelineEvent');
    };
  }, [user?.verifiedCredentials]);

  useImperativeHandle(ref, () => ({
    addEvent: (event: Omit<TimelineEvent, 'id'>) => {
      // Enrich the event with Farcaster profile if available
      const farcasterCred = user?.verifiedCredentials?.find(
        cred => cred.oauthProvider === 'farcaster'
      );
      
      if (farcasterCred) {
        event.user = {
          ...event.user,
          displayName: farcasterCred.oauthDisplayName || undefined,
          username: farcasterCred.oauthUsername,
          pfpUrl: farcasterCred.oauthAccountPhotos?.[0] || undefined
        };
      }

      socket.emit('newTimelineEvent', event);
    },
    getState: () => ({
      events,
      isConnected: socket.connected
    })
  }));

  // Filter events based on selected filter
  const filteredEvents = events.filter(event => {
    if (filter === 'all') return true;
    if (filter === 'camera') return event.type === 'photo_captured' || event.type === 'video_recorded';
    if (filter === 'my' && userAddress) return event.user.address === userAddress;
    return true;
  });

  const displayEvents = variant === 'camera' 
    ? filteredEvents.slice(0, displayCount)
    : filteredEvents;

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

  return (
    <div className="w-full relative" ref={timelineRef}>
      {/* Container with fixed height based on display count */}
      <div 
        className="relative"
        style={{ 
          height: `${displayCount * (window.innerWidth < 640 ? 3 : 3.5)}rem`
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
                    <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gray-100 flex items-center justify-center overflow-hidden">
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
                  <div className={`ml-2 sm:ml-3 flex-left bg-white transition-opacity ${
                    variant === 'camera' && index > 1 ? 'opacity-0' : ''
                  }`}>
                    <p className="text-xs sm:text-sm text-gray-800">
                      <span className="font-medium">
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
    </div>
  );
});

Timeline.displayName = 'Timeline';