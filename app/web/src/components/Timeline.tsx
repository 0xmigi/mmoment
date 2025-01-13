import { useEffect, useState, forwardRef, useImperativeHandle, useRef } from 'react';
import { Camera, Video, Power, User } from 'lucide-react';
import { io } from 'socket.io-client';
import { CONFIG, timelineConfig } from '../config';

export type TimelineEventType =
  | 'initialization'
  | 'user_connected'
  | 'photo_captured'
  | 'video_recorded';

interface TimelineUser {
  address: string;
  username?: string;
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
  variant?: 'camera' | 'full'; // Add variant prop to distinguish between views
}

// Connect to your backend
const socket = io(CONFIG.BACKEND_URL, timelineConfig.wsOptions);

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full' }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Get the display count based on screen width
  const getDisplayCount = () => {
    if (variant !== 'camera') return Infinity;
    if (typeof window === 'undefined') return 13;
    
    return window.innerWidth < 640 ? 23 : 13; // 640px is Tailwind's 'sm' breakpoint
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
      setEvents(prev => {
        const newEvents = [event, ...prev];
        localStorage.setItem('timelineEvents', JSON.stringify(newEvents));
        return newEvents;
      });
    });

    socket.on('recentEvents', (events: TimelineEvent[]) => {
      // Sort events by timestamp, newest first
      const sortedEvents = [...events].sort((a, b) => b.timestamp - a.timestamp);
      setEvents(sortedEvents);
      localStorage.setItem('timelineEvents', JSON.stringify(sortedEvents));
    });

    return () => {
      socket.off('recentEvents');
      socket.off('timelineEvent');
    };
  }, []);

  useImperativeHandle(ref, () => ({
    addEvent: (event: Omit<TimelineEvent, 'id'>) => {
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
      default:
        return <User className="w-4 h-4" />;
    }
  };

  return (
    <div className="w-full relative" ref={timelineRef}>
      {/* Vertical timeline line - adjust positioning */}
      <div className="absolute left-[4px] sm:left-[6px] top-0 bottom-16 w-px bg-gray-200" />
      
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
                  <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gray-100 flex items-center justify-center">
                    <User className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" />
                  </div>
                </div>

                {/* Event details - fade out only in camera view */}
                <div className={`ml-2 sm:ml-3 flex-left bg-white transition-opacity ${
                  variant === 'camera' && index > 1 ? 'opacity-0' : ''
                }`}>
                  <p className="text-xs sm:text-sm text-gray-800">
                    <span className="font-medium">
                      {event.user.address.slice(0, 6)}...
                    </span>
                    {' '}
                    {event.type.replace(/_/g, ' ')}
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

            {/* Profile Stack - only show in camera variant */}
            {variant === 'camera' && (
              <div className="relative pl-4 sm:pl-5 mt-6 sm:mt-8">
                {/* Profile stack */}
                <div className="flex items-center">
                  <div className="flex -space-x-1.5 sm:-space-x-2">
                    {/* Get unique users from all events, not just displayed ones */}
                    {Array.from(new Set(events.map(e => e.user.address)))
                      .slice(0, 6)
                      .map((address, i) => (
                        <div
                          key={address}
                          className="relative w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center"
                          style={{ zIndex: 6 - i }}
                        >
                          <User className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" />
                        </div>
                      ))}
                  </div>
                  {/* Count of total unique users */}
                  <span className="ml-3 text-xs sm:text-sm text-gray-600 font-medium">
                    {new Set(events.map(e => e.user.address)).size} Recently active
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
});

Timeline.displayName = 'Timeline';