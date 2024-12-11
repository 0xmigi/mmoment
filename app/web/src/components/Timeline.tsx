import { useEffect, useState, forwardRef, useImperativeHandle } from 'react';
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
  maxEvents?: number; // Add this to control number of events shown
}

export interface TimelineHandle {
  addEvent: (event: Omit<TimelineEvent, 'id'>) => void;
}

// Connect to your backend
const socket = io(CONFIG.BACKEND_URL, timelineConfig.wsOptions);

// Log connection events for debugging
socket.on('connect', () => {
  console.log('Connected to timeline service at:', CONFIG.BACKEND_URL);
});

socket.on('connect_error', (error) => {
  console.log('Connection error:', error);
});

export const Timeline = forwardRef<any, TimelineProps>(({ maxEvents = 18 }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [uniqueUsers, setUniqueUsers] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Load stored events on mount AND apply maxEvents limit
    const storedEvents = localStorage.getItem('timelineEvents');
    if (storedEvents) {
      const parsedEvents = JSON.parse(storedEvents);
      // Apply the limit here
      setEvents(parsedEvents.slice(0, maxEvents));
    }
  
    socket.on('timelineEvent', (event: TimelineEvent) => {
      setEvents(prev => {
        const newEvents = [event, ...prev].slice(0, maxEvents);
        // Update unique users
        setUniqueUsers(new Set(newEvents.map(e => e.user.address)));
        // Store limited events
        localStorage.setItem('timelineEvents', JSON.stringify(newEvents));
        return newEvents;
      });
    });

    // Cleanup on unmount
    return () => {
      socket.off('recentEvents');
      socket.off('timelineEvent');
    };
  }, [maxEvents]);

  useImperativeHandle(ref, () => ({
    addEvent: (event: Omit<TimelineEvent, 'id'>) => {
      // Emit new event to server
      socket.emit('newTimelineEvent', event);
    }
  }));

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
    <div className="left-0 w-full min-h-[400px] relative">
      {/* Vertical timeline line */}
      <div className="absolute left-0.5 top-0 bottom-16 w-px bg-gray-200" />
      
      <div className="space-y-6 w-full">
        {events.length === 0 ? (
          <p className="text-gray-500 text-sm pl-16">No activity yet</p>
        ) : (
          events.map((event, index) => (
            <div key={event.id} className="flex items-center">
              {/* User avatar and action icon - always visible */}
              <div className="relative flex items-center mr-5 z-10">
              <div className="w-5 h-5 rounded-full bg-white border border-gray-200 -ml-2 mr-2 flex items-center justify-center">
                  {getEventIcon(event.type)}
                </div>
                <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
              </div>

              {/* Event details - only visible for first two events and not overlapped */}
              {index < 2 && (
                <div className="ml-3 flex-left bg-white">
                  <p className="text-sm text-gray-800">
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
              )}
            </div>
          ))
        )}
      </div>
      <div className="relative mt-4 p-4">
        <div className="flex items-center gap-3">
          <div className="flex -space-x-2">
            {Array.from(uniqueUsers).slice(0, 5).map((address, i) => (
              <div 
                key={address} 
                className="w-8 h-8 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center"
                style={{ zIndex: 5 - i }}
              >
                <User className="w-4 h-4 text-gray-600" />
              </div>
            ))}
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            {/* <Users className="w-4 h-4" /> */}
            <span>
              {uniqueUsers.size} {uniqueUsers.size === 1 ? 'person' : 'people'} this session
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

Timeline.displayName = 'Timeline';