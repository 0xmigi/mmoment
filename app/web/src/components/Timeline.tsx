import { useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { Camera, Video, Power, User } from 'lucide-react';
import { io } from 'socket.io-client';

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

export interface TimelineHandle {
  addEvent: (event: Omit<TimelineEvent, 'id'>) => void;
}

// Connect to your backend
const socket = io('http://localhost:3001');

export const Timeline = forwardRef<TimelineHandle>((_, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);

  useEffect(() => {
    // Load stored events on mount
    const storedEvents = localStorage.getItem('timelineEvents');
    if (storedEvents) {
      setEvents(JSON.parse(storedEvents));
    }
  
    socket.on('timelineEvent', (event: TimelineEvent) => {
      setEvents(prev => {
        const newEvents = [event, ...prev].slice(0, 50);
        localStorage.setItem('timelineEvents', JSON.stringify(newEvents));
        return newEvents;
      });
    });
  

    // Cleanup on unmount
    return () => {
      socket.off('recentEvents');
      socket.off('timelineEvent');
    };
  }, []);

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
    <div className="left-0 w-full h-full">
      {/* Vertical timeline line */}
      <div className="absolute left-0.5 -top-10 bottom-0 w-px h-full bg-gray-200" />
      
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
    </div>
  );
});

Timeline.displayName = 'Timeline';