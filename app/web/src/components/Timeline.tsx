import { useEffect, useState, forwardRef, useImperativeHandle, useRef, useCallback } from 'react';
import { Camera, Video, Power, User, Radio, Signal } from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import { CONFIG, timelineConfig } from '../config';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { ProfileModal } from './ProfileModal';

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
}

interface TimelineProps {
  filter?: 'all' | 'camera' | 'my';
  userAddress?: string;
  variant?: 'camera' | 'full';
}

// Debounce function to limit update frequency
const debounce = (fn: Function, ms = 300) => {
  let timeoutId: ReturnType<typeof setTimeout>;
  return function (this: any, ...args: any[]) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), ms);
  };
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

let socket: Socket | null = null;
let connectionAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;

const initializeSocket = () => {
  // Don't try to reconnect if we've exceeded attempts
  if (connectionAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.log('Max WebSocket reconnection attempts reached, falling back to polling');
    return null;
  }

  // If we already have a connected socket, return it
  if (socket?.connected) {
    return socket;
  }

  // Clean up existing socket if any
  if (socket) {
    socket.removeAllListeners();
    socket.close();
    socket = null;
  }

  try {
    socket = io(CONFIG.WS_URL, {
      ...timelineConfig.wsOptions,
      timeout: 5000, // Shorter timeout for faster failure detection
      reconnectionAttempts: 2, // Limit reconnection attempts
      reconnectionDelay: 1000,
      autoConnect: false // We'll handle connection manually
    });

    socket.on('connect', () => {
      console.log('Timeline WebSocket connected');
      connectionAttempts = 0; // Reset attempts on successful connection
    });

    socket.on('connect_error', (error) => {
      console.warn('Timeline WebSocket connection error:', error);
      connectionAttempts++;
      if (connectionAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.log('Max reconnection attempts reached, falling back to polling');
        socket?.close();
        socket = null;
      }
    });

    socket.on('disconnect', (reason) => {
      console.log('Timeline WebSocket disconnected:', reason);
    });

    // Attempt connection
    socket.connect();
    return socket;
  } catch (error) {
    console.warn('Failed to initialize WebSocket:', error);
    return null;
  }
};

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full' }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [selectedUser, setSelectedUser] = useState<TimelineUser | null>(null);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const updatePending = useRef(false);
  const { user } = useDynamicContext();
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout>();

  // Function to fetch events via HTTP if WebSocket isn't available
  const fetchEvents = useCallback(async () => {
    try {
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/timeline/events`);
      if (response.ok) {
        const data = await response.json();
        setEvents(prev => {
          const newEvents = [...data, ...prev].slice(0, 100);
          return newEvents;
        });
      }
    } catch (error) {
      console.warn('Failed to fetch events:', error);
    }
  }, []);

  // Batch updates using requestAnimationFrame
  const batchUpdate = useCallback((newEvents: TimelineEvent[]) => {
    if (!updatePending.current) {
      updatePending.current = true;
      requestAnimationFrame(() => {
        setEvents(newEvents);
        // Save to localStorage in the background
        setTimeout(() => {
          try {
            localStorage.setItem('timelineEvents', JSON.stringify(newEvents));
          } catch (e) {
            console.warn('Failed to save timeline events:', e);
          }
          updatePending.current = false;
        }, 0);
      });
    }
  }, []);

  // Debounced version of batchUpdate
  const debouncedUpdate = useCallback(debounce(batchUpdate, 1000), [batchUpdate]);

  useEffect(() => {
    // Try to initialize WebSocket
    socketRef.current = initializeSocket();

    // If WebSocket fails, fall back to polling
    if (!socketRef.current) {
      console.log('WebSocket unavailable, using polling fallback');
      fetchEvents();
      pollIntervalRef.current = setInterval(fetchEvents, 10000);
    }

    // Load stored events on mount
    try {
      const storedEvents = localStorage.getItem('timelineEvents');
      if (storedEvents) {
        const parsedEvents = JSON.parse(storedEvents);
        batchUpdate(parsedEvents);
      }
    } catch (e) {
      console.warn('Failed to load timeline events:', e);
    }

    // Socket event handlers
    const handleNewEvent = (event: TimelineEvent) => {
      setEvents(prev => {
        const newEvents = [event, ...prev].slice(0, 100); // Limit to 100 events
        debouncedUpdate(newEvents);
        return newEvents;
      });
    };

    socketRef.current?.on('timelineEvent', handleNewEvent);
    return () => {
      if (socketRef.current) {
        socketRef.current.off('timelineEvent', handleNewEvent);
        socketRef.current.removeAllListeners();
        socketRef.current.close();
        socketRef.current = null;
      }
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [batchUpdate, debouncedUpdate, fetchEvents]);

  // Get the display count based on screen width
  const getDisplayCount = () => {
    if (variant !== 'camera') return Infinity;
    if (typeof window === 'undefined') return 13;
    
    return window.innerWidth < 640 ? 23 : 13;  // Show 13 items
  };

  // Get the container height based on screen width (one more than display count)
  const getContainerHeight = () => {
    if (variant !== 'camera') return 'auto';
    if (typeof window === 'undefined') return '49rem'; // 14 * 3.5rem
    
    return window.innerWidth < 640 ? `${23 * 3}rem` : `${14 * 3.5}rem`;  // Height for 14 slots
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
    socketRef.current?.on('recentEvents', (events: TimelineEvent[]) => {
      console.log('Received recent events:', events);
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
      console.log('Setting sorted events:', sortedEvents);
      setEvents(sortedEvents);
      localStorage.setItem('timelineEvents', JSON.stringify(sortedEvents));
    });
  }, [user?.verifiedCredentials]);

  useImperativeHandle(ref, () => ({
    addEvent: (event: Omit<TimelineEvent, 'id'>) => {
      console.log('Adding event through ref:', event);
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

      console.log('Emitting newTimelineEvent:', event);
      socketRef.current?.emit('newTimelineEvent', event);
    },
    getState: () => ({
      events,
      isConnected: socketRef.current?.connected
    })
  }));

  // Filter events based on selected filter
  const filteredEvents = events.filter(event => {
    if (filter === 'all') return true;
    if (filter === 'camera') return event.type === 'photo_captured' || 
                                   event.type === 'video_recorded' || 
                                   event.type === 'stream_started' || 
                                   event.type === 'stream_ended';
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

  const handleProfileClick = (event: TimelineEvent) => {
    setSelectedUser(event.user);
    setIsProfileModalOpen(true);
    setSelectedEvent(event);
  };

  return (
    <div className="w-full relative" ref={timelineRef}>
      {/* Container with fixed height based on display count + 1 */}
      <div 
        className="relative"
        style={{ 
          height: getContainerHeight()
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
                      onClick={() => handleProfileClick(event)}
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
                  <div className={`ml-2 sm:ml-3 flex-left bg-white transition-opacity ${
                    variant === 'camera' && index > 1 ? 'opacity-0' : ''
                  }`}>
                    <p className="text-xs sm:text-sm text-gray-800">
                      <span 
                        onClick={() => handleProfileClick(event)}
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
            ...selectedUser,
            farcasterUsername: selectedUser.username
          }}
          action={{
            type: selectedEvent.type,
            timestamp: selectedEvent.timestamp,
            transactionId: selectedEvent.transactionId
          }}
        />
      )}
    </div>
  );
});

Timeline.displayName = 'Timeline';