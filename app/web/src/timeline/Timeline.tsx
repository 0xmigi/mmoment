/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState, forwardRef, useRef, useCallback, useMemo } from 'react';
import { Camera, Video, Power, User, Radio, Signal } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { ProfileModal } from '../profile/ProfileModal';
import MediaViewer from '../media/MediaViewer';
import { timelineService } from './timeline-service';
import { TimelineEvent, TimelineEventType, TimelineUser } from './timeline-types';
import { IPFSMedia } from '../storage/ipfs/ipfs-service';
import { CONFIG } from '../core/config';

interface TimelineProps {
  filter?: 'all' | 'camera' | 'my';
  userAddress?: string;
  variant?: 'camera' | 'full';
  cameraId?: string;
  mobileOverlay?: boolean;
  /** Pre-populated events for historical/static display (skips real-time subscription) */
  initialEvents?: TimelineEvent[];
  /** Whether to show the profile stack at the bottom (default: true for camera variant) */
  showProfileStack?: boolean;
}

// Get the display count based on screen width
const getDisplayCount = () => {
  if (typeof window === 'undefined') return 13;
  return window.innerWidth < 768 ? 17 : 13;
};

// Get mobile timeline count based on proportional scaling with stream window
const getMobileTimelineCount = () => {
  if (typeof window === 'undefined') return 10;
  const width = window.innerWidth;
  
  // Scale with very aggressive curve - calibrated for iPhone 12 as baseline
  const minWidth = 320;
  const maxWidth = 768;
  const minItems = 14;  // Just a bit longer for iPhone 12
  const maxItems = 30; // Increased proportionally
  
  const linearRatio = Math.min(Math.max((width - minWidth) / (maxWidth - minWidth), 0), 1);
  // Use even more aggressive curve - square the ratio for dramatic scaling
  const curvedRatio = linearRatio * linearRatio;
  return Math.round(minItems + (maxItems - minItems) * curvedRatio);
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
      return 'started a stream';
    case 'check_in':
      return 'checked in to the camera';
    case 'check_out':
      return 'checked out from the camera';
    case 'auto_check_out':
      return 'was checked out by cleanup bot';
    case 'face_enrolled':
      return 'enrolled their face';
    case 'cv_activity':
      return 'completed a CV activity';
    case 'other':
      return 'performed an action';
    default:
      return 'performed an action';
  }
};

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full', cameraId, mobileOverlay = false, initialEvents, showProfileStack }, ref) => {
  const [events, setEvents] = useState<TimelineEvent[]>(initialEvents || []);
  const [selectedUser, setSelectedUser] = useState<TimelineUser | null>(null);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [displayCount, setDisplayCount] = useState(getDisplayCount());
  const [mobileTimelineCount, setMobileTimelineCount] = useState(getMobileTimelineCount());
  const { user } = useDynamicContext();
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [selectedMedia, setSelectedMedia] = useState<IPFSMedia | null>(null);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [userProfiles, setUserProfiles] = useState<Record<string, any>>({});

  // Enrich event with social info if available
  const enrichEventWithUserInfo = useCallback((event: TimelineEvent): TimelineEvent => {
    // First check if we have the profile in our userProfiles map
    const storedProfile = userProfiles[event.user.address];

    if (storedProfile) {
      return {
        ...event,
        user: {
          ...event.user,
          displayName: storedProfile.displayName || event.user.displayName,
          username: storedProfile.username || event.user.username,
          pfpUrl: storedProfile.pfpUrl || event.user.pfpUrl,
          provider: storedProfile.provider || event.user.provider
        }
      };
    }
    
    // If not in userProfiles, check current user's credentials as before
    if (user?.verifiedCredentials) {
      // Try to determine if this is the current user's event
      const isCurrentUser = user.verifiedCredentials.some(cred => cred.address === event.user.address);
      
      if (isCurrentUser) {
        // Find matching social credentials - prioritize Farcaster over Twitter
        const farcasterCred = user.verifiedCredentials.find(
          cred => cred.oauthProvider === 'farcaster'
        );

        const twitterCred = user.verifiedCredentials.find(
          cred => cred.oauthProvider === 'twitter'
        );
        
        // Prioritize Farcaster over Twitter
        const socialCred = farcasterCred || twitterCred;
        
        if (socialCred) {
          // Store this profile for future use
          const newProfile = {
            address: event.user.address,
            displayName: socialCred.oauthDisplayName,
            username: socialCred.oauthUsername,
            pfpUrl: socialCred.oauthAccountPhotos?.[0],
            provider: socialCred.oauthProvider
          };
          
          setUserProfiles(prev => ({
            ...prev,
            [event.user.address]: newProfile
          }));
          
          return {
            ...event,
            user: {
              ...event.user,
              displayName: socialCred.oauthDisplayName || event.user.displayName,
              username: socialCred.oauthUsername || event.user.username,
              pfpUrl: socialCred.oauthAccountPhotos?.[0] || event.user.pfpUrl,
              provider: socialCred.oauthProvider || event.user.provider
            }
          };
        }
      }
    }
    
    return event;
  }, [user?.verifiedCredentials, userProfiles]);

  // Fetch user profiles for all addresses in the events
  useEffect(() => {
    const fetchUserProfiles = async () => {
      // Get unique addresses that we don't already have profiles for
      const addresses = events
        .map(event => event.user.address)
        .filter((address, index, self) =>
          self.indexOf(address) === index && !userProfiles[address]
        );

      if (addresses.length === 0) {
        return;
      }

      try {
        // Fetch profiles from backend in batch
        const response = await fetch(`${CONFIG.BACKEND_URL}/api/profile/batch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ walletAddresses: addresses }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch profiles from backend');
        }

        const data = await response.json();
        const backendProfiles = data.profiles || {};

        // Convert backend profiles to the format we need
        const newProfiles: Record<string, any> = {};
        for (const [address, profile] of Object.entries(backendProfiles)) {
          newProfiles[address] = {
            address,
            displayName: (profile as any).displayName,
            username: (profile as any).username,
            pfpUrl: (profile as any).profileImage,
            provider: (profile as any).provider
          };
        }

        // Update our profiles state
        if (Object.keys(newProfiles).length > 0) {
          setUserProfiles(prev => ({
            ...prev,
            ...newProfiles
          }));
        }
      } catch (error) {
        console.error('[Timeline] Error fetching profiles from backend:', error);
      }
    };

    fetchUserProfiles();
  }, [events]); // Only depend on events, not userProfiles to avoid infinite loop


  // Update display count on window resize
  useEffect(() => {
    if (variant !== 'camera') return;

    const handleResize = () => {
      setDisplayCount(getDisplayCount());
      setMobileTimelineCount(getMobileTimelineCount());
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [variant]);

  // Update events when initialEvents changes (for historical mode)
  useEffect(() => {
    if (initialEvents) {
      setEvents(initialEvents);
    }
  }, [initialEvents]);

  // Timeline service integration (skip if using initialEvents for historical display)
  useEffect(() => {
    if (!cameraId || initialEvents) return;

    // Join the camera room
    timelineService.joinCamera(cameraId);

    // Initialize with existing events from the service (for when navigating back)
    // This fixes events disappearing when switching tabs and coming back
    const existingState = timelineService.getState();
    if (existingState.events.length > 0 && existingState.currentCameraId === cameraId) {
      console.log(`[Timeline] Restoring ${existingState.events.length} existing events for camera ${cameraId}`);
      setEvents(existingState.events.map(enrichEventWithUserInfo));
    }

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
        },
        refreshTimeline: () => {
          timelineService.refreshEvents();
          // Also request from server if connected
          if (cameraId) {
            timelineService.joinCamera(cameraId);
          }
        }
      };
    }

    return () => {
      unsubscribe();
    };
  }, [cameraId, enrichEventWithUserInfo]);

  // Filter events based on selected filter
  const filteredEvents = useMemo(() => {
    // Sort events by timestamp (newest first) before filtering
    const sortedEvents = [...events].sort((a, b) => b.timestamp - a.timestamp);
    return sortedEvents.filter(event => {
      if (filter === 'my' && userAddress) {
        return event.user.address === userAddress;
      }
      return true;
    });
  }, [events, filter, userAddress]);

  // Get display events based on variant and display count
  const displayEvents = useMemo(() => {
    // Apply enrichEventWithUserInfo to all events
    const enrichedEvents = filteredEvents.map(event => enrichEventWithUserInfo(event));
    
    if (variant === 'camera' && !mobileOverlay) {
      return enrichedEvents.slice(0, displayCount); // Desktop: use JavaScript count (perfect!)
    } else if (variant === 'camera' && mobileOverlay) {
      // Mobile: Always create array of mobileTimelineCount length
      // Fill with actual events first, then pad with empty slots
      const result = enrichedEvents.slice(0, mobileTimelineCount);
      // Pad with null entries to always reach mobileTimelineCount
      while (result.length < mobileTimelineCount) {
        result.push(null as any);
      }
      return result;
    }
    return enrichedEvents;
  }, [filteredEvents, variant, displayCount, mobileTimelineCount, userProfiles]);

  const getEventIcon = (type: TimelineEventType, isOverlay = false) => {
    const iconClass = `w-4 h-4 ${isOverlay ? 'text-white' : ''}`;

    switch (type) {
      case 'initialization':
        return <Power className={iconClass} />;
      case 'photo_captured':
        return <Camera className={iconClass} />;
      case 'video_recorded':
        return <Video className={iconClass} />;
      case 'stream_started':
        return <Radio className={`${iconClass} ${isOverlay ? '' : 'text-red-500'}`} />;
      case 'check_in':
        return <User className={`${iconClass} ${isOverlay ? '' : 'text-green-500'}`} />;
      case 'check_out':
        return <User className={`${iconClass} ${isOverlay ? '' : 'text-gray-400'}`} />;
      case 'auto_check_out':
        return <User className={`${iconClass} ${isOverlay ? '' : 'text-orange-500'}`} />;
      case 'face_enrolled':
        return <User className={`${iconClass} ${isOverlay ? '' : 'text-blue-500'}`} />;
      case 'cv_activity':
        return <Signal className={`${iconClass} ${isOverlay ? '' : 'text-purple-500'}`} />;
      default:
        return <User className={iconClass} />;
    }
  };

  const handleProfileClick = (event: TimelineEvent) => {
    // Apply social profile enrichment
    const enrichedEvent = enrichEventWithUserInfo(event);
    setSelectedUser(enrichedEvent.user);
    setIsProfileModalOpen(true);
    setSelectedEvent(enrichedEvent);
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
      {/* Container - desktop uses JavaScript calculation, mobile uses CSS width-based */}
      <div 
        className="relative"
        style={{ 
          height: variant === 'camera' && !mobileOverlay
            ? `${(displayCount + 1) * 3.5}rem`
            : 'auto'
        }}
      >
        {/* Vertical timeline line */}
        <div className="absolute left-[4px] md:left-[6px] top-0 h-full w-px bg-gray-200" />
        
        <div className="space-y-4 md:space-y-6 w-full">
          {displayEvents.length === 0 ? (
            <p className={`text-sm pl-16 ${mobileOverlay ? 'text-white' : 'text-gray-500'}`}>No activity yet</p>
          ) : (
            <>
              {displayEvents.map((event, index) => {
                // Handle null events (empty slots) for mobile overlay
                if (!event) {
                  return (
                    <div key={`empty-${index}`} className="h-8">
                      {/* Empty slot - just space, no visual elements */}
                    </div>
                  );
                }
                
                return (
                  <div key={event.id} className="flex items-center">
                    {/* User avatar and action icon - always visible */}
                    <div className="relative flex items-center mr-3 md:mr-5 z-10">
                      <div className={`w-4 h-4 md:w-5 md:h-5 rounded-full -ml-[4px] md:-ml-[4px] mr-2 flex items-center justify-center ${
                        mobileOverlay ? '' : 'bg-white border border-gray-200'
                      }`}>
                        {getEventIcon(event.type, mobileOverlay)}
                      </div>
                      <div 
                        onClick={(e) => {
                          e.stopPropagation();
                          handleProfileClick(event);
                        }}
                        className="w-6 h-6 md:w-8 md:h-8 rounded-full bg-gray-100 flex items-center justify-center overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all"
                      >
                        {event.user.pfpUrl ? (
                          <img 
                            src={event.user.pfpUrl} 
                            alt={event.user.displayName || event.user.username || 'User'} 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <User className="w-3 h-3 md:w-4 md:h-4 text-gray-600" />
                        )}
                      </div>
                    </div>

                    {/* Event details - fade out only in camera view */}
                    <div 
                      className={`ml-2 md:ml-3 flex-left transition-opacity cursor-pointer ${
                        mobileOverlay ? '' : 'bg-white'
                      } ${
                        variant === 'camera' && index > 1 ? 'opacity-0' : ''
                      }`}
                      onClick={() => event.mediaUrl && handleMediaClick(event)}
                    >
                      <p className={`text-xs md:text-sm ${mobileOverlay ? 'text-white' : 'text-gray-800'}`}>
                        <span 
                          onClick={(e) => {
                            e.stopPropagation();
                            handleProfileClick(event);
                          }}
                          className={`font-medium cursor-pointer transition-colors ${
                            mobileOverlay ? 'hover:text-gray-300' : 'hover:text-blue-600'
                          }`}
                        >
                          {event.user.displayName || event.user.username || 
                           `${event.user.address.slice(0, 6)}...${event.user.address.slice(-4)}`}
                        </span>
                        {' '}
                        {getEventText(event.type)}
                      </p>
                      <p className={`${mobileOverlay ? 'text-[10px]' : 'text-xs'} ${mobileOverlay ? 'text-gray-300' : 'text-gray-500'}`}>
                        {event.timestamp > Date.now() - 60000 
                          ? 'less than a minute ago'
                          : `${Math.floor((Date.now() - event.timestamp) / 60000)} minutes ago`
                        }
                      </p>
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </div>

      {/* Profile Stack with connected timeline */}
      {(showProfileStack ?? variant === 'camera') && (
        <div className="relative">
          {/* Corner and horizontal line container */}
          <div className="absolute left-0 top-0 w-full">
            {/* L-shaped corner with rounded curve using CSS */}
            <div className="absolute left-[4px] md:left-[6px] w-[12px] h-[12px]">
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
          <div className="pl-10 md:pl-14">
            <div className="flex items-center">
              <div className="flex -space-x-1.5 md:-space-x-2">
                {Array.from(new Set(events.map(e => e.user.address)))
                  .slice(0, 6)
                  .map((address, i) => {
                    const event = events.find(e => e.user.address === address);
                    return (
                      <div
                        key={address}
                        className="relative w-6 h-6 md:w-8 md:h-8 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center overflow-hidden"
                        style={{ zIndex: 6 - i }}
                      >
                        {event?.user.pfpUrl ? (
                          <img 
                            src={event.user.pfpUrl} 
                            alt={event.user.displayName || event.user.username || 'User'} 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <User className="w-3 h-3 md:w-4 md:h-4 text-gray-600" />
                        )}
                      </div>
                    );
                  })}
              </div>
              <span className={`ml-3 text-xs md:text-sm font-medium ${mobileOverlay ? 'text-white' : 'text-gray-600'}`}>
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
            provider: selectedUser.provider, // Include provider field from backend
            verifiedCredentials: 
              user?.verifiedCredentials?.some(cred => 
                cred.address === selectedUser.address)
                ? user?.verifiedCredentials?.filter(cred => 
                    cred.oauthProvider === 'farcaster' || cred.oauthProvider === 'twitter'
                  )?.map(cred => ({
                    oauthProvider: cred.oauthProvider as string,
                    oauthDisplayName: cred.oauthDisplayName || undefined,
                    oauthUsername: cred.oauthUsername,
                    oauthAccountPhotos: cred.oauthAccountPhotos
                  }))
                : user?.verifiedCredentials
                  ?.filter(cred => 
                    (cred.oauthProvider === 'farcaster' || cred.oauthProvider === 'twitter') && 
                    cred.address === selectedUser.address
                  )
                  ?.map(cred => ({
                    oauthProvider: cred.oauthProvider as string,
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