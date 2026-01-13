/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState, forwardRef, useRef, useCallback, useMemo } from 'react';
import { Camera, Video, Power, User, Radio, Activity } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { ProfileModal } from '../profile/ProfileModal';
import MediaViewer from '../media/MediaViewer';
import { timelineService } from './timeline-service';
import { TimelineEvent, TimelineEventType, TimelineUser, CVActivityMetadata } from './timeline-types';
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
  /** Show absolute timestamps (e.g. "12:34 PM") instead of relative (e.g. "5 minutes ago") */
  showAbsoluteTime?: boolean;
}

// Get the display count based on screen width
const getDisplayCount = () => {
  if (typeof window === 'undefined') return 10;
  return window.innerWidth < 768 ? 14 : 10;
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

// Format CV activity text based on metadata
const formatCVActivityText = (cvActivity?: CVActivityMetadata): string => {
  if (!cvActivity) {
    return 'completed a CV activity';
  }

  const { app_name, user_stats, results, participant_count } = cvActivity;
  const reps = user_stats?.reps ?? 0;

  // Format app name nicely (pushup -> push-ups)
  const appDisplayName = app_name === 'pushup' ? 'push-ups' :
                         app_name === 'pullup' ? 'pull-ups' :
                         app_name === 'squat' ? 'squats' : app_name;

  // Single participant - just show their count
  if (participant_count === 1) {
    return `completed ${reps} ${appDisplayName}`;
  }

  // Multiple participants - show rank and count
  const userResult = results?.find(r => r.stats?.reps === reps);
  const rank = userResult?.rank ?? 1;
  const rankSuffix = rank === 1 ? 'st' : rank === 2 ? 'nd' : rank === 3 ? 'rd' : 'th';

  return `finished ${rank}${rankSuffix} with ${reps} ${appDisplayName}`;
};

// Add this function before the Timeline component
const getEventText = (type: TimelineEventType, cvActivity?: CVActivityMetadata): string => {
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
      return formatCVActivityText(cvActivity);
    case 'other':
      return 'performed an action';
    default:
      return 'performed an action';
  }
};

// Format absolute timestamp (e.g. "12:34 PM")
const formatAbsoluteTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export const Timeline = forwardRef<any, TimelineProps>(({ filter = 'all', userAddress, variant = 'full', cameraId, mobileOverlay = false, initialEvents, showProfileStack, showAbsoluteTime = false }, ref) => {
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
  // Note: We intentionally exclude enrichEventWithUserInfo from deps to avoid infinite loops
  // Profile enrichment happens separately via displayEvents useMemo
  useEffect(() => {
    if (!cameraId || initialEvents) return;

    // Join the camera room (only once per cameraId)
    timelineService.joinCamera(cameraId);

    // Initialize with existing events from the service (for when navigating back)
    // This fixes events disappearing when switching tabs and coming back
    const existingState = timelineService.getState();
    if (existingState.events.length > 0 && existingState.currentCameraId === cameraId) {
      console.log(`[Timeline] Restoring ${existingState.events.length} existing events for camera ${cameraId}`);
      setEvents(existingState.events);
    }

    // Subscribe to timeline events
    const unsubscribe = timelineService.subscribe((event) => {
      console.log('Timeline event received:', event);
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
          newEvents.push(event);
        } else {
          newEvents.splice(index, 0, event);
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
        }
      };
    }

    return () => {
      unsubscribe();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, initialEvents]);

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
        return <User className={`${iconClass} ${isOverlay ? '' : 'text-primary'}`} />;
      case 'cv_activity':
        return <Activity className={`${iconClass} ${isOverlay ? '' : 'text-purple-500'}`} />;
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
      {/* Container with fixed height for desktop camera variant */}
      <div
        className="relative"
        style={{
          height: variant === 'camera' && !mobileOverlay
            ? '45rem'  // Extends past the gallery on desktop
            : 'auto'
        }}
      >
        {/* Vertical timeline line - stops above the profile stack curve */}
        <div className="absolute left-[4px] md:left-[6px] top-0 bottom-14 w-px bg-gray-200" />

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
                        className="w-6 h-6 md:w-8 md:h-8 rounded-full bg-gray-100 flex items-center justify-center overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary-muted transition-all"
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
                            mobileOverlay ? 'hover:text-gray-300' : 'hover:text-primary'
                          }`}
                        >
                          {event.user.displayName || event.user.username || 
                           `${event.user.address.slice(0, 6)}...${event.user.address.slice(-4)}`}
                        </span>
                        {' '}
                        {getEventText(event.type, event.cvActivity)}
                      </p>
                      <p className={`${mobileOverlay ? 'text-[10px]' : 'text-xs'} ${mobileOverlay ? 'text-gray-300' : 'text-gray-500'}`}>
                        {showAbsoluteTime
                          ? formatAbsoluteTime(event.timestamp)
                          : event.timestamp > Date.now() - 60000
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

        {/* Profile Stack at bottom of timeline with curved connector */}
        {(showProfileStack ?? variant === 'camera') && displayEvents.filter(e => e).length > 0 && (
          <div className="absolute bottom-2 left-0 right-0">
            {/* Vertical connector from main line down to curve - bridges the gap */}
            <div className="absolute left-[4px] md:left-[6px] -top-10 w-px h-10 bg-gray-200" />
            {/* L-shaped corner with rounded curve */}
            <div className="absolute left-[4px] md:left-[6px] top-0 w-[12px] h-[12px]">
              <div className="w-[12px] h-[12px] border-b border-l border-gray-200 rounded-bl-[12px]" />
            </div>
            {/* Horizontal line from curve to avatars */}
            <div className="absolute left-[16px] md:left-[18px] top-[11px] w-[28px] md:w-[36px] h-px bg-gray-200" />

            {/* Profile stack - aligned with curve */}
            <div className="pl-12 md:pl-14 pt-0.5">
              <div className="flex items-center">
                <div className="flex -space-x-1.5 md:-space-x-2">
                  {Array.from(new Set(displayEvents.filter(e => e).map(e => e.user.address)))
                    .slice(0, 6)
                    .map((address, i) => {
                      const event = displayEvents.find(e => e?.user.address === address);
                      const profile = userProfiles[address];
                      const pfpUrl = event?.user.pfpUrl || profile?.pfpUrl;
                      const displayName = event?.user.displayName || profile?.displayName;
                      const username = event?.user.username || profile?.username;

                      return (
                        <div
                          key={address}
                          className="relative w-6 h-6 md:w-8 md:h-8 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center overflow-hidden"
                          style={{ zIndex: 6 - i }}
                        >
                          {pfpUrl ? (
                            <img
                              src={pfpUrl}
                              alt={displayName || username || 'User'}
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
                  {new Set(displayEvents.filter(e => e).map(e => e.user.address)).size || 0} Recently active
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

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
            mediaUrl: selectedEvent.mediaUrl,
            cvActivity: selectedEvent.cvActivity
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