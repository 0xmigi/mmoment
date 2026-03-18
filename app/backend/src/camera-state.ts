// Shared in-memory state accessors for camera data.
// Separated to avoid circular imports between index.ts and router.ts.

interface TimelineEventUser {
  address: string;
  username?: string;
  displayName?: string;
  pfpUrl?: string;
  provider?: string;
}

interface TimelineEvent {
  id: string;
  type: string;
  user: TimelineEventUser;
  timestamp: number;
  cameraId?: string;
  transactionId?: string;
}

// References to the actual data structures in index.ts — set at startup
let _timelineEvents: TimelineEvent[] = [];
let _cameraRooms: Map<string, Set<string>> = new Map();

export function initCameraState(
  timelineEvents: TimelineEvent[],
  cameraRooms: Map<string, Set<string>>,
) {
  _timelineEvents = timelineEvents;
  _cameraRooms = cameraRooms;
}

export function getActiveUsersForCamera(cameraId: string): number {
  const sockets = _cameraRooms.get(cameraId);
  return sockets ? sockets.size : 0;
}

export function getRecentTimelineEvents(cameraId: string, limit: number = 50) {
  return _timelineEvents
    .filter(e => e.cameraId === cameraId)
    .slice(-limit)
    .reverse();
}

export function getCheckedInUsers(cameraId: string): Array<{
  address: string;
  displayName?: string;
  username?: string;
  pfpUrl?: string;
}> {
  const events = _timelineEvents.filter(e => e.cameraId === cameraId);
  const checkedIn = new Map<string, TimelineEventUser>();

  for (const event of events) {
    if (event.type === 'check_in') {
      checkedIn.set(event.user.address, event.user);
    } else if (event.type === 'check_out' || event.type === 'auto_check_out') {
      checkedIn.delete(event.user.address);
    }
  }

  return Array.from(checkedIn.values()).map(u => ({
    address: u.address,
    displayName: u.displayName,
    username: u.username,
    pfpUrl: u.pfpUrl,
  }));
}
