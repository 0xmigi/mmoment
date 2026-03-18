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

// Per-wallet session state: wallet → cameraId
// Written on check-in, cleared on check-out. Persists for session duration.
// Agents read this via GET /v1/me — no scanning needed.
const _walletSessions: Map<string, string> = new Map();

export function initCameraState(
  timelineEvents: TimelineEvent[],
  cameraRooms: Map<string, Set<string>>,
) {
  _timelineEvents = timelineEvents;
  _cameraRooms = cameraRooms;
}

/** Called when a user checks in at a camera */
export function onWalletCheckIn(walletAddress: string, cameraId: string) {
  _walletSessions.set(walletAddress, cameraId);
}

/** Called when a user checks out from a camera */
export function onWalletCheckOut(walletAddress: string) {
  _walletSessions.delete(walletAddress);
}

/** Get the camera a wallet is currently checked in at (if any) */
export function getWalletCamera(walletAddress: string): string | null {
  return _walletSessions.get(walletAddress) || null;
}

// Track recent API-triggered captures: key = "wallet:cameraId", expires after 30s
const _apiCaptures: Map<string, number> = new Map();

/** Mark a capture as API-triggered */
export function markApiCapture(walletAddress: string, cameraId: string) {
  const key = `${walletAddress}:${cameraId}`;
  _apiCaptures.set(key, Date.now());
  // Clean up old entries
  for (const [k, ts] of _apiCaptures) {
    if (Date.now() - ts > 30000) _apiCaptures.delete(k);
  }
}

/** Check if a recent photo from this wallet+camera was API-triggered */
export function wasApiCapture(walletAddress: string, cameraId: string): boolean {
  const key = `${walletAddress}:${cameraId}`;
  const ts = _apiCaptures.get(key);
  if (!ts) return false;
  if (Date.now() - ts > 30000) {
    _apiCaptures.delete(key);
    return false;
  }
  _apiCaptures.delete(key); // consume it
  return true;
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

