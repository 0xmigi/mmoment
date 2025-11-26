// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../timeline';
import { CONFIG, timelineConfig } from '../core/config';

// Key for local storage to persist events
const TIMELINE_EVENTS_STORAGE_KEY = 'timeline_events';
const TIMELINE_CAMERA_ID_KEY = 'timeline_camera_id';
const TIMELINE_SESSION_START_KEY = 'timeline_session_start'; // Timestamp when current session started

// Helper to detect Chrome on mobile
const isChromeOnMobile = () => {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua) && 
         /Chrome\/[.0-9]*/.test(ua) && 
         !/Edge|EdgA|Edg\/|OPR\/|Firefox|Safari/i.test(ua);
};

// Helper to check if backend server is running

class TimelineService {
  private socket!: Socket;  // Use definite assignment assertion
  private listeners: Set<(event: TimelineEvent) => void> = new Set();
  private events: TimelineEvent[] = [];
  private isConnected: boolean = false;
  private currentCameraId: string | null = null;
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private readonly isChromeOnMobile = isChromeOnMobile();
  private heartbeatInterval: any = null;
  private connectionErrorCount: number = 0;
  private forceFallback: boolean = false; // Flag to use local storage only when socket consistently fails

  constructor() {
    // Restore camera ID from localStorage so we can detect same-camera refresh
    const storedCameraId = localStorage.getItem(TIMELINE_CAMERA_ID_KEY);
    const sessionStartStr = localStorage.getItem(TIMELINE_SESSION_START_KEY);
    const sessionStart = sessionStartStr ? parseInt(sessionStartStr, 10) : null;

    if (storedCameraId && sessionStart) {
      this.currentCameraId = storedCameraId;
      // Restore events for this camera, but ONLY events from current session
      const storedEvents = localStorage.getItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${storedCameraId}`);
      if (storedEvents) {
        try {
          const allEvents: TimelineEvent[] = JSON.parse(storedEvents);
          // Filter to only events from AFTER the session started (prevents ghost checkouts)
          this.events = allEvents.filter(e => e.timestamp >= sessionStart);
          console.log(`[Timeline] Restored ${this.events.length}/${allEvents.length} events for current session (started ${new Date(sessionStart).toLocaleTimeString()})`);
        } catch (e) {
          console.error('[Timeline] Failed to parse stored events:', e);
          this.events = [];
        }
      }
    } else {
      // No active session - don't restore any events
      console.log('[Timeline] No active session found, starting fresh');
    }

    // Initialize the socket connection
    this.initializeSocket();

    // Setup heartbeat for Chrome mobile (for connection health only)
    if (this.isChromeOnMobile) {
      this.setupHeartbeat();
    }
  }
  
  private setupHeartbeat() {
    // Clear any existing interval
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    // Set up heartbeat to check connection health in Chrome mobile
    // Don't request recent events - they include old sessions from backend memory
    this.heartbeatInterval = setInterval(() => {
      if (document.visibilityState === 'visible' && this.currentCameraId) {
        if (!this.isConnected || this.forceFallback) {
          // If disconnected or in fallback mode, try to reconnect
          this.tryReconnect();
        }
        // Don't request recent events - live timeline should only show current session
      }
    }, 30000); // Every 30 seconds
  }

  private tryReconnect() {
    if (!this.isConnected && this.socket) {
      console.log('[Timeline] Attempting to reconnect from fallback mode');
      this.socket.connect();
    }
  }

  private initializeSocket() {
    // Always use the timeline WebSocket URL (Railway)
    const timelineWsUrl = CONFIG.TIMELINE_WS_URL;
    
    console.log(`[Timeline] Connecting to timeline service at: ${timelineWsUrl}`);
    
    const socketOptions = {
      ...timelineConfig.wsOptions,
      autoConnect: true,
      // For local development, don't fail on CORS issues
      withCredentials: false,
      // Increase timeout for better reliability
      timeout: 15000,
      // Add reconnection settings
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000
    };
    
    // Chrome mobile performs better with polling first
    if (this.isChromeOnMobile) {
      socketOptions.transports = ['polling', 'websocket'];
    }
    
    try {
      this.socket = io(timelineWsUrl, socketOptions);

      this.socket.on('connect', () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.connectionErrorCount = 0;
        this.forceFallback = false;
        console.log('[Timeline] Connected to timeline service');

        if (this.currentCameraId) {
          this.joinCamera(this.currentCameraId);
          // Don't request recent events - they include old sessions
          // Live timeline should only show events from current session going forward
        }
      });

      this.socket.on('connect_error', (error) => {
        console.error('[Timeline] Connection error:', error);
        this.isConnected = false;
        this.reconnectAttempts++;
        this.connectionErrorCount++;
        
        // If we consistently get errors, enable fallback mode
        if (this.connectionErrorCount > 3) {
          console.log('[Timeline] Too many connection errors, enabling fallback mode');
          this.forceFallback = true;
        }
        
        if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
          console.error('[Timeline] Max reconnection attempts reached');
          if (this.isChromeOnMobile) {
            // Don't disconnect in Chrome mobile, just wait for heartbeat
            console.log('[Timeline] Chrome mobile: will retry reconnection later');
          } else {
            this.socket.disconnect();
          }
        }
      });

      this.socket.on('timelineEvent', (event: TimelineEvent) => {
        console.log('Received timeline event:', event);
        // Process all events, including our own
        if (!this.currentCameraId || event.cameraId === this.currentCameraId) {
          // Check if we already have this event to avoid duplicates in Chrome mobile
          const existingIndex = this.events.findIndex(e => e.id === event.id);
          if (existingIndex !== -1) {
            return; // Skip duplicates
          }
          
          // Insert the event in chronological order (newest first)
          const index = this.events.findIndex(e => e.timestamp < event.timestamp);
          if (index === -1) {
            this.events.push(event);
          } else {
            this.events.splice(index, 0, event);
          }
          // Keep only the last 100 events
          this.events = this.events.slice(0, 100);
          
          // Save to localStorage
          this.saveToLocalStorage();
          
          this.notifyListeners(event);
        }
      });

      // DISABLED: Don't process bulk historical events from backend memory
      // This fixes ghost checkouts from old sessions appearing in live timeline
      // Live timeline only shows real-time events via 'timelineEvent' subscription
      this.socket.on('recentEvents', (events: TimelineEvent[]) => {
        console.log('[Timeline] Ignoring recentEvents - live mode only (received', events?.length || 0, 'events)');
        // Do nothing - we only want real-time events from current session
      });

      this.socket.on('disconnect', () => {
        this.isConnected = false;
        console.log('Disconnected from timeline service');
        
        // For Chrome mobile, try to reconnect when disconnected if page is visible
        if (this.isChromeOnMobile && document.visibilityState === 'visible') {
          setTimeout(() => {
            console.log('[Timeline] Chrome mobile: attempting reconnect after disconnect');
            this.socket.connect();
          }, 3000);
        }
      });

      this.socket.on('error', (error) => {
        console.error('[Timeline] Socket error:', error);
        this.connectionErrorCount++;
        
        // If we consistently get errors, enable fallback mode
        if (this.connectionErrorCount > 3) {
          console.log('[Timeline] Too many socket errors, enabling fallback mode');
          this.forceFallback = true;
        }
      });
    } catch (error) {
      console.error('[Timeline] Error initializing socket:', error);
      this.forceFallback = true;
    }

    // Attempt initial connection
    if (this.socket && !this.socket.connected) {
      this.socket.connect();
    }
  }


  // Method to save events to localStorage
  private saveToLocalStorage() {
    try {
      if (this.currentCameraId) {
        localStorage.setItem(TIMELINE_CAMERA_ID_KEY, this.currentCameraId);
        localStorage.setItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${this.currentCameraId}`, 
          JSON.stringify(this.events));
        console.log(`Saved ${this.events.length} events to localStorage for camera ${this.currentCameraId}`);
      }
    } catch (error) {
      console.error('Error saving timeline events to localStorage:', error);
    }
  }

  joinCamera(cameraId: string) {
    console.log('Joining camera room:', cameraId);

    // If we're already in this camera, just rejoin the socket room to ensure we're connected
    // Keep existing events (for when navigating away and back within current session)
    if (this.currentCameraId === cameraId) {
      console.log('Already in this camera room, keeping current session events');
      if (this.isConnected) {
        this.socket.emit('joinCamera', cameraId);
        // Don't request recent events - they include old sessions from backend memory
        // Current session events are already in this.events
      }
      return;
    }

    // Leave current camera room if any
    if (this.currentCameraId && this.isConnected) {
      this.socket.emit('leaveCamera', this.currentCameraId);
    }

    // Join new camera room - start fresh (no old session events)
    // The live timeline should only show current session, not historical events
    // Historical events belong in Activities view
    this.currentCameraId = cameraId;
    localStorage.setItem(TIMELINE_CAMERA_ID_KEY, cameraId);

    // Clear events - start fresh for new camera/session
    // Don't restore from localStorage as those are from old sessions
    this.events = [];
    console.log('Starting fresh timeline for camera:', cameraId);

    // If we're connected, join the room
    if (this.isConnected) {
      this.socket.emit('joinCamera', cameraId);
      // Don't request recent events from backend - they include old sessions
      // We only want real-time events from the current session going forward
    }
  }

  getState() {
    return {
      events: this.events,
      currentCameraId: this.currentCameraId
    };
  }

  private notifyListeners(event: TimelineEvent) {
    this.listeners.forEach(listener => {
      try {
        listener(event);
      } catch (error) {
        console.error('Error in timeline listener:', error);
      }
    });
  }

  subscribe(callback: (event: TimelineEvent) => void) {
    this.listeners.add(callback);

    // Note: We do NOT send existing events to new listeners anymore.
    // This was causing phantom timeline events (100+ duplicate notifications).
    // Components should use getEvents() to fetch initial data, then
    // subscribe() for LIVE updates only. Events arrive via Socket.IO.

    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  emitEvent(event: Omit<TimelineEvent, 'id'>) {
    if (!this.currentCameraId) {
      console.error('Cannot emit event: no camera selected');
      return;
    }

    // Update to handle new event types without type errors
    const validEventTypes = ['initialization', 'user_connected', 'photo_captured',
      'video_recorded', 'stream_started', 'stream_ended', 'check_in', 'check_out', 'auto_check_out', 'face_enrolled'];

    if (!validEventTypes.includes(event.type)) {
      console.warn(`Unknown event type: ${event.type}, treating as initialization`);
      event = { ...event, type: 'initialization' as any };
    }

    // Ensure the event has the current camera ID and timestamp
    const eventWithCamera = {
      ...event,
      cameraId: this.currentCameraId,
      timestamp: Date.now()
    };

    // If connected, emit to server
    if (this.isSocketConnected()) {
      console.log('Emitting timeline event:', eventWithCamera);
      console.log('[Timeline] Socket state:', {
        connected: this.socket.connected,
        id: this.socket.id,
        transport: this.socket.io.engine?.transport?.name
      });
      this.socket.emit('newTimelineEvent', eventWithCamera);
      console.log('[Timeline] Event emitted to socket');
      
      // Don't automatically request recent events after emitting
      // The server will broadcast the event back to us, which is sufficient
      // Automatic refresh was causing potential duplicates
    } else {
      console.warn('Cannot emit event: socket not connected, storing locally');
      
      // Store the event locally so it's not lost
      const tempEvent = {
        ...eventWithCamera,
        id: `local-${Date.now()}-${Math.random().toString(36).slice(2)}`
      };
      
      // Add to local events
      this.events.unshift(tempEvent);
      this.saveToLocalStorage();
      
      // Notify listeners about the local event
      this.notifyListeners(tempEvent);
    }
  }

  disconnect() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    
    if (this.currentCameraId) {
      // Save events before disconnecting
      this.saveToLocalStorage();
      
      if (this.isSocketConnected()) {
        this.socket.emit('leaveCamera', this.currentCameraId);
      }
      this.currentCameraId = null;
    }
    this.events = [];
    this.listeners.clear();
    if (this.socket) {
      this.socket.disconnect();
      this.isConnected = false;
      console.log('[Timeline] Disconnected from timeline service');
    }
  }
  
  // Explicitly connect to the timeline service
  connect() {
    if (!this.isSocketConnected() && this.socket) {
      console.log('[Timeline] Manually reconnecting to timeline service');
      this.socket.connect();
    }
    return this.isSocketConnected();
  }
  
  // Public method - disabled to prevent loading old session events
  // Live timeline should only show real-time events from current session
  refreshEvents() {
    // Don't request recent events - they include old sessions from backend memory
    // Current session events come through real-time WebSocket subscription
    console.log('[Timeline] refreshEvents disabled - live timeline only shows current session');
  }

  // Clear all events and start a fresh session
  // Call this on check-in to remove any old session events
  clearForNewSession() {
    const now = Date.now();
    console.log('[Timeline] Starting new session at', new Date(now).toLocaleTimeString());
    this.events = [];
    // Set session start timestamp - only events after this will be restored on refresh
    localStorage.setItem(TIMELINE_SESSION_START_KEY, now.toString());
    if (this.currentCameraId) {
      localStorage.removeItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${this.currentCameraId}`);
    }
  }

  // End session - clears session marker so old events won't be restored
  // Call this on checkout
  endSession() {
    console.log('[Timeline] Ending session');
    localStorage.removeItem(TIMELINE_SESSION_START_KEY);
    if (this.currentCameraId) {
      localStorage.removeItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${this.currentCameraId}`);
    }
    this.events = [];
  }

  // For debugging: create a test event
  triggerTestEvent(walletAddress: string) {
    if (!this.currentCameraId) {
      console.error('Cannot create test event: no camera selected');
      return;
    }
    
    const testEvent: Omit<TimelineEvent, 'id'> = {
      type: 'photo_captured',
      user: {
        address: walletAddress || '0xtest',
        username: 'Test User'
      },
      timestamp: Date.now(),
      cameraId: this.currentCameraId
    };
    
    console.log('Creating test timeline event:', testEvent);
    this.emitEvent(testEvent);
    return true;
  }

  // New helper method to check connection status
  isSocketConnected(): boolean {
    return this.isConnected && this.socket && this.socket.connected;
  }
}

export const timelineService = new TimelineService();