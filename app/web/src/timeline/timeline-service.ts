// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../timeline';
import { CONFIG, timelineConfig } from '../core/config';

// Key for local storage (camera ID only - for reconnection to the same room)
const TIMELINE_CAMERA_ID_KEY = 'timeline_camera_id';

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
    // Restore camera ID from localStorage so we can rejoin the same room on refresh
    const storedCameraId = localStorage.getItem(TIMELINE_CAMERA_ID_KEY);
    if (storedCameraId) {
      this.currentCameraId = storedCameraId;
      console.log(`[Timeline] Will rejoin camera room: ${storedCameraId}`);
    }

    // Events will be fetched from Backend (source of truth) when we join the camera room
    // No need to restore from localStorage - Backend has the authoritative data
    console.log('[Timeline] Events will be fetched from Backend on connection');

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

          this.notifyListeners(event);
        }
      });

      // Process recent events from backend (source of truth)
      // Show ALL events at this camera (not filtered by session - camera timeline shows all activity)
      this.socket.on('recentEvents', (events: TimelineEvent[]) => {
        if (!events || events.length === 0) {
          console.log('[Timeline] No recent events from backend');
          return;
        }

        console.log(`[Timeline] Received ${events.length} events from backend (source of truth)`);

        // Sort by timestamp (newest first) - no session filtering for camera timeline
        const sortedEvents = [...events].sort((a, b) => b.timestamp - a.timestamp);
        console.log(`[Timeline] Processing ${sortedEvents.length} events for camera timeline`);

        // Merge with existing events, avoiding duplicates
        sortedEvents.forEach(event => {
          const existingIndex = this.events.findIndex(e => e.id === event.id);
          if (existingIndex === -1) {
            // Insert in chronological order (newest first)
            const index = this.events.findIndex(e => e.timestamp < event.timestamp);
            if (index === -1) {
              this.events.push(event);
            } else {
              this.events.splice(index, 0, event);
            }
          }
        });

        // Keep only the last 100 events
        this.events = this.events.slice(0, 100);

        // Notify listeners of updated events
        this.events.forEach(event => this.notifyListeners(event));
      });

      // Handle timeline event updates (e.g., transaction ID added after cron job writes to chain)
      this.socket.on('timelineEventUpdate', (update: { id: string; transactionId?: string }) => {
        console.log('[Timeline] Received event update:', update);
        const existingEvent = this.events.find(e => e.id === update.id);
        if (existingEvent && update.transactionId) {
          existingEvent.transactionId = update.transactionId;
          console.log(`[Timeline] Updated event ${update.id} with transaction ID: ${update.transactionId.slice(0, 8)}...`);
          // Notify listeners of the updated event
          this.notifyListeners(existingEvent);
        }
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



  joinCamera(cameraId: string) {
    console.log('[Timeline] Joining camera room:', cameraId);

    // Leave current camera room if different
    if (this.currentCameraId && this.currentCameraId !== cameraId && this.isConnected) {
      this.socket.emit('leaveCamera', this.currentCameraId);
      // Clear events when switching cameras
      this.events = [];
    }

    // Update camera ID and persist for reconnection
    this.currentCameraId = cameraId;
    localStorage.setItem(TIMELINE_CAMERA_ID_KEY, cameraId);

    // Join the room - Backend will send recentEvents which we filter by session
    if (this.isConnected) {
      this.socket.emit('joinCamera', cameraId);
      console.log('[Timeline] Joined room, waiting for recentEvents from Backend');
    } else {
      console.log('[Timeline] Not connected yet, will join room when connected');
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
      console.warn('Cannot emit event: socket not connected, showing locally for immediate feedback');

      // Show event locally for immediate UI feedback (not persisted - Backend is source of truth)
      const tempEvent = {
        ...eventWithCamera,
        id: `local-${Date.now()}-${Math.random().toString(36).slice(2)}`
      };

      this.events.unshift(tempEvent);
      this.notifyListeners(tempEvent);
    }
  }

  disconnect() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.currentCameraId) {
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

  // Clear local events when starting a new session
  // Backend is source of truth - events will be fetched fresh
  clearForNewSession() {
    console.log('[Timeline] Clearing local events for new session');
    this.events = [];
  }

  // Clear local events when ending session
  endSession() {
    console.log('[Timeline] Ending session, clearing local events');
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