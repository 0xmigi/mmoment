// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../timeline';
import { CONFIG, timelineConfig } from '../core/config';

// Key for local storage to persist events
const TIMELINE_EVENTS_STORAGE_KEY = 'timeline_events';
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
    // Try to restore events from local storage
    this.restoreFromLocalStorage();

    // Initialize the socket connection
    this.initializeSocket();
    
    // Setup heartbeat for Chrome mobile
    if (this.isChromeOnMobile) {
      this.setupHeartbeat();
    }
  }
  
  private setupHeartbeat() {
    // Clear any existing interval
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    // Set up heartbeat to periodically check connection and refresh in Chrome mobile
    this.heartbeatInterval = setInterval(() => {
      if (document.visibilityState === 'visible' && this.currentCameraId) {
        if (this.isConnected) {
          // Request recent events to force synchronization in Chrome mobile
          this.requestRecentEvents();
        } else if (this.forceFallback) {
          // If in fallback mode, just check if we can reconnect
          this.tryReconnect();
        }
      }
    }, 10000); // Every 10 seconds
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
          
          // Check if we have any pending local events to sync
          const pendingEvents = this.events.filter(e => e.id.startsWith('local-'));
          if (pendingEvents.length > 0) {
            console.log(`[Timeline] Syncing ${pendingEvents.length} pending local events`);
            // We'll just request recent events which should include server-generated IDs
            this.requestRecentEvents();
          }
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

      this.socket.on('recentEvents', (events: TimelineEvent[]) => {
        console.log('Received recent events:', events);
        if (!events || events.length === 0) return;
        
        // Filter for current camera and sort by timestamp (newest first)
        const relevantEvents = this.currentCameraId
          ? events.filter(event => event.cameraId === this.currentCameraId)
          : events;
        
        if (relevantEvents.length === 0) return;

        // For Chrome mobile, we want to be more careful with event merging
        // to ensure we don't miss any events
        const mergedEvents = this.mergeEvents(this.events, relevantEvents);
        
        // Check if we have new events
        const hasNewEvents = mergedEvents.length !== this.events.length ||
          mergedEvents.some(e1 => !this.events.some(e2 => e2.id === e1.id));
          
        // Update events if we have new ones
        if (hasNewEvents) {
          this.events = mergedEvents;
          
          // Save to localStorage
          this.saveToLocalStorage();
          
          // Notify listeners of all new events
          if (this.isChromeOnMobile) {
            // For Chrome mobile, notify about all events to ensure consistency
            this.events.forEach(event => this.notifyListeners(event));
          } else {
            // For other browsers, only notify about events that the client doesn't already have
            relevantEvents
              .filter(event => !this.events.some(e => e.id === event.id && e !== event))
              .forEach(event => this.notifyListeners(event));
          }
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

  // Method to merge events, avoiding duplicates by ID
  private mergeEvents(existingEvents: TimelineEvent[], newEvents: TimelineEvent[]): TimelineEvent[] {
    const eventMap = new Map<string, TimelineEvent>();
    
    // Add existing events to the map
    existingEvents.forEach(event => {
      eventMap.set(event.id, event);
    });
    
    // Add or update with new events
    newEvents.forEach(event => {
      eventMap.set(event.id, event);
    });
    
    // Convert map back to array and sort by timestamp (newest first)
    return Array.from(eventMap.values())
      .sort((a, b) => b.timestamp - a.timestamp);
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

  // Method to restore events from localStorage
  private restoreFromLocalStorage() {
    try {
      // Try to restore camera ID first
      const savedCameraId = localStorage.getItem(TIMELINE_CAMERA_ID_KEY);
      if (savedCameraId) {
        this.currentCameraId = savedCameraId;
        console.log(`Restored camera ID from localStorage: ${savedCameraId}`);
        
        // Now try to restore events for this camera
        const savedEventsString = localStorage.getItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${savedCameraId}`);
        if (savedEventsString) {
          const savedEvents = JSON.parse(savedEventsString) as TimelineEvent[];
          this.events = savedEvents;
          console.log(`Restored ${savedEvents.length} events from localStorage for camera ${savedCameraId}`);
        }
      }
    } catch (error) {
      console.error('Error restoring timeline events from localStorage:', error);
    }
  }

  joinCamera(cameraId: string) {
    if (this.currentCameraId === cameraId) return;

    console.log('Joining camera room:', cameraId);
    
    // Leave current camera room if any
    if (this.currentCameraId && this.isConnected) {
      this.socket.emit('leaveCamera', this.currentCameraId);
    }

    // Join new camera room
    this.currentCameraId = cameraId;
    localStorage.setItem(TIMELINE_CAMERA_ID_KEY, cameraId);
    
    // Try to restore events for this camera from localStorage
    try {
      const savedEventsString = localStorage.getItem(`${TIMELINE_EVENTS_STORAGE_KEY}_${cameraId}`);
      if (savedEventsString) {
        const savedEvents = JSON.parse(savedEventsString) as TimelineEvent[];
        console.log(`Restored ${savedEvents.length} events from localStorage for camera ${cameraId}`);
        
        // If we have saved events, use them and notify listeners
        if (savedEvents.length > 0) {
          this.events = savedEvents;
          setTimeout(() => {
            this.events.forEach(event => this.notifyListeners(event));
          }, 0);
        }
      } else {
        // Clear events if we don't have saved events for this camera
        this.events = [];
      }
    } catch (error) {
      console.error('Error restoring events for camera:', error);
      this.events = [];
    }
    
    // If we're connected, join the room
    if (this.isConnected) {
      this.socket.emit('joinCamera', cameraId);
      // Request recent events for this camera
      this.requestRecentEvents();
    }
  }

  private requestRecentEvents() {
    // Only request if we're connected
    if (this.isSocketConnected() && this.currentCameraId) {
      console.log('Requesting recent events for camera:', this.currentCameraId);
      this.socket.emit('getRecentEvents', { cameraId: this.currentCameraId });
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
    
    // Immediately send existing events to the new listener
    if (this.events.length > 0) {
      setTimeout(() => {
        this.events.forEach(event => {
          try {
            callback(event);
          } catch (error) {
            console.error('Error sending existing events to new listener:', error);
          }
        });
      }, 0);
    }
    
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
      'video_recorded', 'stream_started', 'stream_ended', 'check_in', 'check_out'];
      
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
      this.socket.emit('newTimelineEvent', eventWithCamera);
      
      // Request recent events after emitting to ensure sync
      setTimeout(() => {
        this.requestRecentEvents();
      }, 1000);
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
  
  // Public method that can be called to force refresh for Chrome mobile
  refreshEvents() {
    if (this.isSocketConnected() && this.currentCameraId) {
      this.requestRecentEvents();
    }
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