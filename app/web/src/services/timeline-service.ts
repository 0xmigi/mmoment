// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../types/timeline';
import { CONFIG, timelineConfig } from '../config';

// Key for local storage to persist events
const TIMELINE_EVENTS_STORAGE_KEY = 'timeline_events';
const TIMELINE_CAMERA_ID_KEY = 'timeline_camera_id';

class TimelineService {
  private socket!: Socket;  // Use definite assignment assertion
  private listeners: Set<(event: TimelineEvent) => void> = new Set();
  private events: TimelineEvent[] = [];
  private isConnected: boolean = false;
  private currentCameraId: string | null = null;
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private useFallbackHttp: boolean = false;
  private isMobile: boolean = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
  private lastHttpFetchTime: number = 0;
  private httpFetchInProgress: boolean = false;
  private readonly HTTP_FETCH_INTERVAL = 5000; // 5 seconds

  constructor() {
    // Try to restore events from local storage
    this.restoreFromLocalStorage();

    // Initialize socket with the timeline WebSocket URL
    this.initializeSocket();

    // Check if we should use HTTP fallback immediately (for mobile)
    if (this.isMobile) {
      this.useFallbackHttp = true;
      console.log('[Timeline] Mobile device detected, enabling HTTP fallback');
    }
  }

  private initializeSocket() {
    // Always use the timeline WebSocket URL (Railway)
    const timelineWsUrl = CONFIG.TIMELINE_WS_URL;
    
    console.log(`[Timeline] Connecting to timeline service at: ${timelineWsUrl}`);
    
    this.socket = io(timelineWsUrl, {
      ...timelineConfig.wsOptions,
      autoConnect: true
    });

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log('[Timeline] Connected to timeline service');
      
      if (this.currentCameraId) {
        this.joinCamera(this.currentCameraId);
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('[Timeline] Connection error:', error);
      this.isConnected = false;
      this.reconnectAttempts++;
      
      // Enable HTTP fallback on connection errors
      this.useFallbackHttp = true;
      
      if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
        console.error('[Timeline] Max reconnection attempts reached');
        this.socket.disconnect();
      }
    });

    this.socket.on('timelineEvent', (event: TimelineEvent) => {
      console.log('Received timeline event:', event);
      // Process all events, including our own
      if (!this.currentCameraId || event.cameraId === this.currentCameraId) {
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
      // Filter for current camera and sort by timestamp (newest first)
      const relevantEvents = this.currentCameraId
        ? events.filter(event => event.cameraId === this.currentCameraId)
        : events;

      // Merge with existing events, avoiding duplicates
      const mergedEvents = this.mergeEvents(this.events, relevantEvents);
      this.events = mergedEvents.sort((a, b) => b.timestamp - a.timestamp);
      
      // Save to localStorage
      this.saveToLocalStorage();
      
      // Notify listeners of all events in chronological order
      this.events.forEach(event => this.notifyListeners(event));
    });

    this.socket.on('disconnect', () => {
      this.isConnected = false;
      console.log('Disconnected from timeline service');
      
      // Enable HTTP fallback when disconnected
      this.useFallbackHttp = true;
    });

    // Attempt initial connection
    if (!this.socket.connected) {
      this.socket.connect();
    }
  }

  // HTTP fallback method to fetch events
  private async fetchEventsByHttp(): Promise<void> {
    if (this.httpFetchInProgress) return;
    
    const now = Date.now();
    if (now - this.lastHttpFetchTime < this.HTTP_FETCH_INTERVAL) return;
    
    this.lastHttpFetchTime = now;
    this.httpFetchInProgress = true;
    
    try {
      console.log('[Timeline] Using HTTP fallback to fetch events');
      
      // Get events for a specific camera
      let endpoint = `${CONFIG.BACKEND_URL}/api/timeline/events`;
      
      // Add camera filter if we have a camera ID
      if (this.currentCameraId) {
        endpoint += `?cameraId=${this.currentCameraId}`;
      }

      const response = await fetch(endpoint, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });

      if (response.ok) {
        const events = await response.json() as TimelineEvent[];
        console.log(`[Timeline] HTTP fallback received ${events.length} events`);
        
        // Merge with existing events
        const mergedEvents = this.mergeEvents(this.events, events);
        this.events = mergedEvents.sort((a, b) => b.timestamp - a.timestamp);
        
        // Save to localStorage
        this.saveToLocalStorage();
        
        // Notify listeners of all events
        events.forEach(event => this.notifyListeners(event));
      } else {
        console.error('[Timeline] HTTP fallback failed:', response.status);
      }
    } catch (error) {
      console.error('[Timeline] HTTP fallback error:', error);
    } finally {
      this.httpFetchInProgress = false;
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
    
    // Convert map back to array
    return Array.from(eventMap.values());
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
    if (this.currentCameraId === cameraId) {
      // Even if it's the same camera, attempt HTTP fetch on mobile
      if (this.isMobile && this.useFallbackHttp) {
        this.fetchEventsByHttp();
      }
      return;
    }

    console.log('Joining camera room:', cameraId);
    
    // Leave current camera room if any
    if (this.currentCameraId) {
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
    
    // Attempt to use both WebSocket and HTTP methods for mobile
    this.socket.emit('joinCamera', cameraId);
    this.requestRecentEvents();
    
    // Use HTTP fallback immediately on mobile
    if (this.isMobile) {
      setTimeout(() => {
        this.fetchEventsByHttp();
      }, 500);
    }
  }

  private requestRecentEvents() {
    console.log('Requesting recent events for camera:', this.currentCameraId);
    this.socket.emit('getRecentEvents', { cameraId: this.currentCameraId });
  }

  getState() {
    return {
      events: this.events,
      isConnected: this.isConnected,
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
    
    // If on mobile, immediately try HTTP fallback
    if (this.isMobile && this.currentCameraId) {
      setTimeout(() => {
        this.fetchEventsByHttp();
      }, 300);
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

    // Ensure the event has the current camera ID and timestamp
    const eventWithCamera = {
      ...event,
      cameraId: this.currentCameraId,
      timestamp: Date.now()
    };

    console.log('Emitting timeline event:', eventWithCamera);
    
    // Emit the event
    this.socket.emit('newTimelineEvent', eventWithCamera);
    
    // For mobile, also send via HTTP if WebSocket is not working
    if (this.isMobile && this.useFallbackHttp) {
      this.emitEventByHttp(eventWithCamera);
    }
  }

  // HTTP fallback method to emit events
  private async emitEventByHttp(event: Omit<TimelineEvent, 'id'>) {
    try {
      console.log('[Timeline] Using HTTP fallback to emit event');
      
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/timeline/events`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(event)
      });

      if (response.ok) {
        console.log('[Timeline] HTTP event emit successful');
      } else {
        console.error('[Timeline] HTTP event emit failed:', response.status);
      }
    } catch (error) {
      console.error('[Timeline] HTTP event emit error:', error);
    }
  }

  // Method to force refresh events via HTTP
  public forceRefresh() {
    console.log('[Timeline] Force refreshing events');
    if (this.currentCameraId) {
      this.fetchEventsByHttp();
    }
  }

  disconnect() {
    if (this.currentCameraId) {
      // Save events before disconnecting
      this.saveToLocalStorage();
      
      this.socket.emit('leaveCamera', this.currentCameraId);
      this.currentCameraId = null;
    }
    this.events = [];
    this.listeners.clear();
    this.socket.disconnect();
  }
}

export const timelineService = new TimelineService();