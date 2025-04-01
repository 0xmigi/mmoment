// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../types/timeline';
import { CONFIG, timelineConfig } from '../config';

// Key for local storage to persist events
const TIMELINE_EVENTS_STORAGE_KEY = 'timeline_events';
const TIMELINE_CAMERA_ID_KEY = 'timeline_camera_id';

class TimelineService {
  private socket: Socket;
  private listeners: Set<(event: TimelineEvent) => void> = new Set();
  private events: TimelineEvent[] = [];
  private isConnected: boolean = false;
  private currentCameraId: string | null = null;
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;

  constructor() {
    // Try to restore events and camera ID from local storage
    this.restoreFromLocalStorage();

    // Use the config WS_URL for WebSocket connection
    this.socket = io(CONFIG.WS_URL, {
      ...timelineConfig.wsOptions,
      autoConnect: true
    });

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log('Connected to timeline service');
      
      // If we have a camera ID, join its room immediately after connection
      if (this.currentCameraId) {
        this.joinCamera(this.currentCameraId);
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('Timeline connection error:', error);
      this.isConnected = false;
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
        console.error('Timeline service max reconnection attempts reached');
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
    });

    // Attempt initial connection
    if (!this.socket.connected) {
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
    if (this.currentCameraId === cameraId) return;

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
    
    this.socket.emit('joinCamera', cameraId);

    // Request recent events for this camera
    this.requestRecentEvents();
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