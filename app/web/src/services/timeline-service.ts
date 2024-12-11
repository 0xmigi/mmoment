// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../types/timeline';
import { CONFIG, timelineConfig } from '../config';

class TimelineService {
  private socket: Socket;
  private listeners: Set<(event: TimelineEvent) => void> = new Set();
  private events: TimelineEvent[] = [];
  private isConnected: boolean = false;

  constructor() {
    // Use the config URL instead of hardcoded localhost
    this.socket = io(CONFIG.BACKEND_URL, timelineConfig.wsOptions);

    this.socket.on('connect', () => {
      this.isConnected = true;
      console.log('Connected to timeline service');
    });

    this.socket.on('connect_error', (error) => {
      console.log('Connection error:', error);
    });

    this.socket.on('timelineEvent', (event: TimelineEvent) => {
      this.events.push(event);
      this.notifyListeners(event);
    });

    this.socket.on('recentEvents', (events: TimelineEvent[]) => {
      this.events = events;
      events.forEach(event => this.notifyListeners(event));
    });

    this.socket.on('disconnect', () => {
      this.isConnected = false;
      console.log('Disconnected from timeline service');
    });
  }

  getState() {
    return {
      events: this.events,
      isConnected: this.isConnected
    };
  }

  private notifyListeners(event: TimelineEvent) {
    this.listeners.forEach(listener => listener(event));
  }

  subscribe(callback: (event: TimelineEvent) => void) {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  emitEvent(event: Omit<TimelineEvent, 'id'>) {
    this.socket.emit('newTimelineEvent', event);
  }

  disconnect() {
    this.socket.disconnect();
  }
}

export const timelineService = new TimelineService();