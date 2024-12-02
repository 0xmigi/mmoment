// app/web/src/services/timeline-service.ts
import { io, Socket } from 'socket.io-client';
import { TimelineEvent } from '../types/timeline';

class TimelineService {
  private socket: Socket;
  private listeners: Set<(event: TimelineEvent) => void> = new Set();
  private events: TimelineEvent[] = [];  // Store events locally
  private isConnected: boolean = false;

  constructor() {
    this.socket = io('http://localhost:3001', {
      reconnectionDelay: 1000,
      reconnection: true,
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      this.isConnected = true;
      console.log('Connected to timeline service');
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

  // Add method to get current state
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