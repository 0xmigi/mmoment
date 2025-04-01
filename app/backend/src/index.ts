// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { config } from 'dotenv';

// Load environment variables
config();

const app = express();
const httpServer = createServer(app);

// Add middleware
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  credentials: true
}));

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ status: 'healthy' });
});

// Initialize Socket.IO
const io = new Server(httpServer, {
  cors: {
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    credentials: true
  },
  path: '/socket.io/',
  transports: ['websocket', 'polling'],
  pingTimeout: 90000,
  pingInterval: 25000
});

// Timeline events storage (in-memory)
interface TimelineEvent {
  id: string;
  type: string;
  user: {
    address: string;
    username?: string;
  };
  timestamp: number;
  cameraId?: string;
}

const timelineEvents: TimelineEvent[] = [];
const cameraRooms = new Map<string, Set<string>>();

// Socket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // Handle joining a camera room
  socket.on('joinCamera', (cameraId: string) => {
    console.log(`Socket ${socket.id} joining camera ${cameraId}`);
    
    // Leave any existing camera rooms
    cameraRooms.forEach((sockets, roomId) => {
      if (sockets.has(socket.id)) {
        sockets.delete(socket.id);
        socket.leave(roomId);
      }
    });

    // Join new camera room
    socket.join(cameraId);
    if (!cameraRooms.has(cameraId)) {
      cameraRooms.set(cameraId, new Set());
    }
    cameraRooms.get(cameraId)?.add(socket.id);

    // Send recent events for this camera
    const cameraEvents = timelineEvents
      .filter(event => event.cameraId === cameraId)
      .sort((a, b) => b.timestamp - a.timestamp);
    socket.emit('recentEvents', cameraEvents);
  });

  // Handle leaving a camera room
  socket.on('leaveCamera', (cameraId: string) => {
    console.log(`Socket ${socket.id} leaving camera ${cameraId}`);
    socket.leave(cameraId);
    cameraRooms.get(cameraId)?.delete(socket.id);
  });

  // Handle new timeline events
  socket.on('newTimelineEvent', (event: Omit<TimelineEvent, 'id'>) => {
    const newEvent = {
      ...event,
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: event.timestamp || Date.now()
    };

    // Store the event
    timelineEvents.push(newEvent);
    
    // Keep only last 100 events per camera
    if (event.cameraId) {
      const cameraEvents = timelineEvents.filter(e => e.cameraId === event.cameraId);
      if (cameraEvents.length > 100) {
        const oldestEventIndex = timelineEvents.findIndex(e => e.cameraId === event.cameraId);
        if (oldestEventIndex !== -1) {
          timelineEvents.splice(oldestEventIndex, 1);
        }
      }
      // Broadcast only to sockets in this camera's room
      io.to(event.cameraId).emit('timelineEvent', newEvent);
    } else {
      // If no cameraId, broadcast to all
      io.emit('timelineEvent', newEvent);
    }
  });

  // Handle get recent events request
  socket.on('getRecentEvents', ({ cameraId }: { cameraId?: string }) => {
    console.log(`Socket ${socket.id} requesting recent events${cameraId ? ` for camera ${cameraId}` : ''}`);
    const filteredEvents = cameraId
      ? timelineEvents.filter(event => event.cameraId === cameraId)
      : timelineEvents;
    
    const sortedEvents = [...filteredEvents].sort((a, b) => b.timestamp - a.timestamp);
    socket.emit('recentEvents', sortedEvents);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
    // Clean up camera room memberships
    cameraRooms.forEach((sockets, roomId) => {
      if (sockets.has(socket.id)) {
        sockets.delete(socket.id);
        if (sockets.size === 0) {
          cameraRooms.delete(roomId);
        }
      }
    });
  });
});

const PORT = process.env.PORT || 3001;

httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});