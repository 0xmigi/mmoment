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

// Update CORS configuration for Socket.IO
const io = new Server(httpServer, {
  cors: {
    origin: true, // Allows all origins
    credentials: true,
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"]
  }
});

// Update Express CORS configuration
app.use(cors({
  origin: true, // Allows all origins
  credentials: true,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

// Timeline events storage (in-memory for now)
interface TimelineEvent {
  id: string;
  type: string;
  user: {
    address: string;
    username?: string;
  };
  timestamp: number;
  cameraId: string;
}

const timelineEvents: TimelineEvent[] = [];

// Socket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // Send existing events to new clients
  socket.emit('recentEvents', timelineEvents);

  // Handle new timeline events
  socket.on('newTimelineEvent', (event: Omit<TimelineEvent, 'id'>) => {
    const newEvent = {
      ...event,
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };

    timelineEvents.push(newEvent);

    // Keep only last 50 events
    if (timelineEvents.length > 50) {
      timelineEvents.shift();
    }

    // Broadcast to all clients
    io.emit('timelineEvent', newEvent);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

const PORT = process.env.PORT || 3001;

httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});