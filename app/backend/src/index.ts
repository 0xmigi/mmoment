// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { config } from 'dotenv';
import * as path from 'path';
import mediaRoutes from './routes/media.routes';

// Load environment variables from .env file
config({ path: path.resolve(__dirname, '../.env') });

// Log environment variables (without secrets)
console.log('Environment loaded:', {
  PORT: process.env.PORT,
  FILEBASE_BUCKET: process.env.FILEBASE_BUCKET,
  FILEBASE_KEY_EXISTS: !!process.env.FILEBASE_KEY,
  FILEBASE_SECRET_EXISTS: !!process.env.FILEBASE_SECRET
});

const app = express();
const httpServer = createServer(app);

// Add middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Update CORS configuration to be more permissive for development
app.use(cors({
  origin: '*', // Be more permissive with CORS during development
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Add basic health check endpoint
app.get('/', (req, res) => {
  res.json({ status: 'healthy' });
});

// Mount media routes under /api prefix
app.use('/api', mediaRoutes);

const io = new Server(httpServer, {
  cors: {
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    credentials: true,
    allowedHeaders: ["Content-Type", "Authorization"]
  },
  allowEIO3: true,  // Added for compatibility
  path: '/socket.io/' // Make sure path is explicit
});

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