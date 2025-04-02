// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { config } from 'dotenv';

// Load environment variables
config();

const app = express();

// Create HTTP server with more lenient timeout settings
const httpServer = createServer(app);
httpServer.keepAliveTimeout = 120000; // 2 minutes
httpServer.headersTimeout = 121000; // Slightly higher than keepAliveTimeout

// Basic middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// More permissive CORS configuration for debugging
const corsOptions = {
  origin: true, // Allow all origins temporarily
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['*'], // Allow all headers
  credentials: true,
  maxAge: 86400,
  preflightContinue: true
};

// Add CORS headers middleware
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,HEAD,PUT,PATCH,POST,DELETE');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Max-Age', '86400');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  next();
});

app.use(cors(corsOptions));

// Configure Socket.IO with permissive settings
const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
    credentials: true,
    allowedHeaders: ['*']
  },
  path: '/socket.io/',
  transports: ['websocket', 'polling'],
  pingTimeout: 60000,
  pingInterval: 25000,
  connectTimeout: 60000,
  upgradeTimeout: 30000,
  maxHttpBufferSize: 1e8,
  allowUpgrades: true,
  perMessageDeflate: false,
  destroyUpgrade: false
});

// Trust proxy and handle HTTPS
app.enable('trust proxy');
app.use((req, res, next) => {
  // Increase response timeout
  res.setTimeout(60000, () => {
    res.status(504).json({ error: 'Server timeout' });
  });

  // Only force HTTPS in production and not for health checks
  if (process.env.NODE_ENV === 'production' && !req.secure && !req.path.includes('/health')) {
    return res.redirect(301, `https://${req.headers.host}${req.url}`);
  }
  next();
});

// Add detailed request logging
app.use((req, res, next) => {
  const start = Date.now();
  console.log(`[${new Date().toISOString()}] Incoming ${req.method} ${req.url}`);
  console.log('Headers:', req.headers);
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url} - ${res.statusCode} - ${duration}ms`);
  });
  next();
});

// Health check endpoint with detailed status
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV,
    headers: req.headers,
    secure: req.secure,
    protocol: req.protocol,
    host: req.headers.host,
    origin: req.headers.origin,
    userAgent: req.headers['user-agent']
  });
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

// Error handling for Socket.IO
io.engine.on('connection_error', (err) => {
  console.error('Socket.IO connection error:', err);
});

// Start server with error handling
const port = process.env.PORT || 3001;
httpServer.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log(`Server timeout settings:`, {
    keepAliveTimeout: httpServer.keepAliveTimeout,
    headersTimeout: httpServer.headersTimeout
  });
});