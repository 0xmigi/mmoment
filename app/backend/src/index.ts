// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { config } from 'dotenv';

// Load environment variables
config();

const app = express();

// Create HTTP server with extremely lenient timeout settings
const httpServer = createServer(app);
httpServer.keepAliveTimeout = 300000; // 5 minutes
httpServer.headersTimeout = 301000; // Slightly higher than keepAliveTimeout

// Basic middleware with increased limits
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Super permissive CORS configuration
app.use((req, res, next) => {
  // Allow any origin
  const origin = req.headers.origin || '*';
  res.setHeader('Access-Control-Allow-Origin', origin);
  
  // Allow all methods and headers
  res.setHeader('Access-Control-Allow-Methods', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Max-Age', '86400');
  
  // Add cache control headers
  res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  next();
});

// Use cors middleware with permissive settings
app.use(cors({
  origin: true,
  methods: '*',
  allowedHeaders: '*',
  credentials: true,
  maxAge: 86400,
  preflightContinue: true
}));

// Configure Socket.IO with extremely permissive settings
const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
    credentials: true,
    allowedHeaders: '*'
  },
  path: '/socket.io/',
  transports: ['polling', 'websocket'], // Try polling first
  pingTimeout: 300000, // 5 minutes
  pingInterval: 25000,
  connectTimeout: 300000, // 5 minutes
  upgradeTimeout: 300000, // 5 minutes
  maxHttpBufferSize: 1e8,
  allowUpgrades: true,
  perMessageDeflate: false,
  destroyUpgrade: false
});

// Trust proxy and handle HTTPS
app.enable('trust proxy');

// Add response timeout middleware
app.use((req, res, next) => {
  res.setTimeout(300000, () => { // 5 minutes
    res.status(504).json({ error: 'Server timeout' });
  });
  next();
});

// Debug logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const requestId = Math.random().toString(36).substring(7);
  
  console.log(`[${requestId}] ${new Date().toISOString()} - Incoming ${req.method} ${req.url}`);
  console.log(`[${requestId}] Headers:`, req.headers);
  
  // Log response
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${requestId}] ${new Date().toISOString()} - ${req.method} ${req.url} - ${res.statusCode} - ${duration}ms`);
  });
  
  next();
});

// Add debug endpoints
app.get('/debug/headers', (req, res) => {
  res.json({
    headers: req.headers,
    ip: req.ip,
    ips: req.ips,
    secure: req.secure,
    protocol: req.protocol
  });
});

app.get('/debug/ping', (req, res) => {
  res.json({ pong: Date.now() });
});

// Health check endpoint with connection test
app.get('/health', async (req, res) => {
  try {
    const status = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV,
      headers: req.headers,
      secure: req.secure,
      protocol: req.protocol,
      host: req.headers.host,
      origin: req.headers.origin,
      userAgent: req.headers['user-agent'],
      socketConnections: io.engine.clientsCount,
      memoryUsage: process.memoryUsage(),
      uptime: process.uptime()
    };

    res.json(status);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
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
    // Generate unique ID based on timestamp and random string
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
  console.log('Server configuration:', {
    keepAliveTimeout: httpServer.keepAliveTimeout,
    headersTimeout: httpServer.headersTimeout,
    socketTransports: io.engine.opts.transports,
    socketPingTimeout: io.engine.opts.pingTimeout,
    socketPingInterval: io.engine.opts.pingInterval
  });
});