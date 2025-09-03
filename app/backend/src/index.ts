// backend/src/index.ts
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import { config } from 'dotenv';
import dgram from 'dgram';
import net from 'net';

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

// Device claim storage (in-memory)
interface DeviceClaim {
  userWallet: string;
  created: number;
  expires: number;
  status: 'pending' | 'claimed' | 'expired';
  devicePubkey?: string;
  deviceModel?: string;
}

const timelineEvents: TimelineEvent[] = [];
const cameraRooms = new Map<string, Set<string>>();
const pendingClaims = new Map<string, DeviceClaim>();

// Device claim endpoints for QR-based registration
// 1. Frontend creates a claim token with user wallet and WiFi credentials
app.post('/api/claim/create', (req, res) => {
  try {
    const { userWallet } = req.body;
    
    if (!userWallet) {
      return res.status(400).json({ error: 'userWallet is required' });
    }

    // Generate secure random token
    const crypto = require('crypto');
    const claimToken = crypto.randomBytes(32).toString('hex');
    
    // Create claim record with 10 minute expiry
    const claimData: DeviceClaim = {
      userWallet,
      created: Date.now(),
      expires: Date.now() + (10 * 60 * 1000), // 10 minutes
      status: 'pending'
    };
    
    pendingClaims.set(claimToken, claimData);
    
    console.log(`Created claim token ${claimToken} for wallet ${userWallet}`);
    
    res.json({ 
      claimToken,
      expiresAt: claimData.expires,
      claimEndpoint: `${req.protocol}://${req.get('host')}/api/claim/${claimToken}`
    });
  } catch (error) {
    console.error('Error creating claim token:', error);
    res.status(500).json({ error: 'Failed to create claim token' });
  }
});

// 2. Jetson device claims ownership using the token
app.post('/api/claim/:token', (req, res) => {
  try {
    const { token } = req.params;
    const { device_pubkey, device_model } = req.body;
    
    if (!device_pubkey) {
      return res.status(400).json({ error: 'device_pubkey is required' });
    }
    
    const claim = pendingClaims.get(token);
    
    if (!claim) {
      return res.status(404).json({ error: 'Claim token not found' });
    }
    
    if (claim.expires < Date.now()) {
      claim.status = 'expired';
      return res.status(410).json({ error: 'Claim token has expired' });
    }
    
    if (claim.status !== 'pending') {
      return res.status(409).json({ error: 'Claim token already used' });
    }
    
    // Update claim with device information
    claim.devicePubkey = device_pubkey;
    claim.deviceModel = device_model || 'Unknown Device';
    claim.status = 'claimed';
    
    console.log(`Device ${device_pubkey} claimed token ${token} for wallet ${claim.userWallet}`);
    
    res.json({ 
      success: true,
      userWallet: claim.userWallet,
      message: 'Device successfully claimed'
    });
  } catch (error) {
    console.error('Error processing device claim:', error);
    res.status(500).json({ error: 'Failed to process device claim' });
  }
});

// 3. Frontend polls to check if device has been claimed
app.get('/api/claim/:token/status', (req, res) => {
  try {
    const { token } = req.params;
    const claim = pendingClaims.get(token);
    
    if (!claim) {
      return res.status(404).json({ status: 'not_found' });
    }
    
    // Check if expired
    if (claim.expires < Date.now() && claim.status === 'pending') {
      claim.status = 'expired';
    }
    
    const response = {
      status: claim.status,
      created: claim.created,
      expires: claim.expires,
      devicePubkey: claim.devicePubkey,
      deviceModel: claim.deviceModel
    };
    
    // Clean up expired or claimed tokens after 1 hour
    if ((claim.status === 'claimed' || claim.status === 'expired') && 
        (Date.now() - claim.created) > (60 * 60 * 1000)) {
      pendingClaims.delete(token);
    }
    
    res.json(response);
  } catch (error) {
    console.error('Error checking claim status:', error);
    res.status(500).json({ error: 'Failed to check claim status' });
  }
});

// 4. Notify device of its assigned PDA for Cloudflare tunnel configuration
app.post('/api/claim/:token/assign-pda', (req, res) => {
  try {
    const { token } = req.params;
    const { camera_pda, transaction_id } = req.body;
    
    if (!camera_pda) {
      return res.status(400).json({ error: 'camera_pda is required' });
    }
    
    const claim = pendingClaims.get(token);
    
    if (!claim) {
      return res.status(404).json({ error: 'Claim token not found' });
    }
    
    if (claim.status !== 'claimed') {
      return res.status(400).json({ error: 'Device has not claimed this token yet' });
    }
    
    // Store PDA assignment in claim for device to retrieve
    (claim as any).assignedPda = camera_pda;
    (claim as any).transactionId = transaction_id;
    (claim as any).pdaAssignedAt = Date.now();
    
    console.log(`Assigned PDA ${camera_pda} to device ${claim.devicePubkey} (token: ${token})`);
    
    // In a real implementation, you might want to:
    // 1. Store this in a database
    // 2. Push notify the device via webhook/WebSocket
    // 3. Send to device management service
    
    res.json({ 
      success: true,
      message: 'PDA assigned to device',
      camera_pda,
      subdomain: `${camera_pda.toLowerCase()}.mmoment.xyz`
    });
  } catch (error) {
    console.error('Error assigning PDA to device:', error);
    res.status(500).json({ error: 'Failed to assign PDA to device' });
  }
});

// 5. Device endpoint to retrieve assigned PDA configuration
app.get('/api/device/:device_pubkey/config', (req, res) => {
  try {
    const { device_pubkey } = req.params;
    
    // Find claim by device pubkey
    let foundClaim = null;
    let foundToken = null;
    
    for (const [token, claim] of pendingClaims.entries()) {
      if (claim.devicePubkey === device_pubkey) {
        foundClaim = claim;
        foundToken = token;
        break;
      }
    }
    
    if (!foundClaim) {
      return res.status(404).json({ 
        error: 'No configuration found for this device',
        device_pubkey 
      });
    }
    
    const assignedPda = (foundClaim as any).assignedPda;
    
    if (!assignedPda) {
      return res.status(202).json({ 
        status: 'pending',
        message: 'Device claimed but PDA not yet assigned',
        device_pubkey,
        claimed_at: foundClaim.created
      });
    }
    
    // Return configuration for device to use
    const config = {
      device_pubkey,
      camera_pda: assignedPda,
      subdomain: assignedPda.toLowerCase(),
      full_domain: `${assignedPda.toLowerCase()}.mmoment.xyz`,
      transaction_id: (foundClaim as any).transactionId,
      assigned_at: (foundClaim as any).pdaAssignedAt,
      user_wallet: foundClaim.userWallet,
      api_endpoints: {
        camera_info: `https://${assignedPda.toLowerCase()}.mmoment.xyz/api/camera/info`,
        health: `https://${assignedPda.toLowerCase()}.mmoment.xyz/api/health`,
        status: `https://${assignedPda.toLowerCase()}.mmoment.xyz/api/camera/status`,
        stream: `https://${assignedPda.toLowerCase()}.mmoment.xyz/api/camera/stream`
      }
    };
    
    console.log(`Device ${device_pubkey} retrieved config for PDA ${assignedPda}`);
    
    res.json(config);
  } catch (error) {
    console.error('Error retrieving device config:', error);
    res.status(500).json({ error: 'Failed to retrieve device configuration' });
  }
});

// 6. Cleanup endpoint to remove expired claims (optional, for debugging)
app.post('/api/claim/cleanup', (req, res) => {
  const now = Date.now();
  let cleanedUp = 0;
  
  for (const [token, claim] of pendingClaims.entries()) {
    if (claim.expires < now || (now - claim.created) > (60 * 60 * 1000)) {
      pendingClaims.delete(token);
      cleanedUp++;
    }
  }
  
  res.json({ 
    message: `Cleaned up ${cleanedUp} expired claims`,
    activeClaims: pendingClaims.size 
  });
});

// WebRTC Status endpoint for debugging
app.get('/api/webrtc/status', (req, res) => {
  const peers = Array.from(webrtcPeers.entries()).map(([socketId, peer]) => ({
    socketId,
    ...peer
  }));
  
  const cameras = peers.filter(p => p.type === 'camera');
  const viewers = peers.filter(p => p.type === 'viewer');
  
  res.json({
    totalPeers: peers.length,
    cameras: cameras.length,
    viewers: viewers.length,
    activeCameras: cameras.map(c => c.cameraId),
    peers
  });
});

// WebRTC signaling storage
const webrtcPeers = new Map<string, { cameraId?: string, type: 'camera' | 'viewer' }>();

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

  // WebRTC Signaling Events
  socket.on('register-camera', (data: { cameraId: string }) => {
    console.log(`🎥 Camera ${data.cameraId} registering for WebRTC on socket ${socket.id}`);
    webrtcPeers.set(socket.id, { cameraId: data.cameraId, type: 'camera' });
    socket.join(`webrtc-${data.cameraId}`);
    
    // Debug room state after camera joins
    const roomSockets = io.sockets.adapter.rooms.get(`webrtc-${data.cameraId}`);
    console.log(`🎥 Camera joined room webrtc-${data.cameraId}, now has ${roomSockets ? roomSockets.size : 0} sockets`);
  });

  socket.on('register-viewer', (data: { cameraId: string }) => {
    console.log(`Viewer registering for WebRTC camera ${data.cameraId} on socket ${socket.id}`);
    webrtcPeers.set(socket.id, { cameraId: data.cameraId, type: 'viewer' });
    socket.join(`webrtc-${data.cameraId}`);
    
    // Check how many peers are in the room
    const roomSockets = io.sockets.adapter.rooms.get(`webrtc-${data.cameraId}`);
    console.log(`Room webrtc-${data.cameraId} has ${roomSockets ? roomSockets.size : 0} sockets`);
    
    // Notify camera that a viewer wants to connect
    console.log(`Notifying camera in room webrtc-${data.cameraId} that viewer ${socket.id} wants to connect`);
    socket.to(`webrtc-${data.cameraId}`).emit('viewer-wants-connection', { viewerId: socket.id });
  });

  socket.on('webrtc-offer', (data: { cameraId: string, offer: any, targetId?: string }) => {
    console.log(`WebRTC offer for camera ${data.cameraId}`);
    if (data.targetId) {
      // Direct offer to specific peer
      socket.to(data.targetId).emit('webrtc-offer', { 
        offer: data.offer, 
        senderId: socket.id,
        cameraId: data.cameraId 
      });
    } else {
      // Broadcast offer to camera room
      socket.to(`webrtc-${data.cameraId}`).emit('webrtc-offer', { 
        offer: data.offer, 
        senderId: socket.id,
        cameraId: data.cameraId 
      });
    }
  });

  socket.on('webrtc-answer', (data: { answer: any, targetId: string, cameraId: string }) => {
    console.log(`WebRTC answer for camera ${data.cameraId}`);
    socket.to(data.targetId).emit('webrtc-answer', { 
      answer: data.answer, 
      senderId: socket.id,
      cameraId: data.cameraId 
    });
  });

  socket.on('webrtc-ice-candidate', (data: { candidate: any, targetId: string, cameraId: string }) => {
    console.log(`WebRTC ICE candidate for camera ${data.cameraId}`);
    socket.to(data.targetId).emit('webrtc-ice-candidate', { 
      candidate: data.candidate, 
      senderId: socket.id,
      cameraId: data.cameraId 
    });
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
    // Clean up WebRTC peer tracking
    webrtcPeers.delete(socket.id);
    
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

// Enhanced CoTURN-compatible server implementation for Railway
const TURN_PORT = Number(process.env.PORT) || 8080; // Use Railway's exposed port
const TURN_USERNAME = process.env.TURN_USERNAME || 'mmoment';
const TURN_PASSWORD = process.env.TURN_PASSWORD || 'webrtc123';
const TURN_REALM = process.env.TURN_REALM || 'mmoment.xyz';

// Get external IP for Railway deployment
const EXTERNAL_IP = process.env.RAILWAY_PUBLIC_DOMAIN || 'mmoment-production.up.railway.app';

// Store active TURN allocations
const turnAllocations = new Map<string, {
  clientAddress: string,
  clientPort: number,
  relayPort: number,
  relaySocket: dgram.Socket,
  peers: Map<string, { address: string, port: number }>
}>();

// Create TURN server (TCP for Railway compatibility)
const turnServer = net.createServer();

// Helper function to parse STUN/TURN messages
function parseStunMessage(msg: Buffer) {
  if (msg.length < 20) return null;
  
  const messageType = msg.readUInt16BE(0);
  const messageLength = msg.readUInt16BE(2);
  const magicCookie = msg.readUInt32BE(4);
  const transactionId = msg.slice(8, 20);
  
  return { messageType, messageLength, magicCookie, transactionId };
}

// Helper function to create STUN/TURN response
function createStunResponse(type: number, transactionId: Buffer, attributes: Buffer[] = []): Buffer {
  const totalAttrLength = attributes.reduce((sum, attr) => sum + attr.length, 0);
  const response = Buffer.alloc(20 + totalAttrLength);
  
  response.writeUInt16BE(type, 0); // Message type
  response.writeUInt16BE(totalAttrLength, 2); // Message length
  response.writeUInt32BE(0x2112A442, 4); // Magic cookie
  transactionId.copy(response, 8); // Transaction ID
  
  // Copy attributes
  let offset = 20;
  for (const attr of attributes) {
    attr.copy(response, offset);
    offset += attr.length;
  }
  
  return response;
}

// Helper to create XOR-MAPPED-ADDRESS attribute
function createXorMappedAddress(address: string, port: number): Buffer {
  const attr = Buffer.alloc(12);
  attr.writeUInt16BE(0x0020, 0); // XOR-MAPPED-ADDRESS
  attr.writeUInt16BE(8, 2); // Length
  attr.writeUInt16BE(0x0001, 4); // IPv4 family
  attr.writeUInt16BE(port ^ 0x2112, 6); // XOR'd port
  
  // XOR the IP address
  const ipParts = address.split('.').map(Number);
  const magicCookie = [0x21, 0x12, 0xA4, 0x42];
  for (let i = 0; i < 4; i++) {
    attr.writeUInt8(ipParts[i] ^ magicCookie[i], 8 + i);
  }
  
  return attr;
}

turnServer.on('connection', (socket) => {
  console.log(`TCP TURN connection from ${socket.remoteAddress}:${socket.remotePort}`);
  
  socket.on('data', (data) => {
    try {
      const stunMsg = parseStunMessage(data);
      if (!stunMsg) return;
      
      const { messageType, transactionId } = stunMsg;
      const clientAddress = socket.remoteAddress || '127.0.0.1';
      const clientPort = socket.remotePort || 0;
        
        // Handle STUN Binding Request (0x0001)
        if (messageType === 0x0001) {
          console.log(`STUN Binding request from ${clientAddress}:${clientPort}`);
          
          // Create XOR-MAPPED-ADDRESS attribute
          const xorMappedAddr = createXorMappedAddress(clientAddress, clientPort);
          
          // Send STUN Binding Response (0x0101)
          const response = createStunResponse(0x0101, transactionId, [xorMappedAddr]);
          socket.write(response);
          
          console.log(`STUN Binding response sent to ${clientAddress}:${clientPort}`);
        }
        // Handle TURN Allocate Request (0x0003)
        else if (messageType === 0x0003) {
          console.log(`TURN: Allocation request from ${clientAddress}:${clientPort}`);
        
        // Create a relay socket for this client
        const relaySocket = dgram.createSocket('udp4');
        let relayPort = 40000 + Math.floor(Math.random() * 10000);
        
          relaySocket.bind(relayPort, () => {
            const allocationId = `${clientAddress}:${clientPort}`;
            
            // Store allocation
            turnAllocations.set(allocationId, {
              clientAddress: clientAddress,
              clientPort: clientPort,
              relayPort: relayPort,
              relaySocket: relaySocket,
              peers: new Map()
            });
            
            // Handle relay traffic
            relaySocket.on('message', (relayMsg, relayRinfo) => {
              // Forward relay traffic back to the client
              socket.write(relayMsg);
            });
          
          // Create XOR-RELAYED-ADDRESS attribute
          const relayAddr = createXorMappedAddress('0.0.0.0', relayPort);
          relayAddr.writeUInt16BE(0x0016, 0); // Change type to XOR-RELAYED-ADDRESS
          
            // Create XOR-MAPPED-ADDRESS for the client
            const mappedAddr = createXorMappedAddress(clientAddress, clientPort);
            
            // Create LIFETIME attribute (600 seconds)
            const lifetime = Buffer.alloc(8);
            lifetime.writeUInt16BE(0x000D, 0); // LIFETIME
            lifetime.writeUInt16BE(4, 2); // Length
            lifetime.writeUInt32BE(600, 4); // 600 seconds
            
            // Send Allocate Success Response (0x0103)
            const response = createStunResponse(0x0103, transactionId, [relayAddr, mappedAddr, lifetime]);
            socket.write(response);
            
            console.log(`TURN: Allocated relay port ${relayPort} for ${allocationId}`);
        });
        
        relaySocket.on('error', (err) => {
          console.error(`TURN relay socket error:`, err);
        });
        
      }
        // Handle TURN Refresh Request (0x0004)
        else if (messageType === 0x0004) {
          console.log(`TURN Refresh request from ${clientAddress}:${clientPort}`);
          
          // Send Refresh Success Response (0x0104)
          const lifetime = Buffer.alloc(8);
          lifetime.writeUInt16BE(0x000D, 0); // LIFETIME
          lifetime.writeUInt16BE(4, 2); // Length
          lifetime.writeUInt32BE(600, 4); // 600 seconds
          
          const response = createStunResponse(0x0104, transactionId, [lifetime]);
          socket.write(response);
        }
        // Handle ChannelBind or Send Indication for data relay
        else if (messageType === 0x0009 || messageType === 0x0016) {
          // Handle data relay
          const allocationId = `${clientAddress}:${clientPort}`;
          const allocation = turnAllocations.get(allocationId);
          
          if (allocation) {
            // This is data from the client to be relayed
            // In a full TURN implementation, you'd parse the destination from TURN headers
            // For simplicity, we'll relay to the first peer or back to the client
            
            // Forward to relay socket (which will send to peers)
            allocation.relaySocket.send(data, 0, data.length, allocation.relayPort, 'localhost');
          }
        }
        // Handle other TURN messages
        else {
          console.log(`Unknown STUN/TURN message type: 0x${messageType.toString(16)} from ${clientAddress}:${clientPort}`);
        }
    } catch (error) {
      console.error('TCP TURN server error:', error);
    }
  });

  socket.on('close', () => {
    console.log(`TCP TURN connection closed from ${socket.remoteAddress}:${socket.remotePort}`);
  });

  socket.on('error', (err) => {
    console.error('TCP TURN socket error:', err);
  });
});

turnServer.on('error', (err) => {
  console.error('TURN server error:', err);
});

// TURN server disabled - Railway only provides one port for HTTP
// TCP TURN on the same port as HTTP is not supported
// turnServer.listen(TURN_PORT, '0.0.0.0', () => {
//   console.log(`✅ TCP TURN server listening on 0.0.0.0:${TURN_PORT}`);
//   console.log(`TURN realm: ${TURN_REALM}`);
//   console.log(`TURN credentials: ${TURN_USERNAME}:${TURN_PASSWORD}`);
//   console.log(`External domain: ${EXTERNAL_IP}`);
//   console.log(`Supported messages: STUN Binding, TURN Allocate, TURN Refresh`);
//   console.log(`Transport: TCP (Railway compatible)`);
// });
console.log('⚠️ TURN server disabled - Railway only supports HTTP on port 8080');
console.log('⚠️ Use external TURN server for cross-network WebRTC');

// TURN server info endpoint
app.get('/api/turn/info', (req, res) => {
  const turnInfo = {
    host: req.headers.host?.split(':')[0] || 'localhost',
    port: TURN_PORT,
    username: TURN_USERNAME,
    credential: TURN_PASSWORD,
    urls: [
      `turn:${req.headers.host?.split(':')[0] || 'localhost'}:${TURN_PORT}`,
      `turn:${req.headers.host?.split(':')[0] || 'localhost'}:${TURN_PORT}?transport=udp`
    ]
  };
  
  res.json(turnInfo);
});

// Start HTTP server with error handling
const port = Number(process.env.PORT) || 3001;
httpServer.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log('Server configuration:', {
    keepAliveTimeout: httpServer.keepAliveTimeout,
    headersTimeout: httpServer.headersTimeout,
    socketTransports: io.engine.opts.transports,
    socketPingTimeout: io.engine.opts.pingTimeout,
    socketPingInterval: io.engine.opts.pingInterval,
    turnPort: TURN_PORT,
    turnUsername: TURN_USERNAME
  });
});