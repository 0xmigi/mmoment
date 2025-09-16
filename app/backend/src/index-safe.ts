// Minimal stable backend for MMOMENT
import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";
import { config } from "dotenv";

// Load environment variables
config();

const app = express();
const httpServer = createServer(app);

// Configure server timeouts
httpServer.keepAliveTimeout = 120000; // 2 minutes
httpServer.headersTimeout = 121000;

// Middleware
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(cors({ origin: true, credentials: true }));

// Serve static files
app.use(express.static("../../"));

// Socket.IO setup with stability improvements
const io = new Server(httpServer, {
  cors: {
    origin: true,
    methods: ["GET", "POST"],
    credentials: true,
  },
  transports: ["polling", "websocket"],
  pingTimeout: 60000,
  pingInterval: 25000,
  maxHttpBufferSize: 1e8, // 100 MB
});

// In-memory storage
const pipeAccounts = new Map();
const timelineEvents: any[] = [];
const cameraRooms = new Map<string, Set<string>>();

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Pipe API endpoints
app.get("/api/pipe/credentials", (req, res) => {
  const walletAddress = req.query.wallet as string;

  if (!walletAddress) {
    res.json({
      userId: "7a810586-a79b-4686-a61e-2951e8ab918b",
      userAppKey:
        "aed193f6ec0886bbf103f053712642adc36f18b0f0a6406e3e3adb7c3670581d",
    });
    return;
  }

  const account = pipeAccounts.get(walletAddress);
  if (account) {
    res.json({
      userId: account.userId,
      userAppKey: account.userAppKey,
    });
  } else {
    res.status(404).json({ error: "No Pipe account found for this wallet" });
  }
});

app.post("/api/pipe/create-account", (req, res) => {
  const { walletAddress } = req.body;

  if (!walletAddress) {
    return res.status(400).json({ error: "walletAddress is required" });
  }

  if (pipeAccounts.has(walletAddress)) {
    const account = pipeAccounts.get(walletAddress);
    return res.json({
      userId: account.userId,
      userAppKey: account.userAppKey,
      existing: true,
    });
  }

  const newAccount = {
    userId: "7a810586-a79b-4686-a61e-2951e8ab918b",
    userAppKey:
      "aed193f6ec0886bbf103f053712642adc36f18b0f0a6406e3e3adb7c3670581d",
    createdAt: Date.now(),
  };

  pipeAccounts.set(walletAddress, newAccount);

  res.json({
    userId: newAccount.userId,
    userAppKey: newAccount.userAppKey,
    existing: false,
  });
});

// Pipe API proxy - simplified and stable
app.post("/api/pipe/proxy/*", async (req: any, res) => {
  const endpoint = req.params[0];

  try {
    const headers: any = {
      "Content-Type": "application/json",
    };

    // Add auth headers if present
    if (req.headers["x-user-id"]) {
      headers["X-User-Id"] = req.headers["x-user-id"];
      headers["X-User-App-Key"] = req.headers["x-user-app-key"];
    }

    const response = await fetch(
      `https://us-east-00-firestarter.pipenetwork.com/${endpoint}`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(req.body),
      },
    );

    const text = await response.text();
    res.status(response.status);

    if (text) {
      try {
        res.json(JSON.parse(text));
      } catch {
        res.send(text);
      }
    } else {
      res.json({});
    }
  } catch (error: any) {
    console.error(`Proxy error for ${endpoint}:`, error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: "Proxy request failed" });
    }
  }
});

// Socket.IO connection handling
io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);

  socket.on("joinCamera", (cameraId: string) => {
    socket.join(cameraId);
    if (!cameraRooms.has(cameraId)) {
      cameraRooms.set(cameraId, new Set());
    }
    cameraRooms.get(cameraId)?.add(socket.id);

    const cameraEvents = timelineEvents
      .filter((event) => event.cameraId === cameraId)
      .slice(-50);
    socket.emit("recentEvents", cameraEvents);
  });

  socket.on("leaveCamera", (cameraId: string) => {
    socket.leave(cameraId);
    cameraRooms.get(cameraId)?.delete(socket.id);
  });

  socket.on("newTimelineEvent", (event: any) => {
    const newEvent = {
      ...event,
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: event.timestamp || Date.now(),
    };

    timelineEvents.push(newEvent);

    // Keep only last 100 events per camera
    if (event.cameraId) {
      const cameraEvents = timelineEvents.filter(
        (e) => e.cameraId === event.cameraId,
      );
      if (cameraEvents.length > 100) {
        const oldestIndex = timelineEvents.findIndex(
          (e) => e.cameraId === event.cameraId,
        );
        if (oldestIndex !== -1) {
          timelineEvents.splice(oldestIndex, 1);
        }
      }
      io.to(event.cameraId).emit("timelineEvent", newEvent);
    } else {
      io.emit("timelineEvent", newEvent);
    }
  });

  socket.on("getRecentEvents", ({ cameraId }: { cameraId?: string }) => {
    const events = cameraId
      ? timelineEvents.filter((e) => e.cameraId === cameraId)
      : timelineEvents;
    socket.emit("recentEvents", events.slice(-50));
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.id);
    cameraRooms.forEach((sockets) => {
      sockets.delete(socket.id);
    });
  });
});

// Error handlers
process.on("uncaughtException", (error) => {
  console.error("Uncaught Exception:", error);
});

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled Rejection:", reason);
});

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("SIGTERM received, closing server...");
  httpServer.close(() => {
    process.exit(0);
  });
});

// Start server
const port = Number(process.env.PORT) || 3001;
httpServer.listen(port, "0.0.0.0", () => {
  console.log(`🚀 Stable backend running on port ${port}`);
  console.log(`📍 Health check: http://localhost:${port}/health`);
  console.log(`🔧 API tester: http://localhost:${port}/pipe-api-tester.html`);
});
