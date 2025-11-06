// backend/src/index.ts
import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";
import { config } from "dotenv";
import { createFirestarterSDK } from "firestarter-sdk";
import dgram from "dgram";
import net from "net";
import axios from "axios";
import { Connection, PublicKey } from "@solana/web3.js";
// Using built-in fetch in Node.js 18+

// Load environment variables
config();

// Initialize Solana connection
const connection = new Connection(
  process.env.SOLANA_RPC_URL || "https://api.devnet.solana.com",
  "confirmed"
);

// Import gas sponsorship service
import {
  initializeGasSponsorshipService,
  sponsorTransaction,
  getUserSponsorshipStatus,
  getSponsorshipStats,
  resetUserSponsorship,
  clearAllSponsorships,
  checkFeePayerBalance
} from './gas-sponsorship';

// Import session cleanup cron service
import {
  initializeSessionCleanupCron,
  stopCleanupCron,
  triggerManualCleanup,
  getCleanupCronStatus
} from './session-cleanup-cron';

// Note: Socket.IO server (io) will be passed to session cleanup cron after it's created below

// Initialize the Firestarter SDK with mainnet endpoint
const firestarterSDK = createFirestarterSDK({
  baseUrl: "https://us-west-01-firestarter.pipenetwork.com", // Mainnet
});

// In-memory storage for pipe accounts (simple key-value store)
// The SDK handles auth, we just store the mapping
const pipeAccounts = new Map<
  string,
  {
    userId: string;
    userAppKey: string;
    walletAddress: string;
    created: Date;
  }
>();

// Cached Pipe JWT token for Jetson uploads
let cachedPipeToken: {
  access_token: string;
  user_id: string;
  expires_at: number; // timestamp
} | null = null;

// Helper function to get or refresh Pipe JWT token
async function getPipeJWTToken(): Promise<{ access_token: string; user_id: string }> {
  const now = Date.now();

  // Return cached token if still valid (with 5 min buffer)
  if (cachedPipeToken && cachedPipeToken.expires_at > now + 5 * 60 * 1000) {
    console.log(`✅ Using cached Pipe token (expires in ${Math.floor((cachedPipeToken.expires_at - now) / 1000 / 60)} min)`);
    return {
      access_token: cachedPipeToken.access_token,
      user_id: cachedPipeToken.user_id
    };
  }

  // Token expired or doesn't exist - login fresh
  const pipeUsername = process.env.PIPE_USERNAME || "wallettest1762286471";
  const pipePassword = process.env.PIPE_PASSWORD || "StrongPass123!@#";

  console.log(`🔄 Fetching fresh Pipe JWT token...`);

  const loginResp = await fetch("https://us-west-01-firestarter.pipenetwork.com/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: pipeUsername, password: pipePassword })
  });

  if (!loginResp.ok) {
    throw new Error(`Login failed: ${loginResp.status}`);
  }

  const tokens = await loginResp.json();

  // Get user_id
  const walletResp = await fetch("https://us-west-01-firestarter.pipenetwork.com/checkWallet", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${tokens.access_token}`
    },
    body: JSON.stringify({})
  });

  if (!walletResp.ok) {
    throw new Error(`checkWallet failed: ${walletResp.status}`);
  }

  const walletData = await walletResp.json();

  // Cache the token (assume 1 hour expiry)
  cachedPipeToken = {
    access_token: tokens.access_token,
    user_id: walletData.user_id,
    expires_at: now + 60 * 60 * 1000 // 1 hour from now
  };

  console.log(`✅ Fresh Pipe token cached (user: ${walletData.user_id.slice(0, 20)}...)`);

  return {
    access_token: tokens.access_token,
    user_id: walletData.user_id
  };
}

// Device signature to file mapping (privacy-preserving)
// Maps device signature OR blockchain tx signature → Pipe file metadata
// Supports both: device-signed instant captures AND blockchain tx captures
interface FileMapping {
  signature: string;        // Device signature OR blockchain tx signature
  signatureType: 'device' | 'blockchain';
  walletAddress: string;    // Owner's wallet address
  fileId: string;           // Pipe file ID (blake3 hash)
  fileName: string;         // Stored filename on Pipe
  cameraId: string;         // Which camera captured this
  uploadedAt: Date;         // When Jetson uploaded
  fileType: 'photo' | 'video';
}

const signatureToFileMapping = new Map<string, FileMapping>();

// Wallet to signatures mapping (for gallery queries)
// Maps wallet address → array of signatures (both device and blockchain)
const walletToSignatures = new Map<string, string[]>();

// Legacy support - keep old variable name for backward compatibility
const txToFileMapping = signatureToFileMapping;

const app = express();

// Create HTTP server with extremely lenient timeout settings
const httpServer = createServer(app);
httpServer.keepAliveTimeout = 300000; // 5 minutes
httpServer.headersTimeout = 301000; // Slightly higher than keepAliveTimeout

// Basic middleware with increased limits
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Serve static files from the project root
app.use(express.static("../../"));

// Super permissive CORS configuration
app.use((req, res, next) => {
  // Allow any origin
  const origin = req.headers.origin || "*";
  res.setHeader("Access-Control-Allow-Origin", origin);

  // Allow all methods and headers
  res.setHeader("Access-Control-Allow-Methods", "*");
  res.setHeader("Access-Control-Allow-Headers", "*");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Max-Age", "86400");

  // Add cache control headers
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");

  // Handle preflight
  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }
  next();
});

// Use cors middleware with permissive settings
app.use(
  cors({
    origin: true,
    methods: "*",
    allowedHeaders: "*",
    credentials: true,
    maxAge: 86400,
    preflightContinue: true,
  }),
);

// Configure Socket.IO with extremely permissive settings
const io = new Server(httpServer, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
    credentials: true,
    allowedHeaders: "*",
  },
  path: "/socket.io/",
  transports: ["polling", "websocket"], // Try polling first
  pingTimeout: 300000, // 5 minutes
  pingInterval: 25000,
  connectTimeout: 300000, // 5 minutes
  upgradeTimeout: 300000, // 5 minutes
  maxHttpBufferSize: 1e8,
  allowUpgrades: true,
  perMessageDeflate: false,
  destroyUpgrade: false,
});

// Trust proxy and handle HTTPS
app.enable("trust proxy");

// Add response timeout middleware
app.use((req, res, next) => {
  res.setTimeout(300000, () => {
    // 5 minutes
    res.status(504).json({ error: "Server timeout" });
  });
  next();
});

// Debug logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const requestId = Math.random().toString(36).substring(7);

  console.log(
    `[${requestId}] ${new Date().toISOString()} - Incoming ${req.method} ${req.url}`,
  );
  console.log(`[${requestId}] Headers:`, req.headers);

  // Log response
  res.on("finish", () => {
    const duration = Date.now() - start;
    console.log(
      `[${requestId}] ${new Date().toISOString()} - ${req.method} ${req.url} - ${res.statusCode} - ${duration}ms`,
    );
  });

  next();
});

// Add debug endpoints
app.get("/debug/headers", (req, res) => {
  res.json({
    headers: req.headers,
    ip: req.ip,
    ips: req.ips,
    secure: req.secure,
    protocol: req.protocol,
  });
});

app.get("/debug/ping", (req, res) => {
  res.json({ pong: Date.now() });
});

// Pipe accounts are stored in-memory for simplicity
// In production, this should use a proper database

// Pipe API endpoints - restored working proxy approach
app.get("/api/pipe/credentials", async (req, res) => {
  const walletAddress = req.query.wallet as string;

  if (!walletAddress) {
    res.status(400).json({ error: "walletAddress parameter is required" });
    return;
  }

  try {
    // Check if we have existing Pipe account for this wallet
    const account = pipeAccounts.get(walletAddress);
    if (account) {
      console.log(
        `Found existing Pipe account for wallet: ${walletAddress.slice(0, 8)}...`,
      );
      res.json({
        userId: account.userId,
        userAppKey: account.userAppKey,
      });
    } else {
      res.status(404).json({ error: "No Pipe account found for this wallet" });
    }
  } catch (error) {
    console.error("Error checking credentials:", error);
    res.status(500).json({ error: "Failed to check credentials" });
  }
});

// Debug endpoint to check user info
app.get("/api/pipe/debug-user/:walletAddress", async (req, res) => {
  const walletAddress = req.params.walletAddress;

  try {
    console.log(`\n🔍 Debug user info for wallet: ${walletAddress}`);

    // Check if we have this user
    const user = await firestarterSDK.createUserAccount(walletAddress);
    console.log(`👤 SDK User:`, user);

    // Check if we have stored credentials
    const account = pipeAccounts.get(walletAddress);
    console.log(`💾 Stored Account:`, account);

    res.json({
      walletAddress,
      sdkUser: user,
      storedAccount: account,
      expectedUsername: `mmoment_${walletAddress.slice(0, 16)}`,
    });
  } catch (error) {
    console.error("Debug user error:", error);
    res
      .status(500)
      .json({
        error: error instanceof Error ? error.message : "Unknown error",
      });
  }
});

// Proxy endpoint for Pipe API requests
app.post("/api/pipe/proxy/*", async (req: any, res) => {
  const endpoint = req.params[0]; // Get the endpoint after /proxy/
  const walletAddress = req.headers["x-wallet-address"] as string;

  if (!walletAddress) {
    return res
      .status(400)
      .json({ error: "x-wallet-address header is required" });
  }

  try {
    console.log(
      `🔄 Proxying to Pipe: ${endpoint} for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Handle specific endpoints that frontend expects
    if (endpoint === "checkWallet") {
      // Get balance using SDK
      const balance = await firestarterSDK.getUserBalance(walletAddress);
      res.json({
        balance_sol: balance.sol,
        public_key: balance.publicKey,
      });
      return;
    }

    if (endpoint === "checkCustomToken") {
      // Get PIPE token balance using SDK
      const balance = await firestarterSDK.getUserBalance(walletAddress);
      res.json({
        balance: balance.pipe * 1000000, // Convert to raw units
        ui_amount: balance.pipe,
      });
      return;
    }

    if (endpoint === "exchangeSolForTokens") {
      // Exchange SOL for PIPE using SDK
      const { amount_sol } = req.body;
      if (!amount_sol) {
        return res.status(400).json({ error: "amount_sol is required" });
      }

      const tokensReceived = await firestarterSDK.exchangeSolForPipe(
        walletAddress,
        amount_sol,
      );
      res.json({
        tokens_minted: tokensReceived,
        success: true,
      });
      return;
    }

    // For other endpoints, return basic info
    res.json({
      message: `Proxied ${endpoint}`,
      walletAddress,
    });
  } catch (error) {
    console.error("Pipe proxy error:", error);
    res
      .status(500)
      .json({
        error: error instanceof Error ? error.message : "Proxy request failed",
      });
  }
});

// Create account endpoint - using FirestarterSDK
app.post("/api/pipe/create-account", async (req, res) => {
  const { walletAddress } = req.body;

  if (!walletAddress) {
    return res.status(400).json({ error: "walletAddress is required" });
  }

  try {
    // Check if account already exists locally
    const existingAccount = pipeAccounts.get(walletAddress);
    if (existingAccount) {
      console.log(
        `✅ Existing Pipe account found for ${walletAddress.slice(0, 8)}...`,
      );
      res.json({
        userId: existingAccount.userId,
        userAppKey: existingAccount.userAppKey,
        existing: true,
      });
      return;
    }

    console.log(
      `🔄 Creating new Pipe account for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Use the SDK to create or get the user account
    // The SDK handles all the JWT setup internally
    const pipeUser = await firestarterSDK.createUserAccount(walletAddress);

    // Store account info locally for quick access
    const accountInfo = {
      userId: pipeUser.userId,
      userAppKey: pipeUser.userAppKey || "",
      walletAddress,
      created: new Date(),
    };

    pipeAccounts.set(walletAddress, accountInfo);

    console.log(`✅ Pipe account ready for ${walletAddress.slice(0, 8)}...`);

    res.json({
      userId: accountInfo.userId,
      userAppKey: accountInfo.userAppKey,
      existing: false,
    });
  } catch (error) {
    console.error("Error creating Pipe account:", error);
    res
      .status(500)
      .json({
        error:
          error instanceof Error
            ? error.message
            : "Failed to create Pipe account",
      });
  }
});

// Pipe upload endpoint - using FirestarterSDK
app.post("/api/pipe/upload", async (req, res) => {
  const { walletAddress, imageData, filename, metadata } = req.body;

  if (!walletAddress || !imageData || !filename) {
    return res.status(400).json({
      success: false,
      error: "walletAddress, imageData, and filename are required",
    });
  }

  try {
    // Convert base64 image data to buffer if needed
    let buffer: Buffer;
    if (typeof imageData === "string") {
      // Remove data URL prefix if present (data:image/jpeg;base64,...)
      const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, "");
      buffer = Buffer.from(base64Data, "base64");
    } else {
      buffer = Buffer.from(imageData);
    }

    console.log(
      `📤 Uploading ${filename} to Pipe for ${walletAddress.slice(0, 8)}... (${buffer.length} bytes)`,
    );

    // Use the SDK to upload the file
    // The SDK handles all auth internally including JWT tokens
    const uploadResult = await firestarterSDK.uploadFile(
      walletAddress,
      buffer,
      filename,
      { metadata },
    );

    console.log(
      `✅ Successfully uploaded ${filename} for ${walletAddress.slice(0, 8)}...`,
    );
    console.log(`📝 Upload result:`, {
      fileId: uploadResult.fileId,
      fileName: uploadResult.fileName,
      size: uploadResult.size,
      blake3Hash: uploadResult.blake3Hash,
    });

    res.json({
      success: true,
      result: uploadResult.fileName,
      originalFilename: filename,
      size: buffer.length,
      walletAddress,
      metadata,
      blake3Hash: uploadResult.blake3Hash,
      uploadTimestamp: uploadResult.uploadedAt.toISOString(),
    });
  } catch (error) {
    console.error("Upload failed:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Upload failed",
    });
  }
});

// Get files for a user endpoint
app.get("/api/pipe/files/:walletAddress", async (req, res) => {
  const walletAddress = req.params.walletAddress;

  if (!walletAddress) {
    return res.status(400).json({ error: "Wallet address is required" });
  }

  try {
    console.log(
      `📁 Getting file list for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Get files from SDK upload history first
    const fileRecords = await firestarterSDK.listUserFiles(walletAddress);
    console.log(
      `📁 Found ${fileRecords.length} files for wallet ${walletAddress.slice(0, 8)}:`,
      fileRecords,
    );

    if (fileRecords.length === 0) {
      // Return empty array if no files
      return res.json({
        files: [],
        count: 0,
        walletAddress: walletAddress.slice(0, 8) + "...",
      });
    }

    // Get user credentials to include userAppKey in download URLs
    const user = await firestarterSDK.createUserAccount(walletAddress);

    // For JWT-based downloads, we need to use the SDK's download method instead of direct URLs
    // Transform SDK FileRecord format to frontend PipeFile format
    const files = fileRecords.map((record: any) => ({
      id: record.fileId,
      name: record.originalFileName,
      size: record.size,
      contentType: record.mimeType || "application/octet-stream",
      uploadedAt: record.uploadedAt
        ? record.uploadedAt instanceof Date
          ? record.uploadedAt.toISOString()
          : new Date(record.uploadedAt).toISOString()
        : new Date().toISOString(),
      // Create a backend proxy URL for downloads since we need JWT auth
      url: `http://localhost:3001/api/pipe/download/${walletAddress}/${encodeURIComponent(record.fileId)}`,
      metadata: record.metadata || {},
      blake3Hash: record.blake3Hash,
    }));

    res.json({
      files,
      count: files.length,
      walletAddress: walletAddress.slice(0, 8) + "...",
    });
  } catch (error) {
    console.error("❌ Failed to list files:", error);
    console.error(
      "❌ Stack trace:",
      error instanceof Error ? error.stack : "No stack trace",
    );
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to list files",
    });
  }
});

// Streaming download endpoint for large videos (no buffering)
app.get("/api/pipe/download/:walletAddress/:fileId", async (req, res) => {
  const { walletAddress, fileId } = req.params;

  if (!walletAddress || !fileId) {
    return res
      .status(400)
      .json({ error: "Wallet address and file ID are required" });
  }

  try {
    console.log(
      `📥 Streaming download ${fileId} for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Get JWT token (legacy Pipe account uses JWT Bearer auth, not user_app_key)
    const { access_token } = await getPipeJWTToken();
    const baseUrl = process.env.PIPE_BASE_URL || 'https://us-west-01-firestarter.pipenetwork.com';

    // Determine content type from file extension
    let contentType = "application/octet-stream";
    const fileName = decodeURIComponent(fileId).toLowerCase();
    if (fileName.includes(".mp4") || fileName.includes(".mov")) {
      contentType = "video/mp4";
    } else if (fileName.includes(".jpg") || fileName.includes(".jpeg")) {
      contentType = "image/jpeg";
    } else if (fileName.includes(".png")) {
      contentType = "image/png";
    }

    // Set headers for streaming
    res.setHeader("Content-Type", contentType);
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Cache-Control", "public, max-age=31536000");

    // Pipe priority-upload uses content-addressing: download by hash (fileId), not placeholder fileName
    const downloadUrl = new URL(`${baseUrl}/download-stream`);
    downloadUrl.searchParams.append("file_name", fileId);

    console.log(`📥 Downloading from Pipe using fileId: ${fileId.slice(0, 20)}...`);

    const pipeResponse = await axios.get(downloadUrl.toString(), {
      headers: {
        "Authorization": `Bearer ${access_token}`,
      },
      responseType: "arraybuffer",  // Get full response to parse multipart
      timeout: 300000,
    });

    // Pipe returns multipart form-data, need to extract actual file content
    const responseData = Buffer.from(pipeResponse.data);
    const responseText = responseData.toString('binary');

    // Find boundary in Content-Type header
    const contentTypeHeader = pipeResponse.headers['content-type'] || '';
    const boundaryMatch = contentTypeHeader.match(/boundary=([^;]+)/);

    if (boundaryMatch) {
      // Parse multipart form-data to extract file content
      const boundary = '--' + boundaryMatch[1];
      const parts = responseText.split(boundary);

      // Find the part with file content (skip headers)
      for (const part of parts) {
        if (part.includes('Content-Disposition') && part.includes('filename')) {
          // Extract content after headers (double CRLF separates headers from content)
          const contentStart = part.indexOf('\r\n\r\n') + 4;
          const contentEnd = part.lastIndexOf('\r\n');

          if (contentStart > 3 && contentEnd > contentStart) {
            const fileContent = Buffer.from(part.substring(contentStart, contentEnd), 'binary');
            res.setHeader("Content-Length", fileContent.length);
            return res.send(fileContent);
          }
        }
      }
    }

    // Fallback: if not multipart or parsing failed, send as-is
    res.setHeader("Content-Length", responseData.length);
    res.send(responseData);

  } catch (error) {
    console.error("❌ Download failed:", error);
    if (!res.headersSent) {
      res.status(500).json({
        error: error instanceof Error ? error.message : "Download failed",
      });
    }
  }
});

// Jetson endpoint: Get Pipe credentials for direct uploads
app.get("/api/pipe/jetson/credentials", async (req, res) => {
  try {
    console.log(`🔧 Getting Pipe credentials for Jetson...`);

    // Use the existing JWT token (same as test script)
    const { access_token, user_id } = await getPipeJWTToken();

    // Return JWT auth credentials (like test script)
    res.json({
      user_id: user_id,
      access_token: access_token,
      baseUrl: "https://us-west-01-firestarter.pipenetwork.com",
    });
  } catch (error) {
    console.error("❌ Failed to get Jetson credentials:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to get credentials",
    });
  }
});

// Jetson endpoint: Notify backend after direct upload to Pipe
app.post("/api/pipe/jetson/upload-complete", async (req, res) => {
  const { txSignature, fileName, fileId, blake3Hash, size, cameraId, fileType, metadata } = req.body;

  if (!txSignature || !fileName || !fileId) {
    return res.status(400).json({
      error: "txSignature, fileName, and fileId are required"
    });
  }

  try {
    // Determine signature type (device signature vs blockchain tx)
    // Device signatures are typically longer ed25519 signatures
    // Blockchain tx signatures are base58-encoded (88 chars)
    const signatureType = metadata?.user_wallet ? 'device' : 'blockchain';
    const walletAddress = metadata?.user_wallet || 'unknown';

    console.log(`📝 Jetson uploaded: ${fileName}`);
    console.log(`   Signature Type: ${signatureType}`);
    console.log(`   Signature: ${txSignature.slice(0, 16)}...`);
    console.log(`   Wallet: ${walletAddress.slice(0, 8)}...`);
    console.log(`   Size: ${size} bytes`);

    // Store signature → file mapping
    const mapping: FileMapping = {
      signature: txSignature,
      signatureType,
      walletAddress,
      fileId: blake3Hash || fileId,
      fileName,
      cameraId: cameraId || 'unknown',
      uploadedAt: new Date(),
      fileType: fileType || 'photo',
    };

    signatureToFileMapping.set(txSignature, mapping);

    // Also track by wallet address for gallery queries
    if (walletAddress !== 'unknown') {
      const existingSignatures = walletToSignatures.get(walletAddress) || [];
      existingSignatures.push(txSignature);
      walletToSignatures.set(walletAddress, existingSignatures);

      console.log(`✅ Mapped ${signatureType} signature → file for ${walletAddress.slice(0, 8)}...`);
      console.log(`   Total files for wallet: ${existingSignatures.length}`);
    }

    // Notify connected clients via WebSocket
    io.emit("pipe:upload:complete", {
      txSignature,
      signature: txSignature,
      signatureType,
      walletAddress,
      fileName,
      fileId: mapping.fileId,
      size,
      cameraId,
      timestamp: new Date().toISOString(),
    });

    res.json({
      success: true,
      fileId: mapping.fileId,
      txSignature,
      signature: txSignature,
      signatureType,
    });
  } catch (error) {
    console.error("❌ Failed to process upload notification:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to process upload",
    });
  }
});

// Gallery endpoint: Get user's media (supports both device-signed and blockchain tx captures)
app.get("/api/pipe/gallery/:walletAddress", async (req, res) => {
  const { walletAddress } = req.params;

  if (!walletAddress) {
    return res.status(400).json({ error: "Wallet address required" });
  }

  try {
    console.log(`📸 Fetching gallery for wallet: ${walletAddress.slice(0, 8)}...`);

    const mediaItems = [];

    // Method 1: Get device-signed captures (instant captures, no blockchain tx)
    const deviceSignatures = walletToSignatures.get(walletAddress) || [];
    console.log(`   Device-signed captures: ${deviceSignatures.length}`);

    for (const sig of deviceSignatures) {
      const mapping = signatureToFileMapping.get(sig);
      if (mapping) {
        // Use fileId (hash) for download URL instead of fileName (which may be placeholder)
        const downloadUrl = `/api/pipe/download/${walletAddress}/${mapping.fileId}`;

        mediaItems.push({
          id: mapping.fileId,
          fileId: mapping.fileId,
          fileName: mapping.fileName,
          url: downloadUrl,
          type: mapping.fileType,
          cameraId: mapping.cameraId,
          uploadedAt: mapping.uploadedAt.toISOString(),
          txSignature: mapping.signature,
          signatureType: mapping.signatureType,
          provider: 'pipe'
        });
      }
    }

    // Method 2: Get blockchain tx captures (legacy/privacy mode captures)
    try {
      const userPubkey = new PublicKey(walletAddress);
      const blockchainSignatures = await connection.getSignaturesForAddress(userPubkey, {
        limit: 100, // Last 100 transactions
      });

      console.log(`   Blockchain transactions: ${blockchainSignatures.length}`);

      for (const sig of blockchainSignatures) {
        const mapping = signatureToFileMapping.get(sig.signature);
        if (mapping && mapping.signatureType === 'blockchain') {
          // Avoid duplicates
          const exists = mediaItems.some(item => item.fileId === mapping.fileId);
          if (!exists) {
            // Use fileId (hash) for download URL instead of fileName (which may be placeholder)
            const downloadUrl = `/api/pipe/download/${walletAddress}/${mapping.fileId}`;

            mediaItems.push({
              id: mapping.fileId,
              fileId: mapping.fileId,
              fileName: mapping.fileName,
              url: downloadUrl,
              type: mapping.fileType,
              cameraId: mapping.cameraId,
              uploadedAt: mapping.uploadedAt.toISOString(),
              txSignature: mapping.signature,
              signatureType: mapping.signatureType,
              provider: 'pipe'
            });
          }
        }
      }
    } catch (blockchainError) {
      console.log(`   Blockchain query skipped (may be offline):`, blockchainError);
    }

    // Sort by upload date (newest first)
    mediaItems.sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime());

    console.log(`✅ Found ${mediaItems.length} total media items for user`);
    console.log(`   Device-signed: ${mediaItems.filter(m => m.signatureType === 'device').length}`);
    console.log(`   Blockchain tx: ${mediaItems.filter(m => m.signatureType === 'blockchain').length}`);

    res.json({
      success: true,
      media: mediaItems,
      count: mediaItems.length
    });

  } catch (error) {
    console.error("❌ Failed to fetch gallery:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Failed to fetch gallery",
    });
  }
});

// ========================================
// GAS SPONSORSHIP ENDPOINTS
// ========================================

// Sponsor a transaction for a user
app.post("/api/sponsor-transaction", async (req, res) => {
  try {
    const { userWallet, transaction, action } = req.body;

    if (!userWallet || !transaction || !action) {
      return res.status(400).json({
        success: false,
        error: 'userWallet, transaction, and action are required'
      });
    }

    console.log(`📝 Sponsorship request from ${userWallet.slice(0, 8)}... for action: ${action}`);

    const result = await sponsorTransaction(userWallet, transaction, action);

    if (result.success) {
      res.json({
        success: true,
        transaction: result.transaction,
        remaining: result.remaining,
        message: `Transaction sponsored! ${result.remaining} free interactions remaining.`
      });
    } else {
      res.status(400).json({
        success: false,
        error: result.error,
        requiresUserPayment: result.error?.includes('used all')
      });
    }
  } catch (error) {
    console.error('Sponsor transaction error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to sponsor transaction'
    });
  }
});

// Check user's sponsorship status
app.get("/api/sponsorship-status/:userWallet", (req, res) => {
  try {
    const { userWallet } = req.params;

    if (!userWallet) {
      return res.status(400).json({ error: 'userWallet is required' });
    }

    const status = getUserSponsorshipStatus(userWallet);
    res.json(status);
  } catch (error) {
    console.error('Get sponsorship status error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to get sponsorship status'
    });
  }
});

// Get sponsorship statistics (for monitoring)
app.get("/api/sponsorship-stats", (_req, res) => {
  try {
    const stats = getSponsorshipStats();
    res.json(stats);
  } catch (error) {
    console.error('Get sponsorship stats error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to get sponsorship stats'
    });
  }
});

// Check fee payer balance
app.get("/api/fee-payer-balance", async (_req, res) => {
  try {
    const balance = await checkFeePayerBalance();
    res.json({
      balance,
      publicKey: process.env.FEE_PAYER_SECRET_KEY
        ? '9k5MGiM9Xqx8f2362M1B2rH5uMKFFVNuXaCDKyTsFXep'
        : 'Not configured'
    });
  } catch (error) {
    console.error('Check fee payer balance error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to check balance'
    });
  }
});

// Reset user sponsorship (for testing only - should be protected in production)
app.post("/api/reset-sponsorship/:userWallet", (req, res) => {
  try {
    const { userWallet } = req.params;
    const existed = resetUserSponsorship(userWallet);
    res.json({
      success: true,
      message: existed
        ? `Reset sponsorship for ${userWallet.slice(0, 8)}...`
        : `No sponsorship data found for ${userWallet.slice(0, 8)}...`
    });
  } catch (error) {
    console.error('Reset sponsorship error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to reset sponsorship'
    });
  }
});

// Clear all sponsorships (for testing only - should be protected in production)
app.post("/api/clear-all-sponsorships", (_req, res) => {
  try {
    const count = clearAllSponsorships();
    res.json({
      success: true,
      message: `Cleared sponsorship data for ${count} users`
    });
  } catch (error) {
    console.error('Clear sponsorships error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to clear sponsorships'
    });
  }
});

// ============================================================================
// SESSION CLEANUP CRON ENDPOINTS
// ============================================================================

// Get cleanup cron status
app.get("/api/cleanup-cron/status", (_req, res) => {
  try {
    const status = getCleanupCronStatus();
    res.json(status);
  } catch (error) {
    console.error('Get cron status error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to get cron status'
    });
  }
});

// Trigger manual cleanup (for testing)
app.post("/api/cleanup-cron/trigger", async (_req, res) => {
  try {
    await triggerManualCleanup();
    res.json({
      success: true,
      message: 'Manual cleanup triggered'
    });
  } catch (error) {
    console.error('Trigger manual cleanup error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to trigger cleanup'
    });
  }
});

// Stop cleanup cron
app.post("/api/cleanup-cron/stop", (_req, res) => {
  try {
    stopCleanupCron();
    res.json({
      success: true,
      message: 'Cleanup cron stopped'
    });
  } catch (error) {
    console.error('Stop cron error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to stop cron'
    });
  }
});

// Health check endpoint with connection test
app.get("/health", async (req, res) => {
  try {
    const status = {
      status: "healthy",
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV,
      headers: req.headers,
      secure: req.secure,
      protocol: req.protocol,
      host: req.headers.host,
      origin: req.headers.origin,
      userAgent: req.headers["user-agent"],
      socketConnections: io.engine.clientsCount,
      memoryUsage: process.memoryUsage(),
      uptime: process.uptime(),
    };

    res.json(status);
  } catch (error) {
    res.status(500).json({
      status: "error",
      error: error instanceof Error ? error.message : "Unknown error",
      timestamp: new Date().toISOString(),
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
  status: "pending" | "claimed" | "expired";
  devicePubkey?: string;
  deviceModel?: string;
}

const timelineEvents: TimelineEvent[] = [];
const cameraRooms = new Map<string, Set<string>>();
const pendingClaims = new Map<string, DeviceClaim>();

// Helper function to add timeline events (used by both Socket.IO handlers and cron bot)
function addTimelineEvent(event: Omit<TimelineEvent, "id">, socketServer: Server) {
  // Generate unique ID based on timestamp and random string
  const newEvent = {
    ...event,
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    timestamp: event.timestamp || Date.now(),
  };

  // Store the event
  timelineEvents.push(newEvent);

  // Keep only last 100 events per camera
  if (event.cameraId) {
    const cameraEvents = timelineEvents.filter(
      (e) => e.cameraId === event.cameraId,
    );
    if (cameraEvents.length > 100) {
      const oldestEventIndex = timelineEvents.findIndex(
        (e) => e.cameraId === event.cameraId,
      );
      if (oldestEventIndex !== -1) {
        timelineEvents.splice(oldestEventIndex, 1);
      }
    }
    // Broadcast only to sockets in this camera's room
    socketServer.to(event.cameraId).emit("timelineEvent", newEvent);
  } else {
    // If no cameraId, broadcast to all
    socketServer.emit("timelineEvent", newEvent);
  }

  return newEvent;
}

// Device claim endpoints for QR-based registration
// 1. Frontend creates a claim token with user wallet and WiFi credentials
app.post("/api/claim/create", (req, res) => {
  try {
    const { userWallet } = req.body;

    if (!userWallet) {
      return res.status(400).json({ error: "userWallet is required" });
    }

    // Generate secure random token
    const crypto = require("crypto");
    const claimToken = crypto.randomBytes(32).toString("hex");

    // Create claim record with 10 minute expiry
    const claimData: DeviceClaim = {
      userWallet,
      created: Date.now(),
      expires: Date.now() + 10 * 60 * 1000, // 10 minutes
      status: "pending",
    };

    pendingClaims.set(claimToken, claimData);

    console.log(`Created claim token ${claimToken} for wallet ${userWallet}`);

    res.json({
      claimToken,
      expiresAt: claimData.expires,
      claimEndpoint: `${req.protocol}://${req.get("host")}/api/claim/${claimToken}`,
    });
  } catch (error) {
    console.error("Error creating claim token:", error);
    res.status(500).json({ error: "Failed to create claim token" });
  }
});

// 2. Jetson device claims ownership using the token
app.post("/api/claim/:token", (req, res) => {
  try {
    const { token } = req.params;
    const { device_pubkey, device_model } = req.body;

    if (!device_pubkey) {
      return res.status(400).json({ error: "device_pubkey is required" });
    }

    const claim = pendingClaims.get(token);

    if (!claim) {
      return res.status(404).json({ error: "Claim token not found" });
    }

    if (claim.expires < Date.now()) {
      claim.status = "expired";
      return res.status(410).json({ error: "Claim token has expired" });
    }

    if (claim.status !== "pending") {
      return res.status(409).json({ error: "Claim token already used" });
    }

    // Update claim with device information
    claim.devicePubkey = device_pubkey;
    claim.deviceModel = device_model || "Unknown Device";
    claim.status = "claimed";

    console.log(
      `Device ${device_pubkey} claimed token ${token} for wallet ${claim.userWallet}`,
    );

    res.json({
      success: true,
      userWallet: claim.userWallet,
      message: "Device successfully claimed",
    });
  } catch (error) {
    console.error("Error processing device claim:", error);
    res.status(500).json({ error: "Failed to process device claim" });
  }
});

// 3. Frontend polls to check if device has been claimed
app.get("/api/claim/:token/status", (req, res) => {
  try {
    const { token } = req.params;
    const claim = pendingClaims.get(token);

    if (!claim) {
      return res.status(404).json({ status: "not_found" });
    }

    // Check if expired
    if (claim.expires < Date.now() && claim.status === "pending") {
      claim.status = "expired";
    }

    const response = {
      status: claim.status,
      created: claim.created,
      expires: claim.expires,
      devicePubkey: claim.devicePubkey,
      deviceModel: claim.deviceModel,
    };

    // Clean up expired or claimed tokens after 1 hour
    if (
      (claim.status === "claimed" || claim.status === "expired") &&
      Date.now() - claim.created > 60 * 60 * 1000
    ) {
      pendingClaims.delete(token);
    }

    res.json(response);
  } catch (error) {
    console.error("Error checking claim status:", error);
    res.status(500).json({ error: "Failed to check claim status" });
  }
});

// 4. Notify device of its assigned PDA for Cloudflare tunnel configuration
app.post("/api/claim/:token/assign-pda", (req, res) => {
  try {
    const { token } = req.params;
    const { camera_pda, transaction_id } = req.body;

    if (!camera_pda) {
      return res.status(400).json({ error: "camera_pda is required" });
    }

    const claim = pendingClaims.get(token);

    if (!claim) {
      return res.status(404).json({ error: "Claim token not found" });
    }

    if (claim.status !== "claimed") {
      return res
        .status(400)
        .json({ error: "Device has not claimed this token yet" });
    }

    // Store PDA assignment in claim for device to retrieve
    (claim as any).assignedPda = camera_pda;
    (claim as any).transactionId = transaction_id;
    (claim as any).pdaAssignedAt = Date.now();

    console.log(
      `Assigned PDA ${camera_pda} to device ${claim.devicePubkey} (token: ${token})`,
    );

    // In a real implementation, you might want to:
    // 1. Store this in a database
    // 2. Push notify the device via webhook/WebSocket
    // 3. Send to device management service

    res.json({
      success: true,
      message: "PDA assigned to device",
      camera_pda,
      subdomain: `${camera_pda.toLowerCase()}.mmoment.xyz`,
    });
  } catch (error) {
    console.error("Error assigning PDA to device:", error);
    res.status(500).json({ error: "Failed to assign PDA to device" });
  }
});

// 5. Device endpoint to retrieve assigned PDA configuration
app.get("/api/device/:device_pubkey/config", (req, res) => {
  try {
    const { device_pubkey } = req.params;

    // Find claim by device pubkey
    let foundClaim = null;

    for (const [, claim] of pendingClaims.entries()) {
      if (claim.devicePubkey === device_pubkey) {
        foundClaim = claim;
        break;
      }
    }

    if (!foundClaim) {
      return res.status(404).json({
        error: "No configuration found for this device",
        device_pubkey,
      });
    }

    const assignedPda = (foundClaim as any).assignedPda;

    if (!assignedPda) {
      return res.status(202).json({
        status: "pending",
        message: "Device claimed but PDA not yet assigned",
        device_pubkey,
        claimed_at: foundClaim.created,
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
        stream: `https://${assignedPda.toLowerCase()}.mmoment.xyz/api/camera/stream`,
      },
    };

    console.log(
      `Device ${device_pubkey} retrieved config for PDA ${assignedPda}`,
    );

    res.json(config);
  } catch (error) {
    console.error("Error retrieving device config:", error);
    res.status(500).json({ error: "Failed to retrieve device configuration" });
  }
});

// Debug endpoint to clear Pipe accounts (for testing)
app.post("/api/pipe/clear-accounts", (_req, res) => {
  console.log(`🚨 WARNING: CLEARING ALL PIPE ACCOUNTS!`);
  const clearedCount = pipeAccounts.size;
  pipeAccounts.clear();
  res.json({ message: `Cleared ${clearedCount} Pipe accounts` });
});

// 6. Cleanup endpoint to remove expired claims (optional, for debugging)
app.post("/api/claim/cleanup", (_req, res) => {
  const now = Date.now();
  let cleanedUp = 0;

  for (const [token, claim] of pendingClaims.entries()) {
    if (claim.expires < now || now - claim.created > 60 * 60 * 1000) {
      pendingClaims.delete(token);
      cleanedUp++;
    }
  }

  res.json({
    message: `Cleaned up ${cleanedUp} expired claims`,
    activeClaims: pendingClaims.size,
  });
});

// WebRTC Status endpoint for debugging
app.get("/api/webrtc/status", (_req, res) => {
  const peers = Array.from(webrtcPeers.entries()).map(([socketId, peer]) => ({
    socketId,
    ...peer,
  }));

  const cameras = peers.filter((p) => p.type === "camera");
  const viewers = peers.filter((p) => p.type === "viewer");

  res.json({
    totalPeers: peers.length,
    cameras: cameras.length,
    viewers: viewers.length,
    activeCameras: cameras.map((c) => c.cameraId),
    peers,
  });
});

// WebRTC signaling storage
const webrtcPeers = new Map<
  string,
  { cameraId?: string; type: "camera" | "viewer" }
>();

// Socket connection handling
io.on("connection", (socket) => {
  console.log("Client connected:", socket.id);

  // Handle joining a camera room
  socket.on("joinCamera", (cameraId: string) => {
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
      .filter((event) => event.cameraId === cameraId)
      .sort((a, b) => b.timestamp - a.timestamp);
    socket.emit("recentEvents", cameraEvents);
  });

  // Handle leaving a camera room
  socket.on("leaveCamera", (cameraId: string) => {
    console.log(`Socket ${socket.id} leaving camera ${cameraId}`);
    socket.leave(cameraId);
    cameraRooms.get(cameraId)?.delete(socket.id);
  });

  // Handle new timeline events
  socket.on("newTimelineEvent", (event: Omit<TimelineEvent, "id">) => {
    addTimelineEvent(event, io);
  });

  // Handle get recent events request
  socket.on("getRecentEvents", ({ cameraId }: { cameraId?: string }) => {
    console.log(
      `Socket ${socket.id} requesting recent events${cameraId ? ` for camera ${cameraId}` : ""}`,
    );
    const filteredEvents = cameraId
      ? timelineEvents.filter((event) => event.cameraId === cameraId)
      : timelineEvents;

    const sortedEvents = [...filteredEvents].sort(
      (a, b) => b.timestamp - a.timestamp,
    );
    socket.emit("recentEvents", sortedEvents);
  });

  // WebRTC Signaling Events
  socket.on("register-camera", (data: { cameraId: string }) => {
    console.log(
      `🎥 Camera ${data.cameraId} registering for WebRTC on socket ${socket.id}`,
    );
    webrtcPeers.set(socket.id, { cameraId: data.cameraId, type: "camera" });
    socket.join(`webrtc-${data.cameraId}`);

    // Debug room state after camera joins
    const roomSockets = io.sockets.adapter.rooms.get(`webrtc-${data.cameraId}`);
    console.log(
      `🎥 Camera joined room webrtc-${data.cameraId}, now has ${roomSockets ? roomSockets.size : 0} sockets`,
    );
  });

  socket.on(
    "register-viewer",
    (data: { cameraId: string; cellularMode?: boolean }) => {
      console.log(
        `Viewer registering for WebRTC camera ${data.cameraId} on socket ${socket.id}, cellular mode: ${data.cellularMode || false}`,
      );
      webrtcPeers.set(socket.id, { cameraId: data.cameraId, type: "viewer" });
      socket.join(`webrtc-${data.cameraId}`);

      // Check how many peers are in the room
      const roomSockets = io.sockets.adapter.rooms.get(
        `webrtc-${data.cameraId}`,
      );
      console.log(
        `Room webrtc-${data.cameraId} has ${roomSockets ? roomSockets.size : 0} sockets`,
      );

      // Notify camera that a viewer wants to connect with cellular mode flag
      console.log(
        `Notifying camera in room webrtc-${data.cameraId} that viewer ${socket.id} wants to connect (cellular: ${data.cellularMode || false})`,
      );
      socket.to(`webrtc-${data.cameraId}`).emit("viewer-wants-connection", {
        viewerId: socket.id,
        cellularMode: data.cellularMode || false,
      });
    },
  );

  socket.on(
    "webrtc-offer",
    (data: { cameraId: string; offer: any; targetId?: string }) => {
      console.log(`WebRTC offer for camera ${data.cameraId}`);
      if (data.targetId) {
        // Direct offer to specific peer
        socket.to(data.targetId).emit("webrtc-offer", {
          offer: data.offer,
          senderId: socket.id,
          cameraId: data.cameraId,
        });
      } else {
        // Broadcast offer to camera room
        socket.to(`webrtc-${data.cameraId}`).emit("webrtc-offer", {
          offer: data.offer,
          senderId: socket.id,
          cameraId: data.cameraId,
        });
      }
    },
  );

  socket.on(
    "webrtc-answer",
    (data: { answer: any; targetId: string; cameraId: string }) => {
      console.log(`WebRTC answer for camera ${data.cameraId}`);
      socket.to(data.targetId).emit("webrtc-answer", {
        answer: data.answer,
        senderId: socket.id,
        cameraId: data.cameraId,
      });
    },
  );

  socket.on(
    "webrtc-ice-candidate",
    (data: { candidate: any; targetId: string; cameraId: string }) => {
      console.log(`WebRTC ICE candidate for camera ${data.cameraId}`);
      socket.to(data.targetId).emit("webrtc-ice-candidate", {
        candidate: data.candidate,
        senderId: socket.id,
        cameraId: data.cameraId,
      });
    },
  );

  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.id);
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
io.engine.on("connection_error", (err) => {
  console.error("Socket.IO connection error:", err);
});

// Prevent server crashes
process.on("uncaughtException", (error) => {
  console.error("❌ Uncaught Exception:", error);
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("❌ Unhandled Rejection at:", promise, "reason:", reason);
});

// Enhanced CoTURN-compatible server implementation for Railway
const TURN_PORT = Number(process.env.PORT) || 8080; // Use Railway's exposed port
const TURN_USERNAME = process.env.TURN_USERNAME || "mmoment";
const TURN_PASSWORD = process.env.TURN_PASSWORD || "webrtc123";
const TURN_REALM = process.env.TURN_REALM || "mmoment.xyz";

// Get external IP for Railway deployment
const EXTERNAL_IP =
  process.env.RAILWAY_PUBLIC_DOMAIN || "mmoment-production.up.railway.app";

// Store active TURN allocations
const turnAllocations = new Map<
  string,
  {
    clientAddress: string;
    clientPort: number;
    relayPort: number;
    relaySocket: dgram.Socket;
    peers: Map<string, { address: string; port: number }>;
  }
>();

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
function createStunResponse(
  type: number,
  transactionId: Buffer,
  attributes: Buffer[] = [],
): Buffer {
  const totalAttrLength = attributes.reduce(
    (sum, attr) => sum + attr.length,
    0,
  );
  const response = Buffer.alloc(20 + totalAttrLength);

  response.writeUInt16BE(type, 0); // Message type
  response.writeUInt16BE(totalAttrLength, 2); // Message length
  response.writeUInt32BE(0x2112a442, 4); // Magic cookie
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
  const ipParts = address.split(".").map(Number);
  const magicCookie = [0x21, 0x12, 0xa4, 0x42];
  for (let i = 0; i < 4; i++) {
    attr.writeUInt8(ipParts[i] ^ magicCookie[i], 8 + i);
  }

  return attr;
}

turnServer.on("connection", (socket) => {
  console.log(
    `TCP TURN connection from ${socket.remoteAddress}:${socket.remotePort}`,
  );

  socket.on("data", (data) => {
    try {
      const stunMsg = parseStunMessage(data);
      if (!stunMsg) return;

      const { messageType, transactionId } = stunMsg;
      const clientAddress = socket.remoteAddress || "127.0.0.1";
      const clientPort = socket.remotePort || 0;

      // Handle STUN Binding Request (0x0001)
      if (messageType === 0x0001) {
        console.log(`STUN Binding request from ${clientAddress}:${clientPort}`);

        // Create XOR-MAPPED-ADDRESS attribute
        const xorMappedAddr = createXorMappedAddress(clientAddress, clientPort);

        // Send STUN Binding Response (0x0101)
        const response = createStunResponse(0x0101, transactionId, [
          xorMappedAddr,
        ]);
        socket.write(response);

        console.log(
          `STUN Binding response sent to ${clientAddress}:${clientPort}`,
        );
      }
      // Handle TURN Allocate Request (0x0003)
      else if (messageType === 0x0003) {
        console.log(
          `TURN: Allocation request from ${clientAddress}:${clientPort}`,
        );

        // Create a relay socket for this client
        const relaySocket = dgram.createSocket("udp4");
        let relayPort = 40000 + Math.floor(Math.random() * 10000);

        relaySocket.bind(relayPort, () => {
          const allocationId = `${clientAddress}:${clientPort}`;

          // Store allocation
          turnAllocations.set(allocationId, {
            clientAddress: clientAddress,
            clientPort: clientPort,
            relayPort: relayPort,
            relaySocket: relaySocket,
            peers: new Map(),
          });

          // Handle relay traffic
          relaySocket.on("message", (relayMsg, relayRinfo) => {
            // Forward relay traffic back to the client
            socket.write(relayMsg);
          });

          // Create XOR-RELAYED-ADDRESS attribute
          const relayAddr = createXorMappedAddress("0.0.0.0", relayPort);
          relayAddr.writeUInt16BE(0x0016, 0); // Change type to XOR-RELAYED-ADDRESS

          // Create XOR-MAPPED-ADDRESS for the client
          const mappedAddr = createXorMappedAddress(clientAddress, clientPort);

          // Create LIFETIME attribute (600 seconds)
          const lifetime = Buffer.alloc(8);
          lifetime.writeUInt16BE(0x000d, 0); // LIFETIME
          lifetime.writeUInt16BE(4, 2); // Length
          lifetime.writeUInt32BE(600, 4); // 600 seconds

          // Send Allocate Success Response (0x0103)
          const response = createStunResponse(0x0103, transactionId, [
            relayAddr,
            mappedAddr,
            lifetime,
          ]);
          socket.write(response);

          console.log(
            `TURN: Allocated relay port ${relayPort} for ${allocationId}`,
          );
        });

        relaySocket.on("error", (err) => {
          console.error(`TURN relay socket error:`, err);
        });
      }
      // Handle TURN Refresh Request (0x0004)
      else if (messageType === 0x0004) {
        console.log(`TURN Refresh request from ${clientAddress}:${clientPort}`);

        // Send Refresh Success Response (0x0104)
        const lifetime = Buffer.alloc(8);
        lifetime.writeUInt16BE(0x000d, 0); // LIFETIME
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
          allocation.relaySocket.send(
            data,
            0,
            data.length,
            allocation.relayPort,
            "localhost",
          );
        }
      }
      // Handle other TURN messages
      else {
        console.log(
          `Unknown STUN/TURN message type: 0x${messageType.toString(16)} from ${clientAddress}:${clientPort}`,
        );
      }
    } catch (error) {
      console.error("TCP TURN server error:", error);
    }
  });

  socket.on("close", () => {
    console.log(
      `TCP TURN connection closed from ${socket.remoteAddress}:${socket.remotePort}`,
    );
  });

  socket.on("error", (err) => {
    console.error("TCP TURN socket error:", err);
  });
});

turnServer.on("error", (err) => {
  console.error("TURN server error:", err);
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
console.log(
  "⚠️ TURN server disabled - Railway only supports HTTP on port 8080",
);
console.log("⚠️ Use external TURN server for cross-network WebRTC");

// TURN server info endpoint
app.get("/api/turn/info", (req, res) => {
  const turnInfo = {
    host: req.headers.host?.split(":")[0] || "localhost",
    port: TURN_PORT,
    username: TURN_USERNAME,
    credential: TURN_PASSWORD,
    urls: [
      `turn:${req.headers.host?.split(":")[0] || "localhost"}:${TURN_PORT}`,
      `turn:${req.headers.host?.split(":")[0] || "localhost"}:${TURN_PORT}?transport=udp`,
    ],
  };

  res.json(turnInfo);
});

// Start HTTP server with error handling
const port = Number(process.env.PORT) || 3001;
httpServer.listen(port, "0.0.0.0", () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV}`);
  console.log("Server configuration:", {
    keepAliveTimeout: httpServer.keepAliveTimeout,
    headersTimeout: httpServer.headersTimeout,
    socketTransports: io.engine.opts.transports,
    socketPingTimeout: io.engine.opts.pingTimeout,
    socketPingInterval: io.engine.opts.pingInterval,
    turnPort: TURN_PORT,
    turnUsername: TURN_USERNAME,
  });

  // Initialize Gas Sponsorship Service and Session Cleanup Cron after server starts
  if (process.env.FEE_PAYER_SECRET_KEY && process.env.SOLANA_RPC_URL) {
    initializeGasSponsorshipService(
      process.env.SOLANA_RPC_URL,
      process.env.FEE_PAYER_SECRET_KEY
    );

    // Initialize Session Cleanup Cron with Socket.IO server and timeline event handler
    initializeSessionCleanupCron(
      process.env.SOLANA_RPC_URL,
      process.env.FEE_PAYER_SECRET_KEY,
      io,  // Pass Socket.IO server for timeline event emissions
      addTimelineEvent  // Pass helper function to add timeline events
    );
  } else {
    console.warn('⚠️  Gas sponsorship and cleanup cron not configured - missing FEE_PAYER_SECRET_KEY or SOLANA_RPC_URL');
  }
});
