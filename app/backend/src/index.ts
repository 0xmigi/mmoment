// backend/src/index.ts
import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";
import { config } from "dotenv";
import { PipeClient, generateCredentialsFromAddress, PipeAccount, UploadResult, FileRecord, Balance, PublicLink } from "firestarter-sdk";
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
  getCleanupCronStatus,
  queueAccessKeyForUser
} from './session-cleanup-cron';

// Import database module
import {
  initializeDatabase,
  closeDatabase,
  saveFileMapping,
  getFileMappingBySignature,
  getSignaturesForWallet,
  getFileMappingsForWallet,
  deleteFileMappingByFileName,
  loadAllFileMappingsToMaps,
  getDatabaseStats,
  FileMapping as DBFileMapping,
  saveUserProfile,
  getUserProfile,
  getUserProfiles,
  loadAllUserProfilesToMap,
  deleteUserProfile,
  UserProfile as DBUserProfile,
  saveSessionActivity,
  getSessionActivities,
  clearSessionActivities,
  getSessionBufferStats,
  getUserSessions,
  getCameraActivities,
  getUserActivities,
  getSessionTimelineEvents,
  SessionActivityBuffer,
  SessionSummary,
  SessionTimelineEvent
} from './database';

// Note: Socket.IO server (io) will be passed to session cleanup cron after it's created below

// Initialize the Firestarter SDK (new PipeClient API)
const pipeClient = new PipeClient({
  baseUrl: "https://us-west-01-firestarter.pipenetwork.com", // Mainnet
});

// Cache for PipeAccount objects (wallet address -> PipeAccount)
const pipeAccountCache = new Map<string, PipeAccount>();

// Helper function to get or create PipeAccount for a wallet
async function getPipeAccountForWallet(walletAddress: string): Promise<PipeAccount> {
  // Check cache first
  const cached = pipeAccountCache.get(walletAddress);
  if (cached) {
    return cached;
  }

  // Generate deterministic credentials from wallet address
  const credentials = generateCredentialsFromAddress(walletAddress);

  // Try to login first, create if doesn't exist
  let account: PipeAccount;
  try {
    account = await pipeClient.login(credentials.username, credentials.password);
    console.log(`✅ Logged in to Pipe account for wallet ${walletAddress.slice(0, 8)}...`);
  } catch (error) {
    console.log(`📝 Creating new Pipe account for wallet ${walletAddress.slice(0, 8)}...`);
    account = await pipeClient.createAccount(credentials.username, credentials.password);
    console.log(`✅ Created new Pipe account for wallet ${walletAddress.slice(0, 8)}...`);
  }

  // Cache the account
  pipeAccountCache.set(walletAddress, account);
  return account;
}

// Option A vs Option B configuration
const USE_SHARED_PIPE_ACCOUNT = process.env.USE_SHARED_PIPE_ACCOUNT === 'true';
const SHARED_PIPE_USER_ID = process.env.SHARED_PIPE_USER_ID || '';
const SHARED_PIPE_USER_APP_KEY = process.env.SHARED_PIPE_USER_APP_KEY || '';

console.log(`🔧 Pipe Storage Mode: ${USE_SHARED_PIPE_ACCOUNT ? 'OPTION A (Shared Account)' : 'OPTION B (Per-User Accounts)'}`);

if (USE_SHARED_PIPE_ACCOUNT) {
  if (!SHARED_PIPE_USER_ID || !SHARED_PIPE_USER_APP_KEY) {
    console.error('❌ SHARED_PIPE_USER_ID and SHARED_PIPE_USER_APP_KEY must be set when USE_SHARED_PIPE_ACCOUNT=true');
    process.exit(1);
  }
  console.log(`✅ Shared Pipe Account configured: ${SHARED_PIPE_USER_ID.slice(0, 8)}...`);
}

// In-memory storage for pipe accounts (simple key-value store)
// Only used in Option B mode
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

// Pending login promise to prevent concurrent login attempts (causes 429 rate limits)
let pendingLoginPromise: Promise<{ access_token: string; user_id: string }> | null = null;

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

  // If a login is already in progress, wait for it instead of starting another one
  if (pendingLoginPromise) {
    console.log(`⏳ Login already in progress, waiting...`);
    return pendingLoginPromise;
  }

  // Start a new login and store the promise so concurrent requests can wait on it
  pendingLoginPromise = (async () => {
    try {
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
        expires_at: Date.now() + 60 * 60 * 1000 // 1 hour from now
      };

      console.log(`✅ Fresh Pipe token cached (user: ${walletData.user_id.slice(0, 20)}...)`);

      return {
        access_token: tokens.access_token,
        user_id: walletData.user_id
      };
    } finally {
      // Clear the pending promise so future requests can start a new login if needed
      pendingLoginPromise = null;
    }
  })();

  return pendingLoginPromise;
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

// In-memory caches (backed by SQLite database)
const signatureToFileMapping = new Map<string, FileMapping>();
const walletToSignatures = new Map<string, string[]>();
const userProfilesCache = new Map<string, DBUserProfile>(); // wallet_address -> profile

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
    // Option A: Return shared account credentials
    if (USE_SHARED_PIPE_ACCOUNT) {
      console.log(
        `📦 Returning shared Pipe credentials for wallet: ${walletAddress.slice(0, 8)}...`,
      );
      res.json({
        userId: SHARED_PIPE_USER_ID,
        userAppKey: SHARED_PIPE_USER_APP_KEY,
      });
      return;
    }

    // Option B: Check if we have existing Pipe account for this wallet
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

    // Get or create Pipe account for this wallet
    const pipeAccount = await getPipeAccountForWallet(walletAddress);
    console.log(`👤 Pipe Account:`, {
      username: pipeAccount.username,
      userId: pipeAccount.userId,
    });

    // Check if we have stored credentials
    const account = pipeAccounts.get(walletAddress);
    console.log(`💾 Stored Account:`, account);

    res.json({
      walletAddress,
      pipeAccount: {
        username: pipeAccount.username,
        userId: pipeAccount.userId,
      },
      storedAccount: account,
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
      // Get balance using new SDK
      const account = await getPipeAccountForWallet(walletAddress);
      const balance = await pipeClient.getBalance(account);
      res.json({
        balance_sol: balance.sol,
        public_key: balance.publicKey,
      });
      return;
    }

    if (endpoint === "checkCustomToken") {
      // Get PIPE token balance using new SDK
      const account = await getPipeAccountForWallet(walletAddress);
      const balance = await pipeClient.getBalance(account);
      res.json({
        balance: balance.pipe * 1000000, // Convert to raw units
        ui_amount: balance.pipe,
      });
      return;
    }

    if (endpoint === "exchangeSolForTokens") {
      // Exchange SOL for PIPE using new SDK
      const { amount_sol } = req.body;
      if (!amount_sol) {
        return res.status(400).json({ error: "amount_sol is required" });
      }

      const account = await getPipeAccountForWallet(walletAddress);
      const tokensReceived = await pipeClient.exchangeSolForPipe(
        account,
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

// Create account endpoint - supports both Option A (shared) and Option B (per-user)
app.post("/api/pipe/create-account", async (req, res) => {
  const { walletAddress } = req.body;

  if (!walletAddress) {
    return res.status(400).json({ error: "walletAddress is required" });
  }

  try {
    // OPTION A: Shared account mode - return same credentials for everyone
    if (USE_SHARED_PIPE_ACCOUNT) {
      console.log(
        `✅ [Option A] Returning shared Pipe account for ${walletAddress.slice(0, 8)}...`,
      );
      res.json({
        userId: SHARED_PIPE_USER_ID,
        userAppKey: SHARED_PIPE_USER_APP_KEY,
        existing: true,
        mode: 'shared',
      });
      return;
    }

    // OPTION B: Per-user account mode - create separate account per wallet
    // Check if account already exists locally
    const existingAccount = pipeAccounts.get(walletAddress);
    if (existingAccount) {
      console.log(
        `✅ [Option B] Existing Pipe account found for ${walletAddress.slice(0, 8)}...`,
      );
      res.json({
        userId: existingAccount.userId,
        userAppKey: existingAccount.userAppKey,
        existing: true,
        mode: 'per-user',
      });
      return;
    }

    console.log(
      `🔄 [Option B] Creating new Pipe account for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Use the new SDK to get or create the user account
    const pipeAccount = await getPipeAccountForWallet(walletAddress);

    // Store account info locally for quick access (for backward compatibility)
    const accountInfo = {
      userId: pipeAccount.userId,
      userAppKey: pipeAccount.userAppKey || "",
      walletAddress,
      created: new Date(),
    };

    pipeAccounts.set(walletAddress, accountInfo);

    console.log(`✅ [Option B] Pipe account ready for ${walletAddress.slice(0, 8)}...`);

    res.json({
      userId: accountInfo.userId,
      userAppKey: accountInfo.userAppKey,
      existing: false,
      mode: 'per-user',
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

    // Get the Pipe account for this wallet
    const account = await getPipeAccountForWallet(walletAddress);

    // Use the new SDK to upload the file
    const uploadResult = await pipeClient.uploadFile(
      account,
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

    // Get the Pipe account for this wallet
    const account = await getPipeAccountForWallet(walletAddress);

    // Note: The new SDK's listFiles() returns empty array (API limitation)
    // We need to use local file tracking or backend mapping instead
    const fileRecords = await pipeClient.listFiles(account);
    console.log(
      `📁 SDK returned ${fileRecords.length} files (API may not support listing yet)`,
    );

    if (fileRecords.length === 0) {
      // Return empty array if no files
      // TODO: Consider implementing local file tracking using PipeFileStorage
      return res.json({
        files: [],
        count: 0,
        walletAddress: walletAddress.slice(0, 8) + "...",
      });
    }

    // For JWT-based downloads, we need to use the SDK's download method instead of direct URLs
    // Transform SDK FileRecord format to frontend PipeFile format
    const files = fileRecords.map((record: any) => ({
      id: record.fileId,
      name: record.fileName,
      size: record.size,
      contentType: record.mimeType || "application/octet-stream",
      uploadedAt: record.uploadedAt
        ? record.uploadedAt instanceof Date
          ? record.uploadedAt.toISOString()
          : new Date(record.uploadedAt).toISOString()
        : new Date().toISOString(),
      // Create a backend proxy URL for downloads since we need JWT auth
      // Use fileName (not fileId/hash) because Pipe downloads by original filename
      url: `/api/pipe/download/${walletAddress}/${encodeURIComponent(record.fileName)}`,
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

// Download endpoint - uses shared Pipe account (JWT auth) since all files are stored there
app.get("/api/pipe/download/:walletAddress/:fileId", async (req, res) => {
  const { walletAddress, fileId } = req.params;

  if (!walletAddress || !fileId) {
    return res
      .status(400)
      .json({ error: "Wallet address and file ID are required" });
  }

  const fileName = decodeURIComponent(fileId);
  console.log(
    `📥 Downloading ${fileName.slice(0, 30)}... for wallet: ${walletAddress.slice(0, 8)}...`,
  );

  // Determine content type from file extension
  let contentType = "application/octet-stream";
  const fileNameLower = fileName.toLowerCase();
  if (fileNameLower.includes(".mp4") || fileNameLower.includes(".mov")) {
    contentType = "video/mp4";
  } else if (fileNameLower.includes(".jpg") || fileNameLower.includes(".jpeg")) {
    contentType = "image/jpeg";
  } else if (fileNameLower.includes(".png")) {
    contentType = "image/png";
  } else if (fileNameLower.includes(".webp")) {
    contentType = "image/webp";
  }

  // Primary method: Use shared account JWT (where all Jetson uploads go)
  try {
    console.log("📥 Downloading from shared Pipe account (JWT)...");
    const { access_token } = await getPipeJWTToken();
    const baseUrl = process.env.PIPE_BASE_URL || 'https://us-west-01-firestarter.pipenetwork.com';

    const downloadUrl = new URL(`${baseUrl}/download-stream`);
    downloadUrl.searchParams.append("file_name", fileName);

    const pipeResponse = await axios.get(downloadUrl.toString(), {
      headers: {
        "Authorization": `Bearer ${access_token}`,
      },
      responseType: "arraybuffer",
      timeout: 60000, // 1 minute timeout
    });

    const responseData = Buffer.from(pipeResponse.data);

    // Parse multipart response if needed
    const firstLineEnd = responseData.indexOf('\n'.charCodeAt(0));
    if (firstLineEnd !== -1) {
      const boundary = responseData.slice(0, firstLineEnd).toString('utf8').trim();
      if (boundary.startsWith('--')) {
        // Find file content in multipart (after headers)
        const separator = Buffer.from('\r\n\r\n', 'utf8');
        const headerEnd = responseData.indexOf(separator);
        if (headerEnd !== -1) {
          let fileContent = responseData.slice(headerEnd + 4);
          // Find end boundary - must be preceded by \r\n to avoid false matches in binary data
          const endBoundaryPattern = Buffer.from('\r\n' + boundary, 'utf8');
          const endBoundary = fileContent.indexOf(endBoundaryPattern);
          if (endBoundary !== -1) {
            fileContent = fileContent.slice(0, endBoundary);
          }
          console.log(`✅ Downloaded ${fileContent.length} bytes from shared account (multipart)`);
          res.setHeader("Content-Type", contentType);
          res.setHeader("Content-Length", fileContent.length);
          res.setHeader("Cache-Control", "public, max-age=31536000");
          return res.send(fileContent);
        }
      }
    }

    // Non-multipart response - send directly
    console.log(`✅ Downloaded ${responseData.length} bytes from shared account`);
    res.setHeader("Content-Type", contentType);
    res.setHeader("Content-Length", responseData.length);
    res.setHeader("Cache-Control", "public, max-age=31536000");
    return res.send(responseData);

  } catch (error) {
    console.error("❌ Shared account download failed:", error);

    // Fallback: Try per-user account via SDK (for future per-user uploads)
    try {
      console.log("🔄 Trying fallback download from per-user account (SDK)...");
      const account = await getPipeAccountForWallet(walletAddress);
      const fileData = await pipeClient.downloadFile(account, fileName);
      const buffer = Buffer.from(fileData);

      console.log(`✅ Downloaded ${buffer.length} bytes via SDK (per-user account)`);
      res.setHeader("Content-Type", contentType);
      res.setHeader("Content-Length", buffer.length);
      res.setHeader("Cache-Control", "public, max-age=31536000");
      return res.send(buffer);
    } catch (sdkError) {
      console.error("❌ SDK fallback also failed:", sdkError);
      if (!res.headersSent) {
        res.status(500).json({
          error: "Download failed from both shared and per-user accounts",
        });
      }
    }
  }
});

// Create public share link endpoint
app.post("/api/pipe/share/:walletAddress/:fileId", async (req, res) => {
  const { walletAddress, fileId } = req.params;
  const { title, description } = req.body;

  if (!walletAddress || !fileId) {
    return res
      .status(400)
      .json({ error: "Wallet address and file ID are required" });
  }

  try {
    console.log(
      `🔗 Creating public share link for ${fileId} from wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Get the Pipe account for this wallet
    const account = await getPipeAccountForWallet(walletAddress);

    // Decode the fileId (it might be URL encoded)
    const fileName = decodeURIComponent(fileId);

    // Use the new SDK to create a public link
    const publicLink = await pipeClient.createPublicLink(account, fileName, {
      customTitle: title,
      customDescription: description,
    });

    console.log(
      `✅ Created public share link for ${fileName}: ${publicLink.shareUrl}`,
    );

    res.json({
      success: true,
      linkHash: publicLink.linkHash,
      shareUrl: publicLink.shareUrl,
      fileName: publicLink.fileName,
    });
  } catch (error) {
    console.error("❌ Create share link failed:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Create share link failed",
    });
  }
});

// Delete public share link endpoint
app.delete("/api/pipe/share/:walletAddress/:linkHash", async (req, res) => {
  const { walletAddress, linkHash } = req.params;

  if (!walletAddress || !linkHash) {
    return res
      .status(400)
      .json({ error: "Wallet address and link hash are required" });
  }

  try {
    console.log(
      `🗑️  Deleting public share link ${linkHash} for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Get the Pipe account for this wallet
    const account = await getPipeAccountForWallet(walletAddress);

    // Use the new SDK to delete the public link
    await pipeClient.deletePublicLink(account, linkHash);

    console.log(
      `✅ Successfully deleted public link ${linkHash}`,
    );

    res.json({
      success: true,
      message: "Public link deleted successfully",
      linkHash,
    });
  } catch (error) {
    console.error("❌ Delete public link failed:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Delete public link failed",
    });
  }
});

// Delete file endpoint
app.delete("/api/pipe/delete/:walletAddress/:fileId", async (req, res) => {
  const { walletAddress, fileId } = req.params;

  if (!walletAddress || !fileId) {
    return res
      .status(400)
      .json({ error: "Wallet address and file ID are required" });
  }

  try {
    console.log(
      `🗑️  Deleting file ${fileId} for wallet: ${walletAddress.slice(0, 8)}...`,
    );

    // Get the Pipe account for this wallet
    const account = await getPipeAccountForWallet(walletAddress);

    // Decode the fileId (it might be URL encoded)
    const fileName = decodeURIComponent(fileId);

    // Use the new SDK to delete the file from Pipe storage
    await pipeClient.deleteFile(account, fileName);

    // Also delete from signature mapping (in-memory and database)
    // Find all signatures for this wallet that map to this fileName
    for (const [signature, mapping] of signatureToFileMapping.entries()) {
      if (mapping.fileName === fileName && mapping.walletAddress === walletAddress) {
        signatureToFileMapping.delete(signature);
        console.log(`🗑️  Removed in-memory signature mapping: ${signature.slice(0, 8)}...`);
      }
    }

    // Delete from database
    try {
      const deletedCount = await deleteFileMappingByFileName(fileName, walletAddress);
      console.log(`💾 Deleted ${deletedCount} file mapping(s) from database`);
    } catch (dbError) {
      console.error('Failed to delete file mapping from database:', dbError);
    }

    console.log(
      `✅ Successfully deleted ${fileName} for ${walletAddress.slice(0, 8)}...`,
    );

    res.json({
      success: true,
      message: "File deleted successfully",
      fileName,
    });
  } catch (error) {
    console.error("❌ Delete failed:", error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Delete failed",
    });
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

// Get Pipe account status (balance, storage usage, etc.)
app.get("/api/pipe/account/status", async (req, res) => {
  try {
    console.log('📊 Fetching Pipe account status');

    // Get JWT token
    const tokenData = await getPipeJWTToken();

    // Fetch account balance
    const balanceResponse = await fetch('https://us-west-01-firestarter.pipenetwork.com/checkCustomToken', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokenData.access_token}`
      },
      body: JSON.stringify({
        user_id: tokenData.user_id
      })
    });

    if (!balanceResponse.ok) {
      throw new Error(`Balance check failed: ${balanceResponse.status}`);
    }

    const balanceData = await balanceResponse.json();
    console.log('💰 Pipe balance response:', JSON.stringify(balanceData, null, 2));

    // Fetch file list to calculate storage usage
    const filesResponse = await fetch('https://us-west-01-firestarter.pipenetwork.com/list', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokenData.access_token}`
      },
      body: JSON.stringify({
        user_id: tokenData.user_id
      })
    });

    let totalSize = 0;
    let fileCount = 0;

    if (filesResponse.ok) {
      const filesData = await filesResponse.json();
      if (Array.isArray(filesData.files)) {
        fileCount = filesData.files.length;
        totalSize = filesData.files.reduce((sum: number, file: any) => sum + (file.size || 0), 0);
      }
    }

    res.json({
      success: true,
      data: {
        username: process.env.PIPE_USERNAME,
        userId: process.env.PIPE_USER_ID,
        depositAddress: process.env.PIPE_DEPOSIT_ADDRESS,
        pipeBalance: parseFloat(balanceData.balance || '0'),
        solBalance: parseFloat(balanceData.sol_balance || '0'),
        fileCount,
        storageUsedBytes: totalSize,
        storageUsedMB: (totalSize / (1024 * 1024)).toFixed(2)
      }
    });
  } catch (error: any) {
    console.error('❌ Error fetching Pipe account status:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to fetch account status'
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

    // Save to database for persistence
    try {
      await saveFileMapping(mapping);
      console.log(`💾 Saved file mapping to database`);
    } catch (dbError) {
      console.error('Failed to save file mapping to database:', dbError);
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

    // OPTION A: Shared account mode - use SDK file listing and filter by wallet prefix
    if (USE_SHARED_PIPE_ACCOUNT) {
      console.log(`   [Option A] Fetching from shared Pipe account via SDK...`);

      try {
        // Get the Pipe account for this wallet
        const account = await getPipeAccountForWallet(walletAddress);

        // Get ALL files from shared account
        const allFiles = await pipeClient.listFiles(account);
        console.log(`   Found ${allFiles.length} total files (API may return empty)`);

        // Filter by wallet prefix in filename (e.g., "RsLjCiEi_photo_...")
        const walletPrefix = walletAddress.slice(0, 8);
        const userFiles = allFiles.filter((file: any) =>
          file.fileName.startsWith(walletPrefix)
        );

        console.log(`   Filtered to ${userFiles.length} files for ${walletPrefix}...`);

        // Convert SDK file records to media items
        for (const file of userFiles) {
          const downloadUrl = `/api/pipe/download/${walletAddress}/${encodeURIComponent(file.fileName)}`;

          mediaItems.push({
            id: file.fileId,
            fileId: file.fileId,
            fileName: file.fileName,
            url: downloadUrl,
            type: file.fileName.includes('video') ? 'video' : 'image',
            cameraId: file.metadata?.cameraId || 'unknown',
            uploadedAt: new Date(file.uploadedAt).toISOString(),
            txSignature: undefined,
            signatureType: 'shared-account',
            provider: 'pipe'
          });
        }
      } catch (sdkError) {
        console.error(`   SDK file listing failed:`, sdkError);
        // Fall through to legacy method below
      }
    }

    // OPTION B / FALLBACK: Use backend mappings (device signatures + blockchain)
    if (!USE_SHARED_PIPE_ACCOUNT || mediaItems.length === 0) {
      console.log(`   [Option B/Fallback] Using backend signature mappings...`);

      // Method 1: Get device-signed captures (instant captures, no blockchain tx)
      const deviceSignatures = walletToSignatures.get(walletAddress) || [];
      console.log(`   Device-signed captures: ${deviceSignatures.length}`);

      for (const sig of deviceSignatures) {
        const mapping = signatureToFileMapping.get(sig);
        if (mapping) {
          // Use fileName for download URL (original filename from upload)
          const downloadUrl = `/api/pipe/download/${walletAddress}/${encodeURIComponent(mapping.fileName)}`;

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
    }

    // Sort by upload date (newest first)
    mediaItems.sort((a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime());

    console.log(`✅ Found ${mediaItems.length} total media items for user`);
    if (!USE_SHARED_PIPE_ACCOUNT) {
      console.log(`   Device-signed: ${mediaItems.filter(m => m.signatureType === 'device').length}`);
      console.log(`   Blockchain tx: ${mediaItems.filter(m => m.signatureType === 'blockchain').length}`);
    }

    res.json({
      success: true,
      media: mediaItems,
      count: mediaItems.length,
      mode: USE_SHARED_PIPE_ACCOUNT ? 'shared' : 'per-user'
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

// Timeline events storage (backed by SQLite database)
interface TimelineEvent {
  id: string;
  type: string;
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
    provider?: string;
  };
  timestamp: number;
  cameraId?: string;
  transactionId?: string;
}

// In-memory cache of timeline events
const timelineEvents: TimelineEvent[] = [];

// Device claim storage (in-memory)
interface DeviceClaim {
  userWallet: string;
  created: number;
  expires: number;
  status: "pending" | "claimed" | "expired";
  devicePubkey?: string;
  deviceModel?: string;
}

const cameraRooms = new Map<string, Set<string>>();
const pendingClaims = new Map<string, DeviceClaim>();

// Helper function to add timeline events (used by both Socket.IO handlers and cron bot)
// Now saves to database for persistence and enriches with user profiles
async function addTimelineEvent(event: Omit<TimelineEvent, "id">, socketServer: Server) {
  // Defensive logging before any property access
  console.log(`🔍 addTimelineEvent CALLED`, {
    hasEvent: !!event,
    type: event?.type,
    hasUser: !!event?.user,
    hasAddress: !!event?.user?.address
  });

  // Validate event structure
  if (!event || !event.user || !event.user.address) {
    console.error(`❌ Invalid event data received:`, {
      hasEvent: !!event,
      hasUser: !!event?.user,
      hasAddress: !!event?.user?.address,
      eventPreview: JSON.stringify(event).slice(0, 300)
    });
    return;
  }

  console.log(`🔍 addTimelineEvent START for ${event.type} from ${event.user.address.slice(0, 8)}`);
  // Fetch user profile from database if available
  let enrichedUser = { ...event.user };

  try {
    // Check if incoming event has profile data to save
    const incomingUser = event.user as any;
    if (incomingUser.displayName || incomingUser.username || incomingUser.pfpUrl || incomingUser.provider) {
      // Save or update the user profile in database
      try {
        await saveUserProfile({
          walletAddress: event.user.address,
          displayName: incomingUser.displayName,
          username: incomingUser.username,
          profileImage: incomingUser.pfpUrl,
          provider: incomingUser.provider,
          lastUpdated: new Date()
        });

        // Update cache
        userProfilesCache.set(event.user.address, {
          walletAddress: event.user.address,
          displayName: incomingUser.displayName,
          username: incomingUser.username,
          profileImage: incomingUser.pfpUrl,
          provider: incomingUser.provider,
          lastUpdated: new Date()
        });

        console.log(`💾 Saved user profile for ${event.user.address.slice(0, 8)}...`, {
          displayName: incomingUser.displayName,
          username: incomingUser.username,
          provider: incomingUser.provider,
          pfpUrl: incomingUser.pfpUrl ? 'present' : 'missing'
        });
      } catch (saveError) {
        console.error('Failed to save user profile:', saveError);
      }
    }

    // Check cache first
    let profile = userProfilesCache.get(event.user.address);

    // If not in cache, fetch from database
    if (!profile) {
      const fetchedProfile = await getUserProfile(event.user.address);
      if (fetchedProfile) {
        profile = fetchedProfile;
        userProfilesCache.set(event.user.address, fetchedProfile);
      }
    }

    // Enrich user data with profile if found - preserve all fields from event
    if (profile) {
      enrichedUser = {
        address: event.user.address,
        displayName: profile.displayName || incomingUser.displayName,
        username: profile.username || incomingUser.username,
        pfpUrl: profile.profileImage || incomingUser.pfpUrl,
        provider: profile.provider || incomingUser.provider,
        // Pass through any additional fields from the original event
        ...(event.user as any)
      };

      console.log(`✅ Enriched timeline event with profile for ${event.user.address.slice(0, 8)}... (${profile.displayName || profile.username})`);
    } else {
      // No profile in DB, just use incoming event data
      enrichedUser = { ...incomingUser };
    }
  } catch (error) {
    console.error('Failed to fetch user profile for timeline event:', error);
  }

  // Generate unique ID based on timestamp and random string
  const newEvent = {
    ...event,
    user: enrichedUser,
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    timestamp: event.timestamp || Date.now(),
  };

  // Store the event in memory
  timelineEvents.push(newEvent);

  // Save to session_activity_buffers for persistence
  // Map event type to activity_type (matches Solana ActivityType enum)
  const eventTypeToActivityType: Record<string, number> = {
    'check_in': 0,
    'check_out': 1,
    'auto_check_out': 1,  // Same as check_out
    'photo_captured': 2,
    'video_recorded': 3,
    'stream_started': 4,
    'face_enrolled': 5,
    'cv_activity': 50,
    'initialization': 255,
    'user_connected': 255,
    'other': 255
  };

  const activityType = eventTypeToActivityType[newEvent.type] ?? 255;

  // For frontend events, store content as unencrypted JSON
  // (Jetson events come pre-encrypted via /api/session/activity)
  const eventContent = {
    type: newEvent.type,
    user: enrichedUser,
    timestamp: newEvent.timestamp,
    transactionId: (newEvent as any).transactionId
  };

  try {
    await saveSessionActivity({
      sessionId: newEvent.id,  // Use event ID as session ID for frontend events
      cameraId: newEvent.cameraId || 'unknown',
      userPubkey: newEvent.user.address,
      timestamp: newEvent.timestamp,
      activityType,
      encryptedContent: Buffer.from(JSON.stringify(eventContent), 'utf-8'),  // Not actually encrypted for frontend events
      nonce: Buffer.alloc(12),  // Empty nonce for unencrypted content
      accessGrants: Buffer.from('[]', 'utf-8'),  // Empty grants - content is public
      createdAt: new Date()
    });
    console.log(`💾 Saved activity to session_activity_buffers: type=${activityType} (${newEvent.type})`);
  } catch (error) {
    console.error('Failed to save activity to session_activity_buffers:', error);
  }

  // Keep only last 100 events per camera in memory
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
    const socketsInRoom = cameraRooms.get(event.cameraId);
    console.log(`📤 Broadcasting timeline event ${newEvent.id} (${newEvent.type}) to camera room ${event.cameraId} (${socketsInRoom?.size || 0} sockets)`);
    socketServer.to(event.cameraId).emit("timelineEvent", newEvent);
  } else {
    // If no cameraId, broadcast to all
    console.log(`📤 Broadcasting timeline event ${newEvent.id} (${newEvent.type}) to all clients`);
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

// Database statistics endpoint
app.get("/api/database/stats", async (_req, res) => {
  try {
    const stats = await getDatabaseStats();
    res.json({
      success: true,
      stats,
      inMemory: {
        fileMappings: signatureToFileMapping.size,
        walletMappings: walletToSignatures.size,
        timelineEvents: timelineEvents.length,
        userProfiles: userProfilesCache.size
      }
    });
  } catch (error) {
    console.error('Failed to get database stats:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get stats'
    });
  }
});

// Debug endpoint to inspect actual database contents
app.get("/api/database/debug", async (_req, res) => {
  try {
    // Get recent timeline events from in-memory array
    const recentTimeline = timelineEvents.slice(0, 20);

    // Get file mappings from in-memory (mirrors DB)
    const fileMappings: any[] = [];
    signatureToFileMapping.forEach((mapping, sig) => {
      fileMappings.push({
        signature: sig.slice(0, 16) + '...',
        signatureType: mapping.signatureType,
        walletAddress: mapping.walletAddress.slice(0, 8) + '...',
        fileName: mapping.fileName,
        cameraId: mapping.cameraId,
        uploadedAt: mapping.uploadedAt,
        fileType: mapping.fileType
      });
    });

    // Get user profiles from cache (mirrors DB)
    const profiles: any[] = [];
    userProfilesCache.forEach((profile, addr) => {
      profiles.push({
        walletAddress: addr.slice(0, 8) + '...',
        displayName: profile.displayName,
        username: profile.username,
        provider: profile.provider,
        lastUpdated: profile.lastUpdated
      });
    });

    // Get session buffer stats
    const bufferStats = await getSessionBufferStats();

    res.json({
      success: true,
      timestamp: new Date().toISOString(),
      databasePath: process.env.DATABASE_PATH || '/tmp/mmoment.db',
      data: {
        realtimeEvents: {
          count: recentTimeline.length,
          items: recentTimeline.map((e: any) => ({
            id: e.id,
            type: e.type,
            userAddress: e.user?.address?.slice(0, 8) + '...',
            timestamp: new Date(e.timestamp).toISOString(),
            cameraId: e.cameraId
          }))
        },
        fileMappings: {
          count: fileMappings.length,
          items: fileMappings.slice(0, 20)
        },
        userProfiles: {
          count: profiles.length,
          items: profiles.slice(0, 20)
        },
        sessionBuffers: bufferStats
      }
    });
  } catch (error) {
    console.error('Failed to get database debug info:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get debug info'
    });
  }
});

// ============================================================================
// USER PROFILE ENDPOINTS (for Camera Service)
// ============================================================================

// Save or update user profile (called by camera service when users check in)
app.post("/api/profile/save", async (req, res) => {
  try {
    const { walletAddress, displayName, username, profileImage, provider } = req.body;

    if (!walletAddress) {
      return res.status(400).json({
        success: false,
        error: 'walletAddress is required'
      });
    }

    const profile: DBUserProfile = {
      walletAddress,
      displayName,
      username,
      profileImage,
      provider,
      lastUpdated: new Date()
    };

    // Save to database
    await saveUserProfile(profile);

    // Update cache
    userProfilesCache.set(walletAddress, profile);

    console.log(`✅ Saved user profile for ${walletAddress.slice(0, 8)}... (${displayName || username || 'no name'})`);

    res.json({
      success: true,
      message: 'Profile saved successfully'
    });
  } catch (error) {
    console.error('Failed to save user profile:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to save profile'
    });
  }
});

// Get user profile by wallet address
app.get("/api/profile/:walletAddress", async (req, res) => {
  try {
    const { walletAddress } = req.params;

    // Check cache first
    let profile = userProfilesCache.get(walletAddress);

    // If not in cache, fetch from database
    if (!profile) {
      const fetchedProfile = await getUserProfile(walletAddress);
      if (fetchedProfile) {
        profile = fetchedProfile;
        userProfilesCache.set(walletAddress, fetchedProfile);
      }
    }

    if (profile) {
      res.json({
        success: true,
        profile
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Profile not found'
      });
    }
  } catch (error) {
    console.error('Failed to get user profile:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get profile'
    });
  }
});

// Get multiple user profiles by wallet addresses (batch query)
app.post("/api/profile/batch", async (req, res) => {
  try {
    const { walletAddresses } = req.body;

    if (!Array.isArray(walletAddresses)) {
      return res.status(400).json({
        success: false,
        error: 'walletAddresses must be an array'
      });
    }

    // Fetch profiles from database
    const profilesMap = await getUserProfiles(walletAddresses);

    // Update cache
    for (const [address, profile] of profilesMap.entries()) {
      userProfilesCache.set(address, profile);
    }

    // Convert map to object for JSON response
    const profiles: Record<string, DBUserProfile> = {};
    for (const [address, profile] of profilesMap.entries()) {
      profiles[address] = profile;
    }

    res.json({
      success: true,
      profiles
    });
  } catch (error) {
    console.error('Failed to get user profiles:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get profiles'
    });
  }
});

// Delete user profile (admin/cleanup)
app.delete("/api/profile/:walletAddress", async (req, res) => {
  try {
    const { walletAddress } = req.params;

    if (!walletAddress) {
      return res.status(400).json({
        success: false,
        error: 'walletAddress is required'
      });
    }

    // Delete from database
    const deletedCount = await deleteUserProfile(walletAddress);

    // Remove from cache
    userProfilesCache.delete(walletAddress);

    if (deletedCount > 0) {
      console.log(`🗑️  Deleted profile for ${walletAddress.slice(0, 8)}...`);
      res.json({
        success: true,
        message: 'Profile deleted successfully'
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Profile not found'
      });
    }
  } catch (error) {
    console.error('Failed to delete user profile:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to delete profile'
    });
  }
});

// ============================================================================
// SESSION ACTIVITY BUFFER ENDPOINTS (Privacy-Preserving Timeline)
// ============================================================================

// Receive encrypted activity from Jetson (called during active session)
app.post("/api/session/activity", async (req, res) => {
  try {
    // Extract all possible fields - some are optional depending on activity type
    const {
      sessionId, cameraId, userPubkey, timestamp, activityType,
      encryptedContent, nonce, accessGrants,
      transactionSignature, displayName, username,
      cvActivityMeta  // Optional: CV activity metadata for timeline display
    } = req.body;

    // Validate required fields (common to all activities)
    if (!sessionId || !cameraId || !userPubkey || timestamp === undefined || activityType === undefined) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: sessionId, cameraId, userPubkey, timestamp, activityType'
      });
    }

    // Normalize timestamp: detect seconds vs milliseconds
    const normalizedTimestamp = timestamp < 10000000000 ? timestamp * 1000 : timestamp;

    // Map activity type to timeline event type
    const activityTypeToEventType: Record<number, string> = {
      0: 'check_in',
      1: 'check_out',
      2: 'photo_captured',
      3: 'video_recorded',
      4: 'stream_started',
      5: 'face_enrolled',
      50: 'cv_activity'
    };
    const eventType = activityTypeToEventType[activityType] || 'photo_captured';

    // CHECK_IN (0) and CHECK_OUT (1) - now saved to database for historical timeline
    // These may come with or without encryption from Jetson
    const isCheckInOut = activityType === 0 || activityType === 1;

    if (isCheckInOut) {
      console.log(`📢 ${eventType} for ${userPubkey.slice(0, 8)}... on camera ${cameraId.slice(0, 8)}...`);

      // Save to database for historical timeline queries
      // Use encrypted content if provided (from Jetson), otherwise create minimal plaintext
      let encryptedContentBuffer: Buffer;
      let nonceBuffer: Buffer;
      let accessGrantsBuffer: Buffer;

      if (encryptedContent && nonce && accessGrants) {
        // Jetson sent encrypted data - use it
        encryptedContentBuffer = Buffer.from(encryptedContent, 'base64');
        nonceBuffer = Buffer.from(nonce, 'base64');
        accessGrantsBuffer = Buffer.from(JSON.stringify(accessGrants), 'utf-8');
      } else {
        // Fallback: create minimal plaintext content for database storage
        const plaintextContent = JSON.stringify({
          type: eventType,
          user: userPubkey,
          timestamp: normalizedTimestamp,
          tx_signature: transactionSignature
        });
        encryptedContentBuffer = Buffer.from(plaintextContent, 'utf-8');
        nonceBuffer = Buffer.alloc(12); // Empty nonce for plaintext
        accessGrantsBuffer = Buffer.from('[]', 'utf-8'); // Empty access grants
      }

      const activity: SessionActivityBuffer = {
        sessionId,
        cameraId,
        userPubkey,
        timestamp: normalizedTimestamp,
        activityType,
        encryptedContent: encryptedContentBuffer,
        nonce: nonceBuffer,
        accessGrants: accessGrantsBuffer,
        createdAt: new Date()
      };

      await saveSessionActivity(activity);
      console.log(`✅ Saved ${eventType} to database for session ${sessionId.slice(0, 8)}...`);

      // Create timeline event for real-time display
      const timelineEvent: Record<string, any> = {
        id: `activity-${sessionId}-${normalizedTimestamp}`,
        type: eventType,
        user: {
          address: userPubkey,
          displayName: displayName || undefined,
          username: username || userPubkey.slice(0, 8) + '...'
        },
        timestamp: normalizedTimestamp,
        cameraId: cameraId
      };

      // Include transaction signature for Solscan link if provided
      if (transactionSignature) {
        timelineEvent.transactionId = transactionSignature;
      }

      // Add to in-memory cache for subsequent joinCamera calls
      timelineEvents.push({
        id: timelineEvent.id,
        type: eventType,
        user: {
          address: userPubkey,
          username: username || undefined,
          displayName: displayName || undefined
        },
        timestamp: normalizedTimestamp,
        cameraId: cameraId
      });
      // Keep only last 500 events in memory
      if (timelineEvents.length > 500) {
        timelineEvents.shift();
      }

      // Broadcast to camera room
      const room = io.sockets.adapter.rooms.get(cameraId);
      const socketsInRoom = room ? room.size : 0;
      console.log(`📤 Broadcasting ${eventType} to camera ${cameraId.slice(0, 8)}... (${socketsInRoom} sockets in room)`);

      io.to(cameraId).emit("timelineEvent", timelineEvent);

      return res.json({
        success: true,
        message: `${eventType} saved and broadcast`,
        debug: {
          cameraId,
          socketsInRoom,
          eventType,
          eventId: timelineEvent.id,
          savedToDb: true
        }
      });
    }

    // For all other activities (photos, videos, streams, etc.), encryption is REQUIRED
    if (!encryptedContent || !nonce || !accessGrants) {
      return res.status(400).json({
        success: false,
        error: `encryptedContent, nonce, accessGrants required for activity type ${activityType} (${eventType})`
      });
    }

    // Convert base64 strings to buffers
    const encryptedContentBuffer = Buffer.from(encryptedContent, 'base64');
    const nonceBuffer = Buffer.from(nonce, 'base64');
    const accessGrantsBuffer = Buffer.from(JSON.stringify(accessGrants), 'utf-8');

    // Save to database (include metadata for CV activities)
    const activity: SessionActivityBuffer = {
      sessionId,
      cameraId,
      userPubkey,
      timestamp: normalizedTimestamp,
      activityType,
      encryptedContent: encryptedContentBuffer,
      nonce: nonceBuffer,
      accessGrants: accessGrantsBuffer,
      createdAt: new Date(),
      metadata: cvActivityMeta ? JSON.stringify({ cvActivity: cvActivityMeta }) : undefined
    };

    await saveSessionActivity(activity);

    console.log(`✅ Buffered encrypted activity for session ${sessionId.slice(0, 8)}... (type: ${activityType})`);

    // Create timeline event for real-time display
    const timelineEvent: Record<string, any> = {
      id: `activity-${sessionId}-${normalizedTimestamp}`,
      type: eventType,
      user: {
        address: userPubkey,
        displayName: displayName || undefined,
        username: username || userPubkey.slice(0, 8) + '...'
      },
      timestamp: normalizedTimestamp,
      cameraId: cameraId,
      // Include encrypted data reference for decryption
      encryptedActivity: {
        encryptedContent,
        nonce,
        accessGrants
      }
    };

    // Include transaction signature for check_in/check_out events (for Solscan link)
    if (transactionSignature) {
      timelineEvent.transactionId = transactionSignature;
      console.log(`   📝 Including transaction signature: ${transactionSignature.slice(0, 8)}...`);
    }

    // Include CV activity metadata for timeline display
    if (cvActivityMeta && activityType === 50) {
      timelineEvent.cvActivity = cvActivityMeta;
      console.log(`   🏋️ Including CV activity meta: ${cvActivityMeta.app_name}, ${cvActivityMeta.participant_count} participants`);
    }

    // Broadcast to camera room
    const room = io.sockets.adapter.rooms.get(cameraId);
    const socketsInRoom = room ? room.size : 0;
    console.log(`📤 Broadcasting to camera ${cameraId.slice(0, 8)}... (${socketsInRoom} sockets in room)`);
    console.log(`📤 Event type: ${eventType}, user: ${userPubkey.slice(0, 8)}...`);

    io.to(cameraId).emit("timelineEvent", timelineEvent);

    res.json({
      success: true,
      message: 'Activity buffered successfully',
      debug: {
        cameraId,
        socketsInRoom,
        eventType,
        eventId: timelineEvent.id
      }
    });
  } catch (error) {
    console.error('Failed to buffer session activity:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to buffer activity'
    });
  }
});

// Update a timeline event with a transaction ID (called by cron job after UserSessionChain write)
app.patch("/api/session/activity/transaction", async (req, res) => {
  try {
    const { userPubkey, timestamp, transactionId, eventType } = req.body;

    if (!userPubkey || !timestamp || !transactionId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: userPubkey, timestamp, transactionId'
      });
    }

    console.log(`📝 Updating timeline event with transaction ID:`);
    console.log(`   User: ${userPubkey.slice(0, 8)}...`);
    console.log(`   Timestamp: ${timestamp}`);
    console.log(`   Transaction: ${transactionId.slice(0, 8)}...`);
    console.log(`   Event type: ${eventType || 'check_out'}`);

    // Find the timeline event and get its cameraId
    const matchingEvent = timelineEvents.find(e =>
      e.user?.address === userPubkey &&
      Math.abs(e.timestamp - timestamp) < 60000 && // Within 1 minute
      (eventType ? e.type === eventType : e.type === 'check_out' || e.type === 'auto_check_out')
    );

    if (matchingEvent) {
      // Update in-memory event
      matchingEvent.transactionId = transactionId;
      console.log(`   ✅ Updated in-memory event: ${matchingEvent.id}`);

      // Broadcast update to camera room
      if (matchingEvent.cameraId) {
        io.to(matchingEvent.cameraId).emit("timelineEventUpdate", {
          id: matchingEvent.id,
          transactionId: transactionId
        });
        console.log(`   📤 Broadcasted update to camera room: ${matchingEvent.cameraId.slice(0, 8)}...`);
      }
    } else {
      console.log(`   ⚠️ No matching event found in memory (may have been cleared)`);
    }

    res.json({
      success: true,
      message: 'Transaction ID update processed',
      eventFound: !!matchingEvent
    });
  } catch (error) {
    console.error('Failed to update timeline event transaction:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update transaction'
    });
  }
});

// DEBUG: Test Socket.IO broadcast to a camera room
app.post("/api/debug/broadcast-test", (req, res) => {
  const { cameraId, message } = req.body;

  if (!cameraId) {
    return res.status(400).json({ success: false, error: 'cameraId required' });
  }

  const testEvent = {
    id: `debug-${Date.now()}`,
    type: 'check_in',
    user: {
      address: 'DEBUG_TEST_USER',
      username: 'debug_test'
    },
    timestamp: Date.now(),
    cameraId: cameraId,
    message: message || 'Debug broadcast test'
  };

  // Log room info
  const room = io.sockets.adapter.rooms.get(cameraId);
  const socketsInRoom = room ? room.size : 0;

  console.log(`🧪 DEBUG BROADCAST: cameraId=${cameraId}, socketsInRoom=${socketsInRoom}`);
  console.log(`🧪 Broadcasting test event:`, testEvent);

  io.to(cameraId).emit("timelineEvent", testEvent);

  res.json({
    success: true,
    message: `Broadcast sent to ${socketsInRoom} sockets in room ${cameraId}`,
    socketsInRoom,
    event: testEvent
  });
});

// Fetch buffered activities for a session (called by auto-checkout bot)
app.get("/api/session/activities/:sessionId", async (req, res) => {
  try {
    const { sessionId } = req.params;

    if (!sessionId) {
      return res.status(400).json({
        success: false,
        error: 'sessionId is required'
      });
    }

    // Fetch from database
    const activities = await getSessionActivities(sessionId);

    // Convert buffers to base64 for JSON transport
    const activitiesForResponse = activities.map(activity => ({
      sessionId: activity.sessionId,
      cameraId: activity.cameraId,
      userPubkey: activity.userPubkey,
      timestamp: activity.timestamp,
      activityType: activity.activityType,
      encryptedContent: activity.encryptedContent.toString('base64'),
      nonce: activity.nonce.toString('base64'),
      accessGrants: JSON.parse(activity.accessGrants.toString('utf-8')),
      createdAt: activity.createdAt.toISOString()
    }));

    console.log(`📤 Fetched ${activitiesForResponse.length} buffered activities for session ${sessionId.slice(0, 8)}...`);

    res.json({
      success: true,
      activities: activitiesForResponse,
      count: activitiesForResponse.length
    });
  } catch (error) {
    console.error('Failed to fetch session activities:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fetch activities'
    });
  }
});

// Clear buffered activities after successful checkout (called by auto-checkout bot)
app.delete("/api/session/activities/:sessionId", async (req, res) => {
  try {
    const { sessionId } = req.params;

    if (!sessionId) {
      return res.status(400).json({
        success: false,
        error: 'sessionId is required'
      });
    }

    // Clear from database
    const deletedCount = await clearSessionActivities(sessionId);

    console.log(`🗑️  Cleared ${deletedCount} buffered activities for session ${sessionId.slice(0, 8)}...`);

    res.json({
      success: true,
      message: 'Activities cleared successfully',
      deletedCount
    });
  } catch (error) {
    console.error('Failed to clear session activities:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to clear activities'
    });
  }
});

// Get session buffer statistics (for monitoring)
app.get("/api/session/buffer-stats", async (_req, res) => {
  try {
    const stats = await getSessionBufferStats();

    res.json({
      success: true,
      stats: {
        totalActivities: stats.totalActivities,
        uniqueSessions: stats.uniqueSessions,
        uniqueCameras: stats.uniqueCameras,
        oldestActivity: stats.oldestActivity?.toISOString() || null,
        newestActivity: stats.newestActivity?.toISOString() || null
      }
    });
  } catch (error) {
    console.error('Failed to get session buffer stats:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get buffer stats'
    });
  }
});

// Receive encrypted session access key from Jetson at checkout
// This stores the key so users can decrypt their session activities later
app.post("/api/session/access-key", async (req, res) => {
  try {
    const { user_pubkey, key_ciphertext, nonce, timestamp } = req.body;

    // Validate required fields
    if (!user_pubkey || !key_ciphertext || !nonce) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: user_pubkey, key_ciphertext, nonce'
      });
    }

    // Validate array types
    if (!Array.isArray(key_ciphertext) || !Array.isArray(nonce)) {
      return res.status(400).json({
        success: false,
        error: 'key_ciphertext and nonce must be arrays'
      });
    }

    // Queue the access key for blockchain storage
    const success = await queueAccessKeyForUser(
      user_pubkey,
      key_ciphertext,
      nonce,
      timestamp || Math.floor(Date.now() / 1000)
    );

    if (success) {
      console.log(`✅ Access key queued for user ${user_pubkey.slice(0, 8)}...`);
      res.json({
        success: true,
        message: 'Access key received and queued for storage'
      });
    } else {
      // Still return success - key is queued for retry even if immediate storage failed
      console.log(`⏳ Access key queued for user ${user_pubkey.slice(0, 8)}... (pending retry)`);
      res.json({
        success: true,
        message: 'Access key received and queued for storage (pending)'
      });
    }
  } catch (error) {
    console.error('Failed to receive access key:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to receive access key'
    });
  }
});

// Get sessions for a user (where they have access grants)
app.get("/api/user/:walletAddress/sessions", async (req, res) => {
  try {
    const { walletAddress } = req.params;
    const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);

    if (!walletAddress) {
      return res.status(400).json({
        success: false,
        error: 'walletAddress is required'
      });
    }

    const sessions = await getUserSessions(walletAddress, limit);

    console.log(`📋 Found ${sessions.length} sessions for user ${walletAddress.slice(0, 8)}...`);

    res.json({
      success: true,
      sessions
    });
  } catch (error) {
    console.error('Failed to get user sessions:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get user sessions'
    });
  }
});

// Get all activities for a camera
app.get("/api/camera/:cameraId/activities", async (req, res) => {
  try {
    const { cameraId } = req.params;
    const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);

    if (!cameraId) {
      return res.status(400).json({
        success: false,
        error: 'cameraId is required'
      });
    }

    const activities = await getCameraActivities(cameraId, limit);

    // Convert buffers to base64 for JSON transport
    const activitiesForResponse = activities.map(activity => ({
      sessionId: activity.sessionId,
      cameraId: activity.cameraId,
      userPubkey: activity.userPubkey,
      timestamp: activity.timestamp,
      activityType: activity.activityType,
      encryptedContent: activity.encryptedContent.toString('base64'),
      nonce: activity.nonce.toString('base64'),
      accessGrants: JSON.parse(activity.accessGrants.toString('utf-8')),
      createdAt: activity.createdAt.toISOString()
    }));

    console.log(`📸 Found ${activities.length} activities for camera ${cameraId.slice(0, 8)}...`);

    res.json({
      success: true,
      activities: activitiesForResponse
    });
  } catch (error) {
    console.error('Failed to get camera activities:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get camera activities'
    });
  }
});

// Get full timeline events for a session (all events at camera during session time window)
// This returns ALL events from ALL users at the camera during the user's session
app.get("/api/session/:sessionId/timeline", async (req, res) => {
  try {
    const { sessionId } = req.params;
    const walletAddress = req.query.walletAddress as string;

    if (!sessionId) {
      return res.status(400).json({
        success: false,
        error: 'sessionId is required'
      });
    }

    if (!walletAddress) {
      return res.status(400).json({
        success: false,
        error: 'walletAddress query param is required'
      });
    }

    // First, get the user's sessions to find the session details
    const sessions = await getUserSessions(walletAddress, 100);
    const session = sessions.find(s => s.sessionId === sessionId);

    if (!session) {
      return res.status(404).json({
        success: false,
        error: 'Session not found or user does not have access'
      });
    }

    // Get all timeline events at this camera during the session time window
    const events = await getSessionTimelineEvents(
      session.cameraId,
      session.startTime,
      session.endTime
    );

    console.log(`📜 Found ${events.length} timeline events for session ${sessionId.slice(0, 8)}... at camera ${session.cameraId.slice(0, 8)}...`);

    res.json({
      success: true,
      session: {
        sessionId: session.sessionId,
        cameraId: session.cameraId,
        startTime: session.startTime,
        endTime: session.endTime
      },
      events
    });
  } catch (error) {
    console.error('Failed to get session timeline:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get session timeline'
    });
  }
});

// Get all activities a user has access to
app.get("/api/user/:walletAddress/activities", async (req, res) => {
  try {
    const { walletAddress } = req.params;
    const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);

    if (!walletAddress) {
      return res.status(400).json({
        success: false,
        error: 'walletAddress is required'
      });
    }

    const activities = await getUserActivities(walletAddress, limit);

    // Convert buffers to base64 for JSON transport
    const activitiesForResponse = activities.map(activity => ({
      sessionId: activity.sessionId,
      cameraId: activity.cameraId,
      userPubkey: activity.userPubkey,
      timestamp: activity.timestamp,
      activityType: activity.activityType,
      encryptedContent: activity.encryptedContent.toString('base64'),
      nonce: activity.nonce.toString('base64'),
      accessGrants: JSON.parse(activity.accessGrants.toString('utf-8')),
      createdAt: activity.createdAt.toISOString()
    }));

    console.log(`🔐 Found ${activities.length} accessible activities for user ${walletAddress.slice(0, 8)}...`);

    res.json({
      success: true,
      activities: activitiesForResponse
    });
  } catch (error) {
    console.error('Failed to get user activities:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get user activities'
    });
  }
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
  socket.on("joinCamera", async (cameraId: string) => {
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

    // Query database for persisted events (survives server restarts)
    // Frontend will filter to current session (events after most recent check_in)
    try {
      const dbActivities = await getCameraActivities(cameraId, 100);

      // Map activity_type to event type string
      const activityTypeToEventType: Record<number, string> = {
        0: 'check_in',
        1: 'check_out',
        2: 'photo_captured',
        3: 'video_recorded',
        4: 'stream_started',
        5: 'face_enrolled',
        50: 'cv_activity'
      };

      // Convert database activities to timeline event format
      const dbEvents = dbActivities.map(activity => {
        // Parse metadata if present (contains cvActivity for CV activities)
        let parsedMetadata: { cvActivity?: any } = {};
        if (activity.metadata) {
          try {
            parsedMetadata = JSON.parse(activity.metadata);
          } catch {
            // Ignore parse errors
          }
        }

        return {
          id: `activity-${activity.sessionId}-${activity.timestamp}`,
          type: activityTypeToEventType[activity.activityType] || 'unknown',
          user: {
            address: activity.userPubkey
          },
          timestamp: activity.timestamp,
          cameraId: activity.cameraId,
          // Include CV activity metadata if present
          ...(parsedMetadata.cvActivity && { cvActivity: parsedMetadata.cvActivity })
        };
      });

      // Get in-memory events for this camera (for very recent events not yet persisted)
      const memoryEvents = timelineEvents
        .filter((event) => event.cameraId === cameraId);

      // Merge and deduplicate by ID (prefer memory events as they may have more user info)
      const eventIds = new Set<string>();
      const allEvents: TimelineEvent[] = [];

      // Add memory events first (fresher, may have display names)
      for (const event of memoryEvents) {
        if (!eventIds.has(event.id)) {
          eventIds.add(event.id);
          allEvents.push(event);
        }
      }

      // Add database events that aren't already in memory
      for (const event of dbEvents) {
        if (!eventIds.has(event.id)) {
          eventIds.add(event.id);
          allEvents.push(event as TimelineEvent);
        }
      }

      // Sort by timestamp (newest first) and limit
      const cameraEvents = allEvents
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 100);

      console.log(`[Timeline] Sending ${cameraEvents.length} events (${dbEvents.length} from DB, ${memoryEvents.length} from memory) to socket ${socket.id} for camera ${cameraId}`);
      socket.emit("recentEvents", cameraEvents);
    } catch (error) {
      console.error(`[Timeline] Error fetching camera events from DB:`, error);
      // Fallback to in-memory only
      const cameraEvents = timelineEvents
        .filter((event) => event.cameraId === cameraId)
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 50);
      console.log(`[Timeline] Fallback: Sending ${cameraEvents.length} in-memory events to socket ${socket.id}`);
      socket.emit("recentEvents", cameraEvents);
    }
  });

  // Handle leaving a camera room
  socket.on("leaveCamera", (cameraId: string) => {
    console.log(`Socket ${socket.id} leaving camera ${cameraId}`);
    socket.leave(cameraId);
    cameraRooms.get(cameraId)?.delete(socket.id);
  });

  // Handle new timeline events
  // NOTE: check_in, check_out, and auto_check_out events are BLOCKED from this shortcut
  // They MUST come from the Jetson camera via /api/session/activity with proper encryption
  socket.on("newTimelineEvent", async (event: Omit<TimelineEvent, "id">) => {
    console.log(`📥 Received newTimelineEvent from socket ${socket.id}:`, event.type, event.user?.address?.slice(0, 8) || 'unknown');

    // Block check-in/check-out events - these must go through Jetson for proper encryption
    const blockedEventTypes = ['check_in', 'check_out', 'auto_check_out'];
    if (blockedEventTypes.includes(event.type)) {
      console.warn(`⚠️  BLOCKED ${event.type} event via WebSocket shortcut - must go through Jetson for encryption`);
      return; // Silently drop - Jetson will create the proper encrypted activity
    }

    try {
      await addTimelineEvent(event, io);
    } catch (error) {
      console.error(`❌ Failed to add timeline event:`, error);
    }
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
httpServer.listen(port, "0.0.0.0", async () => {
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

  // Initialize SQLite database and load persisted data
  try {
    console.log('\n📦 Initializing SQLite database...');
    // Use Railway's writable /tmp directory for ephemeral storage
    const dbPath = process.env.DATABASE_PATH ||
                   (process.env.RAILWAY_ENVIRONMENT ? '/tmp/mmoment.db' : './mmoment.db');
    console.log(`📍 Database path: ${dbPath}`);
    await initializeDatabase(dbPath);

    // Load file mappings from database into memory
    console.log('📥 Loading file mappings from database...');
    const { signatureToFileMapping: loadedMappings, walletToSignatures: loadedWalletSigs } =
      await loadAllFileMappingsToMaps();

    // Populate in-memory maps
    for (const [sig, mapping] of loadedMappings.entries()) {
      signatureToFileMapping.set(sig, mapping);
    }
    for (const [wallet, sigs] of loadedWalletSigs.entries()) {
      walletToSignatures.set(wallet, sigs);
    }

    // Load user profiles FIRST (so we can enrich timeline events)
    console.log('📥 Loading user profiles from database...');
    const loadedProfiles = await loadAllUserProfilesToMap();
    for (const [address, profile] of loadedProfiles.entries()) {
      userProfilesCache.set(address, profile);
    }

    // NOTE: Timeline events are now persisted in session_activity_buffers
    // The in-memory timelineEvents array is only for real-time WebSocket broadcasting
    // It starts empty on server restart - historical data is queried from session_activity_buffers
    console.log('📥 Timeline events use session_activity_buffers for persistence');

    // Get database stats
    const stats = await getDatabaseStats();
    console.log('📊 Database statistics:', {
      fileMappings: stats.fileMappings,
      uniqueWallets: stats.uniqueWallets,
      uniqueCameras: stats.uniqueCameras,
      userProfiles: stats.userProfiles,
      sessionActivityBuffers: stats.sessionActivityBuffers
    });

    console.log('✅ Database initialized and data loaded successfully!\n');
  } catch (dbError) {
    console.error('❌ Failed to initialize database:', dbError);
    console.warn('⚠️  Server will continue without persistence');
  }

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

// Graceful shutdown - close database connection
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down gracefully...');
  try {
    await closeDatabase();
    console.log('✅ Database connection closed');
  } catch (err) {
    console.error('❌ Error closing database:', err);
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🛑 Shutting down gracefully...');
  try {
    await closeDatabase();
    console.log('✅ Database connection closed');
  } catch (err) {
    console.error('❌ Error closing database:', err);
  }
  process.exit(0);
});
