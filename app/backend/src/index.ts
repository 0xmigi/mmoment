// backend/src/index.ts
import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import cors from "cors";
import { config } from "dotenv";
// Using built-in fetch in Node.js 18+

// Load environment variables
config();

// Pipe API configuration
const PIPE_API_BASE_URL = "https://us-west-00-firestarter.pipenetwork.com";

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

// Store Pipe accounts in memory (in production, use a database)
interface PipeAccount {
  userId: string;
  userAppKey: string;
  username?: string;
  password?: string;
  accessToken?: string;
  refreshToken?: string;
  tokenExpiry?: number;
  createdAt: number;
}

const pipeAccounts = new Map<string, PipeAccount>();

// Pipe API endpoints
app.get("/api/pipe/credentials", (req, res) => {
  const walletAddress = req.query.wallet as string;

  if (!walletAddress) {
    res.status(400).json({ error: "walletAddress parameter is required" });
    return;
  }

  // Check if account exists for this wallet
  const account = pipeAccounts.get(walletAddress);
  if (account) {
    console.log(`Found existing Pipe account for wallet: ${walletAddress}`);
    res.json({
      userId: account.userId,
      userAppKey: account.userAppKey,
    });
  } else {
    // No account exists yet
    res.status(404).json({ error: "No Pipe account found for this wallet" });
  }
});

// Proxy endpoint for Pipe API requests
app.post("/api/pipe/proxy/*", async (req: any, res) => {
  const endpoint = req.params[0]; // Get the endpoint after /proxy/

  try {
    // Get wallet address from headers to find account
    const walletAddress = req.headers["x-wallet-address"] as string;
    const authHeaders: Record<string, string> = {};
    
    // Check if we have JWT tokens for this wallet
    if (walletAddress && pipeAccounts.has(walletAddress)) {
      const account = pipeAccounts.get(walletAddress)!;
      
      // Check if token is still valid
      if (account.accessToken && account.tokenExpiry && Date.now() < account.tokenExpiry) {
        authHeaders["Authorization"] = `Bearer ${account.accessToken}`;
        console.log(`🔄 Using JWT token for ${endpoint}`);
      } else if (account.refreshToken) {
        console.log(`🔄 Token expired, refreshing for ${endpoint}`);
        try {
          // Refresh the JWT token
          const refreshResponse = await fetch(`${PIPE_API_BASE_URL}/refreshToken`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              refresh_token: account.refreshToken,
            }),
          });

          if (refreshResponse.ok) {
            const refreshData = await refreshResponse.json();
            // Update stored tokens
            account.accessToken = refreshData.access_token;
            account.refreshToken = refreshData.refresh_token || account.refreshToken;
            account.tokenExpiry = Date.now() + (refreshData.expires_in * 1000);

            authHeaders["Authorization"] = `Bearer ${refreshData.access_token}`;
            console.log(`✅ Token refreshed successfully for ${endpoint}`);
          } else {
            console.log(`❌ Token refresh failed, falling back to re-login for ${endpoint}`);
            // Token refresh failed, try to re-login
            const username = `mmoment_${walletAddress.slice(0, 16)}`;
            const password = generateUserPassword(username);

            const loginResponse = await fetch(`${PIPE_API_BASE_URL}/login`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                username,
                password,
              }),
            });

            if (loginResponse.ok) {
              const loginData = await loginResponse.json();
              account.accessToken = loginData.access_token;
              account.refreshToken = loginData.refresh_token;
              account.tokenExpiry = Date.now() + (loginData.expires_in * 1000);

              authHeaders["Authorization"] = `Bearer ${loginData.access_token}`;
              console.log(`✅ Re-logged in successfully for ${endpoint}`);
            } else {
              console.log(`❌ Re-login failed for ${endpoint}, no valid auth available`);
              return res.status(401).json({ error: "Authentication failed - unable to refresh or re-login" });
            }
          }
        } catch (error) {
          console.error(`❌ Error during token refresh/re-login for ${endpoint}:`, error);
          return res.status(500).json({ error: "Authentication refresh failed" });
        }
      } else {
        // Use legacy auth
        authHeaders["X-User-Id"] = account.userId;
        authHeaders["X-User-App-Key"] = account.userAppKey;
      }
    } else {
      // No account found for this wallet - return error
      return res.status(401).json({ error: "No Pipe account found for this wallet. Please create an account first." });
    }
    
    const pipeUrl = `${PIPE_API_BASE_URL}/${endpoint}`;

    console.log(`🔄 Proxying to Pipe: ${endpoint}`);
    console.log(`📤 Request body:`, JSON.stringify(req.body));
    console.log(`📤 Auth headers:`, authHeaders);

    const response = await fetch(pipeUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders,
      },
      body: JSON.stringify(req.body),
    });

    const responseText = await response.text();
    console.log(
      `📥 Pipe response (${response.status}):`,
      responseText.substring(0, 200),
    );

    // Set status and return response
    res.status(response.status);

    // Handle empty responses
    if (!responseText || responseText.trim() === "") {
      return res.json({});
    }

    // Try to parse as JSON, otherwise return as text
    try {
      const jsonData = JSON.parse(responseText);
      return res.json(jsonData);
    } catch {
      return res.send(responseText);
    }
  } catch (error) {
    console.error("Pipe proxy error:", error);
    res.status(500).json({ error: "Proxy request failed" });
  }
});

// Helper function to generate deterministic password (matches your SDK)
function generateUserPassword(userId: string): string {
  const crypto = require('crypto');
  const hasher = crypto.createHash('sha256');
  hasher.update(userId);
  hasher.update('mmoment-pipe-encryption-2024');
  return hasher.digest('hex');
}

app.post("/api/pipe/create-account", async (req, res) => {
  const { walletAddress } = req.body;

  if (!walletAddress) {
    return res.status(400).json({ error: "walletAddress is required" });
  }

  // Check if account already exists
  if (pipeAccounts.has(walletAddress)) {
    const account = pipeAccounts.get(walletAddress)!;
    
    // If account exists but has no JWT tokens, try to set up JWT auth
    if (!account.accessToken) {
      console.log(`🔄 Existing account found but missing JWT tokens for ${walletAddress}`);
      console.log(`   Attempting to set up JWT authentication...`);
      
      // We need to go through the JWT setup process for this existing account
      // For now, we'll treat it as if it doesn't exist and create fresh
    } else {
      console.log(`✅ Returning existing Pipe account with JWT for wallet: ${walletAddress}`);
      return res.json({
        userId: account.userId,
        userAppKey: account.userAppKey,
        existing: true,
      });
    }
  }

  try {
    console.log(`🔄 Starting create account process...`);
    
    // Generate username and password following MMOMENT SDK pattern
    const username = `mmoment_${walletAddress.slice(0, 16)}`;
    console.log(`🔄 Generating password for username: ${username}`);
    const password = generateUserPassword(username);
    console.log(`🔄 Password generated successfully`);
    
    console.log(`🔄 Creating Pipe account for wallet: ${walletAddress}`);
    console.log(`   Username: ${username}`);
    console.log(`   Base URL: ${PIPE_API_BASE_URL}`);
    
    // Step 1: Try to create new user account using correct /users endpoint
    console.log(`🔄 Calling ${PIPE_API_BASE_URL}/users...`);
    let createResponse;
    try {
      createResponse = await fetch(`${PIPE_API_BASE_URL}/users`, {
        method: "POST", 
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
        }),
      });
      console.log(`📥 createUser response: ${createResponse.status} ${createResponse.statusText}`);
    } catch (fetchError) {
      console.error(`❌ Fetch error calling /users:`, fetchError);
      throw fetchError;
    }

    if (createResponse.ok) {
      // Account created successfully
      const userData = await createResponse.json();
      console.log(`✅ Created new Pipe user: ${username}`);
      console.log(`   User ID: ${userData.user_id}`);
      
      // Step 2: Set password for JWT auth
      const setPasswordResponse = await fetch(`${PIPE_API_BASE_URL}/auth/set-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: userData.user_id,
          user_app_key: userData.user_app_key,
          new_password: password,
        }),
      });

      if (setPasswordResponse.ok) {
        console.log(`✅ Password set for ${username}`);
        
        // Step 3: Login to get JWT tokens
        const loginResponse = await fetch(`${PIPE_API_BASE_URL}/auth/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            username,
            password,
          }),
        });

        if (loginResponse.ok) {
          const tokens = await loginResponse.json();
          
          const newAccount: PipeAccount = {
            userId: userData.user_id,
            userAppKey: userData.user_app_key,
            username,
            password,
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
            tokenExpiry: Date.now() + (tokens.expires_in * 1000),
            createdAt: Date.now(),
          };

          pipeAccounts.set(walletAddress, newAccount);
          console.log(`✅ Pipe account ready for ${walletAddress}`);

          return res.json({
            userId: newAccount.userId,
            userAppKey: newAccount.userAppKey,
            existing: false,
          });
        } else {
          console.log(`❌ Login failed after setting password`);
        }
      } else {
        console.log(`❌ Failed to set password for ${username}`);
      }
    } else if (createResponse.status === 409) {
      // User already exists, try to login
      console.log(`ℹ️ User ${username} already exists, attempting login...`);
      
      const loginResponse = await fetch(`${PIPE_API_BASE_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      if (loginResponse.ok) {
        const tokens = await loginResponse.json();

        // Get the actual user_id by calling checkWallet with the JWT token
        const walletResponse = await fetch(`${PIPE_API_BASE_URL}/checkWallet`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${tokens.access_token}`,
          },
          body: JSON.stringify({}),
        });

        let actualUserId = username;
        let actualAppKey = "jwt-based";

        if (walletResponse.ok) {
          const walletData = await walletResponse.json();
          actualUserId = walletData.user_id || username;
          // The public_key is the Solana address for the Pipe wallet
          console.log(`✅ Got actual user_id: ${actualUserId}`);
        }

        const existingAccount: PipeAccount = {
          userId: actualUserId,
          userAppKey: actualAppKey,
          username,
          password,
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token,
          tokenExpiry: Date.now() + (tokens.expires_in * 1000),
          createdAt: Date.now(),
        };

        pipeAccounts.set(walletAddress, existingAccount);
        console.log(`✅ Logged into existing Pipe account for ${walletAddress}`);

        return res.json({
          userId: existingAccount.userId,
          userAppKey: existingAccount.userAppKey,
          existing: true,
        });
      } else {
        console.log(`❌ Login failed for existing user ${username}`);
        return res.status(401).json({ error: "Failed to login to existing account" });
      }
    } else {
      // createUser failed for some other reason
      const status = createResponse.status;
      const errorText = await createResponse.text();
      console.log(`❌ createUser failed with status ${status}: ${errorText}`);
    }
    
    throw new Error("Account creation failed");
    
  } catch (error) {
    console.error("Error creating Pipe account:", error);
    res.status(500).json({ error: "Failed to create Pipe account" });
  }
});

// Pipe upload proxy endpoint for Jetson cameras
app.post("/api/pipe/upload", async (req, res) => {
  const { walletAddress, imageData, filename, metadata } = req.body;

  if (!walletAddress || !imageData || !filename) {
    return res.status(400).json({
      success: false,
      error: "walletAddress, imageData, and filename are required"
    });
  }

  try {
    // Get user's Pipe account
    const account = pipeAccounts.get(walletAddress);
    if (!account) {
      return res.status(404).json({
        success: false,
        error: "Pipe account not found for wallet. Call create-account first."
      });
    }

    // Check if token is expired and refresh if needed
    if (account.tokenExpiry && Date.now() > account.tokenExpiry) {
      console.log(`🔄 Token expired for ${walletAddress}, refreshing...`);

      if (account.refreshToken) {
        try {
          const refreshResponse = await fetch(`${PIPE_API_BASE_URL}/auth/refresh`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh_token: account.refreshToken }),
          });

          if (refreshResponse.ok) {
            const tokens = await refreshResponse.json();
            account.accessToken = tokens.access_token;
            account.refreshToken = tokens.refresh_token;
            account.tokenExpiry = Date.now() + (tokens.expires_in * 1000);
            console.log(`✅ Token refreshed for ${walletAddress}`);
          } else {
            throw new Error('Token refresh failed');
          }
        } catch (refreshError) {
          console.log(`❌ Token refresh failed, upload will fail: ${refreshError}`);
          return res.status(401).json({
            success: false,
            error: "Authentication expired and refresh failed"
          });
        }
      }
    }

    // Convert base64 image data to buffer if needed
    let imageBuffer: Buffer;
    if (typeof imageData === 'string') {
      // Remove data URL prefix if present (data:image/jpeg;base64,...)
      const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, '');
      imageBuffer = Buffer.from(base64Data, 'base64');
    } else {
      imageBuffer = Buffer.from(imageData);
    }

    console.log(`📤 Uploading ${filename} to Pipe for ${walletAddress} (${imageBuffer.length} bytes)`);

    // Upload to Pipe using priorityUpload for speed
    const FormData = require('form-data');
    const form = new FormData();
    form.append('file', imageBuffer, {
      filename: filename,
      contentType: 'image/jpeg'
    });

    const uploadResponse = await fetch(
      `${PIPE_API_BASE_URL}/priorityUpload?user_id=${account.userId}&file_name=${filename}`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${account.accessToken}`,
          ...form.getHeaders()
        },
        body: form
      }
    );

    if (uploadResponse.ok) {
      const resultFilename = await uploadResponse.text();

      console.log(`✅ Successfully uploaded ${filename} → ${resultFilename} for ${walletAddress}`);

      res.json({
        success: true,
        filename: resultFilename.trim(),
        originalFilename: filename,
        size: imageBuffer.length,
        walletAddress,
        metadata,
        uploadTimestamp: new Date().toISOString()
      });
    } else {
      const errorText = await uploadResponse.text();
      console.log(`❌ Pipe upload failed: ${uploadResponse.status} ${errorText}`);

      res.status(uploadResponse.status).json({
        success: false,
        error: `Pipe upload failed: ${errorText}`,
        pipeStatus: uploadResponse.status
      });
    }

  } catch (error) {
    console.error(`❌ Upload error for ${walletAddress}:`, error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Upload failed"
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

// Get user's files from Pipe storage
app.get("/api/pipe/files/:walletAddress", async (req, res) => {
  const { walletAddress } = req.params;

  try {
    const account = pipeAccounts.get(walletAddress);
    if (!account || !account.accessToken) {
      return res.status(404).json({ error: "Pipe account not found or not authenticated" });
    }

    // Check if token needs refresh
    if (account.tokenExpiry && Date.now() >= account.tokenExpiry - 60000) {
      console.log(`🔄 Token about to expire, refreshing for ${walletAddress}`);

      const refreshResponse = await fetch(`${PIPE_API_BASE_URL}/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          refresh_token: account.refreshToken,
        }),
      });

      if (refreshResponse.ok) {
        const tokenData = await refreshResponse.json();
        account.accessToken = tokenData.access_token;
        account.refreshToken = tokenData.refresh_token;
        account.tokenExpiry = Date.now() + 3600 * 1000;
        console.log(`✅ Token refreshed for ${walletAddress}`);
      }
    }

    // List user's files
    const filesResponse = await fetch(`${PIPE_API_BASE_URL}/files`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${account.accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!filesResponse.ok) {
      console.error(`Failed to fetch files: ${filesResponse.status}`);
      return res.status(filesResponse.status).json({
        error: "Failed to fetch files from Pipe"
      });
    }

    const filesData = await filesResponse.json();

    // Transform files to include full URLs
    const transformedFiles = filesData.files?.map((file: any) => ({
      ...file,
      url: `${PIPE_API_BASE_URL}/files/${file.id}/content`,
      // Add a public URL if available
      publicUrl: file.public ? `${PIPE_API_BASE_URL}/public/${file.id}` : null
    })) || [];

    res.json({
      files: transformedFiles,
      total: filesData.total || transformedFiles.length
    });
  } catch (error) {
    console.error("Error fetching Pipe files:", error);
    res.status(500).json({ error: "Failed to fetch files" });
  }
});

// Get a specific file from Pipe storage
app.get("/api/pipe/file/:walletAddress/:fileId", async (req, res) => {
  const { walletAddress, fileId } = req.params;

  try {
    const account = pipeAccounts.get(walletAddress);
    if (!account || !account.accessToken) {
      return res.status(404).json({ error: "Pipe account not found or not authenticated" });
    }

    // Get file metadata
    const fileResponse = await fetch(`${PIPE_API_BASE_URL}/files/${fileId}`, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${account.accessToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!fileResponse.ok) {
      return res.status(fileResponse.status).json({
        error: "File not found"
      });
    }

    const fileData = await fileResponse.json();

    // Add URLs
    const transformedFile = {
      ...fileData,
      url: `${PIPE_API_BASE_URL}/files/${fileId}/content`,
      publicUrl: fileData.public ? `${PIPE_API_BASE_URL}/public/${fileId}` : null
    };

    res.json(transformedFile);
  } catch (error) {
    console.error("Error fetching Pipe file:", error);
    res.status(500).json({ error: "Failed to fetch file" });
  }
});

// Debug endpoint to clear Pipe accounts (for testing)
app.post("/api/pipe/clear-accounts", (_req, res) => {
  pipeAccounts.clear();
  res.json({ message: "All Pipe accounts cleared from memory" });
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
      io.to(event.cameraId).emit("timelineEvent", newEvent);
    } else {
      // If no cameraId, broadcast to all
      io.emit("timelineEvent", newEvent);
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

  socket.on("register-viewer", (data: { cameraId: string }) => {
    console.log(
      `Viewer registering for WebRTC camera ${data.cameraId} on socket ${socket.id}`,
    );
    webrtcPeers.set(socket.id, { cameraId: data.cameraId, type: "viewer" });
    socket.join(`webrtc-${data.cameraId}`);

    // Check how many peers are in the room
    const roomSockets = io.sockets.adapter.rooms.get(`webrtc-${data.cameraId}`);
    console.log(
      `Room webrtc-${data.cameraId} has ${roomSockets ? roomSockets.size : 0} sockets`,
    );

    // Notify camera that a viewer wants to connect
    console.log(
      `Notifying camera in room webrtc-${data.cameraId} that viewer ${socket.id} wants to connect`,
    );
    socket
      .to(`webrtc-${data.cameraId}`)
      .emit("viewer-wants-connection", { viewerId: socket.id });
  });

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

// Start server with error handling
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
  });
});
