// src/config.ts

// Ensure we're using HTTPS in production
const isProduction = import.meta.env.PROD;
const forceHttps = (url: string) => {
  if (isProduction) {
    // Always use HTTPS in production, regardless of input
    return url.replace(/^http:\/\//i, 'https://');
  }
  return url;
};

// Helper to determine if we're behind Cloudflare
const isCloudflareProxy = () => {
  return typeof window !== 'undefined' && 
    (window.location.hostname === 'mmoment.xyz' || 
     window.location.hostname === 'camera.mmoment.xyz');
};

// Get the appropriate API URL based on environment
const getApiUrl = async (): Promise<string> => {
  if (isProduction) {
    return forceHttps("https://camera.mmoment.xyz");
  }
  // In development, check if we're running the API locally
  const localApi = "http://localhost:3001";
  try {
    await fetch(`${localApi}/health`, { 
      method: 'HEAD',
      headers: { 'Cache-Control': 'no-cache' }
    });
    return localApi;
  } catch {
    return forceHttps("https://camera.mmoment.xyz");
  }
};

// Get WebSocket URL based on environment and protocol
const getWebSocketUrl = () => {
  // Check if we're on HTTPS
  const isSecure = window.location.protocol === 'https:' || isProduction;
  
  if (isProduction || isSecure) {
    // Use WSS for production or when on HTTPS
    return "wss://camera.mmoment.xyz";
  }

  // For local development on HTTP
  return "ws://localhost:3001";
};

export const CONFIG = {
  baseUrl: forceHttps(window.location.origin),
  rpcEndpoint: "https://api.devnet.solana.com",
  CAMERA_API_URL: forceHttps("https://camera.mmoment.xyz"), // Will be updated after initialization
  BACKEND_URL: isProduction 
    ? forceHttps("https://mmoment-production.up.railway.app")
    : "http://localhost:3001",
  isProduction,
  isCloudflareProxy: isCloudflareProxy(),
  LIVEPEER_PLAYBACK_ID: process.env.REACT_APP_LIVEPEER_PLAYBACK_ID || '',
  WS_URL: getWebSocketUrl()
};

export const timelineConfig = {
  wsUrl: CONFIG.WS_URL,
  wsOptions: {
    reconnectionDelay: 1000,
    reconnection: true,
    secure: isProduction || window.location.protocol === 'https:',
    path: '/socket.io/',
    rejectUnauthorized: !import.meta.env.DEV,
    transports: ['websocket'],
    upgrade: false,
    timeout: 5000,
    pingTimeout: 90000,
    pingInterval: 25000,
    reconnectionAttempts: 3,
    reconnectionDelayMax: 5000,
    autoConnect: false,
    forceNew: true
  }
};

// Initialize the configuration
(async () => {
  CONFIG.CAMERA_API_URL = await getApiUrl();
})();