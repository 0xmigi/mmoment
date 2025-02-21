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

export const CONFIG = {
  baseUrl: forceHttps(window.location.origin),
  rpcEndpoint: "https://api.devnet.solana.com",
  CAMERA_API_URL: forceHttps("https://camera.mmoment.xyz"),
  BACKEND_URL: isProduction 
    ? forceHttps("https://mmoment-production.up.railway.app")
    : "http://localhost:3001",
  isProduction,
  isCloudflareProxy: isCloudflareProxy()
};

export const timelineConfig = {
  wsUrl: CONFIG.BACKEND_URL,
  wsOptions: {
    reconnectionDelay: 1000,
    reconnection: true,
    secure: isProduction, // Only use secure in production
    path: '/socket.io/',
    rejectUnauthorized: !import.meta.env.DEV, // Allow self-signed certificates in development
    transports: ['websocket'], // Prefer WebSocket transport
    upgrade: true,
    // Additional options for production
    ...(isProduction && {
      timeout: 10000,
      // Cloudflare has a 100s timeout
      pingTimeout: 90000,
      pingInterval: 25000
    })
  }
};