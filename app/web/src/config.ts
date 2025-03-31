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

// Get the cluster from environment variable, defaults to devnet if not specified
const cluster = import.meta.env.VITE_CLUSTER || 'devnet';

// Define alternative RPC endpoints for failover
const devnetEndpoints = [
  'https://api.devnet.solana.com',
  'https://devnet.helius-rpc.com/?api-key=15319106-7d8e-4cf6-8cbe-d3976e537de0',
  'https://devnet.rpcpool.com'
];

let currentEndpointIndex = 0;

// Simple endpoint rotation
const getNextEndpoint = () => {
  currentEndpointIndex = (currentEndpointIndex + 1) % devnetEndpoints.length;
  return devnetEndpoints[currentEndpointIndex];
};

// Set RPC endpoint based on the cluster
const rpcEndpoint = cluster === 'localnet' ? 'http://localhost:8899' : devnetEndpoints[0];

// Get the appropriate API URL based on environment

// Get WebSocket URL based on environment and protocol
const getWebSocketUrl = () => {
  if (isProduction) {
    // Always use WSS in production with the camera domain
    return "wss://camera.mmoment.xyz";
  }

  // For local development
  const localUrl = "ws://localhost:5001"; // Direct to camera API for websockets
  
  // If we're accessing the dev environment through HTTPS, use WSS
  if (window.location.protocol === 'https:') {
    return "wss://camera.mmoment.xyz";
  }
  
  return localUrl;
};

// Export configuration
export const CONFIG = {
  baseUrl: forceHttps(window.location.origin),
  rpcEndpoint,
  devnetEndpoints,
  getNextEndpoint,
  CAMERA_API_URL: isProduction 
    ? "https://middleware.mmoment.xyz"  // Always use the middleware URL in production
    : "http://localhost:5002",
  BACKEND_URL: isProduction 
    ? forceHttps("https://mmoment-production.up.railway.app")
    : "http://localhost:3001",
  isProduction,
  isCloudflareProxy: isCloudflareProxy(),
  LIVEPEER_PLAYBACK_ID: process.env.REACT_APP_LIVEPEER_PLAYBACK_ID || '',
  WS_URL: getWebSocketUrl(),
  CAMERA_PDA: import.meta.env.VITE_CAMERA_PDA || '5onKAv5c6VdBZ8a7D11XqF79Hdzuv3tnysjv4B2pQWZ2'
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