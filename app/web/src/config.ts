// src/config.ts

// Ensure we're using HTTPS in production
const isProduction = import.meta.env.PROD;
const forceHttps = (url: string) => {
  // Always use HTTPS for production and non-localhost URLs
  if (isProduction || !url.includes('localhost')) {
    return url.replace(/^http:\/\//i, 'https://').replace(/^ws:\/\//i, 'wss://');
  }
  return url;
};

// Helper to determine if we're behind Cloudflare
const isCloudflareProxy = () => {
  return typeof window !== 'undefined' && 
    (window.location.hostname === 'mmoment.xyz' || 
     window.location.hostname === 'camera.mmoment.xyz' ||
     window.location.hostname.endsWith('.mmoment.xyz'));
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
const getCameraApiUrl = () => {
  // Always use HTTPS for production and Cloudflare
  if (isProduction || isCloudflareProxy()) {
    return "https://middleware.mmoment.xyz";
  }

  // Only use localhost if we're explicitly in local development
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal) {
    console.log('Using local camera API (forced by VITE_FORCE_LOCAL)');
    return "http://localhost:5002";
  }

  // Default to HTTPS
  return "https://middleware.mmoment.xyz";
};

// Get WebSocket URL based on environment and protocol
const getWebSocketUrl = () => {
  // For production, use Railway URL with WSS
  if (isProduction || isCloudflareProxy()) {
    return forceHttps("wss://mmoment-production.up.railway.app");
  }

  // For local development
  return "ws://localhost:3001";
};

// Export configuration
export const CONFIG = {
  baseUrl: forceHttps(window.location.origin),
  rpcEndpoint,
  devnetEndpoints,
  getNextEndpoint,
  CAMERA_API_URL: getCameraApiUrl(),
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
    secure: true, // Always use secure WebSocket
    path: '/socket.io/',
    rejectUnauthorized: true,
    transports: ['websocket', 'polling'],
    upgrade: true,
    timeout: 20000,
    pingTimeout: 90000,
    pingInterval: 25000,
    reconnectionAttempts: 5,
    reconnectionDelayMax: 5000,
    autoConnect: true,
    forceNew: true,
    extraHeaders: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,HEAD,PUT,PATCH,POST,DELETE",
      "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type, Accept"
    }
  }
};