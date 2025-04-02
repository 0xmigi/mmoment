// src/config.ts

// Ensure we're using HTTPS in production
const isProduction = import.meta.env.PROD;

// Helper to determine if we're on a mobile browser
const isMobileBrowser = () => {
  if (typeof window === 'undefined') return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

const forceHttps = (url: string) => {
  // Always use HTTPS except for localhost
  if (!url.includes('localhost')) {
    // Handle websocket URLs
    if (url.startsWith('ws://')) {
      return url.replace('ws://', 'wss://');
    }
    // Handle HTTP URLs
    if (url.startsWith('http://')) {
      return url.replace('http://', 'https://');
    }
  }
  return url;
};

// Helper to determine if we're behind Cloudflare
const isCloudflareProxy = () => {
  if (typeof window === 'undefined') return false;
  const hostname = window.location.hostname;
  return hostname === 'mmoment.xyz' || 
         hostname === 'www.mmoment.xyz' ||
         hostname === 'camera.mmoment.xyz' ||
         hostname.endsWith('.mmoment.xyz');
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

// Simple endpoint rotation with retry logic
const getNextEndpoint = () => {
  currentEndpointIndex = (currentEndpointIndex + 1) % devnetEndpoints.length;
  return devnetEndpoints[currentEndpointIndex];
};

// Set RPC endpoint based on the cluster
const rpcEndpoint = cluster === 'localnet' ? 'http://localhost:8899' : devnetEndpoints[0];

// Get the appropriate API URL based on environment with fallbacks
const getCameraApiUrl = () => {
  // Primary URL should always be HTTPS
  const primaryUrl = "https://middleware.mmoment.xyz";
  
  // Only use localhost if we're explicitly in local development
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal && window.location.hostname === 'localhost') {
    console.log('Using local camera API (forced by VITE_FORCE_LOCAL)');
    return "http://localhost:5002";
  }

  // Default to HTTPS
  return primaryUrl;
};

// Get WebSocket URL based on environment and protocol with fallback
const getWebSocketUrl = () => {
  // For production or non-localhost, always use WSS
  if (!window.location.hostname.includes('localhost')) {
    return "wss://middleware.mmoment.xyz";
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
    ? "https://mmoment-production.up.railway.app"
    : "http://localhost:3001",
  isProduction,
  isCloudflareProxy: isCloudflareProxy(),
  isMobileBrowser: isMobileBrowser(),
  LIVEPEER_PLAYBACK_ID: process.env.REACT_APP_LIVEPEER_PLAYBACK_ID || '',
  WS_URL: getWebSocketUrl(),
  CAMERA_PDA: import.meta.env.VITE_CAMERA_PDA || '5onKAv5c6VdBZ8a7D11XqF79Hdzuv3tnysjv4B2pQWZ2'
};

// Socket.IO configuration with better mobile support and longer timeouts
export const timelineConfig = {
  wsUrl: CONFIG.WS_URL,
  wsOptions: {
    reconnectionDelay: 10000,
    reconnection: true,
    reconnectionAttempts: 10,
    secure: true,
    path: '/socket.io/',
    rejectUnauthorized: false, // Allow self-signed certs
    transports: ['polling', 'websocket'], // Try polling first
    upgrade: true,
    timeout: 60000,
    pingTimeout: 300000, // 5 minutes
    pingInterval: 25000,
    reconnectionDelayMax: 60000,
    autoConnect: true,
    forceNew: true,
    extraHeaders: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,HEAD,PUT,PATCH,POST,DELETE",
      "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type, Accept",
      "User-Agent": navigator.userAgent,
      "Cache-Control": "no-cache",
      "Pragma": "no-cache"
    }
  }
};