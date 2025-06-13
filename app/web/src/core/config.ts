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
  // Centralized middleware URL for all cameras
  const centralMiddlewareUrl = "https://pi5-middleware.mmoment.xyz";
  
  // Alternative middleware URLs for failover (in order of preference)
  
  // Only use localhost if we're explicitly in local development
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal && window.location.hostname === 'localhost') {
    console.log('Using local camera API (forced by VITE_FORCE_LOCAL)');
    return "http://localhost:5002";
  }
  
  // Override middleware URL if specified in environment
  const overrideUrl = import.meta.env.VITE_CAMERA_API_URL;
  if (overrideUrl) {
    console.log(`Using override camera API URL: ${overrideUrl}`);
    return overrideUrl;
  }

  // Default to centralized middleware
  return centralMiddlewareUrl;
};

// Function to get the direct camera hardware URL
const getCameraHardwareUrl = () => {
  // Central URL for all camera hardware
  const centralCameraUrl = "https://camera.mmoment.xyz";
  
  // Only use localhost if we're explicitly in local development
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal && window.location.hostname === 'localhost') {
    return "http://localhost:5001";
  }
  
  // Override camera URL if specified in environment
  const overrideUrl = import.meta.env.VITE_CAMERA_HARDWARE_URL;
  if (overrideUrl) {
    return overrideUrl;
  }

  return centralCameraUrl;
};

// Function to get the Jetson Orin Nano camera URL
const getJetsonCameraUrl = () => {
  // Override Jetson URL if specified in environment
  const overrideUrl = import.meta.env.VITE_JETSON_CAMERA_URL;
  if (overrideUrl) {
    return overrideUrl;
  }

  // For local development, check if we should use localhost
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal && window.location.hostname === 'localhost') {
    console.log('Using local Jetson camera API (forced by VITE_FORCE_LOCAL)');
    return "http://localhost:5002";
  }

  // Default to the Jetson camera service URL (remote)
  // Note: This should match whatever URL the Jetson is actually accessible from
  return "https://jetson.mmoment.xyz";
};

// Get WebSocket URL for timeline updates from Railway backend
const getTimelineWebSocketUrl = () => {
  // In development, try to connect to the local server first
  if (window.location.hostname.includes('localhost')) {
    // Use HTTP for health check and WS for socket connection
    return "ws://localhost:3001";
  }
  return "wss://mmoment-production.up.railway.app";
};

// For local development: check if we're trying to connect to localhost 
// but with a different port than what's running vite
const isUsingDifferentLocalPorts = () => {
  if (!window.location.hostname.includes('localhost')) return false;
  const vitePort = window.location.port; // e.g., 5173
  const backendPort = '3001';
  return vitePort !== backendPort;
};

// Export configuration
export const CONFIG = {
  baseUrl: forceHttps(window.location.origin),
  rpcEndpoint,
  devnetEndpoints,
  getNextEndpoint,
  // Camera API is your Pi5 device with the Python/Flask server
  CAMERA_API_URL: getCameraApiUrl(),
  CAMERA_HARDWARE_URL: getCameraHardwareUrl(),
  // Jetson Orin Nano camera service
  JETSON_CAMERA_URL: getJetsonCameraUrl(),
  // Timeline backend is your Railway service
  BACKEND_URL: isProduction 
    ? "https://mmoment-production.up.railway.app"
    : "http://localhost:3001",
  isProduction,
  isCloudflareProxy: isCloudflareProxy(),
  isMobileBrowser: isMobileBrowser(),
  LIVEPEER_PLAYBACK_ID: process.env.REACT_APP_LIVEPEER_PLAYBACK_ID || '',
  TIMELINE_WS_URL: getTimelineWebSocketUrl(),
  CAMERA_PDA: import.meta.env.VITE_CAMERA_PDA || 'EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA',
  // Jetson camera PDA
  JETSON_CAMERA_PDA: 'WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD',
  isUsingDifferentLocalPorts: isUsingDifferentLocalPorts()
};

// Socket.IO configuration for timeline (Railway backend)
export const timelineConfig = {
  wsUrl: CONFIG.TIMELINE_WS_URL,
  wsOptions: {
    reconnectionDelay: 1000,
    reconnection: true,
    reconnectionAttempts: 5,
    timeout: 15000, 
    autoConnect: true,
    forceNew: true,
    // For local development, don't use credentials to avoid CORS issues
    withCredentials: !CONFIG.isUsingDifferentLocalPorts,
    // Use appropriate transports based on device
    transports: CONFIG.isMobileBrowser 
      ? ['polling', 'websocket'] // Try polling first on mobile
      : ['websocket', 'polling']  // Try websocket first on desktop
  }
};