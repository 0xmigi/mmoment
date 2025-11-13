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

/**
 * Generate camera API URL based on PDA subdomain system
 * @param cameraPda - The camera's PDA address
 * @returns The API URL for the specific camera
 */
const getCameraApiUrlByPda = (cameraPda: string): string => {
  // For local development
  const forceLocal = import.meta.env.VITE_FORCE_LOCAL === 'true';
  if (forceLocal && window.location.hostname === 'localhost') {
    console.log(`Using local camera API for PDA: ${cameraPda}`);
    return "http://localhost:5002";
  }

  // Convert PDA to lowercase for subdomain
  const pdaSubdomain = cameraPda.toLowerCase();
  
  // Use PDA-based subdomain system
  const pdaUrl = `https://${pdaSubdomain}.mmoment.xyz`;
  
  console.log(`Generated PDA-based URL for ${cameraPda}: ${pdaUrl}`);
  return pdaUrl;
};

/**
 * Utility function to convert PDA to subdomain format
 * Handles edge cases and validation
 */
const pdaToSubdomain = (pda: string): string => {
  if (!pda || typeof pda !== 'string') {
    throw new Error('Invalid PDA: must be a non-empty string');
  }
  
  // Convert to lowercase for subdomain compatibility
  const subdomain = pda.toLowerCase();
  
  // Validate subdomain format (basic check)
  if (subdomain.length < 4 || subdomain.length > 63) {
    throw new Error('Invalid PDA length for subdomain');
  }
  
  // Check for invalid characters (subdomains can only contain alphanumeric and hyphens)
  if (!/^[a-z0-9]+$/.test(subdomain)) {
    console.warn(`PDA ${pda} contains characters that may not be valid for subdomains`);
  }
  
  return subdomain;
};

/**
 * Get camera URL with fallback support
 * @param cameraPda - The camera's PDA address
 * @param fallbackUrl - Optional fallback URL if PDA-based URL fails
 * @returns The API URL for the camera
 */
const getCameraUrlWithFallback = (cameraPda: string, fallbackUrl?: string): string => {
  try {
    return getCameraApiUrlByPda(cameraPda);
  } catch (error) {
    console.warn(`Failed to generate PDA-based URL for ${cameraPda}:`, error);
    if (fallbackUrl) {
      console.log(`Using fallback URL: ${fallbackUrl}`);
      return fallbackUrl;
    }
    throw error;
  }
};

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

// Function to get the Jetson Orin Nano camera URL (legacy - now uses PDA-based URLs)
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

  // Default to the Jetson camera service URL (legacy)
  // This maintains backward compatibility
  return "https://jetson.mmoment.xyz";
};

// Get WebSocket URL for timeline updates from Railway backend
const getTimelineWebSocketUrl = () => {
  // In development, try to connect to the local server first
  if (window.location.hostname.includes('localhost')) {
    // Use HTTP for health check and WS for socket connection
    return "ws://192.168.1.232:3001";
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
  // Jetson Orin Nano camera service (legacy)
  JETSON_CAMERA_URL: getJetsonCameraUrl(),
  // New PDA-based URL generation functions
  getCameraApiUrlByPda,
  pdaToSubdomain,
  getCameraUrlWithFallback,
  // Timeline backend is your Railway service
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || (isProduction
    ? "https://mmoment-production.up.railway.app"
    : "http://localhost:3001"),
  isProduction,
  isCloudflareProxy: isCloudflareProxy(),
  isMobileBrowser: isMobileBrowser(),
  LIVEPEER_PLAYBOOK_ID: process.env.REACT_APP_LIVEPEER_PLAYBOOK_ID || '',
  TIMELINE_WS_URL: getTimelineWebSocketUrl(),
  CAMERA_PDA: import.meta.env.VITE_CAMERA_PDA || 'EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA',
  // Jetson camera PDA
  JETSON_CAMERA_PDA: 'FZ4DgqxLCNpLp1vyvvSZ5A24uyBEUdavvkm5qFE6D54t',
  isUsingDifferentLocalPorts: isUsingDifferentLocalPorts(),
  
  // Known camera configurations with PDA-based URLs
  KNOWN_CAMERAS: {
    // Jetson Orin Nano
    'H1WoNBkWJgNcePeyr65xEEwjFgGDboSpL5UbJan5VyhG': {
      type: 'jetson',
      name: 'Jetson Orin Nano Camera',
      description: 'NVIDIA Jetson Orin Nano with advanced computer vision',
      // Legacy URL for backward compatibility
      legacyUrl: 'https://jetson.mmoment.xyz'
    },
    // Pi5 Camera
    'EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA': {
      type: 'pi5',
      name: 'Raspberry Pi 5 Camera',
      description: 'Raspberry Pi 5 with camera module',
      legacyUrl: 'https://pi5-middleware.mmoment.xyz'
    }
  }
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