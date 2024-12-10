// src/config.ts
export const CONFIG = {
  baseUrl: window.location.origin,
  rpcEndpoint: "https://api.devnet.solana.com",
  CAMERA_API_URL: "https://camera.mmoment.xyz", // Make sure this matches your actual camera API URL
  BACKEND_URL: "mmoment-production.up.railway.app" // Add your actual backend URL here
};

// Timeline service configuration
export const timelineConfig = {
  wsUrl: CONFIG.BACKEND_URL,
  wsOptions: {
    reconnectionDelay: 1000,
    reconnection: true,
    transports: ['websocket'],
    secure: true
  }
};