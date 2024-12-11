// src/config.ts
export const CONFIG = {
  baseUrl: window.location.origin,
  rpcEndpoint: "https://api.devnet.solana.com",
  CAMERA_API_URL: "https://camera.mmoment.xyz",
  BACKEND_URL: "https://mmoment-production.up.railway.app"
};

export const timelineConfig = {
  wsUrl: CONFIG.BACKEND_URL,
  wsOptions: {
    reconnectionDelay: 1000,
    reconnection: true,
    secure: true,
    path: '/socket.io/'
  }
};