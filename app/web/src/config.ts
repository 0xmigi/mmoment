// src/config.ts

export const CAMERA_API_URL = "https://b624-2603-7000-9400-2792-00-12c8.ngrok-free.app";
export const CONFIG = {
    // Use window.location.origin to dynamically get the current origin
    // This helps with both localhost and local network IP testing
    baseUrl: window.location.origin,
    
    // RPC endpoints
    rpcEndpoint: "https://api.devnet.solana.com",
    
    // Camera API
    CAMERA_API_URL,
    
    // Add any other config values here
  };
  
  // Wallet adapter configuration
  export const walletConfig = {
    // List of allowed origins for wallet connections
    allowedOrigins: [
      'http://localhost:5173',
      'http://127.0.0.1:5173',
      window.location.origin, // Dynamically add current origin
    ],
  };