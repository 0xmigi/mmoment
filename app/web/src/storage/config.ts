// Storage configuration
// This file manages API credentials in a way that doesn't expose them in source code

// Default development credentials - these will be stored in localStorage, not in the source code
const DEV_CREDENTIALS = {
  PINATA_JWT: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiI4MTUxMzVjMi0xYzI0LTRiMWItOTEwOC00MjU2MGFlYzJhYzMiLCJlbWFpbCI6IjB4bWlnaS5ldGhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6ImJhNWJlZTBmYzM5YTM1ZTQ0MmI2Iiwic2NvcGVkS2V5U2VjcmV0IjoiOTVhNGRmOTIwMjU3ZTUwNDE5MWY5MDI1NzE3MjZjMDgzZDNlY2ZiZWEwOGMxMzY2MjhlNTgzMDc0NzFkMTlhOCIsImV4cCI6MTc3OTI4ODk3Mn0.e9zzPNZIeA49R4NG3zT1rAOqDEHO60QJHc1FFBTBiik',
  PINATA_API_KEY: '***REMOVED***',
  PINATA_API_SECRET: '***REMOVED***'
};

// Credential storage key
const STORAGE_KEY = 'pinata_dev_credentials';

// Store initial credentials in localStorage if they don't exist
function initializeCredentials() {
  if (typeof window !== 'undefined') {
    const storedCreds = localStorage.getItem(STORAGE_KEY);
    if (!storedCreds) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEV_CREDENTIALS));
      console.log('📦 Initialized development credentials in localStorage');
    }
  }
}

// Initialize on load
initializeCredentials();

// Get the appropriate credentials (environment variables or localStorage)
export function getPinataCredentials() {
  // First check environment variables
  const envJwt = import.meta.env.VITE_PINATA_JWT;
  const envApiKey = import.meta.env.VITE_PINATA_API_KEY;
  const envApiSecret = import.meta.env.VITE_PINATA_API_SECRET;

  // If environment variables are set, use those (production mode)
  if (envJwt || envApiKey) {
    return {
      PINATA_JWT: envJwt,
      PINATA_API_KEY: envApiKey,
      PINATA_API_SECRET: envApiSecret
    };
  }

  // Otherwise check localStorage (development mode)
  if (typeof window !== 'undefined') {
    try {
      const storedCreds = localStorage.getItem(STORAGE_KEY);
      const creds = storedCreds ? JSON.parse(storedCreds) : DEV_CREDENTIALS;
      return creds;
    } catch (error) {
      console.warn('⚠️ Error retrieving credentials from localStorage:', error);
    }
  }

  // Fallback to empty values
  return {
    PINATA_JWT: '',
    PINATA_API_KEY: '',
    PINATA_API_SECRET: ''
  };
}

// Update credentials in localStorage (only usable in development)
export function updatePinataCredentials(newCredentials: {
  PINATA_JWT?: string;
  PINATA_API_KEY?: string;
  PINATA_API_SECRET?: string;
}) {
  if (typeof window !== 'undefined') {
    try {
      const storedCreds = localStorage.getItem(STORAGE_KEY);
      const currentCreds = storedCreds ? JSON.parse(storedCreds) : DEV_CREDENTIALS;
      const updatedCreds = { ...currentCreds, ...newCredentials };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedCreds));
      console.log('✅ Updated Pinata credentials in localStorage');
      return true;
    } catch (error) {
      console.error('❌ Error updating credentials in localStorage:', error);
    }
  }
  return false;
} 