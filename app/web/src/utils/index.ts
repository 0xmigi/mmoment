// Utils Module Exports

// Helper to check if browser is mobile
export function isMobileBrowser(): boolean {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

// Format date for display
export function formatDate(timestamp: number): string {
  return new Date(timestamp).toLocaleString();
}

// Format short address
export function formatShortAddress(address: string): string {
  if (!address) return '';
  if (address.length <= 10) return address;
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

// Safely parse JSON
export function safeParseJSON<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json) as T;
  } catch (e) {
    return fallback;
  }
}

// Generate a random ID
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
} 