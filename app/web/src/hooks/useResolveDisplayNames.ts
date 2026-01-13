import { useState, useEffect, useRef, useCallback } from 'react';
import { profileService } from '../services/profile-service';

/**
 * Checks if a display name looks like a truncated wallet address
 * e.g., "RsLjCiEi...kdLT" or "0x1234...5678"
 */
function looksLikeWalletAddress(displayName: string | undefined): boolean {
  if (!displayName) return true;

  // Pattern: starts with alphanumeric, has "..." in middle, ends with alphanumeric
  // This catches both Solana (base58) and Ethereum (0x...) style truncations
  const truncatedWalletPattern = /^[A-Za-z0-9]{4,8}\.{3}[A-Za-z0-9]{4,6}$/;
  return truncatedWalletPattern.test(displayName);
}

/**
 * Formats a wallet address for display when no display name is available
 */
function formatWalletAddress(wallet: string): string {
  if (wallet.length <= 12) return wallet;
  return `${wallet.substring(0, 6)}...${wallet.substring(wallet.length - 4)}`;
}

interface DisplayNameCache {
  [walletAddress: string]: string;
}

// Global cache to avoid refetching across component instances
const globalCache: DisplayNameCache = {};

/**
 * Hook that resolves wallet addresses to display names from the backend.
 *
 * When data from Jetson/camera only includes wallet addresses (or truncated wallet
 * addresses as display names), this hook fetches the real display names from the
 * backend profile service.
 *
 * @param walletAddresses - Array of wallet addresses to resolve
 * @returns Object with resolved display names and loading state
 */
export function useResolveDisplayNames(walletAddresses: string[]) {
  const [displayNames, setDisplayNames] = useState<DisplayNameCache>({});
  const [loading, setLoading] = useState(false);
  const fetchedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const resolveNames = async () => {
      // Filter to addresses we haven't fetched yet and aren't in cache
      const addressesToFetch = walletAddresses.filter(addr =>
        addr &&
        !globalCache[addr] &&
        !fetchedRef.current.has(addr)
      );

      if (addressesToFetch.length === 0) {
        // Just use cached values
        const cached: DisplayNameCache = {};
        walletAddresses.forEach(addr => {
          if (addr && globalCache[addr]) {
            cached[addr] = globalCache[addr];
          }
        });
        setDisplayNames(prev => ({ ...prev, ...cached }));
        return;
      }

      setLoading(true);

      // Mark as fetching to prevent duplicate requests
      addressesToFetch.forEach(addr => fetchedRef.current.add(addr));

      try {
        const profiles = await profileService.getProfiles(addressesToFetch);

        const newNames: DisplayNameCache = {};
        addressesToFetch.forEach(addr => {
          const profile = profiles[addr];
          if (profile?.displayName) {
            newNames[addr] = profile.displayName;
            globalCache[addr] = profile.displayName;
          } else {
            // Use formatted wallet as fallback
            newNames[addr] = formatWalletAddress(addr);
            globalCache[addr] = newNames[addr];
          }
        });

        setDisplayNames(prev => ({ ...prev, ...newNames }));
      } catch (error) {
        console.error('[useResolveDisplayNames] Error fetching profiles:', error);
        // Use formatted wallet addresses as fallback
        const fallbacks: DisplayNameCache = {};
        addressesToFetch.forEach(addr => {
          fallbacks[addr] = formatWalletAddress(addr);
        });
        setDisplayNames(prev => ({ ...prev, ...fallbacks }));
      } finally {
        setLoading(false);
      }
    };

    if (walletAddresses.length > 0) {
      resolveNames();
    }
  }, [walletAddresses.join(',')]); // Re-run when addresses change

  /**
   * Get display name for a wallet address.
   * Returns the resolved name, or a formatted wallet address if not found.
   */
  const getDisplayName = useCallback((walletAddress: string, fallbackName?: string): string => {
    // Check global cache first
    if (globalCache[walletAddress]) {
      return globalCache[walletAddress];
    }

    // Check local state
    if (displayNames[walletAddress]) {
      return displayNames[walletAddress];
    }

    // If fallback doesn't look like a wallet, use it
    if (fallbackName && !looksLikeWalletAddress(fallbackName)) {
      return fallbackName;
    }

    // Last resort: format the wallet address
    return formatWalletAddress(walletAddress);
  }, [displayNames]);

  /**
   * Check if a display name needs resolution (looks like a wallet address)
   */
  const needsResolution = useCallback((displayName: string | undefined): boolean => {
    return looksLikeWalletAddress(displayName);
  }, []);

  return {
    displayNames,
    loading,
    getDisplayName,
    needsResolution,
  };
}

/**
 * Utility function to clear the global cache (useful for testing or logout)
 */
export function clearDisplayNameCache(): void {
  Object.keys(globalCache).forEach(key => delete globalCache[key]);
}
