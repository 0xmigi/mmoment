/**
 * Walrus Storage Section for Account Page
 *
 * Shows user's Walrus/Sui wallet balances (SUI + WAL) and blob storage info
 */

import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { CONFIG } from "../../core/config";
import {
  HardDrive,
  Coins,
  Loader2,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Database,
} from "lucide-react";
import { useState, useEffect, useCallback } from "react";

// Sui RPC endpoint for mainnet
const SUI_RPC_URL = "https://fullnode.mainnet.sui.io:443";

// Coin types
const SUI_COIN_TYPE = "0x2::sui::SUI";
const WAL_COIN_TYPE = "0x356a26eb9e012a68958082340d4c4116e7f55615cf27affcff209cf0ae544f59::wal::WAL";

// The Walrus address to display
const WALRUS_ADDRESS = "0x785cdcca561738ae78a971b2f50d6af87b768af133f20d635361dd20134ba26d";

interface StatusMessage {
  message: string;
  type: "success" | "error" | "info";
}

interface WalrusWalletInfo {
  suiBalance: number;
  walBalance: number;
  blobCount: number;
  storageUsedMB: number;
}

/**
 * Query a single coin balance from Sui RPC
 */
async function getSuiCoinBalance(address: string, coinType: string): Promise<number> {
  try {
    const response = await fetch(SUI_RPC_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "suix_getBalance",
        params: [address, coinType],
      }),
    });

    if (!response.ok) {
      throw new Error(`RPC request failed: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      console.error("Sui RPC error:", data.error);
      return 0;
    }

    // Balance is returned in smallest units (MIST for SUI, similar for WAL)
    // 1 SUI = 10^9 MIST
    const totalBalance = BigInt(data.result?.totalBalance || "0");
    return Number(totalBalance) / 1e9;
  } catch (error) {
    console.error(`Failed to fetch ${coinType} balance:`, error);
    return 0;
  }
}

/**
 * Get all balances for an address
 */
async function getAllBalances(address: string): Promise<{ sui: number; wal: number }> {
  const [suiBalance, walBalance] = await Promise.all([
    getSuiCoinBalance(address, SUI_COIN_TYPE),
    getSuiCoinBalance(address, WAL_COIN_TYPE),
  ]);

  return { sui: suiBalance, wal: walBalance };
}

export function WalrusStorageSection() {
  const { primaryWallet } = useDynamicContext();
  const [walrusInfo, setWalrusInfo] = useState<WalrusWalletInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);

  const loadWalrusInfo = useCallback(async () => {
    try {
      console.log("📊 Fetching Walrus account status...");

      // Fetch balances from Sui RPC
      const balances = await getAllBalances(WALRUS_ADDRESS);

      // Fetch blob count from backend (if available)
      let blobCount = 0;
      let storageUsedMB = 0;

      try {
        // Try to get blob count from backend API
        const response = await fetch(
          `${CONFIG.BACKEND_URL}/api/walrus/gallery/${primaryWallet?.address || WALRUS_ADDRESS}?includeShared=false`
        );

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            blobCount = data.ownedCount || data.items?.length || 0;
            // Estimate storage based on average blob size (rough estimate)
            storageUsedMB = blobCount * 0.5; // Assume ~0.5MB average per photo/video
          }
        }
      } catch (err) {
        console.warn("Could not fetch blob count from backend:", err);
      }

      setWalrusInfo({
        suiBalance: balances.sui,
        walBalance: balances.wal,
        blobCount,
        storageUsedMB,
      });

      console.log("✅ Loaded Walrus account info:", {
        sui: balances.sui.toFixed(4),
        wal: balances.wal.toFixed(4),
        blobs: blobCount,
      });
    } catch (error) {
      console.error("Failed to load Walrus info:", error);
      setStatusMessage({
        type: "error",
        message: "Failed to fetch wallet info",
      });
      setWalrusInfo(null);
    } finally {
      setLoading(false);
    }
  }, [primaryWallet?.address]);

  // Load wallet info on mount and periodically refresh
  useEffect(() => {
    loadWalrusInfo();

    // Refresh every 30 seconds
    const interval = setInterval(loadWalrusInfo, 30000);

    return () => clearInterval(interval);
  }, [loadWalrusInfo]);

  const formatBalance = (balance: number, decimals: number = 4): string => {
    if (balance === 0) return "0";
    if (balance < 0.0001) return "<0.0001";
    return balance.toFixed(decimals);
  };

  const shortenAddress = (address: string): string => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  if (loading) {
    return (
      <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-600">Loading Walrus info...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <HardDrive className="w-5 h-5 text-gray-700" />
        <h2 className="text-lg font-medium">Walrus Storage</h2>
      </div>

      {/* Status Message */}
      {statusMessage && (
        <div
          className={`mb-4 p-3 rounded-lg ${
            statusMessage.type === "success"
              ? "bg-green-50 text-green-700 border border-green-200"
              : statusMessage.type === "error"
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-primary-light text-primary border border-primary-light"
          }`}
        >
          <div className="flex items-center gap-2">
            {statusMessage.type === "success" && (
              <CheckCircle className="w-4 h-4" />
            )}
            {statusMessage.type === "error" && (
              <AlertCircle className="w-4 h-4" />
            )}
            <p className="text-sm">{statusMessage.message}</p>
          </div>
        </div>
      )}

      {walrusInfo ? (
        <>
          {/* Balance Info */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-white rounded-lg p-3 border">
              <div className="flex items-center gap-2 mb-1">
                <Coins className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium">SUI</span>
              </div>
              <div className="text-lg font-semibold">
                {formatBalance(walrusInfo.suiBalance)}
              </div>
            </div>
            <div className="bg-white rounded-lg p-3 border">
              <div className="flex items-center gap-2 mb-1">
                <HardDrive className="w-4 h-4 text-purple-500" />
                <span className="text-sm font-medium">WAL</span>
              </div>
              <div className="text-lg font-semibold">
                {formatBalance(walrusInfo.walBalance)}
              </div>
            </div>
          </div>

          {/* Blob Storage Stats */}
          {walrusInfo.blobCount > 0 && (
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-gray-500" />
                  <span className="text-sm text-gray-600">Blobs Stored</span>
                </div>
                <span className="text-sm font-medium">
                  {walrusInfo.blobCount} blob{walrusInfo.blobCount !== 1 ? 's' : ''}
                </span>
              </div>
              {walrusInfo.storageUsedMB > 0 && (
                <div className="text-xs text-gray-500">
                  ~{walrusInfo.storageUsedMB.toFixed(1)} MB estimated
                </div>
              )}
            </div>
          )}

          {/* Wallet Address Info */}
          <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <h4 className="text-sm font-medium text-gray-700 mb-2">
              Sui Wallet
            </h4>
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex items-center justify-between">
                <span className="font-medium">Address:</span>
                <a
                  href={`https://suiscan.xyz/mainnet/account/${WALRUS_ADDRESS}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-primary hover:underline font-mono"
                >
                  {shortenAddress(WALRUS_ADDRESS)}
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">Network:</span>
                <span>Sui Mainnet</span>
              </div>
            </div>
          </div>

          {/* Refresh Button */}
          <button
            onClick={() => {
              setLoading(true);
              loadWalrusInfo();
            }}
            className="w-full py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
          >
            Refresh
          </button>

          {/* Info Text */}
          <div className="mt-4 p-3 bg-primary-light rounded-lg border border-primary-light">
            <p className="text-xs text-primary">
              Your photos are stored on Walrus decentralized storage with client-side encryption. Only you can access them.
            </p>
          </div>
        </>
      ) : (
        /* Walrus Not Available */
        <div className="text-center py-6">
          <HardDrive className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <h3 className="font-medium text-gray-700 mb-2">
            Walrus Storage
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Could not fetch wallet information from Sui network.
          </p>
          <button
            onClick={() => {
              setLoading(true);
              setStatusMessage(null);
              loadWalrusInfo();
            }}
            className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-hover transition-colors"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
