/**
 * Pipe Storage Section for Account Page
 *
 * Shows user's storage balance and allows purchasing more storage
 */

import { pipeService } from "../../storage/pipe/pipe-service";
import {
  pipeWalletBridge,
  PipeWalletInfo,
  StoragePurchaseOption,
} from "../../storage/pipe/pipe-wallet-bridge";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { isSolanaWallet } from "@dynamic-labs/solana";
import { useConnection } from "@solana/wallet-adapter-react";
import {
  HardDrive,
  Coins,
  ShoppingCart,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { useState, useEffect } from "react";

interface StatusMessage {
  message: string;
  type: "success" | "error" | "info";
}

export function PipeStorageSection() {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [pipeInfo, setPipeInfo] = useState<PipeWalletInfo | null>(null);
  const [pipeCredentials, setPipeCredentials] = useState<{
    userId: string;
    userAppKey: string;
    username?: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [purchasing, setPurchasing] = useState(false);
  const [selectedOption, setSelectedOption] =
    useState<StoragePurchaseOption | null>(null);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );

  // Load user's Pipe storage info
  useEffect(() => {
    loadPipeInfo();
  }, [primaryWallet?.address]);

  const loadPipeInfo = async () => {
    if (!primaryWallet?.address) {
      setLoading(false);
      return;
    }

    try {
      // Ensure Pipe service is initialized first
      await pipeService.ensureInitialized();

      // Always try to create/get account (backend will return existing if it exists)
      console.log("🔍 Checking for Pipe account...");
      const credentials = await pipeService.createOrGetAccount(
        primaryWallet.address
      );

      if (credentials) {
        // Store the credentials for debugging display
        setPipeCredentials({
          userId: credentials.userId,
          userAppKey: credentials.userAppKey,
          username: `mmoment_${primaryWallet.address.slice(0, 16)}`,
        });

        // Now get the wallet info
        const info = await pipeWalletBridge.getUserPipeInfo(
          primaryWallet.address
        );
        setPipeInfo(info);
      } else {
        console.log("❌ Could not get/create Pipe account");
        setPipeInfo(null);
        setPipeCredentials(null);
      }
    } catch (error) {
      console.error("Failed to load Pipe info:", error);
    } finally {
      setLoading(false);
    }
  };

  const handlePurchaseStorage = async (option: StoragePurchaseOption) => {
    if (!primaryWallet?.address || !connection) {
      setStatusMessage({
        type: "error",
        message: "Wallet or connection not available",
      });
      return;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setStatusMessage({ type: "error", message: "Not a Solana wallet" });
      return;
    }

    setPurchasing(true);
    setSelectedOption(option);
    setStatusMessage(null);

    try {
      // Copy the EXACT pattern from TransactionModal that works
      const signTransaction = async (tx: any) => {
        // Don't override blockhash - it's already set in pipe-wallet-bridge

        console.log(
          "🔐 Attempting to sign transaction with wallet:",
          primaryWallet.address
        );

        // Use wallet to sign - exact pattern from TransactionModal
        if (!isSolanaWallet(primaryWallet)) {
          throw new Error("Not a Solana wallet");
        }

        const signer = await primaryWallet.getSigner();
        const signedTx = await signer.signTransaction(tx);

        return signedTx;
      };

      const result = await pipeWalletBridge.purchaseStorage(
        primaryWallet.address,
        option.sizeGB,
        option.priceSol,
        signTransaction
      );

      if (result.success) {
        setStatusMessage({
          type: "success",
          message: `Successfully purchased ${option.sizeGB}GB of storage!`,
        });

        // Reload storage info
        await loadPipeInfo();
      } else {
        setStatusMessage({
          type: "error",
          message: result.error || "Purchase failed",
        });
      }
    } catch (error) {
      setStatusMessage({
        type: "error",
        message: error instanceof Error ? error.message : "Purchase failed",
      });
    } finally {
      setPurchasing(false);
      setSelectedOption(null);
    }
  };

  const storageOptions = pipeWalletBridge.getStorageOptions();
  const storageUsedPercent = pipeInfo
    ? (pipeInfo.storageUsed / pipeInfo.storageLimit) * 100
    : 0;

  if (loading) {
    return (
      <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
          <span className="ml-2 text-gray-600">Loading storage info...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <HardDrive className="w-5 h-5 text-gray-700" />
        <h2 className="text-lg font-medium">Pipe Storage</h2>
      </div>

      {/* Status Message */}
      {statusMessage && (
        <div
          className={`mb-4 p-3 rounded-lg ${
            statusMessage.type === "success"
              ? "bg-green-50 text-green-700 border border-green-200"
              : statusMessage.type === "error"
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-blue-50 text-blue-700 border border-blue-200"
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

      {pipeInfo ? (
        <>
          {/* Storage Usage */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Storage Used</span>
              <span className="text-sm font-medium">
                {pipeInfo.storageUsed}MB / {pipeInfo.storageLimit}MB
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(storageUsedPercent, 100)}%` }}
              />
            </div>
          </div>

          {/* Balance Info */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-white rounded-lg p-3 border">
              <div className="flex items-center gap-2 mb-1">
                <Coins className="w-4 h-4 text-yellow-600" />
                <span className="text-sm font-medium">SOL</span>
              </div>
              <div className="text-lg font-semibold">
                {pipeInfo.solBalance.toFixed(3)}
              </div>
            </div>
            <div className="bg-white rounded-lg p-3 border">
              <div className="flex items-center gap-2 mb-1">
                <HardDrive className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium">PIPE</span>
              </div>
              <div className="text-lg font-semibold">
                {pipeInfo.pipeBalance.toFixed(0)}
              </div>
            </div>
          </div>

          {/* Debug Account Info - Temporary for development */}
          {pipeCredentials && (
            <div className="mb-6 p-3 bg-gray-50 rounded-lg border border-gray-200">
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                🔧 Pipe Account Debug Info
              </h4>
              <div className="space-y-1 text-xs text-gray-600">
                <div>
                  <span className="font-medium">Username:</span> {pipeCredentials.username}
                </div>
                <div>
                  <span className="font-medium">User ID:</span> {pipeCredentials.userId}
                </div>
                <div>
                  <span className="font-medium">Pipe Wallet:</span> {pipeInfo.pipeWalletAddress}
                </div>
                <div>
                  <span className="font-medium">Connected Wallet:</span> {primaryWallet?.address?.slice(0, 16)}...
                </div>
              </div>
            </div>
          )}

          {/* Purchase Options */}
          <div>
            <h3 className="text-md font-medium mb-3">Buy More Storage</h3>
            <div className="space-y-2">
              {storageOptions.map((option) => (
                <button
                  key={option.label}
                  onClick={() => handlePurchaseStorage(option)}
                  disabled={purchasing}
                  className={`w-full p-3 rounded-lg border text-left transition-all ${
                    option.recommended
                      ? "border-blue-200 bg-blue-50 hover:bg-blue-100"
                      : "border-gray-200 bg-white hover:bg-gray-50"
                  } ${purchasing ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{option.label}</span>
                        {option.recommended && (
                          <span className="px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
                            Recommended
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-600">
                        {option.priceSol} SOL (
                        {(option.priceSol / option.sizeGB).toFixed(3)} SOL/GB)
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {purchasing && selectedOption?.label === option.label ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <ShoppingCart className="w-4 h-4" />
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Info Text */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-700">
              💡 Your photos are stored in your own Pipe account with
              client-side encryption. Only you can access them.
            </p>
          </div>
        </>
      ) : (
        /* Pipe Not Available */
        <div className="text-center py-6">
          <HardDrive className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <h3 className="font-medium text-gray-700 mb-2">
            Pipe Storage Not Available
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Pipe Network provides decentralized storage for your camera
            captures.
          </p>
          <div className="text-xs text-gray-500 bg-gray-100 rounded-lg p-3">
            To enable: Camera system needs Pipe credentials configured.
          </div>
        </div>
      )}
    </div>
  );
}
