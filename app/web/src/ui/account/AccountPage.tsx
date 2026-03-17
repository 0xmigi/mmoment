import { WalrusStorageSection } from "./WalrusStorageSection";
import { RecognitionTokenModal } from "./RecognitionTokenModal";
import { WalletBalanceModal } from "./WalletBalanceModal";
import {
  useDynamicContext,
  useEmbeddedWallet,
} from "@dynamic-labs/sdk-react-core";
import { useConnection } from "@solana/wallet-adapter-react";
import { PublicKey, LAMPORTS_PER_SOL } from "@solana/web3.js";
import {
  User,
  LogOut,
  KeyRound,
  Globe,
  Lock,
  CheckCircle,
  AlertCircle,
  Loader2,
  ChevronRight,
  Camera,
  Pencil,
} from "lucide-react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useFacialEmbeddingStatus } from "../../hooks/useFacialEmbeddingStatus";
import { useUserSessionChain } from "../../hooks/useUserSessionChain";
import { profileService } from "../../services/profile-service";
import { useDisplayProfile } from "../../auth/useDisplayProfile";

interface StatusMessage {
  message: string;
  type: "success" | "error" | "info";
}

export function AccountPage() {
  const { primaryWallet, handleLogOut, user } = useDynamicContext();
  const { revealWalletKey } = useEmbeddedWallet();
  const { connection } = useConnection();
  const navigate = useNavigate();
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );
  const [showBackupOptions, setShowBackupOptions] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [showRecognitionModal, setShowRecognitionModal] = useState(false);
  const [showWalletModal, setShowWalletModal] = useState(false);
  const [solBalance, setSolBalance] = useState<number | null>(null);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editNameValue, setEditNameValue] = useState('');
  const [isSavingName, setIsSavingName] = useState(false);

  // Get facial embedding status from blockchain
  const facialEmbeddingStatus = useFacialEmbeddingStatus();

  // Get session keychain status from blockchain
  const sessionChainStatus = useUserSessionChain();

  // Fetch SOL balance
  useEffect(() => {
    const fetchBalance = async () => {
      if (!primaryWallet?.address || !connection) return;

      try {
        const publicKey = new PublicKey(primaryWallet.address);
        const balance = await connection.getBalance(publicKey);
        setSolBalance(balance / LAMPORTS_PER_SOL);
      } catch (error) {
        console.error('Error fetching balance:', error);
      }
    };

    fetchBalance();
    // Refresh balance every 30 seconds
    const interval = setInterval(fetchBalance, 30000);
    return () => clearInterval(interval);
  }, [primaryWallet?.address, connection]);

  const handleSignOut = async () => {
    try {
      await handleLogOut();
      navigate("/");
    } catch (err) {
      console.error("Failed to sign out:", err);
    }
  };


  const handleExportWallet = async (type: "recoveryPhrase" | "privateKey") => {
    setIsExporting(true);
    setExportError(null);
    try {
      await revealWalletKey({
        htmlContainerId: "wallet-export-container",
        type,
      });
      setStatusMessage({
        type: "success",
        message: `${
          type === "privateKey" ? "Private key" : "Recovery phrase"
        } displayed.`,
      });
    } catch (error) {
      console.error("Failed to export wallet:", error);
      setExportError("Failed to export wallet. Please try again.");
      setStatusMessage({ type: "error", message: "Failed to export wallet." });
    } finally {
      setIsExporting(false);
    }
  };

  if (!primaryWallet?.address) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-neutral-600">Loading...</div>
      </div>
    );
  }

  // Resolved display profile — single source of truth, never contains email
  const displayProfile = useDisplayProfile();
  const profileImageUrl = displayProfile?.profileImage;
  const displayName = displayProfile?.name || primaryWallet.address.slice(0, 6) + "..." + primaryWallet.address.slice(-4);
  const primarySocialProvider = displayProfile?.hasSocialAuth ? displayProfile.providerLabel : null;

  // Check if the wallet is an embedded wallet (not Phantom)
  const isEmbeddedWallet =
    primaryWallet.connector?.name.toLowerCase() !== "phantom";

  // Define identity items using displayProfile (never leaks email)
  const socials = displayProfile?.socials || {};
  const identities = [
    {
      id: "google",
      label: "Google",
      value: socials.google?.displayName || 'Connected',
      connected: !!socials.google,
      isPublic: false,
      icon: <Globe className="w-3 h-3 mr-1" />,
    },
    {
      id: "twitter",
      label: "X / Twitter",
      value: socials.twitter?.username,
      connected: !!socials.twitter,
      isPublic: true,
      icon: <Globe className="w-3 h-3 mr-1" />,
    },
    {
      id: "farcaster",
      label: "Farcaster",
      value: socials.farcaster?.username,
      connected: !!socials.farcaster,
      isPublic: true,
      icon: <Globe className="w-3 h-3 mr-1" />,
    },
    {
      id: "email",
      label: "Email",
      value: user?.email ? 'Connected' : undefined,
      connected: !!user?.email,
      isPublic: false,
      icon: <Lock className="w-3 h-3 mr-1" />,
    },
    {
      id: "recognition",
      label: "Recognition Token",
      value: facialEmbeddingStatus.hasEmbedding ? "Active" : "Not Enrolled",
      connected: facialEmbeddingStatus.hasEmbedding,
      isPublic: false,
      isRecognition: true,
      status: facialEmbeddingStatus,
      icon: facialEmbeddingStatus.isLoading
        ? <Loader2 className="w-3 h-3 mr-1 animate-spin text-primary" />
        : facialEmbeddingStatus.hasEmbedding
        ? <CheckCircle className="w-3 h-3 mr-1 text-green-500" />
        : <AlertCircle className="w-3 h-3 mr-1 text-orange-500" />,
    },
    {
      id: "wallet",
      label: "Solana Wallet",
      value: primaryWallet.address,
      shortValue: `${primaryWallet.address.slice(
        0,
        6
      )}...${primaryWallet.address.slice(-4)}`,
      connected: true,
      isPublic: true,
      isWallet: true,
      icon: <Globe className="w-3 h-3 mr-1" />,
    },
    {
      id: "sessionKeychain",
      label: "Session Keychain",
      value: sessionChainStatus.hasSessionChain
        ? `${sessionChainStatus.sessionCount} key${sessionChainStatus.sessionCount !== 1 ? 's' : ''} stored`
        : "Not set up",
      connected: sessionChainStatus.hasSessionChain,
      isPublic: false,
      isSessionKeychain: true,
      status: sessionChainStatus,
      icon: sessionChainStatus.isLoading
        ? <Loader2 className="w-3 h-3 mr-1 animate-spin text-primary" />
        : sessionChainStatus.hasSessionChain
        ? <CheckCircle className="w-3 h-3 mr-1 text-green-500" />
        : <AlertCircle className="w-3 h-3 mr-1 text-orange-500" />,
    },
  ].filter(
    (item) =>
      item.connected || ["farcaster", "email", "twitter", "recognition", "sessionKeychain"].includes(item.id)
  );

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto pt-8 px-4">
        <div className="bg-white mb-6">
          <h1 className="text-xl font-semibold">Account</h1>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div
            className={`mb-4 p-3 rounded-lg ${
              statusMessage.type === "success"
                ? "bg-green-50 text-green-700"
                : statusMessage.type === "error"
                ? "bg-red-50 text-red-700"
                : "bg-primary-light text-primary"
            }`}
          >
            <p>{statusMessage.message}</p>
          </div>
        )}

        {/* Identity Section */}
        <div className="bg-neutral-100 rounded-xl p-4 sm:p-6 mb-6">
          <div className="flex items-baseline justify-between mb-6">
            <h2 className="text-lg font-medium">Identity</h2>
            {solBalance !== null && (
              <div className="text-sm font-medium text-neutral-600">
                {solBalance.toFixed(2)} SOL
                <span className="text-xs text-neutral-400 ml-2">
                  ${(solBalance * 150).toFixed(0)}
                </span>
              </div>
            )}
          </div>

          {/* Profile Container */}
          <div className="mb-6">
            <div className="flex items-center mb-6">
              {/* Profile Avatar */}
              {profileImageUrl ? (
                <img
                  src={profileImageUrl}
                  alt={displayName}
                  referrerPolicy="no-referrer"
                  className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border-2 border-neutral-200"
                />
              ) : (
                <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-neutral-200 flex items-center justify-center">
                  <User className="w-8 h-8 sm:w-10 sm:h-10 text-neutral-400" />
                </div>
              )}

              {/* Profile name — editable if no social provider set it */}
              <div className="ml-4 flex-1">
                {isEditingName ? (
                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!editNameValue.trim() || !primaryWallet?.address) return;
                      setIsSavingName(true);
                      try {
                        await profileService.saveProfile({
                          walletAddress: primaryWallet.address,
                          displayName: editNameValue.trim(),
                        });
                        displayProfile?.refresh();
                        setIsEditingName(false);
                        setStatusMessage({ type: 'success', message: 'Display name saved!' });
                      } catch {
                        setStatusMessage({ type: 'error', message: 'Failed to save name' });
                      } finally {
                        setIsSavingName(false);
                      }
                    }}
                    className="flex items-center gap-2"
                  >
                    <input
                      type="text"
                      value={editNameValue}
                      onChange={(e) => setEditNameValue(e.target.value)}
                      placeholder="Enter your name"
                      autoFocus
                      maxLength={50}
                      className="flex-1 text-sm border border-neutral-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-primary"
                    />
                    <button
                      type="submit"
                      disabled={isSavingName || !editNameValue.trim()}
                      className="text-xs bg-primary text-white px-3 py-1.5 rounded-lg disabled:opacity-50"
                    >
                      {isSavingName ? '...' : 'Save'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsEditingName(false)}
                      className="text-xs text-neutral-500 px-2 py-1.5"
                    >
                      Cancel
                    </button>
                  </form>
                ) : (
                  <div className="flex items-center gap-2">
                    <div>
                      <div className="font-medium text-neutral-900">{displayName}</div>
                      <div className="text-sm text-neutral-600">
                        {primarySocialProvider || "Wallet Address"}
                      </div>
                    </div>
                    {!displayProfile?.hasSocialAuth && (
                      <button
                        onClick={() => {
                          setEditNameValue(displayName.includes('...') ? '' : displayName);
                          setIsEditingName(true);
                        }}
                        className="p-1 text-neutral-400 hover:text-neutral-600 transition-colors"
                        title="Edit display name"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Identity List */}
            <div className="space-y-3">
              {identities.map((identity, idx) => {
                const isClickable = identity.isWallet || identity.isRecognition;
                const handleClick = () => {
                  if (identity.isWallet) {
                    setShowWalletModal(true);
                  } else if (identity.isRecognition) {
                    setShowRecognitionModal(true);
                  }
                };

                return (
                  <div
                    key={idx}
                    className={`flex justify-between items-center py-3 px-3 -mx-3 rounded-lg ${
                      isClickable
                        ? 'cursor-pointer hover:bg-white/60 transition-colors'
                        : ''
                    }`}
                    onClick={isClickable ? handleClick : undefined}
                  >
                    {/* Identity Info */}
                    <div className="flex items-center flex-1">
                      <div className="flex items-center justify-center w-8 h-8 mr-3">
                        {identity.icon}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-neutral-900 text-sm">
                          {identity.label}
                        </div>
                        <div className="text-xs text-neutral-400 mt-0.5">
                          {identity.connected ? (
                            identity.isWallet ? (
                              <span className="font-mono">{identity.shortValue}</span>
                            ) : identity.isRecognition ? (
                              <span className={identity.status.hasEmbedding ? 'text-green-600' : 'text-orange-600'}>
                                {identity.value}
                              </span>
                            ) : identity.isSessionKeychain ? (
                              <span className={identity.status.hasSessionChain ? 'text-green-600' : 'text-orange-600'}>
                                {identity.value}
                              </span>
                            ) : (
                              <>
                                {identity.id === "twitter" && "@"}
                                {identity.value}
                                {identity.isPublic && <span className="ml-2 text-neutral-400">• Public</span>}
                                {!identity.isPublic && identity.value && <span className="ml-2 text-neutral-400">• Private</span>}
                              </>
                            )
                          ) : identity.isSessionKeychain ? (
                            <span className="text-neutral-400">Created on first check-in</span>
                          ) : (
                            <span className="text-neutral-400">Not connected</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Action Indicator */}
                    {isClickable && (
                      <ChevronRight className="w-4 h-4 text-neutral-400 ml-2" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Walrus Storage Section */}
        <WalrusStorageSection />

        {/* Wallet Backup Section - responsive padding */}
        {isEmbeddedWallet && (
          <div className="bg-neutral-100 rounded-xl px-4 py-4 mb-4">
            <div className="text-sm">
              <div className="font-medium mb-3">Wallet Backup</div>
              <button
                onClick={() => setShowBackupOptions(!showBackupOptions)}
                className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-neutral-600 text-white rounded-lg text-sm font-medium hover:bg-neutral-600/90 transition-colors"
              >
                <KeyRound className="w-4 h-4" />
                Back up Wallet
              </button>
              {showBackupOptions && (
                <div className="mt-4 space-y-2">
                  <button
                    onClick={() => handleExportWallet("recoveryPhrase")}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-neutral-200 text-neutral-600 rounded-lg text-sm font-medium hover:bg-neutral-100 transition-colors disabled:opacity-50"
                  >
                    Show Recovery Phrase
                  </button>
                  <button
                    onClick={() => handleExportWallet("privateKey")}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-neutral-200 text-neutral-600 rounded-lg text-sm font-medium hover:bg-neutral-100 transition-colors disabled:opacity-50"
                  >
                    Show Private Key
                  </button>
                </div>
              )}
              {exportError && (
                <p className="text-sm text-red-600 mt-2">{exportError}</p>
              )}
              <div
                id="wallet-export-container"
                className="mt-2 p-3 sm:p-4 bg-neutral-100 rounded-lg font-mono text-xs break-all"
              />
            </div>
          </div>
        )}

        {/* Register Camera Link */}
        <div className="bg-neutral-100 rounded-xl px-4 py-4 mb-4">
          <button
            onClick={() => navigate('/app/register')}
            className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-neutral-600 text-white rounded-lg text-sm font-medium hover:bg-neutral-600/90 transition-colors"
          >
            <Camera className="w-4 h-4" />
            Register New Camera
          </button>
        </div>

        {/* Sign Out Button - responsive padding */}
        <div className="bg-neutral-100 rounded-xl px-4 py-4 mb-8">
          <button
            onClick={handleSignOut}
            className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>

        {/* Modals */}
        <RecognitionTokenModal
          isOpen={showRecognitionModal}
          onClose={() => setShowRecognitionModal(false)}
          status={facialEmbeddingStatus}
          onStatusUpdate={() => {
            // Force refresh of facial embedding status after enrollment
            facialEmbeddingStatus.refetch();
          }}
        />

        <WalletBalanceModal
          isOpen={showWalletModal}
          onClose={() => setShowWalletModal(false)}
        />
      </div>
    </div>
  );
}
