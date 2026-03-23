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
  CheckCircle,
  AlertCircle,
  Loader2,
  ChevronRight,
  ChevronDown,
  Camera,
  Pencil,
  Mail,
} from "lucide-react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useFacialEmbeddingStatus } from "../../hooks/useFacialEmbeddingStatus";
import { useUserSessionChain } from "../../hooks/useUserSessionChain";
import { profileService } from "../../services/profile-service";
import { useDisplayProfile } from "../../auth/useDisplayProfile";
import { AgentSetupSection } from "./AgentSetupSection";

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
  const [showConnections, setShowConnections] = useState(false);

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

  // Social connections
  const socials = displayProfile?.socials || {};
  const socialConnections = [
    {
      label: "Google",
      detail: socials.google?.displayName || "Sign in with Google",
      connected: !!socials.google,
      icon: <Globe className="w-4 h-4" />,
    },
    {
      label: "X / Twitter",
      detail: socials.twitter?.username ? `@${socials.twitter.username}` : "Connect your X account",
      connected: !!socials.twitter,
      icon: <Globe className="w-4 h-4" />,
    },
    {
      label: "Farcaster",
      detail: socials.farcaster?.username || "Connect your Farcaster",
      connected: !!socials.farcaster,
      icon: <Globe className="w-4 h-4" />,
    },
    {
      label: "Email",
      detail: user?.email || "No email configured",
      connected: !!user?.email,
      isPrivate: true,
      icon: <Mail className="w-4 h-4" />,
    },
  ];
  const connectedCount = socialConnections.filter(s => s.connected).length;

  // On-chain / account items — full display
  const accountItems = [
    {
      id: "recognition",
      label: "Recognition Token",
      value: facialEmbeddingStatus.hasEmbedding ? "Active" : "Not enrolled",
      active: facialEmbeddingStatus.hasEmbedding,
      loading: facialEmbeddingStatus.isLoading,
      onClick: () => setShowRecognitionModal(true),
    },
    {
      id: "wallet",
      label: "Solana Wallet",
      value: `${primaryWallet.address.slice(0, 6)}...${primaryWallet.address.slice(-4)}`,
      active: true,
      loading: false,
      onClick: () => setShowWalletModal(true),
    },
    {
      id: "sessionKeychain",
      label: "Session Keychain",
      value: sessionChainStatus.hasSessionChain
        ? `${sessionChainStatus.sessionCount} key${sessionChainStatus.sessionCount !== 1 ? 's' : ''} stored`
        : "Created on first check-in",
      active: sessionChainStatus.hasSessionChain,
      loading: sessionChainStatus.isLoading,
    },
  ];

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

        {/* Profile Section */}
        <div className="bg-[#F3F3EF] rounded-xl p-4 sm:p-6 mb-4">
          <div className="flex items-center">
            {/* Profile Avatar */}
            {profileImageUrl ? (
              <img
                src={profileImageUrl}
                alt={displayName}
                referrerPolicy="no-referrer"
                className="w-14 h-14 rounded-full border-2 border-[#E8E8E3]"
              />
            ) : (
              <div className="w-14 h-14 rounded-full bg-[#E8E8E3] flex items-center justify-center">
                <User className="w-7 h-7 text-[#8A8A82]" />
              </div>
            )}

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
                    className="flex-1 text-sm border border-[#E8E8E3] rounded-lg px-3 py-1.5 focus:outline-none focus:border-[#D97706] bg-white"
                  />
                  <button
                    type="submit"
                    disabled={isSavingName || !editNameValue.trim()}
                    className="text-xs bg-[#1A1A18] text-white px-3 py-1.5 rounded-lg disabled:opacity-50"
                  >
                    {isSavingName ? '...' : 'Save'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsEditingName(false)}
                    className="text-xs text-[#8A8A82] px-2 py-1.5"
                  >
                    Cancel
                  </button>
                </form>
              ) : (
                <div className="flex items-center gap-2">
                  <div>
                    <div className="font-medium text-[#1A1A18]">{displayName}</div>
                    <div className="text-sm text-[#8A8A82]">
                      {primarySocialProvider || "Wallet Address"}
                    </div>
                  </div>
                  {!displayProfile?.hasSocialAuth && (
                    <button
                      onClick={() => {
                        setEditNameValue(displayName.includes('...') ? '' : displayName);
                        setIsEditingName(true);
                      }}
                      className="p-1 text-[#8A8A82] hover:text-[#5C5C56] transition-colors"
                      title="Edit display name"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Connections — collapsible */}
          <div className="mt-4 pt-4 border-t border-[#E8E8E3]">
            <button
              onClick={() => setShowConnections(!showConnections)}
              className="flex items-center justify-between w-full text-left"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-[#1A1A18]">Sign-in methods</span>
                <span className="text-xs text-[#8A8A82]">
                  {connectedCount} connected
                </span>
              </div>
              {showConnections ? (
                <ChevronDown className="w-4 h-4 text-[#8A8A82]" />
              ) : (
                <ChevronRight className="w-4 h-4 text-[#8A8A82]" />
              )}
            </button>

            {showConnections && (
              <div className="mt-3 space-y-1">
                {socialConnections.map((s) => (
                  <div
                    key={s.label}
                    className="flex items-center justify-between py-2.5 px-3 -mx-3 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`${s.connected ? 'text-[#5C5C56]' : 'text-[#8A8A82]'}`}>
                        {s.icon}
                      </div>
                      <div>
                        <div className="text-sm font-medium text-[#1A1A18]">{s.label}</div>
                        <div className="text-xs text-[#8A8A82]">
                          {s.connected ? s.detail : s.detail}
                          {s.connected && s.isPrivate && (
                            <span className="ml-1.5">· Private</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <span className={`text-xs font-medium ${
                      s.connected ? 'text-[#2F7D3E]' : 'text-[#8A8A82]'
                    }`}>
                      {s.connected ? 'Connected' : 'Not connected'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* On-chain / Account Section */}
        <div className="bg-[#F3F3EF] rounded-xl p-4 sm:p-6 mb-4">
          <div className="flex items-baseline justify-between mb-4">
            <h2 className="text-xs font-semibold tracking-[0.15em] uppercase text-[#8A8A82]">Account</h2>
            {solBalance !== null && (
              <div className="text-sm font-medium text-[#5C5C56]">
                {solBalance.toFixed(2)} SOL
                <span className="text-xs text-[#8A8A82] ml-1.5">
                  ${(solBalance * 150).toFixed(0)}
                </span>
              </div>
            )}
          </div>

          <div className="space-y-1">
            {accountItems.map((item) => (
              <div
                key={item.id}
                className={`flex items-center justify-between py-3 px-3 -mx-3 rounded-lg ${
                  item.onClick ? 'cursor-pointer hover:bg-white/60 transition-colors' : ''
                }`}
                onClick={item.onClick}
              >
                <div className="flex items-center gap-3">
                  <div className="w-5 flex justify-center">
                    {item.loading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-[#8A8A82]" />
                    ) : item.active ? (
                      <CheckCircle className="w-3.5 h-3.5 text-[#2F7D3E]" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5 text-[#D97706]" />
                    )}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-[#1A1A18]">{item.label}</div>
                    <div className={`text-xs mt-0.5 ${
                      item.active ? 'text-[#2F7D3E]' : 'text-[#8A8A82]'
                    }`}>
                      {item.id === "wallet" ? (
                        <span className="font-mono">{item.value}</span>
                      ) : (
                        item.value
                      )}
                    </div>
                  </div>
                </div>
                {item.onClick && (
                  <ChevronRight className="w-4 h-4 text-[#8A8A82]" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Agent Access */}
        {primaryWallet?.address && (
          <AgentSetupSection walletAddress={primaryWallet.address} />
        )}

        {/* Walrus Storage Section */}
        <WalrusStorageSection />

        {/* Wallet Backup Section */}
        {isEmbeddedWallet && (
          <div className="bg-[#F3F3EF] rounded-xl px-4 py-4 mb-4">
            <div className="text-sm">
              <div className="font-medium text-[#1A1A18] mb-3">Wallet Backup</div>
              <button
                onClick={() => setShowBackupOptions(!showBackupOptions)}
                className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-[#1A1A18] text-[#FAFAF8] rounded-lg text-sm font-medium hover:bg-[#1A1A18]/90 transition-colors"
              >
                <KeyRound className="w-4 h-4" />
                Back up Wallet
              </button>
              {showBackupOptions && (
                <div className="mt-4 space-y-2">
                  <button
                    onClick={() => handleExportWallet("recoveryPhrase")}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-[#E8E8E3] text-[#5C5C56] rounded-lg text-sm font-medium hover:bg-[#F3F3EF] transition-colors disabled:opacity-50"
                  >
                    Show Recovery Phrase
                  </button>
                  <button
                    onClick={() => handleExportWallet("privateKey")}
                    disabled={isExporting}
                    className="w-full px-3 sm:px-4 py-2 bg-white border border-[#E8E8E3] text-[#5C5C56] rounded-lg text-sm font-medium hover:bg-[#F3F3EF] transition-colors disabled:opacity-50"
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
        <div className="bg-[#F3F3EF] rounded-xl px-4 py-4 mb-4">
          <button
            onClick={() => navigate('/app/register')}
            className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-[#1A1A18] text-[#FAFAF8] rounded-lg text-sm font-medium hover:bg-[#1A1A18]/90 transition-colors"
          >
            <Camera className="w-4 h-4" />
            Register New Camera
          </button>
        </div>

        {/* Sign Out */}
        <div className="bg-[#F3F3EF] rounded-xl px-4 py-4 mb-8">
          <button
            onClick={handleSignOut}
            className="w-full flex justify-center items-center gap-2 px-3 sm:px-4 py-2 bg-[#C73A3A] text-white rounded-lg text-sm font-medium hover:bg-[#C73A3A]/90 transition-colors"
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
