import { useState, useEffect, useCallback } from "react";
import { CONFIG } from "../../core/config";
import { Bot, Copy, Check, Key, X, Trash2 } from "lucide-react";

interface AgentKey {
  id: string;
  key_prefix: string;
  name: string;
  created_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
}

interface AgentSetupSectionProps {
  walletAddress: string;
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function AgentSetupSection({ walletAddress }: AgentSetupSectionProps) {
  const [keys, setKeys] = useState<AgentKey[]>([]);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  const backendUrl = CONFIG.BACKEND_URL;

  // Self-contained prompt — no URL fetching required by the AI
  const buildPrompt = (key: string) => `You have access to a physical camera network. Use these API endpoints to interact with it.

Base URL: ${backendUrl}
Auth: Authorization: Bearer ${key}

Step 1 — call GET /v1/me to find which camera your user is at. Use the checked_in_at value as {camera_id}.

Endpoints:
- GET /v1/cameras/{camera_id}/status — camera info
- GET /v1/cameras/{camera_id}/presence — who's checked in
- POST /v1/cameras/{camera_id}/capture — take a photo (returns photo_url you can display inline)
- POST /v1/cameras/{camera_id}/record/start — start recording (optional: {"duration": 10})
- POST /v1/cameras/{camera_id}/record/stop — stop recording
- GET /v1/cameras/{camera_id}/activities — recent activity

The photo_url from /capture is a public URL — fetch and display it inline.`;

  const fetchKeys = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/v1/keys/wallet/${walletAddress}`);
      if (res.ok) {
        const data = await res.json();
        setKeys(data.data || []);
      }
    } catch (err) {
      console.error("[AgentSetup] Failed to fetch keys:", err);
    }
  }, [backendUrl, walletAddress]);

  useEffect(() => { fetchKeys(); }, [fetchKeys]);

  const generateKey = useCallback(async () => {
    setIsGenerating(true);
    try {
      const res = await fetch(`${backendUrl}/v1/keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wallet_address: walletAddress }),
      });
      const data = await res.json();
      if (data.data?.key) {
        setNewKey(data.data.key);
        await fetchKeys();
      }
    } catch (err) {
      console.error("[AgentSetup] Failed to generate key:", err);
    } finally {
      setIsGenerating(false);
    }
  }, [backendUrl, walletAddress, fetchKeys]);

  const revokeKey = useCallback(async (keyId: string) => {
    try {
      await fetch(`${backendUrl}/v1/keys/wallet/${walletAddress}/${keyId}`, { method: "DELETE" });
      await fetchKeys();
    } catch (err) {
      console.error("[AgentSetup] Failed to revoke key:", err);
    }
  }, [backendUrl, walletAddress, fetchKeys]);

  const revokeAll = useCallback(async () => {
    try {
      await fetch(`${backendUrl}/v1/keys/wallet/${walletAddress}`, { method: "DELETE" });
      setNewKey(null);
      await fetchKeys();
    } catch (err) {
      console.error("[AgentSetup] Failed to revoke all:", err);
    }
  }, [backendUrl, walletAddress, fetchKeys]);

  const copyCommand = () => {
    if (!newKey) return;
    navigator.clipboard.writeText(buildPrompt(newKey));
    setCopied(true);
    setNewKey(null);
    setTimeout(() => setCopied(false), 2000);
  };

  const activeKeys = keys.filter(k => !k.revoked_at);

  return (
    <div className="bg-[#F3F3EF] rounded-xl p-4 sm:p-6 mb-4">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-[#5C5C56]" />
          <h3 className="text-lg font-medium text-[#1A1A18]">Agent Access</h3>
        </div>
        {activeKeys.length > 0 && (
          <span className="text-xs text-[#8A8A82]">
            {activeKeys.length} active
          </span>
        )}
      </div>

      <p className="text-sm text-[#8A8A82] mb-4">
        Generate a key, copy the prompt, and paste it into any AI chat.
      </p>

      {/* Newly generated key — show once to copy */}
      {newKey && (
        <div className="mb-4">
          <div className="bg-white border border-[#E8E8E3] rounded-lg p-3.5 mb-2 max-h-32 overflow-y-auto">
            <pre className="text-xs text-[#5C5C56] whitespace-pre-wrap leading-relaxed font-mono">
              {buildPrompt(newKey)}
            </pre>
          </div>
          <button
            onClick={copyCommand}
            className="w-full flex justify-center items-center gap-2 px-4 py-2.5 bg-[#1A1A18] text-[#FAFAF8] rounded-lg text-sm font-medium hover:bg-[#1A1A18]/90 transition-colors"
          >
            {copied ? (
              <><Check className="w-4 h-4" /> Copied</>
            ) : (
              <><Copy className="w-4 h-4" /> Copy prompt</>
            )}
          </button>
          <p className="text-xs text-[#D97706] text-center mt-1.5">
            Copy this now — the full key won't be shown again.
          </p>
        </div>
      )}

      {/* Active keys list — show up to 5 most recent */}
      {activeKeys.length > 0 && (
        <div className="space-y-1.5 mb-3">
          {activeKeys.slice(0, 5).map(k => (
            <div key={k.id} className="bg-white rounded-lg px-3 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#2F7D3E] shrink-0" />
                <div>
                  <span className="text-xs font-mono text-[#5C5C56]">{k.key_prefix}...</span>
                  <div className="text-[10px] text-[#8A8A82]">
                    Created {timeAgo(k.created_at)}
                    {k.last_used_at && <> · Used {timeAgo(k.last_used_at)}</>}
                  </div>
                </div>
              </div>
              <button
                onClick={() => revokeKey(k.id)}
                className="p-1 text-[#E8E8E3] hover:text-[#C73A3A] transition-colors"
                title="Revoke key"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
          {activeKeys.length > 5 && (
            <div className="text-[10px] text-[#8A8A82] px-1">
              +{activeKeys.length - 5} more
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={generateKey}
          disabled={isGenerating}
          className="flex-1 flex justify-center items-center gap-2 px-4 py-2.5 bg-white border border-[#E8E8E3] text-[#5C5C56] rounded-lg text-sm font-medium hover:bg-[#FAFAF8] transition-colors disabled:opacity-50"
        >
          <Key className="w-4 h-4" />
          {isGenerating ? "Generating..." : "Generate new key"}
        </button>
        {activeKeys.length > 1 && (
          <button
            onClick={revokeAll}
            className="flex items-center gap-1.5 px-3 py-2.5 bg-white border border-[#E8E8E3] text-[#8A8A82] rounded-lg text-sm hover:text-[#C73A3A] hover:border-[#C73A3A]/30 transition-colors"
            title="Revoke all keys"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
