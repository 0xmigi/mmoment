import { useState, useEffect, useCallback } from "react";
import { CONFIG } from "../../core/config";
import { Bot, Copy, Check, Key, X } from "lucide-react";

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
  const skillUrl = `${backendUrl}/agent-skill.md`;

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
      await fetch(`${backendUrl}/v1/keys/wallet/${walletAddress}/${keyId}`, {
        method: "DELETE",
      });
      await fetchKeys();
    } catch (err) {
      console.error("[AgentSetup] Failed to revoke key:", err);
    }
  }, [backendUrl, walletAddress, fetchKeys]);

  const copyCommand = () => {
    if (!newKey) return;
    navigator.clipboard.writeText(`Set up ${skillUrl} with key ${newKey}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const activeKeys = keys.filter(k => !k.revoked_at);
  const revokedKeys = keys.filter(k => k.revoked_at);

  return (
    <div className="bg-neutral-100 rounded-xl p-4 sm:p-6 mb-6">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-neutral-600" />
          <h3 className="text-lg font-medium text-neutral-900">Agent Access</h3>
        </div>
        <span className="text-xs text-neutral-400">
          {activeKeys.length} active {activeKeys.length === 1 ? "key" : "keys"}
        </span>
      </div>

      <p className="text-sm text-neutral-500 mb-4">
        Generate a key and paste the setup command into your agent.
      </p>

      {/* Newly generated key — show once to copy */}
      {newKey && (
        <div className="mb-4">
          <div className="bg-white border border-neutral-200 rounded-lg p-3.5 mb-2">
            <code className="text-sm text-neutral-800 break-all leading-relaxed">
              Set up {skillUrl} with key {newKey}
            </code>
          </div>
          <button
            onClick={copyCommand}
            className="w-full flex justify-center items-center gap-2 px-4 py-2.5 bg-neutral-900 text-white rounded-lg text-sm font-medium hover:bg-neutral-800 transition-colors"
          >
            {copied ? (
              <><Check className="w-4 h-4" /> Copied</>
            ) : (
              <><Copy className="w-4 h-4" /> Copy to clipboard</>
            )}
          </button>
          <p className="text-xs text-amber-600 text-center mt-1.5">
            Copy this now — the full key won't be shown again.
          </p>
        </div>
      )}

      {/* Active keys list */}
      {activeKeys.length > 0 && (
        <div className="space-y-1.5 mb-3">
          {activeKeys.map(k => (
            <div key={k.id} className="bg-white rounded-lg px-3 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
                <div>
                  <span className="text-xs font-mono text-neutral-700">{k.key_prefix}...</span>
                  <div className="text-[10px] text-neutral-400">
                    Created {timeAgo(k.created_at)}
                    {k.last_used_at && <> · Used {timeAgo(k.last_used_at)}</>}
                  </div>
                </div>
              </div>
              <button
                onClick={() => revokeKey(k.id)}
                className="p-1 text-neutral-300 hover:text-red-500 transition-colors"
                title="Revoke key"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Revoked keys count */}
      {revokedKeys.length > 0 && (
        <div className="text-[10px] text-neutral-400 mb-3">
          {revokedKeys.length} revoked
        </div>
      )}

      {/* Generate button */}
      <button
        onClick={generateKey}
        disabled={isGenerating}
        className="w-full flex justify-center items-center gap-2 px-4 py-2.5 bg-white border border-neutral-200 text-neutral-700 rounded-lg text-sm font-medium hover:bg-neutral-50 transition-colors disabled:opacity-50"
      >
        <Key className="w-4 h-4" />
        {isGenerating ? "Generating..." : "Generate new key"}
      </button>
    </div>
  );
}
