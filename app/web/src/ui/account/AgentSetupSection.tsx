import { useState, useCallback } from "react";
import { CONFIG } from "../../core/config";
import { Bot, Copy, Check, Key, RefreshCw, Trash2 } from "lucide-react";

interface ApiKeyInfo {
  id: string;
  key_prefix: string;
  name: string;
  created_at: number;
  last_used_at: number | null;
}

interface AgentSetupSectionProps {
  walletAddress: string;
}

export function AgentSetupSection({ walletAddress }: AgentSetupSectionProps) {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [existingKeys, setExistingKeys] = useState<ApiKeyInfo[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState<"key" | "command" | null>(null);
  const [showKeys, setShowKeys] = useState(false);

  const backendUrl = CONFIG.BACKEND_URL;
  const skillUrl = `${backendUrl}/agent-skill.md`;

  // Fetch existing keys
  const fetchKeys = useCallback(async (rawKey: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${backendUrl}/v1/keys`, {
        headers: { Authorization: `Bearer ${rawKey}` },
      });
      if (res.ok) {
        const data = await res.json();
        setExistingKeys(data.data || []);
      }
    } catch (err) {
      console.error("[AgentSetup] Failed to fetch keys:", err);
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl]);

  // Generate new API key
  const generateKey = useCallback(async () => {
    setIsGenerating(true);
    try {
      const res = await fetch(`${backendUrl}/v1/keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wallet_address: walletAddress }),
      });
      const data = await res.json();
      if (data.data?.raw_key) {
        setApiKey(data.data.raw_key);
        setShowKeys(true);
      }
    } catch (err) {
      console.error("[AgentSetup] Failed to generate key:", err);
    } finally {
      setIsGenerating(false);
    }
  }, [backendUrl, walletAddress]);

  // Revoke a key
  const revokeKey = useCallback(async (keyId: string) => {
    if (!apiKey) return;
    try {
      await fetch(`${backendUrl}/v1/keys/${keyId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      setExistingKeys(prev => prev.filter(k => k.id !== keyId));
    } catch (err) {
      console.error("[AgentSetup] Failed to revoke key:", err);
    }
  }, [backendUrl, apiKey]);

  const copyToClipboard = (text: string, type: "key" | "command") => {
    navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(null), 2000);
  };

  const setupCommand = apiKey
    ? `Set up ${skillUrl} with key ${apiKey}`
    : null;

  return (
    <div className="bg-neutral-100 rounded-xl p-4 sm:p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Bot className="w-5 h-5 text-neutral-600" />
        <h3 className="text-lg font-medium text-neutral-900">Agent Access</h3>
      </div>

      <p className="text-sm text-neutral-500 mb-4">
        Connect your AI agent to the MMOMENT camera network. Generate an API key and give your agent the setup command below.
      </p>

      {!apiKey ? (
        <button
          onClick={generateKey}
          disabled={isGenerating}
          className="flex items-center gap-2 px-4 py-2.5 bg-neutral-900 text-white rounded-lg text-sm font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50"
        >
          {isGenerating ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Key className="w-4 h-4" />
          )}
          {isGenerating ? "Generating..." : "Generate API Key"}
        </button>
      ) : (
        <div className="space-y-4">
          {/* Setup command — the main thing to copy */}
          <div>
            <label className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2 block">
              Give this to your agent
            </label>
            <div className="bg-white border border-neutral-200 rounded-lg p-3 flex items-start gap-2">
              <code className="flex-1 text-sm text-neutral-800 break-all leading-relaxed">
                {setupCommand}
              </code>
              <button
                onClick={() => copyToClipboard(setupCommand!, "command")}
                className="shrink-0 p-1.5 hover:bg-neutral-100 rounded transition-colors"
                title="Copy setup command"
              >
                {copied === "command" ? (
                  <Check className="w-4 h-4 text-emerald-600" />
                ) : (
                  <Copy className="w-4 h-4 text-neutral-400" />
                )}
              </button>
            </div>
          </div>

          {/* Raw API key (collapsible) */}
          <div>
            <label className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2 block">
              API Key
            </label>
            <div className="bg-white border border-neutral-200 rounded-lg p-3 flex items-center gap-2">
              <code className="flex-1 text-xs font-mono text-neutral-600 break-all">
                {apiKey}
              </code>
              <button
                onClick={() => copyToClipboard(apiKey, "key")}
                className="shrink-0 p-1.5 hover:bg-neutral-100 rounded transition-colors"
                title="Copy API key"
              >
                {copied === "key" ? (
                  <Check className="w-4 h-4 text-emerald-600" />
                ) : (
                  <Copy className="w-4 h-4 text-neutral-400" />
                )}
              </button>
            </div>
            <p className="text-xs text-amber-600 mt-1.5">
              Save this key now — it won't be shown again.
            </p>
          </div>

          {/* Generate another */}
          <button
            onClick={generateKey}
            disabled={isGenerating}
            className="text-xs text-neutral-500 hover:text-neutral-700 flex items-center gap-1.5 transition-colors"
          >
            <Key className="w-3 h-3" />
            Generate another key
          </button>
        </div>
      )}
    </div>
  );
}
