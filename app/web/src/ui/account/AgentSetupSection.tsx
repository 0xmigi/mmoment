import { useState, useEffect } from "react";
import { CONFIG } from "../../core/config";
import { Bot, Copy, Check } from "lucide-react";

interface AgentSetupSectionProps {
  walletAddress: string;
}

export function AgentSetupSection({ walletAddress }: AgentSetupSectionProps) {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const backendUrl = CONFIG.BACKEND_URL;
  const skillUrl = `${backendUrl}/agent-skill.md`;

  // Generate key immediately on mount
  useEffect(() => {
    const generate = async () => {
      try {
        const res = await fetch(`${backendUrl}/v1/keys`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ wallet_address: walletAddress }),
        });
        const data = await res.json();
        if (data.data?.key) {
          setApiKey(data.data.key);
        }
      } catch (err) {
        console.error("[AgentSetup] Failed to generate key:", err);
      }
    };
    generate();
  }, [backendUrl, walletAddress]);

  const setupCommand = apiKey
    ? `Set up ${skillUrl} with key ${apiKey}`
    : "Generating...";

  const copyCommand = () => {
    if (!apiKey) return;
    navigator.clipboard.writeText(setupCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-neutral-100 rounded-xl p-4 sm:p-6 mb-6">
      <div className="flex items-center gap-2 mb-1">
        <Bot className="w-5 h-5 text-neutral-600" />
        <h3 className="text-lg font-medium text-neutral-900">Agent Access</h3>
      </div>

      <p className="text-sm text-neutral-500 mb-4">
        Paste this prompt into your agent to connect it to the camera network.
      </p>

      <div className="bg-white border border-neutral-200 rounded-lg p-3.5 mb-3">
        <code className="text-sm text-neutral-800 break-all leading-relaxed">
          {setupCommand}
        </code>
      </div>

      <button
        onClick={copyCommand}
        disabled={!apiKey}
        className="w-full flex justify-center items-center gap-2 px-4 py-2.5 bg-neutral-900 text-white rounded-lg text-sm font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50"
      >
        {copied ? (
          <>
            <Check className="w-4 h-4" />
            Copied
          </>
        ) : (
          <>
            <Copy className="w-4 h-4" />
            Copy to clipboard
          </>
        )}
      </button>
    </div>
  );
}
