/**
 * SessionChainSection Component
 *
 * Read-only display of the user's session chain status.
 * Session chain creation is handled automatically during check-in flow.
 * This section is for reference/visibility only.
 */

import { useUserSessionChain } from '../../hooks/useUserSessionChain';
import {
  KeyRound,
  CheckCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
} from 'lucide-react';

export function SessionChainSection() {
  const { hasSessionChain, sessionChainPda, sessionCount, isLoading } = useUserSessionChain();

  return (
    <div className="bg-gray-50 rounded-xl p-4 mb-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <KeyRound className="w-4 h-4 text-gray-600" />
          <span className="font-medium text-sm">Session Keychain</span>
        </div>
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin text-primary" />
        ) : hasSessionChain ? (
          <CheckCircle className="w-4 h-4 text-green-500" />
        ) : (
          <AlertCircle className="w-4 h-4 text-orange-500" />
        )}
      </div>

      {isLoading ? (
        <div className="text-sm text-gray-500">Checking status...</div>
      ) : hasSessionChain ? (
        <div className="space-y-2">
          <div className="text-sm text-green-600">
            Active - {sessionCount} session key{sessionCount !== 1 ? 's' : ''} stored
          </div>
          {sessionChainPda && (
            <a
              href={`https://explorer.solana.com/address/${sessionChainPda}?cluster=devnet`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-primary hover:text-primary-hover"
            >
              <ExternalLink className="w-3 h-3" />
              View on Explorer
            </a>
          )}
        </div>
      ) : (
        <div className="text-sm text-gray-500">
          Will be created automatically on your first check-in
        </div>
      )}
    </div>
  );
}
