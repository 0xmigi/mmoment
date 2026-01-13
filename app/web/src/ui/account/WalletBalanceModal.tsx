import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, Copy, Check, ExternalLink } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { CONFIG } from '../../core/config';

interface WalletBalanceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface BalanceData {
  sol: number;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

interface SponsorshipQuota {
  used: number;
  total: number;
}

export function WalletBalanceModal({ isOpen, onClose }: WalletBalanceModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [balanceData, setBalanceData] = useState<BalanceData>({
    sol: 0,
    loading: true,
    error: null,
    lastUpdated: null,
  });
  const [copied, setCopied] = useState(false);
  const [sponsorshipQuota, setSponsorshipQuota] = useState<SponsorshipQuota>({ used: 0, total: 10 });
  const [sponsoredGasEnabled, setSponsoredGasEnabled] = useState(() => {
    return localStorage.getItem('sponsoredGasEnabled') === 'true';
  });

  const fetchBalance = async () => {
    if (!primaryWallet?.address || !connection) {
      setBalanceData(prev => ({
        ...prev,
        loading: false,
        error: 'Wallet or connection not available',
      }));
      return;
    }

    try {
      setBalanceData(prev => ({ ...prev, loading: true, error: null }));

      const publicKey = new PublicKey(primaryWallet.address);
      const balance = await connection.getBalance(publicKey);
      const solBalance = balance / LAMPORTS_PER_SOL;

      setBalanceData({
        sol: solBalance,
        loading: false,
        error: null,
        lastUpdated: new Date(),
      });
    } catch (error) {
      console.error('Error fetching balance:', error);
      setBalanceData(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch balance',
      }));
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchBalance();
      fetchSponsorshipQuota();
    }
  }, [isOpen, primaryWallet?.address, connection]);

  const fetchSponsorshipQuota = async () => {
    if (!primaryWallet?.address) return;

    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/sponsorship-status/${primaryWallet.address}`);
      if (response.ok) {
        const data = await response.json();
        // API returns { count: X, remaining: Y } where count = used transactions
        setSponsorshipQuota({ used: data.count || 0, total: 10 });
      }
    } catch (error) {
      console.error('Error fetching sponsorship quota:', error);
      // Keep default 0/10 on error
    }
  };

  const handleCopyAddress = async () => {
    if (!primaryWallet?.address) return;

    try {
      await navigator.clipboard.writeText(primaryWallet.address);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy address:', err);
    }
  };

  const handleExplorerClick = () => {
    if (primaryWallet?.address) {
      window.open(`https://solscan.io/account/${primaryWallet.address}?cluster=devnet`, '_blank');
    }
  };

  const handleToggleSponsoredGas = () => {
    const newValue = !sponsoredGasEnabled;
    setSponsoredGasEnabled(newValue);
    localStorage.setItem('sponsoredGasEnabled', String(newValue));
  };

  const formatAddress = (address: string, start = 6, end = 6) => {
    if (!address) return '';
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  };

  if (!isOpen) return null;

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Wallet
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="p-4">
            {/* Simple Address Display */}
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-200">
              <div className="flex-1">
                <div className="text-xs text-gray-500 mb-1">Address</div>
                <div className="text-sm font-mono text-gray-700">
                  {primaryWallet?.address ? formatAddress(primaryWallet.address, 6, 6) : 'Not connected'}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyAddress}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  title="Copy address"
                >
                  {copied ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
                <button
                  onClick={handleExplorerClick}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  title="View on explorer"
                >
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Sponsored Gas Banner */}
            <div className="mb-4 pb-3 border-b border-gray-200">
              <div className="bg-gradient-to-r from-primary-light to-purple-50 rounded-lg p-3 border border-primary-light">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <div className="text-xs font-medium text-primary mb-1">🎁 Free Transactions</div>
                    <div className="text-xs text-primary">
                      First 10 transactions on us!
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-primary">
                      {sponsorshipQuota.total - sponsorshipQuota.used}/{sponsorshipQuota.total}
                    </div>
                    <div className="text-xs text-primary">remaining</div>
                  </div>
                </div>
                <div className="w-full bg-primary-light/50 rounded-full h-1.5 mb-2">
                  <div
                    className="bg-primary h-1.5 rounded-full transition-all"
                    style={{ width: `${((sponsorshipQuota.total - sponsorshipQuota.used) / sponsorshipQuota.total) * 100}%` }}
                  />
                </div>
                {sponsorshipQuota.total - sponsorshipQuota.used > 0 ? (
                  <button
                    onClick={handleToggleSponsoredGas}
                    className={`w-full py-2 px-3 rounded-md text-xs font-medium transition-colors ${
                      sponsoredGasEnabled
                        ? 'bg-green-500 text-white hover:bg-green-600'
                        : 'bg-white text-primary border border-primary-light hover:bg-primary-light'
                    }`}
                  >
                    {sponsoredGasEnabled ? '✓ Sponsored Gas Enabled' : 'Enable Sponsored Gas'}
                  </button>
                ) : (
                  <div className="text-xs text-center text-orange-600 py-2">
                    All free transactions used. Add SOL to continue.
                  </div>
                )}
              </div>
            </div>

            {/* Token List */}
            <div className="space-y-3">
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Tokens</div>

              {/* SOL Token */}
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-primary flex items-center justify-center mr-3">
                    <span className="text-white text-xs font-bold">SOL</span>
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">Solana</div>
                    <div className="text-xs text-gray-500">SOL</div>
                  </div>
                </div>
                <div className="text-right">
                  {balanceData.loading ? (
                    <div className="text-sm text-gray-400">...</div>
                  ) : (
                    <>
                      <div className="font-medium text-gray-900">
                        {balanceData.sol.toFixed(4)}
                      </div>
                      <div className="text-xs text-gray-500">
                        ${(balanceData.sol * 150).toFixed(2)}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* USDC Token (Coming Soon) */}
              <div className="flex items-center justify-between py-2 opacity-50">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center mr-3">
                    <span className="text-white text-sm font-bold">$</span>
                  </div>
                  <div>
                    <div className="font-medium text-gray-600">USD Coin</div>
                    <div className="text-xs text-gray-400">USDC</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-gray-400">--</div>
                  <div className="text-xs text-gray-400">Coming soon</div>
                </div>
              </div>
            </div>

            {/* Network Badge */}
            <div className="mt-6 pt-3 border-t border-gray-200">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Network</span>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-orange-500 rounded-full mr-1.5"></div>
                  <span>Solana DevNet</span>
                </div>
              </div>
            </div>

            {/* Error Display */}
            {balanceData.error && (
              <div className="mt-3 text-xs text-red-600">
                {balanceData.error}
              </div>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}