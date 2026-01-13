/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState, useMemo } from 'react';
import { X, DollarSign, Trophy } from 'lucide-react';
import { unifiedCameraService } from './unified-camera-service';
import { useResolveDisplayNames } from '../hooks/useResolveDisplayNames';

interface CompetitorStats {
  display_name: string;
  wallet_address?: string;
  stats: Record<string, any>;
}

interface CompetitionState {
  active: boolean;
  competitors: CompetitorStats[];
}

interface EscrowInfo {
  pda: string;
  stakeSol: number;
  totalPool?: number;
  participants?: number;
  status?: string;
  winners?: string[];
}

interface CompetitionScoreboardProps {
  cameraId: string;
  walletAddress?: string;
  onClose?: () => void;
  escrowInfo?: EscrowInfo | null;
}

export function CompetitionScoreboard({
  cameraId,
  walletAddress,
  onClose,
  escrowInfo
}: CompetitionScoreboardProps) {
  const [competitionState, setCompetitionState] = useState<CompetitionState | null>(null);

  // Extract wallet addresses from competitors for display name resolution
  const competitorWallets = useMemo(() =>
    (competitionState?.competitors || [])
      .map(c => c.wallet_address)
      .filter((w): w is string => !!w),
    [competitionState?.competitors]
  );

  // Resolve display names from backend when Jetson only provides wallet addresses
  const { getDisplayName } = useResolveDisplayNames(competitorWallets);

  useEffect(() => {
    const checkAppStatus = async () => {
      try {
        const result = await unifiedCameraService.getAppStatus(cameraId);

        if (result.success && result.data?.active_app) {
          if (result.data.state?.competitors?.length) {
            setCompetitionState(result.data.state);
          } else {
            const competitorsJson = sessionStorage.getItem('competition_competitors');
            if (competitorsJson) {
              const competitors = JSON.parse(competitorsJson);
              setCompetitionState({
                active: result.data.state?.active || false,
                competitors: competitors.map((c: any) => ({
                  display_name: c.display_name,
                  wallet_address: c.wallet_address,
                  stats: { reps: 0 }
                }))
              });
            }
          }
        } else {
          setCompetitionState(null);
        }
      } catch (error) {
        console.error('[Scoreboard] Error:', error);
      }
    };

    checkAppStatus();
    const pollInterval = setInterval(checkAppStatus, 2000);
    return () => clearInterval(pollInterval);
  }, [cameraId]);

  if (!competitionState?.competitors?.length) {
    return null;
  }

  // Get primary metric (reps, score, points, etc.)
  const getPrimaryMetric = (stats: Record<string, any>) => {
    return stats.reps ?? stats.score ?? stats.points ?? 0;
  };

  const getMetricLabel = () => {
    const firstCompetitor = competitionState.competitors[0];
    if (!firstCompetitor?.stats) return 'points';

    if ('reps' in firstCompetitor.stats) return 'reps';
    if ('score' in firstCompetitor.stats) return 'score';
    if ('points' in firstCompetitor.stats) return 'points';
    return 'points';
  };

  // Sort competitors by primary metric (descending)
  const sortedCompetitors = [...competitionState.competitors].sort((a, b) =>
    getPrimaryMetric(b.stats) - getPrimaryMetric(a.stats)
  );

  const metricLabel = getMetricLabel();
  const isSettled = escrowInfo?.status === 'settled';
  const hasWinners = (escrowInfo?.winners?.length ?? 0) > 0;

  // Render competitor scores - for white header background
  const renderScores = () => {
    // Single competitor - stacked layout
    if (sortedCompetitors.length === 1) {
      const competitor = sortedCompetitors[0];
      const stats = competitor.stats || {};
      const isCurrentUser = walletAddress && competitor.wallet_address === walletAddress;
      const resolvedName = competitor.wallet_address
        ? getDisplayName(competitor.wallet_address, competitor.display_name)
        : competitor.display_name;

      return (
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">
            {resolvedName}
            {isCurrentUser && <span className="ml-1 text-primary">(you)</span>}
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-black tabular-nums">
              {getPrimaryMetric(stats)}
            </span>
            <span className="text-sm text-gray-500">{metricLabel}</span>
          </div>
        </div>
      );
    }

    // Two competitors - versus layout
    if (sortedCompetitors.length === 2) {
      const [competitor1, competitor2] = sortedCompetitors;
      const stats1 = competitor1.stats || {};
      const stats2 = competitor2.stats || {};
      const score1 = getPrimaryMetric(stats1);
      const score2 = getPrimaryMetric(stats2);
      const isUser1 = walletAddress && competitor1.wallet_address === walletAddress;
      const isUser2 = walletAddress && competitor2.wallet_address === walletAddress;
      const name1 = competitor1.wallet_address
        ? getDisplayName(competitor1.wallet_address, competitor1.display_name)
        : competitor1.display_name;
      const name2 = competitor2.wallet_address
        ? getDisplayName(competitor2.wallet_address, competitor2.display_name)
        : competitor2.display_name;

      return (
        <div className="flex items-center gap-4">
          {/* Competitor 1 */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700 truncate max-w-[80px]">
              {name1}
              {isUser1 && <span className="text-primary">*</span>}
            </span>
            <span className="text-xl font-bold text-black tabular-nums">
              {score1}
            </span>
          </div>

          {/* VS divider */}
          <span className="text-sm font-bold text-gray-400">vs</span>

          {/* Competitor 2 */}
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold text-black tabular-nums">
              {score2}
            </span>
            <span className="text-sm font-medium text-gray-700 truncate max-w-[80px]">
              {name2}
              {isUser2 && <span className="text-primary">*</span>}
            </span>
          </div>
        </div>
      );
    }

    // 3+ competitors - horizontal list
    return (
      <div className="flex items-center gap-4">
        {sortedCompetitors.slice(0, 3).map((competitor, index) => {
          const stats = competitor.stats || {};
          const isFirst = index === 0;
          const isCurrentUser = walletAddress && competitor.wallet_address === walletAddress;
          const resolvedName = competitor.wallet_address
            ? getDisplayName(competitor.wallet_address, competitor.display_name)
            : competitor.display_name;

          return (
            <div key={index} className="flex items-center gap-1.5">
              <span className={`text-sm font-bold ${isFirst ? 'text-green-600' : 'text-gray-400'}`}>
                #{index + 1}
              </span>
              <span className="text-sm text-gray-700 truncate max-w-[60px]">
                {resolvedName}
                {isCurrentUser && <span className="text-primary">*</span>}
              </span>
              <span className="text-lg font-bold text-black tabular-nums">
                {getPrimaryMetric(stats)}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="fixed top-2 left-2 right-2 z-[60] sm:left-4 sm:right-4 md:left-1/2 md:-translate-x-1/2 md:max-w-3xl md:w-full">
      <div className="bg-white rounded-xl shadow-lg p-1">
        <div className="border-2 border-amber-500 rounded-lg px-4 py-4 flex items-start justify-between">
          {/* Left side - Scores */}
          <div className="flex flex-col gap-2">
            {renderScores()}

            {/* Escrow info */}
            {escrowInfo && (
              <div className={`flex items-center gap-1.5 text-sm ${
                isSettled
                  ? hasWinners ? 'text-green-600' : 'text-gray-500'
                  : 'text-primary'
              }`}>
                {isSettled ? (
                  <>
                    <Trophy className="w-4 h-4" />
                    <span>
                      {hasWinners
                        ? `Winner: ${escrowInfo.winners![0].slice(0, 6)}...`
                        : 'Settled'
                      }
                    </span>
                  </>
                ) : (
                  <>
                    <DollarSign className="w-4 h-4" />
                    <span>
                      {escrowInfo.totalPool?.toFixed(2) ?? escrowInfo.stakeSol.toFixed(2)} SOL
                    </span>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Right side - Exit button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              aria-label="Exit competition"
            >
              <X className="w-5 h-5 text-gray-700" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
