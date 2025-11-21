import { useEffect, useState } from 'react';
import { unifiedCameraService } from './unified-camera-service';

interface CompetitorStats {
  display_name: string;
  wallet_address?: string;
  stats: Record<string, any>;
}

interface CompetitionState {
  active: boolean;
  competitors: CompetitorStats[];
}

interface CompetitionScoreboardProps {
  cameraId: string;
  walletAddress?: string;
}

export function CompetitionScoreboard({ cameraId, walletAddress }: CompetitionScoreboardProps) {
  const [competitionState, setCompetitionState] = useState<CompetitionState | null>(null);

  console.log('[Scoreboard] Component render - cameraId:', cameraId, 'competitionState:', competitionState);

  useEffect(() => {
    console.log('[Scoreboard] useEffect mounted for cameraId:', cameraId);
    // Check immediately on mount
    const checkAppStatus = async () => {
      try {
        console.log('[Scoreboard] Checking app status...');
        const result = await unifiedCameraService.getAppStatus(cameraId);
        console.log('[Scoreboard] App status result:', result);

        if (result.success && result.data?.active_app) {
          console.log('[Scoreboard] Active app detected:', result.data.active_app);

          // Check if backend has competition state with competitors
          if (result.data.state?.competitors?.length) {
            console.log('[Scoreboard] Setting competition state from backend:', result.data.state);
            setCompetitionState(result.data.state);
          } else {
            // No competitors in backend state - use sessionStorage
            const competitorsJson = sessionStorage.getItem('competition_competitors');
            console.log('[Scoreboard] No competitors in backend, checking sessionStorage:', competitorsJson);
            if (competitorsJson) {
              const competitors = JSON.parse(competitorsJson);
              console.log('[Scoreboard] Creating placeholder state for competitors:', competitors);
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
          console.log('[Scoreboard] No active app, clearing state');
          setCompetitionState(null);
        }
      } catch (error) {
        console.error('[Scoreboard] Error checking app status:', error);
      }
    };

    // Check immediately
    checkAppStatus();

    // Poll every 500ms
    const pollInterval = setInterval(checkAppStatus, 500);

    return () => clearInterval(pollInterval);
  }, [cameraId]);

  // Don't show if no competition state
  if (!competitionState?.competitors?.length) {
    console.log('[Scoreboard] Returning null - no competitors. competitionState:', competitionState);
    return null;
  }

  console.log('[Scoreboard] Rendering scoreboard with competitors:', competitionState.competitors);

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

  // Single competitor - centered layout
  if (sortedCompetitors.length === 1) {
    const competitor = sortedCompetitors[0];
    const stats = competitor.stats || {};
    const isCurrentUser = walletAddress && competitor.wallet_address === walletAddress;

    return (
      <div className="w-full bg-white rounded-lg px-4 py-2">
        <div className="flex flex-col items-center gap-1">
          <span className="text-xs font-medium text-gray-900">
            {competitor.display_name}
            {isCurrentUser && <span className="ml-1 text-blue-600">(you)</span>}
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-bold text-gray-900 tabular-nums">
              {getPrimaryMetric(stats)}
            </span>
            <span className="text-sm text-gray-500">{metricLabel}</span>
          </div>
        </div>
      </div>
    );
  }

  // Two competitors - versus layout (MMA style)
  if (sortedCompetitors.length === 2) {
    const [competitor1, competitor2] = sortedCompetitors;
    const stats1 = competitor1.stats || {};
    const stats2 = competitor2.stats || {};
    const score1 = getPrimaryMetric(stats1);
    const score2 = getPrimaryMetric(stats2);
    const total = score1 + score2;
    const percentage1 = total > 0 ? Math.round((score1 / total) * 100) : 50;
    const percentage2 = total > 0 ? 100 - percentage1 : 50;
    const isUser1 = walletAddress && competitor1.wallet_address === walletAddress;
    const isUser2 = walletAddress && competitor2.wallet_address === walletAddress;

    return (
      <div className="w-full bg-white rounded-lg px-4 py-2">
        <div className="flex items-center justify-between gap-4">
          {/* Competitor 1 - Left */}
          <div className="flex flex-col items-start flex-1">
            <span className="text-xs font-medium text-gray-900 truncate max-w-full">
              {competitor1.display_name}
              {isUser1 && <span className="ml-1 text-blue-600">(you)</span>}
            </span>
            <span className="text-2xl font-bold text-gray-900 tabular-nums">
              {score1}
            </span>
          </div>

          {/* VS divider with percentages */}
          <div className="flex flex-col items-center gap-1 flex-shrink-0">
            <div className="flex items-center gap-2">
              <span className={`text-xs font-bold ${score1 > score2 ? 'text-green-600' : 'text-gray-400'}`}>
                {percentage1}%
              </span>
              <span className="text-xs font-medium text-gray-400">VS</span>
              <span className={`text-xs font-bold ${score2 > score1 ? 'text-green-600' : 'text-gray-400'}`}>
                {percentage2}%
              </span>
            </div>
            <span className="text-xs text-gray-500">{metricLabel}</span>
          </div>

          {/* Competitor 2 - Right */}
          <div className="flex flex-col items-end flex-1">
            <span className="text-xs font-medium text-gray-900 truncate max-w-full">
              {competitor2.display_name}
              {isUser2 && <span className="ml-1 text-blue-600">(you)</span>}
            </span>
            <span className="text-2xl font-bold text-gray-900 tabular-nums">
              {score2}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // 3+ competitors - horizontal list
  return (
    <div className="w-full bg-white rounded-lg px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        {sortedCompetitors.map((competitor, index) => {
          const stats = competitor.stats || {};
          const isFirst = index === 0;
          const isCurrentUser = walletAddress && competitor.wallet_address === walletAddress;

          return (
            <div key={index} className="flex items-center gap-2">
              <span className={`text-xs font-bold ${isFirst ? 'text-green-600' : 'text-gray-500'}`}>
                #{index + 1}
              </span>
              <span className="text-xs font-medium text-gray-900">
                {competitor.display_name}
                {isCurrentUser && <span className="ml-1 text-blue-600">(you)</span>}
              </span>
              <span className="text-lg font-bold text-gray-900 tabular-nums">
                {getPrimaryMetric(stats)}
              </span>
              <span className="text-xs text-gray-500">{metricLabel}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
