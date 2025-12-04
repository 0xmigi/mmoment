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
            {isCurrentUser && <span className="ml-1 text-primary">(you)</span>}
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
              {isUser1 && <span className="ml-1 text-primary">(you)</span>}
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
              {isUser2 && <span className="ml-1 text-primary">(you)</span>}
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
                {isCurrentUser && <span className="ml-1 text-primary">(you)</span>}
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
