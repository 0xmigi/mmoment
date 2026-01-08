import { useState, useEffect, useCallback } from 'react';
import { Loader2, DollarSign, Trophy } from 'lucide-react';
import { unifiedCameraService } from './unified-camera-service';
import { useCompetitionEscrow, type ActiveCompetition } from '../hooks/useCompetitionEscrow';

interface CompetitionControlsProps {
  cameraId: string;
}

export function CompetitionControls({ cameraId }: CompetitionControlsProps) {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoadedApp, setHasLoadedApp] = useState(false);
  const [escrowInfo, setEscrowInfo] = useState<{
    pda: string;
    stakeSol: number;
  } | null>(null);
  const [escrowCompetition, setEscrowCompetition] = useState<ActiveCompetition | null>(null);

  const {
    startCompetition: startEscrowCompetition,
    cancelCompetition: cancelEscrowCompetition,
    fetchCompetition,
    loading: escrowLoading,
    error: escrowError,
  } = useCompetitionEscrow();

  // Load escrow info from session storage
  useEffect(() => {
    const escrowPda = sessionStorage.getItem('competition_escrow_pda');
    const stakeSol = sessionStorage.getItem('competition_stake_sol');

    if (escrowPda && stakeSol) {
      setEscrowInfo({
        pda: escrowPda,
        stakeSol: parseFloat(stakeSol),
      });
    } else {
      setEscrowInfo(null);
    }
  }, []);

  // Fetch escrow competition status when we have escrow info
  const refreshEscrowStatus = useCallback(async () => {
    if (!escrowInfo?.pda) return;

    try {
      const competition = await fetchCompetition(escrowInfo.pda);
      setEscrowCompetition(competition);
    } catch (err) {
      console.error('[CompetitionControls] Error fetching escrow status:', err);
    }
  }, [escrowInfo?.pda, fetchCompetition]);

  useEffect(() => {
    if (escrowInfo?.pda) {
      refreshEscrowStatus();
      // Poll for updates every 5 seconds
      const interval = setInterval(refreshEscrowStatus, 5000);
      return () => clearInterval(interval);
    }
  }, [escrowInfo?.pda, refreshEscrowStatus]);

  useEffect(() => {
    // Check if app is loaded and ready
    const checkAppStatus = async () => {
      try {
        const result = await unifiedCameraService.getAppStatus(cameraId);
        if (result.success && result.data?.active_app) {
          setHasLoadedApp(true);
          // Check if competition is already running
          if (result.data.state?.active) {
            setIsActive(true);
          }
        } else {
          setHasLoadedApp(false);
          setIsActive(false);
        }
      } catch (error) {
        console.error('Error checking app status:', error);
      }
    };

    checkAppStatus();
    const interval = setInterval(checkAppStatus, 2000);
    return () => clearInterval(interval);
  }, [cameraId]);

  const handleStart = async () => {
    setIsLoading(true);
    try {
      // Get competitors and duration from sessionStorage
      const competitorsJson = sessionStorage.getItem('competition_competitors');
      const durationStr = sessionStorage.getItem('competition_duration');

      if (!competitorsJson) {
        console.error('No competitors found in storage');
        return;
      }

      const competitors = JSON.parse(competitorsJson);
      const duration = durationStr ? parseInt(durationStr) : 300;

      // If we have an escrow, start it on-chain first
      if (escrowInfo?.pda) {
        console.log('[CompetitionControls] Starting on-chain competition...');
        const escrowStarted = await startEscrowCompetition(escrowInfo.pda);
        if (!escrowStarted) {
          console.error('Failed to start on-chain competition:', escrowError);
          // Continue anyway for CV tracking, but log the error
        } else {
          console.log('[CompetitionControls] On-chain competition started');
          await refreshEscrowStatus();
        }
      }

      // Start the CV competition on Jetson
      const result = await unifiedCameraService.startCompetition(
        cameraId,
        competitors,
        duration
      );

      if (result.success) {
        setIsActive(true);
      } else {
        console.error('Failed to start competition:', result.error);
      }
    } catch (error) {
      console.error('Error starting competition:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async () => {
    setIsLoading(true);
    try {
      // Build competition metadata from sessionStorage
      const competitionMode = sessionStorage.getItem('competition_mode') || 'none';
      const escrowPda = sessionStorage.getItem('competition_escrow_pda');
      const stakeSol = sessionStorage.getItem('competition_stake_sol');
      const targetPushups = sessionStorage.getItem('competition_target_pushups');

      const competitionMeta = competitionMode !== 'none' ? {
        mode: competitionMode,
        escrow_pda: escrowPda || undefined,
        stake_amount_sol: stakeSol ? parseFloat(stakeSol) : undefined,
        target_reps: targetPushups ? parseInt(targetPushups) : undefined,
      } : undefined;

      console.log('[CompetitionControls] Ending competition with meta:', competitionMeta);

      // End the CV competition on Jetson with competition metadata
      // Note: The Jetson will handle calling settle_competition on-chain
      // since it has the camera's device key required to sign the settlement
      const result = await unifiedCameraService.endCompetition(cameraId, competitionMeta);
      if (result.success) {
        setIsActive(false);
        // Deactivate the app
        await unifiedCameraService.deactivateApp(cameraId);
        setHasLoadedApp(false);

        // Refresh escrow status to see settlement results
        if (escrowInfo?.pda) {
          await refreshEscrowStatus();
        }

        // Clear session storage
        clearSessionStorage();
      }
    } catch (error) {
      console.error('Error stopping competition:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExit = async () => {
    setIsLoading(true);
    try {
      // If competition is active, end it first
      if (isActive) {
        await unifiedCameraService.endCompetition(cameraId);
      }

      // If we have an escrow that's still pending, cancel it
      if (escrowInfo?.pda && escrowCompetition?.status === 'pending') {
        console.log('[CompetitionControls] Cancelling on-chain competition...');
        await cancelEscrowCompetition(escrowInfo.pda, 'User cancelled');
      }

      // Deactivate the app
      await unifiedCameraService.deactivateApp(cameraId);
      setHasLoadedApp(false);
      setIsActive(false);

      // Clear session storage
      clearSessionStorage();
    } catch (error) {
      console.error('Error exiting competition mode:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearSessionStorage = () => {
    sessionStorage.removeItem('competition_competitors');
    sessionStorage.removeItem('competition_duration');
    sessionStorage.removeItem('competition_app');
    sessionStorage.removeItem('competition_escrow_pda');
    sessionStorage.removeItem('competition_escrow_created_at');
    sessionStorage.removeItem('competition_stake_sol');
    setEscrowInfo(null);
    setEscrowCompetition(null);
  };

  // Don't show if no app is loaded
  if (!hasLoadedApp) {
    return null;
  }

  const isEscrowSettled = escrowCompetition?.status === 'settled';
  const hasWinners = (escrowCompetition?.winners?.length ?? 0) > 0;

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center space-y-2">
      {/* Escrow Status Banner */}
      {escrowInfo && (
        <div className={`px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2 ${
          isEscrowSettled
            ? hasWinners ? 'bg-green-500 text-white' : 'bg-gray-500 text-white'
            : 'bg-primary text-white'
        }`}>
          {isEscrowSettled ? (
            <>
              <Trophy className="w-4 h-4" />
              <span className="text-sm font-medium">
                {hasWinners
                  ? `Winner: ${escrowCompetition.winners[0].slice(0, 6)}...`
                  : 'Competition Settled'
                }
              </span>
            </>
          ) : (
            <>
              <DollarSign className="w-4 h-4" />
              <span className="text-sm font-medium">
                Pot: {escrowCompetition?.totalPool?.toFixed(2) ?? escrowInfo.stakeSol.toFixed(2)} SOL
              </span>
              {escrowCompetition && (
                <span className="text-xs opacity-75">
                  ({escrowCompetition.participants.length} joined)
                </span>
              )}
            </>
          )}
        </div>
      )}

      {/* Control Buttons */}
      <div className="flex items-center space-x-2">
        <button
          onClick={!isActive ? handleStart : handleStop}
          disabled={isLoading || escrowLoading}
          className={`w-24 py-2 flex items-center justify-center bg-white rounded-lg shadow-lg transition-colors disabled:bg-gray-200 ${
            !isActive ? 'text-green-600 hover:bg-gray-50' : 'text-red-600 hover:bg-gray-50'
          }`}
        >
          {isLoading || escrowLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <span className="text-sm font-medium">{!isActive ? 'Start' : 'Stop'}</span>
          )}
        </button>

        <button
          onClick={handleExit}
          disabled={isLoading || escrowLoading}
          className="w-24 py-2 flex items-center justify-center bg-white text-black hover:bg-gray-50 rounded-lg shadow-lg transition-colors disabled:bg-gray-200"
        >
          <span className="text-sm font-medium">Cancel</span>
        </button>
      </div>

      {/* Escrow Error */}
      {escrowError && (
        <div className="px-3 py-1 bg-red-100 text-red-700 text-xs rounded-lg">
          {escrowError}
        </div>
      )}
    </div>
  );
}
