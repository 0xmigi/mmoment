import { useState, useEffect, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import { unifiedCameraService } from './unified-camera-service';
import { useCompetitionEscrow, type ActiveCompetition } from '../hooks/useCompetitionEscrow';

interface CompetitionControlsProps {
  cameraId: string;
  onEscrowChange?: (escrowInfo: EscrowInfo | null) => void;
  onHasLoadedAppChange?: (hasLoaded: boolean) => void;
}

export interface EscrowInfo {
  pda: string;
  stakeSol: number;
  totalPool?: number;
  participants?: number;
  status?: string;
  winners?: string[];
}

export function CompetitionControls({
  cameraId,
  onEscrowChange,
  onHasLoadedAppChange
}: CompetitionControlsProps) {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoadedApp, setHasLoadedApp] = useState(false);
  const [escrowInfo, setEscrowInfo] = useState<{
    pda: string;
    stakeSol: number;
  } | null>(null);
  const [escrowCompetition, setEscrowCompetition] = useState<ActiveCompetition | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const {
    startCompetition: startEscrowCompetition,
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

  // Notify parent of escrow changes
  useEffect(() => {
    if (onEscrowChange) {
      if (escrowInfo && escrowCompetition) {
        onEscrowChange({
          pda: escrowInfo.pda,
          stakeSol: escrowInfo.stakeSol,
          totalPool: escrowCompetition.totalPool,
          participants: escrowCompetition.participants.length,
          status: escrowCompetition.status,
          winners: escrowCompetition.winners,
        });
      } else if (escrowInfo) {
        onEscrowChange({
          pda: escrowInfo.pda,
          stakeSol: escrowInfo.stakeSol,
        });
      } else {
        onEscrowChange(null);
      }
    }
  }, [escrowInfo, escrowCompetition, onEscrowChange]);

  // Notify parent of hasLoadedApp changes
  useEffect(() => {
    if (onHasLoadedAppChange) {
      onHasLoadedAppChange(hasLoadedApp);
    }
  }, [hasLoadedApp, onHasLoadedAppChange]);

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
        // Only consider competition "loaded" if we have competitors in sessionStorage
        // This prevents stale app state on Jetson from affecting the UI
        const competitorsJson = sessionStorage.getItem('competition_competitors');
        const hasCompetitors = !!competitorsJson;

        if (!hasCompetitors) {
          // No competition setup in progress - don't show competition UI
          setHasLoadedApp(false);
          setIsActive(false);
          return;
        }

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
    setLocalError(null);
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

      // Read escrow info directly from sessionStorage (more reliable than state)
      const escrowPdaFromStorage = sessionStorage.getItem('competition_escrow_pda');
      const stakeFromStorage = sessionStorage.getItem('competition_stake_sol');

      console.log('[CompetitionControls] Escrow info from storage:', {
        escrowPdaFromStorage,
        stakeFromStorage,
        escrowInfoState: escrowInfo
      });

      // If we have an escrow, start it on-chain first
      if (escrowPdaFromStorage) {
        console.log('[CompetitionControls] Starting on-chain competition...');

        // First, fetch and log the current escrow state for debugging
        const currentState = await fetchCompetition(escrowPdaFromStorage);
        console.log('[CompetitionControls] Current escrow state before start:', {
          status: currentState?.status,
          participants: currentState?.participants,
          pendingInvites: currentState?.pendingInvites,
          totalPool: currentState?.totalPool,
        });

        const escrowStarted = await startEscrowCompetition(escrowPdaFromStorage);
        if (!escrowStarted) {
          const errorMsg = escrowError || 'Failed to start on-chain escrow (no participants?)';
          console.error('[CompetitionControls] Failed to start on-chain competition:', errorMsg);
          console.error('[CompetitionControls] Escrow state was:', currentState);
          // DON'T continue - if escrow can't be started, settlement will fail
          // Show error to user and abort
          setLocalError(`Escrow start failed: ${errorMsg}. Check console for details.`);
          setIsLoading(false);
          return;
        }

        // VERIFY the escrow is actually Active now (don't trust the return value)
        console.log('[CompetitionControls] Verifying escrow status after start...');
        const verifyState = await fetchCompetition(escrowPdaFromStorage);
        console.log('[CompetitionControls] Escrow state AFTER start:', {
          status: verifyState?.status,
          participants: verifyState?.participants,
          totalPool: verifyState?.totalPool,
        });

        if (verifyState?.status !== 'active') {
          console.error('[CompetitionControls] Escrow NOT active after startCompetition! Status:', verifyState?.status);
          setLocalError(`Escrow failed to activate. Status: ${verifyState?.status}. Check Solscan.`);
          setIsLoading(false);
          return;
        }

        console.log('[CompetitionControls] On-chain competition VERIFIED active!');
        await refreshEscrowStatus();
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
      // The Jetson handles settlement and includes result in cv_activity_meta for timeline
      const result = await unifiedCameraService.endCompetition(cameraId, competitionMeta);

      if (result.success) {
        setIsActive(false);

        // Log settlement result (will appear in timeline via cv_activity_meta)
        const jetsonResult = result.data?.result;
        const settlement = jetsonResult?.settlement;
        const settlementError = jetsonResult?.settlement_error;

        if (settlement?.success) {
          console.log('[CompetitionControls] Settlement succeeded:', settlement);
        } else if (settlementError) {
          console.error('[CompetitionControls] Settlement failed:', settlementError);
        }

        // Deactivate the app
        await unifiedCameraService.deactivateApp(cameraId);
        setHasLoadedApp(false);

        // Clear session storage
        clearSessionStorage();
      }
    } catch (error) {
      console.error('Error stopping competition:', error);
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
    sessionStorage.removeItem('competition_user_wallet');
    sessionStorage.removeItem('competition_mode');
    sessionStorage.removeItem('competition_target_pushups');
    setEscrowInfo(null);
    setEscrowCompetition(null);
  };

  // Don't show if no app is loaded
  if (!hasLoadedApp) {
    return null;
  }

  return (
    <>
      {/* Start/Stop Button - Bottom center */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
        <button
          onClick={!isActive ? handleStart : handleStop}
          disabled={isLoading || escrowLoading}
          className={`px-8 py-3 flex items-center justify-center rounded-xl shadow-lg transition-all disabled:opacity-50 font-bold text-white ${
            !isActive
              ? 'bg-green-500 hover:bg-green-600'
              : 'bg-red-500 hover:bg-red-600'
          }`}
        >
          {isLoading || escrowLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : !isActive ? (
            'START'
          ) : (
            'STOP'
          )}
        </button>
      </div>

      {/* Error Toast - Bottom Center */}
      {(escrowError || localError) && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 z-50 px-4 py-2 bg-red-100 text-red-700 text-sm rounded-lg shadow-lg max-w-xs text-center">
          {localError || escrowError}
        </div>
      )}
    </>
  );
}

// Export a hook for the exit handler to be used by the scoreboard
export function useCompetitionExit(cameraId: string) {
  const [isLoading, setIsLoading] = useState(false);
  const { cancelCompetition, fetchCompetition } = useCompetitionEscrow();

  const handleExit = useCallback(async () => {
    setIsLoading(true);
    try {
      const escrowPda = sessionStorage.getItem('competition_escrow_pda');

      // Check if competition is active
      const appStatus = await unifiedCameraService.getAppStatus(cameraId);
      if (appStatus.success && appStatus.data?.state?.active) {
        await unifiedCameraService.endCompetition(cameraId);
      }

      // If we have an escrow that's still pending, cancel it
      if (escrowPda) {
        const competition = await fetchCompetition(escrowPda);
        if (competition?.status === 'pending') {
          console.log('[useCompetitionExit] Cancelling on-chain competition...');
          await cancelCompetition(escrowPda, 'User cancelled');
        }
      }

      // Deactivate the app
      await unifiedCameraService.deactivateApp(cameraId);

      // Clear session storage
      sessionStorage.removeItem('competition_competitors');
      sessionStorage.removeItem('competition_duration');
      sessionStorage.removeItem('competition_app');
      sessionStorage.removeItem('competition_escrow_pda');
      sessionStorage.removeItem('competition_escrow_created_at');
      sessionStorage.removeItem('competition_stake_sol');
    } catch (error) {
      console.error('Error exiting competition mode:', error);
    } finally {
      setIsLoading(false);
    }
  }, [cameraId, cancelCompetition, fetchCompetition]);

  return { handleExit, isLoading };
}
