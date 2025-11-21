import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { unifiedCameraService } from './unified-camera-service';

interface CompetitionControlsProps {
  cameraId: string;
}

export function CompetitionControls({ cameraId }: CompetitionControlsProps) {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoadedApp, setHasLoadedApp] = useState(false);

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

      // Start the competition
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
      const result = await unifiedCameraService.endCompetition(cameraId);
      if (result.success) {
        setIsActive(false);
        // Deactivate the app
        await unifiedCameraService.deactivateApp(cameraId);
        setHasLoadedApp(false);
        // Clear session storage
        sessionStorage.removeItem('competition_competitors');
        sessionStorage.removeItem('competition_duration');
        sessionStorage.removeItem('competition_app');
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
      // Deactivate the app
      await unifiedCameraService.deactivateApp(cameraId);
      setHasLoadedApp(false);
      setIsActive(false);
      // Clear session storage
      sessionStorage.removeItem('competition_competitors');
      sessionStorage.removeItem('competition_duration');
      sessionStorage.removeItem('competition_app');
    } catch (error) {
      console.error('Error exiting competition mode:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Don't show if no app is loaded
  if (!hasLoadedApp) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center space-x-2">
      <button
        onClick={!isActive ? handleStart : handleStop}
        disabled={isLoading}
        className={`w-24 py-2 flex items-center justify-center bg-white rounded-lg shadow-lg transition-colors disabled:bg-gray-200 ${
          !isActive ? 'text-green-600 hover:bg-gray-50' : 'text-red-600 hover:bg-gray-50'
        }`}
      >
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <span className="text-sm font-medium">{!isActive ? 'Start' : 'Stop'}</span>
        )}
      </button>

      <button
        onClick={handleExit}
        disabled={isLoading}
        className="w-24 py-2 flex items-center justify-center bg-white text-black hover:bg-gray-50 rounded-lg shadow-lg transition-colors disabled:bg-gray-200"
      >
        <span className="text-sm font-medium">Cancel</span>
      </button>
    </div>
  );
}
