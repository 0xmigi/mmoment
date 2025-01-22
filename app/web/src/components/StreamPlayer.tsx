import { Player } from "@livepeer/react";
import { useState, useEffect } from "react";
import { CONFIG } from "../config";

interface StreamInfo {
  playbackId: string;
  isActive: boolean;
}

export function StreamPlayer() {
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStreamInfo = async () => {
    try {
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`);
      if (!response.ok) {
        throw new Error('Failed to fetch stream info');
      }
      const data = await response.json();
      console.log('Received stream info:', data);
      setStreamInfo({
        playbackId: data.playbackId,
        isActive: data.isActive
      });
      setError(null);
    } catch (err) {
      console.error('Failed to get stream info:', err);
      setError('Failed to get stream info');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStreamInfo();
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchStreamInfo, 5000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="aspect-video border-2 border-gray-200 rounded-lg flex items-center justify-center">
        <p className="text-gray-500">Loading stream...</p>
      </div>
    );
  }

  if (error || !streamInfo?.playbackId) {
    return (
      <div className="aspect-video border-2 border-gray-200 rounded-lg flex items-center justify-center">
        <p className="text-gray-500">Stream not available</p>
      </div>
    );
  }

  return (
    <div className="aspect-video bg-black rounded-lg overflow-hidden">
      <Player 
        title="Camera Stream"
        playbackId={streamInfo.playbackId}
        autoPlay
        muted
      />
    </div>
  );
}