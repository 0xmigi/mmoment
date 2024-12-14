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

  useEffect(() => {
    fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`)
      .then(res => res.json())
      .then(data => {
        if (data.playbackId) {
          setStreamInfo({
            playbackId: data.playbackId,
            isActive: data.isActive
          });
        }
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to get stream info:', err);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) {
    return (
      <div className="aspect-video bg-gray-200 rounded-lg flex items-center justify-center">
        <p className="text-gray-500">Loading stream...</p>
      </div>
    );
  }

  if (!streamInfo?.playbackId) {
    return (
      <div className="aspect-video bg-gray-200 rounded-lg flex items-center justify-center">
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