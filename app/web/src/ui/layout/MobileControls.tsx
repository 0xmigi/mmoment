import { useState, useCallback } from 'react';

interface CameraControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  onToggleStream?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
  isRecording?: boolean;
}

export function CameraControls({ onTakePicture, onRecordVideo, isLoading, isRecording }: CameraControlsProps) {
  const [isBlinking, setIsBlinking] = useState(false);

  const handlePhoto = useCallback(() => {
    if (isLoading) return;
    setIsBlinking(true);
    onTakePicture();
    // Reset after animation completes
    setTimeout(() => setIsBlinking(false), 300);
  }, [isLoading, onTakePicture]);

  return (
    <>
      {/* Shape-based controls — bottom right, thumb-friendly */}
      <div className="md:hidden fixed right-4 bottom-20 z-40 flex flex-col gap-3.5 items-center">
        {/* Photo: circle — blinks like an eye on tap */}
        <button
          onClick={handlePhoto}
          disabled={isLoading}
          className={`w-[42px] h-[42px] rounded-full transition-all duration-150 ${
            isLoading
              ? 'bg-white/50 cursor-not-allowed'
              : isBlinking
                ? 'animate-blink'
                : 'bg-white shadow-[0_0_8px_rgba(0,0,0,0.3)] active:scale-90'
          }`}
          aria-label="Take Picture"
        />

        {/* Video: rounded-square — turns red when recording */}
        <button
          onClick={onRecordVideo}
          disabled={isLoading}
          className={`w-[42px] h-[42px] rounded-xl transition-all duration-200 ${
            isLoading
              ? 'bg-white/50 cursor-not-allowed'
              : isRecording
                ? 'bg-red-500 scale-[0.9] shadow-lg shadow-red-500/30'
                : 'bg-white shadow-[0_0_8px_rgba(0,0,0,0.3)] active:scale-90'
          }`}
          aria-label={isRecording ? "Stop Recording" : "Record Video"}
        />
      </div>
    </>
  );
}
