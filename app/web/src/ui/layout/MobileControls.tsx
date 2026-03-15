import { useState, useCallback } from 'react';

function PhotoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 1576.3 1356.35" fill="currentColor">
      <path d="M785.97,1356.24c-177.8.34-356.08.72-533.09-18.25-47.99-5.96-96.7-13.36-139.22-38.02-62.07-34.5-99.32-96.12-104.06-165.43-5.93-62.11-8.13-124.74-9.18-187.13-1.02-132.83-.67-265.71,6.1-398.4,9.27-169.16,94.24-271.07,264.21-294.04,30.21-7.31,64.66-7.11,92.53-20.76,12.91-12.08,18.98-30.23,29.54-44.32C458.38,91.15,535.08,4.02,662.54,2.1c64.38-3.3,128.98-1.15,193.43-2.1,54.02.51,110.07,1.85,160.13,24.39,71.58,33.67,121.34,98.61,164.91,162.32,14.9,20.39,23.19,51.97,52.35,54.94,53.45,11.16,108.16,17.6,160.04,35.12,131.85,45.93,175.12,172.13,176.09,301.45,8,171.67,10.04,344.02.12,515.66-.85,93.78-34.65,172.62-123.23,214.35-58.29,27.89-124.26,27.6-187.21,35.3-157.18,13.38-315.36,13.07-473.01,12.72h-.19ZM786.91,389.55c-100.93-2.33-206.24,44.21-274.03,122.36-68.38,76.3-97.86,173.74-88.24,274.89,12.51,178.33,171.28,328.39,349.63,330.17,30.71.94,62.62-1.87,92.54-8.52,172.05-35.85,296.01-202.26,282.4-381.88-11.19-186.6-173.19-341.01-362.11-337.03h-.2Z" />
      <path d="M536,749.61c.98-180.64,188.07-297.25,352.19-226.56,91.8,38.94,153.05,134.89,150.27,234.53-.91,140.43-122.53,259.02-266.99,247.26-134.96-7.34-237.95-121.1-235.47-255.03v-.2Z" />
    </svg>
  );
}

function VideoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 1559.61 1073.32" fill="currentColor">
      <path d="M1156.91,762.4c-7.13,6.55-4.91,20.22-7.21,29.17-1.97,14.6-3.78,29.13-6.52,43.62-23.81,131.41-138.41,211.62-266.31,224.53-75.64,9.42-151.83,12.79-228.02,12.72-81.09.4-162.34,2.64-243.33-2.01-59.11-4.59-119.05-6.85-176.91-20.67C80.92,1009.86,13.54,905.97,7.68,756.75c-4.51-74.31-9.52-148.87-7.01-223.33-1.7-97.9,1.62-196.51,16.56-293.41C35.09,135.83,114.04,56.43,213.9,27.39c43.93-13.33,90.05-16.33,135.53-20.41C463.65-2.33,578.56.46,693.05.19c25.49.27,50.94,1.87,76.35,3.79,57.3,4.61,115.67,6.48,171.37,21.87,98.7,26.29,179.65,102.69,200.76,204.38,5.78,25.71,8.3,51.44,11.45,77.73.74,4.13,1.13,10.25,5.24,12.12,4.86,2.16,12.49-3.42,17.18-5.89,63.71-36.59,126.94-74.01,191.15-109.72,36.9-19.9,75.14-44.53,119.13-35.7,55.99,8.42,63.01,60.95,66.07,108.22,11.07,121.27,6.11,243.01,7.86,364.58-.72,72.1-3.65,144.41-14.45,215.76-10.57,55.24-75.8,70.97-121.79,49.6-86.6-41.72-169.38-91.26-252.02-140.21-3.87-2.29-10.65-6.61-14.35-4.39l-.1.05Z" />
    </svg>
  );
}

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
    setTimeout(() => setIsBlinking(false), 300);
  }, [isLoading, onTakePicture]);

  return (
    <>
      {/* Shape-based controls — bottom right, thumb-friendly */}
      <div className="md:hidden fixed right-4 bottom-20 z-40 flex flex-col gap-3.5 items-center">
        {/* Photo: camera shape — blinks on tap */}
        <button
          onClick={handlePhoto}
          disabled={isLoading}
          className={`w-[42px] h-[42px] flex items-center justify-center transition-all duration-150 ${
            isLoading
              ? 'opacity-50 cursor-not-allowed'
              : isBlinking
                ? 'animate-photo-shutter'
                : 'active:scale-90'
          }`}
          aria-label="Take Picture"
        >
          <PhotoIcon className={`w-full h-full drop-shadow-[0_0_6px_rgba(0,0,0,0.3)] ${
            isLoading ? 'text-white/50' : 'text-white'
          }`} />
        </button>

        {/* Video: camcorder shape — turns red when recording */}
        <button
          onClick={onRecordVideo}
          disabled={isLoading}
          className={`w-[46px] h-[34px] flex items-center justify-center transition-all duration-200 ${
            isLoading
              ? 'opacity-50 cursor-not-allowed'
              : isRecording
                ? 'scale-[0.9] animate-record-pulse'
                : 'active:scale-90'
          }`}
          aria-label={isRecording ? "Stop Recording" : "Record Video"}
        >
          <VideoIcon className={`w-full h-full ${
            isLoading
              ? 'text-white/50'
              : isRecording
                ? 'text-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.4)]'
                : 'text-white drop-shadow-[0_0_6px_rgba(0,0,0,0.3)]'
          }`} />
        </button>
      </div>
    </>
  );
}
