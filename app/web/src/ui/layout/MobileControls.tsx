import { Camera, Video } from 'lucide-react';

// Renamed from MobileControls to CameraControls
interface CameraControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  onToggleStream?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
  isRecording?: boolean;
}

export function CameraControls({ onTakePicture, onRecordVideo, isLoading, isRecording }: CameraControlsProps) {
  return (
    <>
      {/* TikTok-style vertical controls for mobile - positioned in bottom right to match hamburger menu */}
      <div className="md:hidden fixed right-4 bottom-20 z-40 flex flex-col gap-2">
        <button
          onClick={onTakePicture}
          disabled={isLoading}
          className={`w-9 h-9 flex items-center justify-center rounded-lg transition-colors ${
            isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-500 hover:bg-gray-400'
          }`}
          aria-label="Take Picture"
        >
          <Camera className={`w-5 h-5 ${isLoading ? 'text-gray-600' : 'text-gray-50'}`} />
        </button>

        <button
          onClick={onRecordVideo}
          disabled={isLoading}
          className={`w-9 h-9 flex items-center justify-center rounded-lg transition-all duration-200 ${
            isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : isRecording
                ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/30'
                : 'bg-gray-500 hover:bg-gray-400'
          }`}
          aria-label={isRecording ? "Stop Recording" : "Record Video"}
        >
          {isRecording ? (
            <span className="w-3 h-3 bg-white rounded-[3px]" />
          ) : (
            <Video className={`w-5 h-5 ${isLoading ? 'text-gray-600' : 'text-gray-50'}`} />
          )}
        </button>
      </div>
    </>
  );
}