import { Camera, Video, Play, Square } from 'lucide-react';

// Renamed from MobileControls to CameraControls
interface CameraControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  onToggleStream?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
}

export function CameraControls({ onTakePicture, onRecordVideo, onToggleStream, isLoading, isStreaming }: CameraControlsProps) {
  return (
    <div className="md:hidden flex gap-2">
      <button 
        onClick={onTakePicture}
        disabled={isLoading}
        className={`flex-1 h-10 rounded-lg ${isLoading ? 'bg-gray-200 cursor-not-allowed' : 'bg-gray-50'} flex items-center justify-center gap-2`}
        aria-label="Take Picture"
      >
        <Camera className="w-4 h-4 text-gray-600 hover:text-stone-800" />
        <span className="text-sm font-medium text-gray-600 hover:text-stone-800">Photo</span>
      </button>
      
      <button
        onClick={onRecordVideo}
        disabled={isLoading}
        className={`flex-1 h-10 rounded-lg ${isLoading ? 'bg-gray-200 cursor-not-allowed' : 'bg-gray-50'} flex items-center justify-center gap-2`}
        aria-label="Record Video"
      >
        <Video className="w-4 h-4 text-gray-600 hover:text-stone-800" />
        <span className="text-sm font-medium text-gray-600 hover:text-stone-800">Video</span>
      </button>

      {onToggleStream && (
        <button
          onClick={onToggleStream}
          disabled={isLoading}
          className={`flex-1 h-10 rounded-lg ${
            isLoading 
              ? 'bg-gray-200 cursor-not-allowed' 
              : isStreaming 
                ? 'bg-red-500 hover:bg-red-600' 
                : 'bg-gray-50'
          } flex items-center justify-center gap-2`}
          aria-label={isStreaming ? "Stop Stream" : "Start Stream"}
        >
          {isStreaming ? (
            <>
              <Square className="w-4 h-4 text-white" />
              <span className="text-sm font-medium text-white">Stop</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-gray-600 hover:text-stone-800" />
              <span className="text-sm font-medium text-gray-600 hover:text-stone-800">Stream</span>
            </>
          )}
        </button>
      )}
    </div>
  );
}