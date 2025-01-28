import { Camera, Video, Play, Square } from 'lucide-react';

// In MobileControls.tsx
interface MobileControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  onToggleStream?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
}

export function MobileControls({ onTakePicture, onRecordVideo, onToggleStream, isLoading, isStreaming }: MobileControlsProps) {
  return (
    <div className="sm:hidden fixed right-4 bottom-10 flex flex-col gap-4 z-50">
      <button 
        onClick={onTakePicture}
        disabled={isLoading}
        className={`w-11 h-11 rounded-lg ${isLoading ? 'bg-stone-400 cursor-not-allowed' : ' bg-[#e7eeff] hover:bg-stone-500'} drop-shadow-lg flex items-center justify-center`}
        aria-label="Take Picture"
      >
        <Camera className="w-6 h-6 text-gray-800" />
      </button>
      
      <button
        onClick={onRecordVideo}
        disabled={isLoading}
        className={`w-11 h-11 rounded-lg ${isLoading ? 'bg-gray-400 cursor-not-allowed' : ' bg-[#e7eeff] hover:bg-gray-500'} drop-shadow-lg flex items-center justify-center`}
        aria-label="Record Video"
      >
        <Video className="w-6 h-6 text-gray-800" />
      </button>

      {onToggleStream && (
        <button
          onClick={onToggleStream}
          disabled={isLoading}
          className={`w-11 h-11 rounded-lg ${isLoading ? 'bg-gray-400 cursor-not-allowed' : ' bg-[#e7eeff] hover:bg-gray-500'} drop-shadow-lg flex items-center justify-center`}
          aria-label={isStreaming ? "Stop Stream" : "Start Stream"}
        >
          {isStreaming ? (
            <Square className="w-6 h-6 text-gray-800" />
          ) : (
            <Play className="w-6 h-6 text-gray-800" />
          )}
        </button>
      )}
    </div>
  );
}