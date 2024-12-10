import { Camera, Video } from 'lucide-react';

// In MobileControls.tsx
interface MobileControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  isLoading?: boolean;
}

export function MobileControls({ onTakePicture, onRecordVideo, isLoading }: MobileControlsProps) {
  return (
    <div className="sm:hidden fixed right-4 bottom-20 flex flex-col gap-4 z-50">
      <button 
        onClick={onTakePicture}
        disabled={isLoading}
        className={`w-12 h-12 rounded-lg ${isLoading ? 'bg-stone-400 cursor-not-allowed' : 'bg-stone-400 hover:bg-stone-500'} shadow-lg flex items-center justify-center`}
        aria-label="Take Picture"
      >
        <Camera className="w-6 h-6 text-white" />
      </button>
      
      <button
        onClick={onRecordVideo}
        disabled={isLoading}
        className={`w-12 h-12 rounded-lg ${isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-400 hover:bg-gray-500'} shadow-lg flex items-center justify-center`}
        aria-label="Record Video"
      >
        <Video className="w-6 h-6 text-white" />
      </button>
    </div>
  );
}