import { Camera, Video } from 'lucide-react';

// Renamed from MobileControls to CameraControls
interface CameraControlsProps {
  onTakePicture: () => void;
  onRecordVideo: () => void;
  onToggleStream?: () => void;
  isLoading?: boolean;
  isStreaming?: boolean;
}

export function CameraControls({ onTakePicture, onRecordVideo, isLoading }: CameraControlsProps) {
  return (
    <>
      {/* TikTok-style vertical controls for mobile - positioned in bottom right to match hamburger menu */}
      <div className="md:hidden fixed right-4 bottom-20 z-40 flex flex-col gap-2">
        <button 
          onClick={onTakePicture}
          disabled={isLoading}
          className={`p-2 rounded-lg ${
            isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#999999] hover:bg-[#d1dfff]'
          } transition-colors`}
          aria-label="Take Picture"
        >
          <Camera className={`w-5 h-5 ${isLoading ? 'text-gray-600' : 'text-gray-50'}`} />
        </button>
        
        <button
          onClick={onRecordVideo}
          disabled={isLoading}
          className={`p-2 rounded-lg ${
            isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#999999] hover:bg-[#d1dfff]'
          } transition-colors`}
          aria-label="Record Video"
        >
          <Video className={`w-5 h-5 ${isLoading ? 'text-gray-600' : 'text-gray-50'}`} />
        </button>
      </div>

    </>
  );
}