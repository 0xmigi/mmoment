import { Play, StopCircle, Loader } from 'lucide-react';
import { useCamera } from '../camera/CameraProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

interface StreamControlsProps {
    timelineRef?: React.MutableRefObject<any>;
    onStreamToggle?: () => void;
    isStreaming?: boolean;  // Add streaming state as prop
    isLoading?: boolean;    // Add loading state as prop
}

export const StreamControls = ({ onStreamToggle, isStreaming = false, isLoading = false }: StreamControlsProps) => {
    const { isInitialized, loading: initLoading } = useCamera();
    useDynamicContext();

    // Handle the button click directly
    const handleStreamClick = () => {
        console.log("STREAM BUTTON CLICKED IN COMPONENT - direct handler");
        if (onStreamToggle) {
            onStreamToggle();
        }
    };

    return (
        <div className="group h-1/2 relative">
            <button
                onClick={handleStreamClick}
                disabled={isLoading || initLoading || !isInitialized}
                className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-black transition-colors rounded-xl"
                aria-label={isStreaming ? "Stop Stream" : "Start Stream"}
            >
                {isLoading || initLoading ? (
                    <Loader className="w-5 h-5 animate-spin" />
                ) : isStreaming ? (
                    <StopCircle className="w-5 h-5" />
                ) : (
                    <Play className="w-5 h-5" />
                )}
            </button>

            {/* Status tooltip */}
            <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                {isLoading || initLoading ? 'Processing...' : isStreaming ? 'Stop Stream' : 'Start Stream'}
            </span>
        </div>
    );
};