import { useState, useEffect } from 'react';
import { Play, StopCircle } from 'lucide-react';
import { CONFIG } from '../config';

export const StreamControls = () => {
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    // Check stream status on mount
    useEffect(() => {
        checkStreamStatus();
        // Poll every 10 seconds
        const interval = setInterval(checkStreamStatus, 10000);
        return () => clearInterval(interval);
    }, []);

    const checkStreamStatus = async () => {
        try {
            const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`);
            const data = await response.json();
            setIsStreaming(data.isActive);
        } catch (error) {
            console.error('Failed to check stream status:', error);
        }
    };

    const handleStartStream = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/start`, {
                method: 'POST'
            });
            if (response.ok) {
                setIsStreaming(true);
            }
        } catch (error) {
            console.error('Failed to start stream:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStopStream = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/stop`, {
                method: 'POST'
            });
            if (response.ok) {
                setIsStreaming(false);
            }
        } catch (error) {
            console.error('Failed to stop stream:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="group h-1/2 relative">
            <button
                onClick={isStreaming ? handleStopStream : handleStartStream}
                disabled={isLoading}
                className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-black transition-colors rounded-xl"
                aria-label={isStreaming ? "Stop Stream" : "Start Stream"}
            >
                {isLoading ? (
                    <div className="animate-spin rounded-full h-5 w-5" />
                ) : isStreaming ? (
                    <StopCircle className="w-5 h-5" />
                ) : (
                    <Play className="w-5 h-5" />
                )}
            </button>

            {/* Status tooltip */}
            <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                {isLoading ? 'Processing...' : isStreaming ? 'Stop Stream' : 'Start Stream'}
            </span>

            {/* Stream status indicator */}
            {isStreaming && (
                <div className="absolute -top-2 -right-2">
                    <div className="flex items-center gap-1 bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                        LIVE
                    </div>
                </div>
            )}
        </div>
    );
};