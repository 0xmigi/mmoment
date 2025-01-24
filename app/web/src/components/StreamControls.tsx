import { useState, useEffect } from 'react';
import { Play, StopCircle, Loader } from 'lucide-react';
import { CONFIG } from '../config';
import { useCamera } from './CameraProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useProgram } from '../anchor/setup';
import { PublicKey, SystemProgram } from '@solana/web3.js';
import { BN } from '@coral-xyz/anchor';

interface StreamControlsProps {
    timelineRef?: React.MutableRefObject<any>;
}

export const StreamControls = ({ timelineRef }: StreamControlsProps) => {
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const { cameraKeypair, isInitialized, loading: initLoading } = useCamera();
    const { primaryWallet } = useDynamicContext();
    const program = useProgram();

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
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setIsStreaming(data.isActive);
        } catch (error) {
            console.error('Failed to check stream status:', error);
        }
    };

    const handleStartStream = async () => {
        if (!primaryWallet?.address || !program || !isInitialized) {
            return;
        }
        setIsLoading(true);

        try {
            await program.methods.activateCamera(new BN(100))
                .accounts({
                    cameraAccount: cameraKeypair.publicKey,
                    user: new PublicKey(primaryWallet.address),
                    systemProgram: SystemProgram.programId,
                })
                .rpc();

            const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/start`, {
                method: 'POST'
            });

            if (response.ok) {
                setIsStreaming(true);
                
                if (timelineRef?.current) {
                    const event = {
                        type: 'stream_started' as const,
                        timestamp: Date.now(),
                        user: { address: primaryWallet.address }
                    };
                    timelineRef.current.addEvent(event);
                }
            } else {
                const errorText = await response.text();
                console.error("Stream start failed:", errorText);
            }
        } catch (error) {
            console.error('Failed to start stream:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStopStream = async () => {
        if (!primaryWallet?.address || !program || !isInitialized) return;
        setIsLoading(true);

        try {
            await program.methods.activateCamera(new BN(100))
                .accounts({
                    cameraAccount: cameraKeypair.publicKey,
                    user: new PublicKey(primaryWallet.address),
                    systemProgram: SystemProgram.programId,
                })
                .rpc();

            const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/stop`, {
                method: 'POST'
            });
            if (response.ok) {
                setIsStreaming(false);
                
                if (timelineRef?.current) {
                    const event = {
                        type: 'stream_ended' as const,
                        timestamp: Date.now(),
                        user: { address: primaryWallet.address }
                    };
                    timelineRef.current.addEvent(event);
                }
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