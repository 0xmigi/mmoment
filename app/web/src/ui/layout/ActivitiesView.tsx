import { Timeline } from '../../timeline/Timeline';
import { useRef, useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useParams } from 'react-router-dom';

type FilterType = 'all' | 'camera' | 'my';

export function ActivitiesView() {
    const timelineRef = useRef<any>(null);
    const { primaryWallet } = useDynamicContext();
    const [activeFilter, setActiveFilter] = useState<FilterType>('all');
    const { cameraId } = useParams<{ cameraId?: string }>();
    
    useEffect(() => {
        console.log('ActivitiesView with cameraId:', cameraId);
        // Store cameraId in localStorage for persistence if available
        if (cameraId) {
            localStorage.setItem('directCameraId', cameraId);
            // Set filter to 'camera' when a specific camera is selected
            if (activeFilter !== 'camera') {
                setActiveFilter('camera');
            }
        }
    }, [cameraId, activeFilter]);

    const filters: Array<{ id: FilterType; label: string }> = [
        { id: 'all', label: 'All' },
        { id: 'camera', label: 'Camera' },
        { id: 'my', label: 'Mine' },
    ];

    return (
        <div className="pb-20">
            <div className="max-w-3xl mx-auto pt-8">
                <div className="bg-white rounded-lg p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-semibold">Activities</h2>
                        <div className="flex gap-2">
                            {filters.map((filter) => (
                                <button
                                    key={filter.id}
                                    onClick={() => setActiveFilter(filter.id)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                                        ${activeFilter === filter.id
                                            ? 'bg-gray-900 text-white'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }`}
                                >
                                    {filter.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="bg-gray-50 rounded-lg p-4">
                            <Timeline
                                ref={timelineRef}
                                filter={activeFilter}
                                userAddress={primaryWallet?.address}
                                variant="full"
                                cameraId={cameraId}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}