import { Timeline } from '../Timeline';
import { useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

export function ActivitiesView() {
    const timelineRef = useRef<any>(null);
    const { primaryWallet } = useDynamicContext();

    return (
        <div className="h-full overflow-y-auto pb-20">
            <div className="max-w-3xl mx-auto pt-8 ">
                <div className="bg-white rounded-lg p-6">
                    <h2 className="text-xl font-semibold mb-6">Activites</h2>
                    <div className="space-y-6">
                        <div className="bg-gray-50 rounded-lg p-4">
                            <h3 className="text-sm font-medium text-gray-600 mb-4">Camera Activity</h3>
                            <Timeline ref={timelineRef} maxEvents={50} />
                        </div>

                        <div className="bg-gray-50 rounded-lg p-4">
                            <h3 className="text-sm font-medium text-gray-600 mb-4">
                                Wallet Activity ({primaryWallet?.address?.slice(0, 6)}...{primaryWallet?.address?.slice(-4)})
                            </h3>
                            {/* We'll add wallet-specific activity here later */}
                            <p className="text-sm text-gray-500">Wallet activity coming soon...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ActivitiesView;