import { useState } from 'react';
import { Smartphone, Eye, Video, Users, Lock, Zap, X, Check } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useFacialEmbeddingStatus } from '../hooks/useFacialEmbeddingStatus';
import { PhoneSelfieEnrollment } from './PhoneSelfieEnrollment';
import { useParams } from 'react-router-dom';

interface IRLAppsButtonProps {
  cameraId: string;
  walletAddress?: string;
  onEnrollmentComplete?: () => void;
}

export function IRLAppsButton({ cameraId, walletAddress, onEnrollmentComplete }: IRLAppsButtonProps) {
  const [showAppsModal, setShowAppsModal] = useState(false);
  const [showEnrollment, setShowEnrollment] = useState(false);
  const { primaryWallet } = useDynamicContext();
  const facialEmbeddingStatus = useFacialEmbeddingStatus();

  // Store the camera PDA in localStorage when this component mounts
  const { cameraId: cameraIdFromUrl } = useParams<{ cameraId: string }>();
  if (cameraIdFromUrl && typeof window !== 'undefined') {
    localStorage.setItem('lastAccessedCameraPDA', cameraIdFromUrl);
  }

  const handleIRLAppsClick = () => {
    // Always show the app store first - users can see what's available
    setShowAppsModal(true);
  };

  // Define available CV apps for this camera
  const availableApps = [
    {
      id: 'face_recognition',
      name: 'Face Recognition',
      description: 'Automatic detection and identification',
      icon: <Eye className="w-5 h-5" />,
      enabled: true
    },
    {
      id: 'gesture_controls',
      name: 'Gesture Controls',
      description: 'Take photos with hand gestures',
      icon: <Users className="w-5 h-5" />,
      enabled: true
    },
    {
      id: 'auto_streaming',
      name: 'Auto Streaming',
      description: 'Start/stop streams with presence',
      icon: <Video className="w-5 h-5" />,
      enabled: false // Example: not all cameras support this
    }
  ];

  // Only show the button if we have a wallet address
  if (!walletAddress) {
    return null;
  }

  return (
    <>
      <button
        onClick={handleIRLAppsClick}
        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg shadow-lg transition-colors"
        title={facialEmbeddingStatus.hasEmbedding ? "Access IRL Apps" : "Create Recognition Token for IRL Apps"}
      >
        <Zap className="w-4 h-4" />
        <span className="text-sm">IRL Apps</span>
        {!facialEmbeddingStatus.hasEmbedding && (
          <Lock className="w-3 h-3 opacity-70" />
        )}
      </button>

      {/* Apps Drawer - Full screen with proper mobile sizing */}
      {showAppsModal && (
        <div className="fixed inset-0 bg-black/30 z-[200]">
          <div className="fixed inset-x-0 top-16 bottom-0 bg-white rounded-t-2xl">
            {/* Header - Mobile sized like CameraModal */}
            <div className="flex items-center justify-between p-3 border-b border-gray-100">
              <h3 className="text-base font-medium">IRL Apps</h3>
              <button
                onClick={() => setShowAppsModal(false)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>

            {/* Content */}
            <div className="flex flex-col" style={{ height: 'calc(100vh - 120px)' }}>
              <div className="flex-1 p-4 overflow-y-auto">
                <div className="space-y-3 mb-4">
                  {availableApps.map((app) => {
                    const isAccessible = facialEmbeddingStatus.hasEmbedding && app.enabled;
                    const needsToken = !facialEmbeddingStatus.hasEmbedding;

                    return (
                      <div key={app.id} className="flex items-center mb-4">
                        <div className={`w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center overflow-hidden ${
                          isAccessible ? 'bg-green-50' : needsToken ? 'bg-orange-50' : 'bg-gray-50'
                        }`}>
                          {app.icon}
                        </div>
                        <div className="ml-3 flex-1">
                          <div className="text-sm text-gray-700">{app.name}</div>
                          <div className="text-xs text-gray-500">{app.description}</div>
                        </div>
                        <div className={`text-xs px-2 py-1 rounded-full ${
                          isAccessible ? 'bg-green-100 text-green-700' :
                          needsToken ? 'bg-orange-100 text-orange-700' : 'bg-gray-100 text-gray-600'
                        }`}>
                          {isAccessible ? 'Available' : needsToken ? 'Locked' : 'Offline'}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {facialEmbeddingStatus.hasEmbedding && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <h3 className="text-sm font-medium text-green-800 mb-1">Recognition Token Active</h3>
                    <p className="text-xs text-green-700">All apps unlocked across the network</p>
                  </div>
                )}

                {!facialEmbeddingStatus.hasEmbedding && (
                  <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
                    <h3 className="text-sm font-medium text-orange-800 mb-1">Apps Locked</h3>
                    <p className="text-xs text-orange-700">Create a recognition token to unlock</p>
                  </div>
                )}
              </div>

              {/* Bottom Button */}
              {!facialEmbeddingStatus.hasEmbedding && (
                <div className="p-4 border-t border-gray-100">
                  <button
                    onClick={() => {
                      setShowAppsModal(false);
                      setShowEnrollment(true);
                    }}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    Create Recognition Token
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Enrollment Drawer - Full screen camera drawer */}
      {showEnrollment && (
        <div className="fixed inset-0 bg-black/30 z-[200]">
          <div className="fixed inset-x-0 top-16 bottom-0 bg-white rounded-t-2xl">
            {/* Header - Mobile sized like CameraModal */}
            <div className="flex items-center justify-between p-3 border-b border-gray-100">
              <h3 className="text-base font-medium">Create Recognition Token</h3>
              <button
                onClick={() => setShowEnrollment(false)}
                className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>

            {/* Camera Content - Fill remaining drawer space with padding and rounded corners */}
            <div className="p-2" style={{ height: 'calc(100vh - 120px)' }}>
              <div className="h-full rounded-lg overflow-hidden">
                <PhoneSelfieEnrollment
                  onEnrollmentComplete={(result) => {
                    if (result.success) {
                      setShowEnrollment(false);
                      if (onEnrollmentComplete) {
                        onEnrollmentComplete();
                      }
                      // Automatically show apps after successful enrollment
                      setTimeout(() => setShowAppsModal(true), 500);
                    }
                  }}
                  onCancel={undefined} // No duplicate X button
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}