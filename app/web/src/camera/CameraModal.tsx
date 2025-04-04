import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  camera: {
    id: string;
    owner: string;
    ownerDisplayName?: string;
    ownerPfpUrl?: string;
    isLive: boolean;
    isStreaming: boolean;
    status: 'ok' | 'error' | 'offline';
    lastSeen?: number;
    activityCounter?: number;
    model?: string;
    // New properties for development info
    showDevInfo?: boolean;
    defaultDevCamera?: string;
  };
}

export function CameraModal({ isOpen, onClose, camera }: CameraModalProps) {
  const { primaryWallet } = useDynamicContext();
  if (!isOpen) return null;

  // Check if current user is the owner
  const isOwner = primaryWallet?.address === camera.owner;

  const handleCameraExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.id}?cluster=devnet`, '_blank');
  };

  const handleOwnerExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.owner}?cluster=devnet`, '_blank');
  };

  // Format address for display
  const formatAddress = (address: string, start = 6, end = 6) => {
    if (!address) return '';
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  };

  const handleDevCameraClick = () => {
    if (camera.defaultDevCamera) {
      // Redirect to the correct camera page URL
      const baseUrl = window.location.origin;
      window.location.href = `${baseUrl}/app/camera/${camera.defaultDevCamera}`;
    }
  };

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Camera Details
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Camera Content */}
          <div className="p-4">
            {camera.showDevInfo ? (
              // Development section when no camera is connected
              <div className="space-y-4">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
                  <h3 className="text-sm font-medium text-yellow-800 mb-1">Development Mode</h3>
                  <p className="text-xs text-yellow-700 mb-2">
                    No camera is currently connected. Connect to the physical camera below:
                  </p>
                  <button
                    onClick={handleDevCameraClick}
                    className="text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                  >
                    Connect to Camera <ExternalLink className="h-3 w-3 ml-1" />
                  </button>
                </div>
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Camera PDA</div>
                  <div className="flex items-center">
                    <div className="text-sm font-medium">
                      {formatAddress(camera.defaultDevCamera || '')}
                    </div>
                    <button
                      onClick={() => window.open(`https://solscan.io/account/${camera.defaultDevCamera}?cluster=devnet`, '_blank')}
                      className="ml-1 text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                    >
                      <ExternalLink className="w-3 h-3 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              // Original camera details layout
              <>
                {/* Camera PDA */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-50 flex-shrink-0 flex items-center justify-center overflow-hidden">
                    <Camera className="w-5 h-5 text-blue-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">{formatAddress(camera.id)}</div>
                  </div>
                  <button
                    onClick={handleCameraExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Owner */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-gray-100 flex-shrink-0 flex items-center justify-center overflow-hidden ml-0">
                    <User className="w-5 h-5 text-gray-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Owner</div>
                    <div className="text-sm font-medium">{formatAddress(camera.owner, 9, 5)}</div>
                  </div>
                  <button
                    onClick={handleOwnerExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Camera Name */}
                {camera.ownerDisplayName && (
                  <div className="mb-4">
                    <div className="text-sm text-gray-700">Camera Name</div>
                    <div className="text-sm">{camera.ownerDisplayName}</div>
                  </div>
                )}

                {/* Type */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Type</div>
                  <div className="text-sm">{camera.model || "pi5"}</div>
                </div>

                {/* Activity */}
                {camera.activityCounter !== undefined && (
                  <div className="mb-4">
                    <div className="text-sm text-gray-700">Activity</div>
                    <div className="text-sm">{camera.activityCounter} total interactions</div>
                  </div>
                )}

                {/* Your Status */}
                <div>
                  <div className="text-sm text-gray-700">Your Status</div>
                  <div className="text-sm">
                    {isOwner ? (
                      <span className="text-green-600">You own this camera</span>
                    ) : (
                      <span className="text-gray-600">Viewer</span>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
} 