import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { X, Calendar, ExternalLink, Camera } from 'lucide-react';
import { IPFSMedia } from '../services/ipfs-service';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

interface MediaViewerProps {
  isOpen: boolean;
  onClose: () => void;
  media: IPFSMedia | null;
}

export default function MediaViewer({ isOpen, onClose, media }: MediaViewerProps) {
  const { user, primaryWallet } = useDynamicContext();
  if (!media) return null;

  // Get user details from Dynamic auth
  const username = user?.verifiedCredentials?.[0]?.oauthUsername;
  const displayName = user?.verifiedCredentials?.[0]?.oauthDisplayName;
  const profileImage = user?.verifiedCredentials?.[0]?.oauthAccountPhotos?.[0];
  
  // Format wallet address for display
  const walletAddress = primaryWallet?.address || media.walletAddress;
  const truncatedAddress = walletAddress ? 
    `${walletAddress.slice(0, 4)}...${walletAddress.slice(-4)}` : 
    'Unknown';

  const ActionInfo = () => (
    <div className="border-t border-gray-100 pt-3 space-y-2">
      <div className="text-sm text-gray-600">Action</div>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-sm">
            {media.type === 'video' ? 'Video Recorded' : 'Photo Captured'}
          </div>
          <a
            href={`${media.provider === 'Filebase' ? 'https://console.filebase.com/object/' : 'https://gateway.pinata.cloud/ipfs/'}${media.id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-500 hover:text-blue-600 flex items-center gap-1"
          >
            View Source
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>

        <div className="flex flex-col gap-1.5 text-xs text-gray-500">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1">
              <Camera className="h-3 w-3" />
              Camera Source
            </span>
            <a
              href={media.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600"
            >
              View
            </a>
          </div>

          <div className="flex items-center justify-between">
            <span>Transaction</span>
            <a
              href={`https://solscan.io/tx/${media.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600"
            >
              View Tx
            </a>
          </div>

          <div className="flex items-center justify-between">
            <span>IPFS Storage ({media.provider})</span>
            <span className="text-gray-400 font-mono">
              {media.id.slice(0, 6)}...{media.id.slice(-6)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        {/* Background overlay */}
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/25" />
        </Transition.Child>

        <div className="fixed inset-0">
          {/* Mobile layout */}
          <div className="absolute inset-x-0 bottom-0 sm:hidden p-2">
            <div className="relative w-full bg-white rounded-2xl overflow-hidden shadow-xl">
              {/* Media section */}
              <div className="w-full">
                {media.type === 'video' ? (
                  <video
                    src={media.url}
                    className="w-full max-h-[70vh] object-contain"
                    controls
                    autoPlay
                  />
                ) : (
                  <img
                    src={media.url}
                    alt="Media preview"
                    className="w-full max-h-[70vh] object-contain"
                  />
                )}
              </div>

              {/* Close button - top right */}
              <div className="absolute right-2 top-2">
                <button
                  onClick={onClose}
                  className="rounded-full p-2 bg-white/80 hover:bg-white/90"
                >
                  <X className="h-5 w-5 text-gray-600" />
                </button>
              </div>

              {/* Info section */}
              <div className="px-4 py-4 space-y-4 bg-white">
                {/* Timestamp */}
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-gray-400" />
                  <span className="text-sm text-gray-600">
                    {new Date(media.timestamp).toLocaleString()}
                  </span>
                </div>

                {/* Profile info */}
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gray-100 overflow-hidden flex-shrink-0">
                    {profileImage && (
                      <img 
                        src={profileImage} 
                        alt={username || truncatedAddress}
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium">
                      {displayName || username || truncatedAddress}
                    </div>
                    <div className="text-xs text-gray-500">
                      Wallet
                    </div>
                  </div>
                </div>

                <ActionInfo />
              </div>
            </div>
          </div>

          {/* Desktop layout */}
          <div className="hidden sm:flex min-h-full items-center justify-center p-4">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded-xl bg-white shadow-xl">
                <div className="relative">
                  <button
                    onClick={onClose}
                    className="absolute right-4 top-4 z-10 rounded-full bg-white/80 p-2 hover:bg-white/90"
                  >
                    <X className="h-5 w-5 text-gray-600" />
                  </button>
                  
                  {media.type === 'video' ? (
                    <video
                      src={media.url}
                      className="w-full object-contain max-h-[80vh]"
                      controls
                      autoPlay
                    />
                  ) : (
                    <img
                      src={media.url}
                      alt="Media preview"
                      className="w-full object-contain max-h-[80vh]"
                    />
                  )}

                  {/* Info section - Desktop */}
                  <div className="px-6 py-4 space-y-4 bg-white">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-600">
                        {new Date(media.timestamp).toLocaleString()}
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-gray-100 overflow-hidden">
                        {profileImage && (
                          <img 
                            src={profileImage} 
                            alt={username || truncatedAddress}
                            className="w-full h-full object-cover"
                          />
                        )}
                      </div>
                      <div>
                        <div className="text-sm font-medium">
                          {displayName || username || truncatedAddress}
                        </div>
                        <div className="text-xs text-gray-500">
                          Wallet
                        </div>
                      </div>
                    </div>

                    <ActionInfo />
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}