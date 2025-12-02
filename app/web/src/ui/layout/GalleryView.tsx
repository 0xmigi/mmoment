import { useState, useEffect } from 'react';
import MediaGallery from '../../media/Gallery';
import { useCamera } from '../../camera/CameraProvider';

export function GalleryView() {
  const { selectedCamera } = useCamera();
  const [cameraFilter, setCameraFilter] = useState<string | null>(null);

  // Get connected camera ID from context
  const connectedCameraId = selectedCamera?.publicKey || localStorage.getItem('directCameraId');

  useEffect(() => {
    console.log('GalleryView - Connected camera:', connectedCameraId);
  }, [connectedCameraId]);

  // Determine which cameraId to pass to MediaGallery
  const activeCameraFilter = cameraFilter === 'connected' ? connectedCameraId : cameraFilter;

  return (
    <div className="pb-20">
      <div className="max-w-3xl mx-auto pt-8 px-4">
        {/* Title */}
        <h2 className="text-xl font-semibold mb-4">Gallery</h2>

        {/* Simple filter toggle - only show if camera is connected */}
        {connectedCameraId && (
          <div className="flex items-center gap-2 mb-6">
            <button
              onClick={() => setCameraFilter(null)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                ${cameraFilter === null
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              All
            </button>
            <button
              onClick={() => setCameraFilter('connected')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                ${cameraFilter === 'connected'
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              This Camera
            </button>
          </div>
        )}
      </div>

      <MediaGallery
        mode="archive"
        cameraId={activeCameraFilter || undefined}
        hideTitle={true}
      />
    </div>
  );
}