import { useState, useEffect } from 'react';
import { Camera, Video } from 'lucide-react';
import MediaGallery from '../../media/Gallery';
import { useCamera } from '../../camera/CameraProvider';

type MediaTypeFilter = 'all' | 'photos' | 'videos';

export function GalleryView() {
  const { selectedCamera } = useCamera();
  const [cameraFilter, setCameraFilter] = useState<string | null>(null);
  const [mediaTypeFilter, setMediaTypeFilter] = useState<MediaTypeFilter>('all');

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

        {/* Filters */}
        <div className="flex items-center gap-2 mb-6">
          {/* Camera filter - only show if camera is connected */}
          {connectedCameraId && (
            <button
              onClick={() => setCameraFilter(cameraFilter === 'connected' ? null : 'connected')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap
                ${cameraFilter === 'connected'
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              This Camera
            </button>
          )}

          {/* Media type filters */}
          <button
            onClick={() => setMediaTypeFilter(mediaTypeFilter === 'photos' ? 'all' : 'photos')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center gap-1
              ${mediaTypeFilter === 'photos'
                ? 'bg-gray-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            <Camera className="w-3 h-3" />
            Photos
          </button>

          <button
            onClick={() => setMediaTypeFilter(mediaTypeFilter === 'videos' ? 'all' : 'videos')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center gap-1
              ${mediaTypeFilter === 'videos'
                ? 'bg-gray-700 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
          >
            <Video className="w-3 h-3" />
            Videos
          </button>
        </div>
      </div>

      {/* MediaGallery with padding wrapper like in CameraView */}
      <div className="px-4">
        <MediaGallery
          mode="archive"
          cameraId={activeCameraFilter || undefined}
          hideTitle={true}
          mediaType={mediaTypeFilter}
        />
      </div>
    </div>
  );
}