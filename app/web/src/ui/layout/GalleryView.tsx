import MediaGallery from '../../media/Gallery';
import { useParams } from 'react-router-dom';
import { useEffect } from 'react';

// GalleryView.tsx
export function GalleryView() {
  const { cameraId } = useParams<{ cameraId?: string }>();
  
  useEffect(() => {
    console.log('GalleryView with cameraId:', cameraId);
    // Store cameraId in localStorage for persistence if available
    if (cameraId) {
      localStorage.setItem('directCameraId', cameraId);
    }
  }, [cameraId]);
  
  console.log('Rendering GalleryView with mode: archive');
  return (
    <div className="px-6 pt-12 pb-20">
      <MediaGallery mode="archive" cameraId={cameraId} />
    </div>
  );
}