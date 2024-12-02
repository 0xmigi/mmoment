import MediaGallery from '../ImageGallery';

// GalleryView.tsx
export function GalleryView() {
  console.log('Rendering GalleryView with mode: archive');
  return (
    <div className="flex flex-col items-center w-full max-w-md mx-auto">
      <MediaGallery mode="archive" />
    </div>
  );
}