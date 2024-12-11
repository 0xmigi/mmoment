import MediaGallery from '../ImageGallery';

// GalleryView.tsx
export function GalleryView() {
  console.log('Rendering GalleryView with mode: archive');
  return (
    <div className="h-full overflow-y-auto px-6 pt-12 pb-20">
      <MediaGallery mode="archive" />
    </div>
  );
}