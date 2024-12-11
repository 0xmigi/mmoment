import { useEffect } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface MediaViewerProps {
  isOpen: boolean;
  onClose: () => void;
  media: {
    url: string;
    type: 'image' | 'video';
    timestamp: string;
  } | null;
}

export default function MediaViewer({ isOpen, onClose, media }: MediaViewerProps) {
  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }
    
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!media) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90"  // increased z-index to 100
          onClick={onClose}
        >
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 text-white rounded-full hover:bg-white/10"
          >
            <X className="w-6 h-6" />
          </button>

          {/* Media container */}
          <div 
            className="w-full h-full p-4 flex flex-col items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="relative max-w-4xl w-full h-full flex items-center justify-center">
              {media.type === 'video' ? (
                // biome-ignore lint/a11y/useMediaCaption: <explanation>
                <video
                  src={media.url}
                  className="max-h-full max-w-full object-contain"
                  controls
                  autoPlay
                />
              ) : (
                <img
                  src={media.url}
                  alt={`Captured at ${media.timestamp}`}
                  className="max-h-full max-w-full object-contain"
                  loading="lazy"
                />
              )}
              
              {/* Timestamp */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                <p className="text-sm text-white/70 bg-black/50 px-3 py-1 rounded-full">
                  {media.timestamp}
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}