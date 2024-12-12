// Create a new TooltipPortal.tsx component
import { createPortal } from 'react-dom';

interface TooltipPortalProps {
  show: boolean;
  text: string;
  anchorRef: React.RefObject<HTMLElement>;
}

const TooltipPortal = ({ show, text, anchorRef }: TooltipPortalProps) => {
  if (!show || !anchorRef.current) return null;

  const rect = anchorRef.current.getBoundingClientRect();

  return createPortal(
    <div 
      className="fixed px-3 py-2 bg-black/75 text-white text-sm rounded whitespace-nowrap transition-opacity"
      style={{
        top: rect.top + rect.height / 2,
        right: window.innerWidth - rect.left + 12,
        transform: 'translateY(-50%)',
        zIndex: 9999,
      }}
    >
      {text}
    </div>,
    document.body
  );
};

export default TooltipPortal;