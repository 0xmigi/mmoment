import React, { RefObject } from 'react';
import { createPortal } from 'react-dom';

interface TooltipPortalProps {
  show: boolean;
  text: string;
  anchorRef: RefObject<HTMLButtonElement>;
}

const TooltipPortal: React.FC<TooltipPortalProps> = ({ show, text, anchorRef }) => {
  if (!show || !anchorRef.current) return null;

  const buttonRect = anchorRef.current.getBoundingClientRect();
  const tooltipStyle = {
    position: 'fixed' as const,
    top: `${buttonRect.top - 40}px`,
    left: `${buttonRect.left + buttonRect.width / 2}px`,
    transform: 'translateX(-50%)',
    zIndex: 1000,
  };

  return createPortal(
    <div style={tooltipStyle}>
      {text}
    </div>,
    document.body
  );
};

export default TooltipPortal;