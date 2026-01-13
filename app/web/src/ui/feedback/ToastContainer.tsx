import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, CheckCircle2, XCircle } from 'lucide-react';

interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
}

interface ToastContainerProps {
  message: ToastMessage | null;  // Changed from messages array to single message
  onDismiss: () => void;
}

const Toast = ({ message, onDismiss }: { message: ToastMessage; onDismiss: () => void }) => {
  useEffect(() => {
    // Simple auto-dismiss: all messages disappear after 4 seconds except critical errors
    let timeout = 4000; // Default 4 seconds
    
    // Make deletion-related errors disappear faster (they're often transient)
    if (message.type === 'error' && (
      message.message.includes('delete') ||
      message.message.includes('unpin') ||
      message.message.includes('IPFS')
    )) {
      timeout = 2000; // Deletion errors disappear after 2 seconds
    }
    
    // Only keep critical errors until manually dismissed
    if (message.type === 'error' && (
      message.message.includes('Failed to create blockchain transaction') ||
      message.message.includes('No camera') ||
      message.message.includes('check in')
    )) {
      timeout = 0; // Critical errors stay until dismissed
    }
    
    if (timeout > 0) {
      const timer = setTimeout(onDismiss, timeout);
      return () => clearTimeout(timer);
    }
  }, [message.id]); // Only depend on message.id, not onDismiss

  const icons = {
    success: <CheckCircle2 className="w-4 h-4 text-green-500" />,
    error: <XCircle className="w-4 h-4 text-red-500" />,
    info: <AlertCircle className="w-4 h-4 text-primary" />
  };

  const bgColors = {
    success: 'bg-green-50',
    error: 'bg-red-50',
    info: 'bg-primary-light'
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className={`
        flex items-center justify-between gap-2 
        px-4 py-3
        w-full
        sm:w-auto 
        sm:min-w-[320px] 
        sm:max-w-md 
        rounded-lg 
        ${bgColors[message.type]}
      `}
    >
      <div className="flex items-center gap-2">
        {icons[message.type]}
        <span className="text-sm text-gray-600">{message.message}</span>
      </div>
      <button
        onClick={onDismiss}
        className="text-gray-400 hover:text-gray-600"
        aria-label="Dismiss notification"
      >
        ×
      </button>
    </motion.div>
  );
};


export const ToastContainer: React.FC<ToastContainerProps> = ({ message, onDismiss }) => {
  return (
    <div className="
      fixed 
      top-3
      left-4
      right-4 
      sm:left-1/2 
      sm:right-auto 
      sm:top-20 
      sm:-translate-x-1/2 
      z-50
    ">
      <AnimatePresence>
        {message && (
          <Toast message={message} onDismiss={onDismiss} />
        )}
      </AnimatePresence>
    </div>
  );
};