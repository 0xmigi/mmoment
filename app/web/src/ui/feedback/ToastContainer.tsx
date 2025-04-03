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
    // Set different timeouts based on message type and content
    let timeout = 0;
    
    if (message.type === 'success') {
      timeout = 5000; // Success messages disappear after 5 seconds
    } else if (message.type === 'info') {
      // For streaming-related info messages, we want them to disappear quickly
      if (message.message.includes('Starting stream') || 
          message.message.includes('stream transaction')) {
        timeout = 3000; // Stream-related info messages disappear after 3 seconds
      } else {
        timeout = 4000; // Regular info messages after 4 seconds
      }
    }
    
    // Only set a timeout if we have one (errors stay until dismissed)
    if (timeout > 0) {
      const timer = setTimeout(() => {
        onDismiss();
      }, timeout);
      return () => clearTimeout(timer);
    }
  }, [message, onDismiss]);

  const icons = {
    success: <CheckCircle2 className="w-4 h-4 text-green-500" />,
    error: <XCircle className="w-4 h-4 text-red-500" />,
    info: <AlertCircle className="w-4 h-4 text-blue-500" />
  };

  const bgColors = {
    success: 'bg-green-50',
    error: 'bg-red-50',
    info: 'bg-blue-50'
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