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
      // Only start the dismiss timer after a success message
      if (message.type === 'success') {
        const timer = setTimeout(() => {
          onDismiss();
        }, 6000);
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
            flex items-center gap-2 
            px-4 py-3
            w-full
            sm:w-auto 
            sm:min-w-[320px] 
            sm:max-w-md 
            rounded-lg 
            shadow-lg 
            ${bgColors[message.type]}
          `}
        >
          {icons[message.type]}
          <span className="text-sm text-gray-600">{message.message}</span>
        </motion.div>
      );
    };
    
  
export const ToastContainer: React.FC<ToastContainerProps> = ({ message, onDismiss }) => {
  return (
    <div className="
      fixed 
      top-24
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