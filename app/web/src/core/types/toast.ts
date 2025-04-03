// src/types/toast.ts
export interface ToastMessage {
    id: string;
    type: 'success' | 'error' | 'info';
    message: string;
  }