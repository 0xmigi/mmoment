import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

type QuickActionType = 'PHOTO' | 'VIDEO' | 'CUSTOM';

interface QuickAction {
  type: QuickActionType;
  enabled: boolean;
}

interface QuickActionsContextType {
  quickActions: Record<QuickActionType, QuickAction>;
  enableQuickAction: (type: QuickActionType) => void;
  disableQuickAction: (type: QuickActionType) => void;
  isQuickActionEnabled: (type: QuickActionType) => boolean;
  saveSettings: () => void;
  hasUnsavedChanges: boolean;
}

const STORAGE_KEY = 'quickActionsSettings';

const QuickActionsContext = createContext<QuickActionsContextType | undefined>(undefined);

const QuickActionsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Load initial state from localStorage or use defaults
  const loadInitialState = () => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
    return {
      PHOTO: { type: 'PHOTO', enabled: false },
      VIDEO: { type: 'VIDEO', enabled: false },
      CUSTOM: { type: 'CUSTOM', enabled: false },
    };
  };

  const [quickActions, setQuickActions] = useState<Record<QuickActionType, QuickAction>>(loadInitialState);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const enableQuickAction = (type: QuickActionType) => {
    setQuickActions(prev => ({
      ...prev,
      [type]: { ...prev[type], enabled: true },
    }));
    setHasUnsavedChanges(true);
  };

  const disableQuickAction = (type: QuickActionType) => {
    setQuickActions(prev => ({
      ...prev,
      [type]: { ...prev[type], enabled: false },
    }));
    setHasUnsavedChanges(true);
  };

  const isQuickActionEnabled = (type: QuickActionType) => {
    return quickActions[type]?.enabled || false;
  };

  const saveSettings = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(quickActions));
    setHasUnsavedChanges(false);
  };

  // Load settings from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      setQuickActions(JSON.parse(saved));
    }
  }, []);

  return (
    <QuickActionsContext.Provider
      value={{
        quickActions,
        enableQuickAction,
        disableQuickAction,
        isQuickActionEnabled,
        saveSettings,
        hasUnsavedChanges
      }}
    >
      {children}
    </QuickActionsContext.Provider>
  );
};

const useQuickActions = () => {
  const context = useContext(QuickActionsContext);
  if (context === undefined) {
    throw new Error('useQuickActions must be used within a QuickActionsProvider');
  }
  return context;
};

export { QuickActionsProvider, useQuickActions }; 