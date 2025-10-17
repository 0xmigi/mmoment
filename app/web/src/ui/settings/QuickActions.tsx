import { useQuickActions } from '../context/QuickActionsProvider';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Switch } from '@headlessui/react';

export const QuickActions = () => {
  const { quickActions, enableQuickAction, disableQuickAction, saveSettings, hasUnsavedChanges } = useQuickActions();
  const { primaryWallet } = useDynamicContext();

  const handleToggle = async (type: 'PHOTO' | 'VIDEO' | 'CUSTOM', enabled: boolean) => {
    try {
      if (!primaryWallet?.address) {
        throw new Error('Please connect your wallet first');
      }

      if (enabled) {
        enableQuickAction(type);
      } else {
        disableQuickAction(type);
      }
    } catch (error) {
      console.error('Failed to toggle quick action:', error);
    }
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-lg font-semibold">Quick Actions</h2>
          <p className="text-sm text-gray-600">
            Enable one-click actions without requiring transaction signatures.
          </p>
        </div>
        {hasUnsavedChanges && (
          <button
            onClick={saveSettings}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Save Changes
          </button>
        )}
      </div>

      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Photo</h3>
            <p className="text-sm text-gray-500">Take photos without signing transactions</p>
          </div>
          <Switch
            checked={quickActions.PHOTO.enabled}
            onChange={(enabled) => handleToggle('PHOTO', enabled)}
            className={`${
              quickActions.PHOTO.enabled ? 'bg-blue-600' : 'bg-gray-200'
            } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
          >
            <span
              className={`${
                quickActions.PHOTO.enabled ? 'translate-x-6' : 'translate-x-1'
              } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </Switch>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Video</h3>
            <p className="text-sm text-gray-500">Record videos without signing transactions</p>
          </div>
          <Switch
            checked={quickActions.VIDEO.enabled}
            onChange={(enabled) => handleToggle('VIDEO', enabled)}
            className={`${
              quickActions.VIDEO.enabled ? 'bg-blue-600' : 'bg-gray-200'
            } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
          >
            <span
              className={`${
                quickActions.VIDEO.enabled ? 'translate-x-6' : 'translate-x-1'
              } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </Switch>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium">Custom</h3>
            <p className="text-sm text-gray-500">Custom actions without signing transactions</p>
          </div>
          <Switch
            checked={quickActions.CUSTOM.enabled}
            onChange={(enabled) => handleToggle('CUSTOM', enabled)}
            className={`${
              quickActions.CUSTOM.enabled ? 'bg-blue-600' : 'bg-gray-200'
            } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
          >
            <span
              className={`${
                quickActions.CUSTOM.enabled ? 'translate-x-6' : 'translate-x-1'
              } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
            />
          </Switch>
        </div>
      </div>

      <p className="mt-6 text-xs text-gray-500">
        Note: Quick actions are secured and can be disabled at any time.
      </p>
    </div>
  );
};