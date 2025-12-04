import { useState } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useCamera } from '../../camera/CameraProvider';
import { TransactionModal } from '../../auth/components/TransactionModal';
import { Switch } from '@headlessui/react';

export const Settings = () => {
  useDynamicContext();
  const { cameraKeypair, quickActions, updateQuickAction } = useCamera();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentAction, setCurrentAction] = useState<{
    type: 'photo' | 'video' | 'stream';
    cameraAccount: string;
  } | null>(null);

  const handleQuickActionToggle = (type: 'photo' | 'video' | 'stream') => {
    if (!quickActions[type]) {
      setCurrentAction({
        type,
        cameraAccount: cameraKeypair.publicKey.toString()
      });
      setIsModalOpen(true);
    } else {
      updateQuickAction(type, false);
    }
  };

  const handleSuccess = () => {
    if (currentAction) {
      updateQuickAction(currentAction.type, true);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow">
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-4">Settings</h2>
          <p className="text-sm text-gray-600 mb-6">
            Enable one-click actions without requiring transaction signatures each time.
            This requires a one-time approval transaction to enable each action.
          </p>

          <div className="space-y-6">
            {/* Photo Quick Action */}
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium">Quick Photo</h3>
                <p className="text-sm text-gray-500">Take photos without signing transactions</p>
              </div>
              <Switch
                checked={quickActions.photo}
                onChange={() => handleQuickActionToggle('photo')}
                className={`${
                  quickActions.photo ? 'bg-primary' : 'bg-gray-200'
                } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2`}
              >
                <span
                  className={`${
                    quickActions.photo ? 'translate-x-6' : 'translate-x-1'
                  } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                />
              </Switch>
            </div>

            {/* Video Quick Action */}
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium">Quick Video</h3>
                <p className="text-sm text-gray-500">Start recording without signing transactions</p>
              </div>
              <Switch
                checked={quickActions.video}
                onChange={() => handleQuickActionToggle('video')}
                className={`${
                  quickActions.video ? 'bg-primary' : 'bg-gray-200'
                } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2`}
              >
                <span
                  className={`${
                    quickActions.video ? 'translate-x-6' : 'translate-x-1'
                  } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                />
              </Switch>
            </div>

            {/* Stream Quick Action */}
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium">Quick Stream</h3>
                <p className="text-sm text-gray-500">Start streaming without signing transactions</p>
              </div>
              <Switch
                checked={quickActions.stream}
                onChange={() => handleQuickActionToggle('stream')}
                className={`${
                  quickActions.stream ? 'bg-primary' : 'bg-gray-200'
                } relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2`}
              >
                <span
                  className={`${
                    quickActions.stream ? 'translate-x-6' : 'translate-x-1'
                  } inline-block h-4 w-4 transform rounded-full bg-white transition-transform`}
                />
              </Switch>
            </div>
          </div>

          <p className="mt-6 text-xs text-gray-500">
            Note: Quick actions are secured by your wallet and can be disabled at any time.
            Each action requires a one-time approval transaction.
          </p>
        </div>
      </div>

      <TransactionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        transactionData={currentAction || undefined}
        onSuccess={handleSuccess}
      />
    </div>
  );
}; 