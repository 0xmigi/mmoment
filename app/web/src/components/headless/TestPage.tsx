import { HeadlessAuthButton } from './auth/AuthButton';
import { QuickActions } from './settings/QuickActions';
import { QuickActionsProvider } from './context/QuickActionsProvider';

export const TestPage = () => {
  return (
    <QuickActionsProvider>
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-end mb-8">
            <HeadlessAuthButton />
          </div>
          
          <div className="bg-white rounded-lg shadow">
            <QuickActions />
          </div>
        </div>
      </div>
    </QuickActionsProvider>
  );
}; 