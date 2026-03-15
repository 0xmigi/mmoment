import { useState, useEffect } from 'react';
import { Code, Settings, X, Terminal } from 'lucide-react';

const CV_DEV_MODE_KEY = 'mmoment_cv_dev_mode';

/**
 * Get CV Dev Mode status from localStorage
 */
export function getCVDevModeEnabled(): boolean {
  return localStorage.getItem(CV_DEV_MODE_KEY) === 'true';
}

/**
 * Set CV Dev Mode status in localStorage
 */
export function setCVDevModeEnabled(enabled: boolean): void {
  localStorage.setItem(CV_DEV_MODE_KEY, enabled ? 'true' : 'false');
}

/**
 * Component for managing developer settings
 * Includes CV App Dev Mode toggle for testing CV apps with pre-recorded video
 */
export function DeveloperSettings() {
  const [isOpen, setIsOpen] = useState(false);
  const [cvDevMode, setCvDevMode] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  // Load settings on mount
  useEffect(() => {
    setCvDevMode(getCVDevModeEnabled());
  }, []);

  // Handle CV Dev Mode toggle
  const handleCVDevModeToggle = () => {
    const newValue = !cvDevMode;
    setCvDevMode(newValue);
    setCVDevModeEnabled(newValue);
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 2000);

    // Dispatch custom event so other components can react
    window.dispatchEvent(new CustomEvent('cvDevModeChanged', { detail: { enabled: newValue } }));
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <button
          onClick={() => setIsOpen(true)}
          className="bg-gray-800 hover:bg-gray-700 text-white rounded-full p-3 shadow-lg transition-colors"
          title="Developer Settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center px-6 py-4 bg-gray-50 border-b">
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5 text-gray-600" />
            <h2 className="text-lg font-semibold text-gray-900">Developer Settings</h2>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* CV App Dev Mode Section */}
          <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-yellow-600" />
                  <h3 className="font-medium text-gray-900">CV App Dev Mode</h3>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {cvDevMode
                    ? 'Using pre-recorded video for CV app testing'
                    : 'Using live camera feed'}
                </p>
              </div>
              <button
                type="button"
                onClick={handleCVDevModeToggle}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  cvDevMode ? 'bg-yellow-500' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform shadow ${
                    cvDevMode ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {cvDevMode && (
              <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm">
                <p className="text-yellow-800 font-medium mb-1">Dev Mode Active</p>
                <p className="text-yellow-700 text-xs">
                  A video control panel will appear below the stream player.
                  Load test videos to iterate on CV apps without needing live camera access.
                </p>
              </div>
            )}
          </div>

          {/* Info Box */}
          <div className="p-3 bg-blue-50 text-blue-800 rounded border border-blue-200">
            <p className="text-sm">
              <strong>Note:</strong> CV Dev Mode requires the Jetson to be running with{' '}
              <code className="bg-blue-100 px-1 rounded text-xs">CV_DEV_MODE=true</code>
              {' '}environment variable.
            </p>
          </div>

          {/* Saved indicator */}
          {isSaved && (
            <div className="flex items-center justify-center text-green-600 text-sm">
              <span>Settings saved</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
