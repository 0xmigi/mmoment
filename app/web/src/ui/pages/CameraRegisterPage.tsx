/**
 * Camera Registration Page - Single location for device setup
 * 
 * Integrates local device discovery with existing registration flow
 * Path: /app/register
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DeviceSetupWizard } from '../setup/DeviceSetupWizard';

type RegistrationMode = 'wizard' | 'manual';

export function CameraRegisterPage() {
  const [registrationMode, setRegistrationMode] = useState<RegistrationMode>('wizard');
  const [setupComplete, setSetupComplete] = useState(false);
  const [registeredCamera, setRegisteredCamera] = useState<any>(null);
  const navigate = useNavigate();

  const handleSetupComplete = (cameraData: any) => {
    setSetupComplete(true);
    setRegisteredCamera(cameraData);
    console.log('Camera setup completed:', cameraData);
    
    // Optionally navigate to camera view
    setTimeout(() => {
      // Calculate camera PDA from registration
      navigate(`/app/camera/${cameraData.device.deviceId}`);
    }, 3000);
  };

  const handleSetupError = (error: string) => {
    console.error('Setup error:', error);
    // Could show error modal or switch to manual mode
  };

  if (setupComplete && registeredCamera) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
            <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            Camera Setup Complete!
          </h1>
          
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <h3 className="font-medium text-gray-900 mb-2">Your New Camera</h3>
            <p className="text-sm text-gray-600 mb-1">
              <strong>Name:</strong> {registeredCamera.cameraName}
            </p>
            <p className="text-sm text-gray-600 mb-1">
              <strong>Device:</strong> {registeredCamera.device.model}
            </p>
            <p className="text-sm text-gray-600 mb-1">
              <strong>IP:</strong> {registeredCamera.device.ip}
            </p>
            {registeredCamera.devicePubkey && (
              <p className="text-sm text-gray-600">
                <strong>DePIN Status:</strong> 
                <span className="ml-1 bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">
                  Authenticated
                </span>
              </p>
            )}
          </div>

          <div className="space-y-3">
            <button
              onClick={() => navigate(`/app/camera/${registeredCamera.device.deviceId}`)}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
            >
              Go to Camera View
            </button>
            
            <button
              onClick={() => navigate('/app')}
              className="w-full bg-gray-200 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-300"
            >
              Back to Dashboard
            </button>

            <button
              onClick={() => {
                setSetupComplete(false);
                setRegisteredCamera(null);
              }}
              className="w-full text-sm text-gray-500 hover:text-gray-700"
            >
              Register Another Camera
            </button>
          </div>

          <div className="mt-6 p-3 bg-blue-50 rounded-lg">
            <p className="text-xs text-blue-800">
              Your camera is now configuring its public API endpoint. 
              It will be available for remote access shortly.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Register New Camera
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Set up your MMOMENT camera in minutes. We'll discover your device, 
            configure the network, and register it on the blockchain.
          </p>
        </div>

        {/* Registration Mode Selection */}
        <div className="mb-8">
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => setRegistrationMode('wizard')}
              className={`px-6 py-3 rounded-lg font-medium ${
                registrationMode === 'wizard'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 border border-gray-300'
              }`}
            >
              🎯 Quick Setup (Recommended)
            </button>
            <button
              onClick={() => setRegistrationMode('manual')}
              className={`px-6 py-3 rounded-lg font-medium ${
                registrationMode === 'manual'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 border border-gray-300'
              }`}
            >
              ⚙️ Manual Registration
            </button>
          </div>
        </div>

        {/* Registration Content */}
        <div className="flex justify-center">
          {registrationMode === 'wizard' ? (
            <div>
              <div className="mb-6 text-center">
                <h2 className="text-xl font-semibold text-gray-900 mb-2">
                  Automated Device Setup
                </h2>
                <p className="text-gray-600 mb-4">
                  Make sure your camera is powered on and you're connected to the same network.
                </p>
                
                {/* Feature highlights */}
                <div className="grid grid-cols-2 gap-4 max-w-md mx-auto mb-6">
                  <div className="bg-white rounded-lg p-3 border">
                    <div className="text-2xl mb-1">🔍</div>
                    <div className="text-sm font-medium">Auto Discovery</div>
                    <div className="text-xs text-gray-500">Finds your device automatically</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border">
                    <div className="text-2xl mb-1">📶</div>
                    <div className="text-sm font-medium">WiFi Setup</div>
                    <div className="text-xs text-gray-500">Configure network connection</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border">
                    <div className="text-2xl mb-1">🔗</div>
                    <div className="text-sm font-medium">Blockchain</div>
                    <div className="text-xs text-gray-500">Register on-chain securely</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border">
                    <div className="text-2xl mb-1">🛡️</div>
                    <div className="text-sm font-medium">DePIN Auth</div>
                    <div className="text-xs text-gray-500">Cryptographic device signing</div>
                  </div>
                </div>
              </div>

              <DeviceSetupWizard
                onComplete={handleSetupComplete}
                onError={handleSetupError}
              />
            </div>
          ) : (
            <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Manual Registration
              </h2>
              <p className="text-gray-600 mb-4">
                For advanced users or custom setups. You'll need to manually configure 
                your device and provide connection details.
              </p>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Device IP Address
                  </label>
                  <input
                    type="text"
                    className="w-full p-2 border border-gray-300 rounded-md"
                    placeholder="192.168.1.100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Camera Name
                  </label>
                  <input
                    type="text"
                    className="w-full p-2 border border-gray-300 rounded-md"
                    placeholder="Living Room Camera"
                  />
                </div>

                <button 
                  className="w-full bg-gray-400 text-white py-2 px-4 rounded-md cursor-not-allowed"
                  disabled
                >
                  Manual Setup (Coming Soon)
                </button>
              </div>

              <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
                <p className="text-sm text-yellow-800">
                  💡 <strong>Tip:</strong> The Quick Setup mode handles everything automatically 
                  and is much easier for first-time setup!
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Security Notice */}
        <div className="mt-12 max-w-2xl mx-auto">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800">
                  Local Network Security
                </h3>
                <div className="mt-1 text-sm text-blue-700">
                  <p>
                    • Device setup requires local network access for security<br/>
                    • Your camera will only work on your home network initially<br/>
                    • Geofencing prevents remote misuse and unauthorized access<br/>
                    • All communications are cryptographically signed
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}