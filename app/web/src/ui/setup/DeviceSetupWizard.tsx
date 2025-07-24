/**
 * Device Setup Wizard for MMOMENT Cameras
 * 
 * Handles complete flow from local discovery to on-chain registration
 * Designed for /app/register page - single location setup experience
 */

import { useState, useEffect } from 'react';
import { PublicKey } from '@solana/web3.js';
import { useAnchorProgram } from '../../blockchain/anchor-client';
import { localDeviceDiscovery, LocalDevice, WiFiNetwork } from '../../camera/local-device-discovery';
import { hasValidDeviceSignature, logDeviceSignature } from '../../camera/device-signature-utils';

type SetupStep = 'scan' | 'select' | 'wifi' | 'register' | 'complete';

interface SetupWizardProps {
  onComplete?: (cameraData: any) => void;
  onError?: (error: string) => void;
}

export function DeviceSetupWizard({ onComplete, onError }: SetupWizardProps) {
  // Wizard state
  const [currentStep, setCurrentStep] = useState<SetupStep>('scan');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Discovery state
  const [discoveredDevices, setDiscoveredDevices] = useState<LocalDevice[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<LocalDevice | null>(null);
  const [availableNetworks, setAvailableNetworks] = useState<WiFiNetwork[]>([]);

  // Setup state
  const [wifiCredentials, setWifiCredentials] = useState({ ssid: '', password: '' });
  const [cameraName, setCameraName] = useState('');
  const [setupProgress, setSetupProgress] = useState<string>('');

  // Blockchain
  const { program } = useAnchorProgram();

  /**
   * Step 1: Scan for local devices
   */
  const handleDeviceScan = async () => {
    setLoading(true);
    setError(null);
    setSetupProgress('Scanning local network for MMOMENT devices...');

    try {
      // Verify user is on local network (security check)
      const localPresenceVerified = await localDeviceDiscovery.verifyLocalPresence();
      if (!localPresenceVerified) {
        throw new Error('Device setup must be performed on the same local network');
      }

      // Scan for devices
      const devices = await localDeviceDiscovery.scanForDevices();
      
      if (devices.length === 0) {
        setError('No MMOMENT devices found on local network. Please ensure device is powered on and connected.');
        return;
      }

      // Filter to only local devices for security
      const localDevices = devices.filter(device => localDeviceDiscovery.isLocalDevice(device));
      
      setDiscoveredDevices(localDevices);
      setCurrentStep('select');
      setSetupProgress(`Found ${localDevices.length} device(s) on local network`);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Device discovery failed';
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Step 2: Select device and check setup status
   */
  const handleDeviceSelect = async (device: LocalDevice) => {
    setSelectedDevice(device);
    setLoading(true);
    setSetupProgress(`Connecting to device ${device.deviceId}...`);

    try {
      if (device.setupRequired) {
        // Device needs WiFi configuration
        const networks = await localDeviceDiscovery.getAvailableNetworks(device);
        setAvailableNetworks(networks);
        setCurrentStep('wifi');
        setSetupProgress('Device found - WiFi setup required');
      } else {
        // Device already configured, can register directly
        setCameraName(`Camera-${device.deviceId.slice(-6)}`); // Suggest name
        setCurrentStep('register');
        setSetupProgress('Device ready for registration');
      }
    } catch (err) {
      setError('Failed to connect to selected device');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Step 3: Configure WiFi
   */
  const handleWifiSetup = async () => {
    if (!selectedDevice || !wifiCredentials.ssid || !wifiCredentials.password) {
      setError('Please select a network and enter password');
      return;
    }

    setLoading(true);
    setSetupProgress('Configuring device WiFi...');

    try {
      const success = await localDeviceDiscovery.configureDeviceWiFi(
        selectedDevice,
        wifiCredentials.ssid,
        wifiCredentials.password
      );

      if (success) {
        // Wait for device to restart and connect to new network
        setSetupProgress('WiFi configured. Device reconnecting...');
        await new Promise(resolve => setTimeout(resolve, 10000)); // 10 second wait

        // Try to rediscover device on new network
        const devices = await localDeviceDiscovery.scanForDevices();
        const reconnectedDevice = devices.find(d => d.deviceId === selectedDevice.deviceId);

        if (reconnectedDevice) {
          setSelectedDevice(reconnectedDevice);
          setCameraName(`Camera-${reconnectedDevice.deviceId.slice(-6)}`);
          setCurrentStep('register');
          setSetupProgress('WiFi setup complete - ready for registration');
        } else {
          throw new Error('Device did not reconnect after WiFi setup');
        }
      } else {
        throw new Error('WiFi configuration failed');
      }

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'WiFi setup failed';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Step 4: Register device on-chain
   */
  const handleDeviceRegistration = async () => {
    if (!selectedDevice || !program || !cameraName.trim()) {
      setError('Missing required information for registration');
      return;
    }

    setLoading(true);
    setSetupProgress('Registering camera on Solana blockchain...');

    try {
      // Get fresh device info with signature
      const deviceInfoResponse = await fetch(`http://${selectedDevice.ip}:5002/api/device-info`);
      if (!deviceInfoResponse.ok) {
        throw new Error('Could not fetch device information');
      }

      const deviceInfo = await deviceInfoResponse.json();
      
      // Verify device signature
      if (!hasValidDeviceSignature(deviceInfo)) {
        throw new Error('Device signature verification failed - device may not be authentic');
      }

      logDeviceSignature(deviceInfo, 'registration');

      // Prepare registration arguments
      const registerArgs = {
        name: cameraName.trim(),
        model: deviceInfo.model || 'MMOMENT Camera',
        location: null,
        description: `Camera registered via local setup wizard`,
        features: {
          faceRecognition: true,
          gestureControl: true,
          videoRecording: true,
          liveStreaming: true,
          messaging: false
        },
        devicePubkey: deviceInfo.device_pubkey ? new PublicKey(deviceInfo.device_pubkey) : null
      };

      // Register using existing program
      const tx = await program.methods
        .registerCamera(registerArgs)
        .accounts({
          // This will use your existing PDA generation logic
        })
        .rpc();

      setSetupProgress('Registration transaction sent...');

      // Wait for confirmation
      await program.provider.connection.confirmTransaction(tx);

      setCurrentStep('complete');
      setSetupProgress('Camera registration complete!');

      // Notify parent component
      onComplete?.({
        device: selectedDevice,
        cameraName,
        transactionId: tx,
        devicePubkey: deviceInfo.device_pubkey
      });

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Registration failed';
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // Auto-start scan when component mounts
  useEffect(() => {
    if (currentStep === 'scan') {
      handleDeviceScan();
    }
  }, [currentStep]);

  const renderStepContent = () => {
    switch (currentStep) {
      case 'scan':
        return (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h3 className="text-lg font-semibold mb-2">Discovering Devices</h3>
            <p className="text-gray-600 mb-4">Scanning your local network for MMOMENT cameras...</p>
            <p className="text-sm text-gray-500">Make sure your camera is powered on and connected to your network</p>
          </div>
        );

      case 'select':
        return (
          <div>
            <h3 className="text-lg font-semibold mb-4">Select Your Camera</h3>
            <div className="space-y-3">
              {discoveredDevices.map((device) => (
                <div
                  key={device.ip}
                  className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                  onClick={() => handleDeviceSelect(device)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">{device.model}</h4>
                      <p className="text-sm text-gray-600">IP: {device.ip}</p>
                      <p className="text-sm text-gray-600">Device ID: {device.deviceId}</p>
                    </div>
                    <div className="text-right">
                      {device.setupRequired ? (
                        <span className="bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded">
                          Setup Required
                        </span>
                      ) : (
                        <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded">
                          Ready
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'wifi':
        return (
          <div>
            <h3 className="text-lg font-semibold mb-4">Configure WiFi</h3>
            <p className="text-gray-600 mb-4">
              Connect your camera to your WiFi network to enable internet access.
            </p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  WiFi Network
                </label>
                <select
                  value={wifiCredentials.ssid}
                  onChange={(e) => setWifiCredentials(prev => ({ ...prev, ssid: e.target.value }))}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  <option value="">Select a network...</option>
                  {availableNetworks.map((network) => (
                    <option key={network.ssid} value={network.ssid}>
                      {network.ssid} ({network.security})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  WiFi Password
                </label>
                <input
                  type="password"
                  value={wifiCredentials.password}
                  onChange={(e) => setWifiCredentials(prev => ({ ...prev, password: e.target.value }))}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  placeholder="Enter WiFi password"
                />
              </div>

              <button
                onClick={handleWifiSetup}
                disabled={loading || !wifiCredentials.ssid || !wifiCredentials.password}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300"
              >
                {loading ? 'Configuring WiFi...' : 'Configure WiFi'}
              </button>
            </div>
          </div>
        );

      case 'register':
        return (
          <div>
            <h3 className="text-lg font-semibold mb-4">Register Camera</h3>
            <p className="text-gray-600 mb-4">
              Almost done! Give your camera a name and register it on the blockchain.
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Camera Name
                </label>
                <input
                  type="text"
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  placeholder="e.g., Living Room Camera"
                />
              </div>

              {selectedDevice && (
                <div className="bg-gray-50 p-3 rounded-md">
                  <h4 className="font-medium text-sm text-gray-700 mb-2">Device Information</h4>
                  <p className="text-sm text-gray-600">Model: {selectedDevice.model}</p>
                  <p className="text-sm text-gray-600">IP: {selectedDevice.ip}</p>
                  {selectedDevice.devicePubkey && (
                    <p className="text-sm text-gray-600">
                      Device Key: {selectedDevice.devicePubkey.slice(0, 8)}...
                      <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">
                        DePIN Authenticated
                      </span>
                    </p>
                  )}
                </div>
              )}

              <button
                onClick={handleDeviceRegistration}
                disabled={loading || !cameraName.trim()}
                className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-300"
              >
                {loading ? 'Registering Camera...' : 'Register Camera'}
              </button>
            </div>
          </div>
        );

      case 'complete':
        return (
          <div className="text-center py-8">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
              <svg className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-green-800 mb-2">Setup Complete!</h3>
            <p className="text-gray-600 mb-4">
              Your camera has been successfully registered and is ready to use.
            </p>
            <p className="text-sm text-gray-500">
              The device will now configure its public API endpoint and be available for remote access.
            </p>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
      {/* Progress indicator */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm text-gray-500 mb-2">
          <span className={currentStep === 'scan' ? 'text-blue-600 font-medium' : ''}>Discover</span>
          <span className={currentStep === 'select' ? 'text-blue-600 font-medium' : ''}>Select</span>
          <span className={currentStep === 'wifi' ? 'text-blue-600 font-medium' : ''}>Configure</span>
          <span className={currentStep === 'register' ? 'text-blue-600 font-medium' : ''}>Register</span>
          <span className={currentStep === 'complete' ? 'text-green-600 font-medium' : ''}>Complete</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ 
              width: `${
                currentStep === 'scan' ? '20%' :
                currentStep === 'select' ? '40%' :
                currentStep === 'wifi' ? '60%' :
                currentStep === 'register' ? '80%' :
                currentStep === 'complete' ? '100%' : '0%'
              }` 
            }}
          />
        </div>
      </div>

      {/* Step content */}
      {renderStepContent()}

      {/* Progress status */}
      {setupProgress && (
        <div className="mt-4 p-3 bg-blue-50 rounded-md">
          <p className="text-sm text-blue-800">{setupProgress}</p>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 rounded-md">
          <p className="text-sm text-red-800">{error}</p>
          <button
            onClick={() => {
              setError(null);
              if (currentStep === 'scan') handleDeviceScan();
            }}
            className="mt-2 text-sm text-red-600 hover:text-red-500"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}