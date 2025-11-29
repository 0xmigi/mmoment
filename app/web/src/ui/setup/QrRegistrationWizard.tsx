/**
 * QR Registration Wizard - New approach using QR codes shown to Jetson camera
 * 
 * Flow:
 * 1. User enters WiFi credentials
 * 2. Generate QR code with WiFi + claim token
 * 3. Show QR to Jetson camera
 * 4. Poll for device claim completion
 * 5. Complete on-chain registration
 */

import { useState, useEffect, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { PublicKey, SystemProgram, Connection } from '@solana/web3.js';
import { AnchorProvider, Program, Idl } from '@coral-xyz/anchor';
import { IDL } from '../../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../../anchor/setup';
import QRCode from 'qrcode';

type QrStep = 'wifi' | 'qr' | 'scanning' | 'register' | 'complete';

interface QrRegistrationProps {
  onComplete?: (cameraData: any) => void;
  onError?: (error: string) => void;
  backendUrl?: string;
}

interface ClaimStatus {
  status: 'pending' | 'claimed' | 'expired' | 'not_found';
  created?: number;
  expires?: number;
  devicePubkey?: string;
  deviceModel?: string;
}

export function QrRegistrationWizard({ 
  onComplete, 
  onError,
  backendUrl = 'https://mmoment-backend-production.up.railway.app'
}: QrRegistrationProps): JSX.Element {
  // Wizard state
  const [currentStep, setCurrentStep] = useState<QrStep>('wifi');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressMessage, setProgressMessage] = useState<string>('');

  // Form state
  const [wifiCredentials, setWifiCredentials] = useState({
    ssid: '',
    password: ''
  });
  const [cameraName, setCameraName] = useState('');

  // QR code state
  const [qrCodeData, setQrCodeData] = useState<string>('');
  const [claimToken, setClaimToken] = useState<string>('');
  const [claimExpiry, setClaimExpiry] = useState<number>(0);

  // Device state
  const [claimedDevice, setClaimedDevice] = useState<any>(null);

  // Polling
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Blockchain
  const { primaryWallet } = useDynamicContext();
  const [program, setProgram] = useState<Program<Idl> | null>(null);
  const [programLoading, setProgramLoading] = useState(true);
  const [connection] = useState(new Connection('https://api.devnet.solana.com'));
  
  // Get wallet address from Dynamic wallet (same pattern as all other pages)
  const walletAddress = primaryWallet?.address;

  // Helper function to check if wallet is Solana wallet
  const isSolanaWallet = (wallet: any): boolean => {
    return wallet && typeof wallet.getSigner === 'function';
  };

  // Initialize program when wallet is connected (copied from debug page)
  useEffect(() => {
    if (!primaryWallet?.address || !connection) {
      setProgramLoading(false);
      return;
    }

    try {
      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: any) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            const signer = await (primaryWallet as any).getSigner();
            return await signer.signTransaction(tx);
          },
          signAllTransactions: async (txs: any[]) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            const signer = await (primaryWallet as any).getSigner();
            return await signer.signAllTransactions(txs);
          },
        },
        { commitment: 'confirmed' }
      );

      // Create the program with the general Idl type
      const prog = new Program(IDL as Idl, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      setProgram(prog);
      console.log('Program initialized with ID:', prog.programId.toString());
    } catch (err) {
      console.error('Failed to initialize program:', err);
    } finally {
      setProgramLoading(false);
    }
  }, [primaryWallet?.address, connection]);

  /**
   * Step 1: Collect WiFi credentials and create claim token
   */
  const handleWifiSubmit = async () => {
    if (!wifiCredentials.ssid.trim() || !wifiCredentials.password.trim()) {
      setError('Please enter both WiFi network name and password');
      return;
    }

    if (!walletAddress) {
      setError('Please connect your wallet first');
      return;
    }

    setLoading(true);
    setError(null);
    setProgressMessage('Creating device claim token...');

    try {
      console.log(`Attempting to create claim token at: ${backendUrl}/api/claim/create`);
      
      // Create claim token on backend
      const response = await fetch(`${backendUrl}/api/claim/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userWallet: walletAddress
        })
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error (${response.status}): ${errorText}`);
      }

      const { claimToken, expiresAt, claimEndpoint } = await response.json();
      
      // Generate QR code data
      const qrData = {
        wifi_ssid: wifiCredentials.ssid,
        wifi_password: wifiCredentials.password,
        claim_endpoint: claimEndpoint,
        user_wallet: walletAddress,
        expires: expiresAt
      };

      // Generate QR code image
      const qrString = await QRCode.toDataURL(JSON.stringify(qrData), {
        width: 400,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#FFFFFF'
        }
      });

      setClaimToken(claimToken);
      setClaimExpiry(expiresAt);
      setQrCodeData(qrString);
      setCurrentStep('qr');
      setProgressMessage('QR code generated - ready for device scanning');

    } catch (err) {
      console.error('Error creating claim token:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to create claim token';
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Step 2: Start polling for device claim
   */
  const startDevicePolling = () => {
    setCurrentStep('scanning');
    setProgressMessage('Show the QR code to your camera. Waiting for device to scan...');
    
    // Start polling every 2 seconds
    pollIntervalRef.current = setInterval(async () => {
      try {
        console.log(`Polling claim status: ${backendUrl}/api/claim/${claimToken}/status`);
        const response = await fetch(`${backendUrl}/api/claim/${claimToken}/status`);
        if (!response.ok) {
          console.log('Polling response not ok:', response.status);
          return;
        }
        
        const status: ClaimStatus = await response.json();
        console.log('Claim status response:', status);
        
        if (status.status === 'claimed' && status.devicePubkey) {
          // Device has been claimed!
          clearInterval(pollIntervalRef.current!);
          
          setClaimedDevice({
            devicePubkey: status.devicePubkey,
            deviceModel: status.deviceModel || 'MMOMENT Camera'
          });
          
          setCameraName(`Camera-${status.devicePubkey.slice(-6)}`);
          setCurrentStep('register');
          setProgressMessage('Device claimed successfully! Ready for blockchain registration.');
          
        } else if (status.status === 'expired') {
          clearInterval(pollIntervalRef.current!);
          setError('QR code has expired. Please start over.');
        }
      } catch (err) {
        // Silent fail - keep polling
        console.warn('Polling error:', err);
      }
    }, 2000);

    // Set timeout for overall process
    setTimeout(() => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        setError('Device scanning timed out. Please try again.');
      }
    }, 10 * 60 * 1000); // 10 minutes
  };

  /**
   * Step 3: Register device on blockchain
   */
  const handleBlockchainRegistration = async () => {
    console.log('Blockchain registration debug:', {
      program: !!program,
      programLoading,
      claimedDevice,
      cameraName: cameraName.trim(),
      walletAddress
    });
    
    if (programLoading) {
      setError('Blockchain program is still loading. Please wait and try again.');
      return;
    }
    
    if (!program || !claimedDevice || !cameraName.trim()) {
      setError(`Missing required information for registration. Program: ${!!program}, Device: ${!!claimedDevice}, Name: ${!!cameraName.trim()}`);
      return;
    }

    setLoading(true);
    setProgressMessage('Registering camera on Solana blockchain...');

    try {
      // Get wallet public key
      const ownerPublicKey = new PublicKey(walletAddress!);

      // Find the camera PDA - EXACTLY as in the working debug page
      const [cameraPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('camera'),
          Buffer.from(cameraName.trim()),
          ownerPublicKey.toBuffer()
        ],
        program.programId
      );

      // Find the registry PDA
      const [registryPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('camera-registry')],
        program.programId
      );

      const registerCameraArgs = {
        name: cameraName.trim(),
        model: claimedDevice.deviceModel,
        location: null,
        description: `Camera registered via QR code setup`,
        features: {
          faceRecognition: true,
          gestureControl: true,
          videoRecording: true,
          liveStreaming: true,
          messaging: false
        },
        devicePubkey: new PublicKey(claimedDevice.devicePubkey)
      };

      console.log('Camera registration args:', registerCameraArgs);
      console.log('Camera PDA:', cameraPda.toString());
      console.log('Registry PDA:', registryPda.toString());
      console.log('Owner:', ownerPublicKey.toString());

      // IMPORTANT: Exactly match the working pattern
      const tx = await program.methods
        .registerCamera(registerCameraArgs)
        .accounts({
          owner: ownerPublicKey,
          cameraRegistry: registryPda,
          camera: cameraPda,
          systemProgram: SystemProgram.programId,
        })
        .rpc();

      setProgressMessage('Waiting for blockchain confirmation...');
      await program.provider.connection.confirmTransaction(tx);

      // Use the camera PDA from the transaction/registration
      
      // Notify backend about PDA assignment so device can configure tunnel
      try {
        setProgressMessage('Configuring device tunnel...');
        const pdaAssignResponse = await fetch(`${backendUrl}/api/claim/${claimToken}/assign-pda`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            camera_pda: cameraPda.toString(),
            transaction_id: tx
          })
        });

        if (pdaAssignResponse.ok) {
          const pdaAssignResult = await pdaAssignResponse.json();
          console.log('PDA assigned to device:', pdaAssignResult);
          setProgressMessage(`Camera registered and tunnel configured: ${pdaAssignResult.subdomain}`);
        } else {
          console.warn('Failed to assign PDA to device:', await pdaAssignResponse.text());
          setProgressMessage('Camera registered (tunnel configuration may be pending)');
        }
      } catch (pdaError) {
        console.error('Error assigning PDA to device:', pdaError);
        setProgressMessage('Camera registered (tunnel configuration may be pending)');
      }

      setCurrentStep('complete');

      // Notify parent with refresh instruction
      onComplete?.({
        device: claimedDevice,
        cameraName,
        transactionId: tx,
        devicePubkey: claimedDevice.devicePubkey,
        cameraPda: cameraPda.toString(), // Include the PDA for reference
        shouldRefreshCameras: true // Signal that camera lists need refresh
      });

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Blockchain registration failed';
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const renderStepContent = () => {
    switch (currentStep) {
      case 'wifi':
        return (
          <div>
            <h3 className="text-lg font-semibold mb-4">WiFi Configuration</h3>
            <p className="text-gray-600 mb-6">
              Enter your WiFi network details. This will be sent to your camera via QR code.
            </p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  WiFi Network Name (SSID)
                </label>
                <input
                  type="text"
                  value={wifiCredentials.ssid}
                  onChange={(e) => setWifiCredentials(prev => ({ ...prev, ssid: e.target.value }))}
                  className="w-full p-3 border border-gray-300 rounded-md"
                  placeholder="Your WiFi network name"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  WiFi Password
                </label>
                <input
                  type="password"
                  value={wifiCredentials.password}
                  onChange={(e) => setWifiCredentials(prev => ({ ...prev, password: e.target.value }))}
                  className="w-full p-3 border border-gray-300 rounded-md"
                  placeholder="Your WiFi password"
                />
              </div>

              <button
                onClick={handleWifiSubmit}
                disabled={loading || !wifiCredentials.ssid.trim() || !wifiCredentials.password.trim()}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300"
              >
                {loading ? 'Generating QR Code...' : 'Generate Setup QR Code'}
              </button>
            </div>
          </div>
        );

      case 'qr':
        return (
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-4">Show QR Code to Camera</h3>
            <p className="text-gray-600 mb-6">
              Point your MMOMENT camera at this QR code. The camera will automatically:
              <br />• Connect to your WiFi network
              <br />• Register itself to your account
            </p>
            
            <div className="bg-white p-4 rounded-lg border-2 border-gray-200 inline-block mb-6">
              <img src={qrCodeData} alt="Device Setup QR Code" className="mx-auto" />
            </div>
            
            <div className="space-y-3">
              <button
                onClick={startDevicePolling}
                className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700"
              >
                📱 Ready - Camera Can Scan Now
              </button>
              
              <button
                onClick={() => setCurrentStep('wifi')}
                className="w-full bg-gray-200 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-300"
              >
                ← Back to WiFi Settings
              </button>
            </div>
            
            <div className="mt-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-800">
              QR code expires in {Math.ceil((claimExpiry - Date.now()) / 60000)} minutes
            </div>
          </div>
        );

      case 'scanning':
        return (
          <div className="text-center py-8">
            <div className="animate-pulse">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-semibold mb-2">Waiting for Camera</h3>
            <p className="text-gray-600 mb-4">
              Your camera should scan the QR code and connect to WiFi automatically.
              This usually takes 30-60 seconds.
            </p>
            <div className="text-sm text-gray-500">
              💡 Make sure your camera has a clear view of the QR code
            </div>
          </div>
        );

      case 'register':
        return (
          <div>
            <h3 className="text-lg font-semibold mb-4">Complete Registration</h3>
            <p className="text-gray-600 mb-4">
              Your camera is connected! Give it a name and complete the blockchain registration.
            </p>

            <div className="space-y-4">
              <div className="bg-green-50 p-3 rounded-md">
                <h4 className="font-medium text-sm text-green-800 mb-2">Device Connected ✅</h4>
                <p className="text-sm text-green-700">Model: {claimedDevice?.deviceModel}</p>
                <p className="text-sm text-green-700">Device Key: {claimedDevice?.devicePubkey?.slice(0, 8)}...</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Camera Name
                </label>
                <input
                  type="text"
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-md"
                  placeholder="e.g., Living Room Camera"
                />
              </div>

              <button
                onClick={handleBlockchainRegistration}
                disabled={loading || !cameraName.trim()}
                className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-300"
              >
                {loading ? 'Registering on Blockchain...' : 'Complete Registration'}
              </button>
            </div>
          </div>
        );

      case 'complete':
        return (
          <div className="text-center py-8">
            <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
              <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-green-800 mb-4">Success!</h3>
            <p className="text-gray-600 mb-4">
              Your camera has been registered and is ready to use. The device is now connected to 
              your WiFi network and authenticated on the blockchain.
            </p>
            <p className="text-sm text-gray-500">
              You can now access your camera through the MMOMENT app.
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
          <span className={currentStep === 'wifi' ? 'text-blue-600 font-medium' : ''}>WiFi</span>
          <span className={currentStep === 'qr' ? 'text-blue-600 font-medium' : ''}>QR Code</span>
          <span className={currentStep === 'scanning' ? 'text-blue-600 font-medium' : ''}>Scan</span>
          <span className={currentStep === 'register' ? 'text-blue-600 font-medium' : ''}>Register</span>
          <span className={currentStep === 'complete' ? 'text-green-600 font-medium' : ''}>Done</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ 
              width: `${
                currentStep === 'wifi' ? '20%' :
                currentStep === 'qr' ? '40%' :
                currentStep === 'scanning' ? '60%' :
                currentStep === 'register' ? '80%' :
                currentStep === 'complete' ? '100%' : '0%'
              }` 
            }}
          />
        </div>
      </div>

      {/* Step content */}
      {renderStepContent()}

      {/* Progress message */}
      {progressMessage && (
        <div className="mt-4 p-3 bg-blue-50 rounded-md">
          <p className="text-sm text-blue-800">{progressMessage}</p>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 rounded-md">
          <p className="text-sm text-red-800">{error}</p>
          <button
            onClick={() => {
              setError(null);
              if (currentStep === 'wifi') {
                // Reset to WiFi step
                setCurrentStep('wifi');
              }
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