/* eslint-disable @typescript-eslint/no-explicit-any, react-hooks/exhaustive-deps */
import { useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, SystemProgram, Transaction, Connection } from '@solana/web3.js';
import { Program, AnchorProvider, Idl } from '@coral-xyz/anchor';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../../anchor/setup';
import { IDL } from '../../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { useNavigate } from 'react-router-dom';
import { useCamera } from '../../camera/CameraProvider';

// Define IDL type for stricter checking
// type CameraActivationIdl = typeof IDL;

// Define account types from IDL
// type CameraAccountData = IdlAccounts<CameraActivationIdl>['cameraAccount'];
// type FaceDataAccount = IdlAccounts<CameraActivationIdl>['faceData']; // Define FaceData type
// type UserSessionAccount = IdlAccounts<CameraActivationIdl>['userSession']; // Define UserSession type
// type CameraRegistryAccount = IdlAccounts<CameraActivationIdl>['cameraRegistry']; // Define Registry type

// Define a type for the program
// type CameraProgram = Program<Idl>; // Removed unused type

// Simple cache for account info to reduce RPC calls
const accountInfoCache = new Map<string, { data: unknown, timestamp: number }>(); // Use unknown instead of any
const CACHE_TTL = 30000; // 30 seconds

// Simplified helper to get account info with caching
async function getCachedAccountInfo(connection: Connection, pubkey: PublicKey) {
  const key = pubkey.toString();
  const now = Date.now();
  const cached = accountInfoCache.get(key);

  if (cached && now - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }

  try {
    const info = await connection.getAccountInfo(pubkey);
    accountInfoCache.set(key, { data: info, timestamp: now });
    return info;
  } catch (error) {
    console.error('Error fetching account info:', error);
    throw error;
  }
}

// Anchor instruction discriminators (first 8 bytes of sha256 hash of method name)

// Add these types for camera data
// interface ActivityType { // Removed unused type
//   photoCapture?: Record<string, never>; // Use Record<string, never> for empty objects
//   videoRecord?: Record<string, never>;
//   liveStream?: Record<string, never>;
// }

interface CameraData {
  publicKey: string;
  owner: string;
  isActive: boolean;
  name: string;
  model: string;
  location?: [number, number] | null;
  activeSessions: number;
  totalSessions: number;
  registrationDate: number;
  lastActivity: number;
  userCheckedIn?: boolean;
  devicePubkey?: string; // Device signing key for DePIN authentication
}

// Add a helper for transaction confirmation with retries and timeouts
const confirmTransactionWithRetry = async (connection: Connection, signature: string, maxRetries = 3, timeoutMs = 60000) => {
  console.log(`Confirming transaction ${signature} with ${maxRetries} retries and ${timeoutMs}ms timeout`);
  
  // Create a timeout promise
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => {
      reject(new Error(`Transaction confirmation timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  
  // Function to attempt confirmation with retries
  const attemptConfirmation = async (attemptsLeft: number): Promise<void> => {
    try {
      // Get the latest blockhash
      const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
      
      // Set up confirmation options with higher commitment
      const confirmationOptions = {
        blockhash,
        lastValidBlockHeight,
        signature,
        commitment: 'confirmed' as const
      };
      
      // Race the confirmation against the timeout
      await Promise.race([
        connection.confirmTransaction(confirmationOptions),
        timeoutPromise
      ]);
      
      console.log(`Transaction ${signature} confirmed successfully`);
      
      // Verify transaction success by checking status
      const status = await connection.getSignatureStatus(signature);
      if (status?.value?.err) {
        throw new Error(`Transaction failed with error: ${JSON.stringify(status.value.err)}`);
      }
      
      return;
    } catch (error) {
      if (attemptsLeft <= 1) {
        console.error(`All ${maxRetries} confirmation attempts failed for ${signature}`);
        throw error;
      }
      
      console.log(`Confirmation attempt failed, ${attemptsLeft - 1} retries left for ${signature}`);
      return attemptConfirmation(attemptsLeft - 1);
    }
  };
  
  return attemptConfirmation(maxRetries);
};

export function SolDevNetDebug() {
  const dynamicContext = useDynamicContext();
  const { primaryWallet } = dynamicContext;
  const { connection } = useConnection();
  const navigate = useNavigate();
  const { onCameraListRefresh, triggerCameraListRefresh } = useCamera();

  // State variables
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txResult, setTxResult] = useState<string | null>(null);
  const [program, setProgram] = useState<Program<Idl> | null>(null);

  // Form state
  const [cameraName, setCameraName] = useState('');
  const [cameraModel, setCameraModel] = useState('');

  // Status messages
  const [statusMessage, setStatusMessage] = useState('');
  const [statusType, setStatusType] = useState<'info' | 'success' | 'error'>('info');

  // Add state for registered cameras
  const [registeredCameras, setRegisteredCameras] = useState<CameraData[]>([]);
  const [loadingCameras, setLoadingCameras] = useState(false);
  
  // Add this to the state variables near the top of the component
  const [useFaceRecognition, setUseFaceRecognition] = useState<{[key: string]: boolean}>({});

  // Add state for user sessions
  const [userSessions, setUserSessions] = useState<{[key: string]: boolean}>({});
  
  // Add state for active users per camera analytics
  const [activeUsersPerCamera, setActiveUsersPerCamera] = useState<{[key: string]: number}>({});
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);

  // --- Face Enrollment State ---
  const [jetsonStreamUrl] = useState<string>(''); // TODO: Set this to your Jetson stream URL - Removed unused setter
  const [capturedEmbedding, setCapturedEmbedding] = useState<Uint8Array | null>(null);
  const [enrollmentLoading, setEnrollmentLoading] = useState<boolean>(false);
  const [enrollmentStatus, setEnrollmentStatus] = useState<string>('');
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null);
  // ----------------------------

  // Initialize program when wallet is connected
   
  useEffect(() => {
    if (!primaryWallet?.address || !connection) return;

    try {
      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          // Add proper signTransaction and signAllTransactions methods
          signTransaction: async (tx: Transaction) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            // Use Dynamic's getSigner method
            const signer = await primaryWallet.getSigner();
            return await signer.signTransaction(tx);
          },
          signAllTransactions: async (txs: Transaction[]) => {
            if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
            }
            // Use Dynamic's getSigner method
            const signer = await primaryWallet.getSigner();
            return await Promise.all(txs.map(tx => signer.signTransaction(tx)));
          },
         
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program with the general Idl type
      const prog = new Program(IDL as Idl, CAMERA_ACTIVATION_PROGRAM_ID, provider);
      setProgram(prog);
      console.log('Program initialized with ID:', prog.programId.toString());
    } catch (err) {
      console.error('Failed to initialize program:', err);
      setError('Failed to initialize program');
    }
  }, [primaryWallet?.address, connection]);

  // Check if registry is initialized
   
  useEffect(() => {
    if (!primaryWallet?.address || !connection || !program) return;

    const checkRegistry = async () => {
      try {
        setLoading(true);

        // Find the registry address
        const [registryAddress] = await PublicKey.findProgramAddress(
          [Buffer.from('camera-registry')],
          CAMERA_ACTIVATION_PROGRAM_ID
        );

        // Try to check if the registry account exists
        try {
          // Check if the registry account exists with caching
          const registryAccountInfo = await getCachedAccountInfo(connection, registryAddress) as { data: { length: number } } | null;
          const isInitialized = !!registryAccountInfo && registryAccountInfo.data && registryAccountInfo.data.length > 0;

          setInitialized(isInitialized);
          setStatusMessage(isInitialized ? 'Registry is initialized' : 'Registry needs initialization');
          setStatusType(isInitialized ? 'success' : 'info');
        } catch (err) {
          console.warn('Error checking registry account, assuming program is initialized:', err);
          // If we can't check the registry account, assume the program is initialized
          // since you mentioned it was upgraded
          setInitialized(true);
          setStatusMessage('Program is assumed to be initialized after upgrade');
          setStatusType('success');
        }

        // Fetch cameras right away if we have a connection
        fetchRegisteredCameras();
      } catch (err) {
        console.error('Error checking registry:', err);
        setError(err instanceof Error ? err.message : 'Failed to check registry');
        setStatusMessage('Failed to check registry: ' + (err instanceof Error ? err.message : 'Unknown error'));
        setStatusType('error');
      } finally {
        setLoading(false);
      }
    };

    checkRegistry();
  }, [primaryWallet?.address, connection, program]);

  // Initialize registry
  const initializeRegistry = async () => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      setStatusMessage('Initializing camera registry...');
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const authorityPublicKey = new PublicKey(primaryWallet.address);

      // Find the registry PDA
      const [registryPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      console.log('Registry PDA:', registryPda.toString());

      // Initialize the registry
      const tx = await program.methods
        .initialize()
        .accounts({
          authority: authorityPublicKey,
          cameraRegistry: registryPda,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
      
      console.log('Registry initialization sent with signature:', tx);
      
      // Get the connection
      const connection = await primaryWallet.getConnection();
      
      // Use improved confirmation handling
      try {
        await confirmTransactionWithRetry(connection, tx);
        console.log('Registry initialized successfully, signature:', tx);
        setTxResult(tx);
        setStatusMessage(`Registry initialized successfully! Transaction: ${tx}`);
        setStatusType('success');
        setInitialized(true);
      } catch (confirmError) {
        console.error('Transaction confirmation error:', confirmError);
        setStatusMessage(`Transaction sent but confirmation timed out. It may still succeed. Check the explorer: ${tx}`);
        setStatusType('info');
      }

    } catch (err) {
      console.error('Error initializing registry:', err);

      let errorMessage = 'Failed to initialize registry';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Register camera
  const registerCamera = async () => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    if (!cameraName || !cameraModel) {
      setStatusMessage('Please fill in all camera details');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      setStatusMessage('Registering camera...');
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const ownerPublicKey = new PublicKey(primaryWallet.address);

      // Find the camera PDA - EXACTLY as in the test scripts
      const [cameraPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('camera'),
          Buffer.from(cameraName.trim()),
          ownerPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      console.log('Camera PDA:', cameraPda.toString());

      // Find the registry PDA
      const [registryPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      console.log('Registry PDA:', registryPda.toString());

      // Get device public key from camera before registration
      setStatusMessage('Fetching device public key from camera...');
      let devicePubkey: PublicKey;
      
      // For scalable production: Get device info from PDA-based URL
      // In production, this PDA would come from QR code or user input
      let targetCameraPda: string;
      
      // Demo: Ask user for camera PDA or use default
      const userProvidedPda = prompt('Enter camera PDA (or press OK for demo device):');
      if (userProvidedPda && userProvidedPda.trim()) {
        targetCameraPda = userProvidedPda.trim();
      } else {
        // Demo fallback - in production this would be from QR/NFC
        targetCameraPda = 'HYq5rnk9r92eTLsEF7cZ2ZWcCLNp4KHE9qGFTEKxwt4L';
      }
      
      try {
        // Generate PDA-based URL for this specific device
        const deviceApiUrl = `https://${targetCameraPda.toLowerCase()}.mmoment.xyz`;
        console.log(`Fetching device info from: ${deviceApiUrl}/api/device-info`);
        
        const deviceInfoResponse = await fetch(`${deviceApiUrl}/api/device-info`);
        if (deviceInfoResponse.ok) {
          const deviceInfo = await deviceInfoResponse.json();
          if (deviceInfo.device_pubkey) {
            devicePubkey = new PublicKey(deviceInfo.device_pubkey);
            console.log(`✅ Fetched device pubkey from ${targetCameraPda}:`, devicePubkey.toString());
          } else {
            throw new Error('No device_pubkey in response');
          }
        } else {
          throw new Error(`HTTP error: ${deviceInfoResponse.status}`);
        }
      } catch (pdaError) {
        console.warn('PDA-based URL failed, trying legacy URL:', pdaError);
        
        // Fallback: Try legacy URL for existing demo device
        try {
          const LEGACY_JETSON_URL = 'https://jetson.mmoment.xyz';
          const deviceInfoResponse = await fetch(`${LEGACY_JETSON_URL}/api/device-info`);
          if (deviceInfoResponse.ok) {
            const deviceInfo = await deviceInfoResponse.json();
            if (deviceInfo.device_pubkey) {
              devicePubkey = new PublicKey(deviceInfo.device_pubkey);
              console.log('✅ Fetched device pubkey via legacy URL:', devicePubkey.toString());
            } else {
              throw new Error('No device_pubkey in legacy response');
            }
          } else {
            throw new Error(`Legacy HTTP error: ${deviceInfoResponse.status}`);
          }
        } catch (legacyError) {
          console.warn('All HTTP attempts failed, using demo device key:', legacyError);
          // Final fallback for demo
          devicePubkey = new PublicKey('BXqMyo3Uh6SiLr3xh9iEBCY9AgV1aUciymK37SpNgbNE');
          setStatusMessage('Using demo device key (device unavailable)');
        }
      }

      // Use `any` for args type for simplicity
      const registerCameraArgs: {
        name: string;
        model: string;
        location: null;
        description: string;
        features: {
          faceRecognition: boolean;
          gestureControl: boolean;
          videoRecording: boolean;
          liveStreaming: boolean;
          messaging: boolean;
        };
        devicePubkey: PublicKey | null;
      } = {
        name: cameraName.trim(),
        model: cameraModel.trim(),
        location: null,  // Explicitly null for no location
        description: "Camera registered via debug interface",
        features: {
          faceRecognition: true,
          gestureControl: true,
          videoRecording: true,
          liveStreaming: true,
          messaging: false
        },
        devicePubkey: devicePubkey // Will be Some(devicePubkey) in Rust
      };

      console.log('Camera registration args:', registerCameraArgs);

      // Log all the important data for debugging
      console.log('Camera PDA:', cameraPda.toString());
      console.log('Registry PDA:', registryPda.toString());
      console.log('Owner:', ownerPublicKey.toString());
      console.log('Program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());
        
      // IMPORTANT: Exactly match the working test script pattern
      const tx = await program.methods
        .registerCamera(registerCameraArgs)
        .accounts({
          owner: ownerPublicKey,
          cameraRegistry: registryPda,  // Changed from 'registry' to 'cameraRegistry' to match IDL
          camera: cameraPda,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
      
      console.log('Camera registration sent with signature:', tx);
      
      // Get the connection
      const connection = await primaryWallet.getConnection();
      
      // Use improved confirmation handling
      try {
        await confirmTransactionWithRetry(connection, tx);
      console.log('Camera registered successfully, signature:', tx);
      setTxResult(tx);
      setStatusMessage(`Camera registered successfully! Transaction: ${tx}`);
      setStatusType('success');
      } catch (confirmError) {
        console.error('Transaction confirmation error:', confirmError);
        setStatusMessage(`Transaction sent but confirmation timed out. It may still succeed. Check the explorer: ${tx}`);
        setStatusType('info');
      }

      // Clear form
      setCameraName('');
      setCameraModel('');

      // Refresh list of cameras and notify other components
      await fetchRegisteredCameras();
      triggerCameraListRefresh();
    } catch (err) {
      console.error('Error registering camera:', err);

      let errorMessage = 'Failed to register camera';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Function to create face data if it doesn't exist
  const createFaceData = async (userPublicKey: PublicKey) => {
    if (!program) {
      throw new Error('Program not available');
    }

    console.log('Creating face data for user:', userPublicKey.toString());
    
    // Find the recognition token PDA
    const [faceDataPda] = PublicKey.findProgramAddressSync(
      [
        Buffer.from('recognition-token'),
        userPublicKey.toBuffer()
      ],
      CAMERA_ACTIVATION_PROGRAM_ID
    );
    console.log('Face Data PDA:', faceDataPda.toString());
    
    // Generate mock embedding data (random 128 bytes)
    // Just like in the face-recognition-test.js script
    const mockEmbedding = Array(32).fill(0).map(() => Math.floor(Math.random() * 256));
    console.log('Generated mock face embedding data');
    
    // Get the connection for transaction submission
    if (!primaryWallet || !isSolanaWallet(primaryWallet)) {
      throw new Error('Invalid wallet');
    }
    
    const connection = await primaryWallet.getConnection();
    
    try {
      // Create the method call with proper account naming
      const tx = await program.methods
        .upsertRecognitionToken(
          mockEmbedding,
          "Debug Token", // display_name
          0  // source: phone_selfie
        )
        .accounts({
          user: userPublicKey,
          recognitionToken: faceDataPda,
          systemProgram: SystemProgram.programId,
        })
        .rpc();
      
      console.log('Face data creation transaction sent with signature:', tx);
      
      // Confirm transaction
      await connection.confirmTransaction(tx, 'confirmed');
      console.log('Face data creation successful');
      
      return faceDataPda;
    } catch (createErr) { // Use error variable
      console.error('Error creating face data:', createErr);
      // Rethrow or handle as needed, e.g., set an error state
      throw createErr; // Rethrowing the error might be better
    }
  };

  // Add a function to check user session status for all cameras
  const checkUserSessionStatus = async () => {
    if (!primaryWallet?.address || !program || !connection) return;
    
    try {
      const userPublicKey = new PublicKey(primaryWallet.address);
      
      // Create a map to store session status
      const sessionStatus: {[key: string]: boolean} = {};
      
      // Check each camera
      for (const camera of registeredCameras) {
        const cameraPublicKey = new PublicKey(camera.publicKey);
        
        // Find the session PDA
        const [sessionPda] = PublicKey.findProgramAddressSync(
          [
            Buffer.from('session'),
            userPublicKey.toBuffer(),
            cameraPublicKey.toBuffer()
          ],
          CAMERA_ACTIVATION_PROGRAM_ID
        );
        
        // Check if session exists
        try {
          await program.account.userSession.fetch(sessionPda);
          sessionStatus[camera.publicKey] = true;
        } catch {
          sessionStatus[camera.publicKey] = false;
        }
      }
      
      setUserSessions(sessionStatus);
      
      // Update the cameras with session status
      setRegisteredCameras(cameras => 
        cameras.map(camera => ({
          ...camera,
          userCheckedIn: sessionStatus[camera.publicKey] || false
        }))
      );
    } catch (err) {
      console.error('Error checking session status:', err);
    }
  };
  
  // Update useEffect to check session status after fetching cameras
   
  useEffect(() => {
    if (registeredCameras.length > 0 && primaryWallet?.address) {
      checkUserSessionStatus();
    }
  }, [registeredCameras.length, primaryWallet]);

  // Update takePicture to update session status after check-in
  const takePicture = async (camera: CameraData, useFaceRec: boolean) => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    // Check if the camera is active
    if (!camera.isActive) {
      setStatusMessage('Camera is inactive. Cannot take picture.');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      setStatusMessage(`Checking in to camera ${camera.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.publicKey);

      console.log('User address:', userPublicKey.toString());
      console.log('Camera address:', cameraPublicKey.toString());
      console.log('Using face recognition:', useFaceRec);

      // Find the session PDA - EXACTLY as in the tests
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      
      console.log('Session PDA:', sessionPda.toString());

      // Find the recognition token PDA
      const [faceDataPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('recognition-token'),
          userPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
      console.log('Face Data PDA:', faceDataPda.toString());

      // Check if face data exists
      let faceDataExists = false;
      try {
        // Fetch with generic account type
         
        const faceAccount = await program.account.recognitionToken.fetch(faceDataPda) as any;
        if (faceAccount) { // Check if account data is valid
           faceDataExists = true;
           console.log('Face data exists');
        }
      } catch (faceFetchErr) { // Use the error variable
        console.log('No face data found:', faceFetchErr);
        
        // If face recognition is requested but no data exists, create it
        if (useFaceRec) {
          try {
            console.log('Creating face data for face recognition...');
            await createFaceData(userPublicKey);
            faceDataExists = true;
            console.log('Face data created successfully');
          } catch (createErr) {
            console.error('Error creating face data:', createErr);
            setStatusMessage('Failed to create face recognition data. Proceeding without it.');
            setStatusType('error');
            // Continue without face recognition
            useFaceRec = false;
          }
        }
      }
      
      // Get the connection to send directly
      const connection = await primaryWallet.getConnection();
      
      try {
        // MATCH THE EXACT APPROACH FROM THE JAVASCRIPT TEST SCRIPTS
        
        // Create a method call that matches the face-recognition-checkin.js script
        const methodCall = program.methods.checkIn(useFaceRec);
        
        // Define the accounts like in the test script
        const accountsObj: Record<string, PublicKey> = {
          user: userPublicKey,
          camera: cameraPublicKey,
          session: sessionPda,
          systemProgram: SystemProgram.programId
        };

        // Only include recognition token if it exists
        if (faceDataExists) {
          accountsObj.recognitionToken = faceDataPda;
        }
        
        console.log('Check-in accounts:', accountsObj);
        
        // Directly send the transaction using the RPC method like in the test scripts
        const tx = await methodCall
          .accounts(accountsObj)
        .rpc();
      
      console.log('Check-in transaction sent with signature:', tx);
      
        // Use improved confirmation handling
        try {
          await confirmTransactionWithRetry(connection, tx);
      console.log('Check-in successful, signature:', tx);
      setTxResult(tx);
          const faceRecText = useFaceRec ? ' with face recognition' : '';
          setStatusMessage(`Successfully checked in${faceRecText}! Transaction: ${tx}`);
      setStatusType('success');
        } catch (confirmError) {
          console.error('Transaction confirmation error:', confirmError);
          setStatusMessage(`Check-in sent but confirmation timed out. It may still succeed. Check the explorer: ${tx}`);
          setStatusType('info');
        }
        
        // Update session status after sending transaction
        setUserSessions(prev => ({
          ...prev,
          [camera.publicKey]: true
        }));
        
        // Update camera in the UI to show checked in
        setRegisteredCameras(cameras => 
          cameras.map(c => c.publicKey === camera.publicKey ? 
            {...c, userCheckedIn: true} : c
          )
        );

      // Refresh cameras to see updated activity count
      await fetchRegisteredCameras();
      
      // Refresh active users analytics
      await fetchActiveUsersPerCamera();
      } catch (txErr) {
        console.error('Transaction error:', txErr);
        throw txErr;
      }
    } catch (err) {
      console.error('Error checking in:', err);

      let errorMessage = 'Failed to check in';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;

        // Special error messages
        if (err.message.includes('CameraInactive')) {
          errorMessage = 'Camera is currently inactive. Please activate it first.';
        } else if (err.message.includes('Unauthorized')) {
          errorMessage = 'You are not authorized to use this camera.';
        } else if (err.message.includes('SessionAlreadyActive')) {
          errorMessage = 'You already have an active session with this camera.';
        } else if (err.message.includes('0xbc0')) {
          errorMessage = 'Transaction simulation failed. This may be due to program constraints or insufficient funds.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Update checkOutCamera to update session status after check-out
  const checkOutCamera = async (camera: CameraData) => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      setStatusMessage(`Checking out from camera ${camera.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.publicKey);

      // Find the session PDA - EXACTLY as in the tests
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      console.log('Session PDA for checkout:', sessionPda.toString());

      // Check if the session exists before trying to check out
      try {
        // Fetch with generic account type
        await program.account.userSession.fetch(sessionPda);
        console.log('Found session, proceeding with checkout');
      } catch (checkoutFetchErr) { // Use the error variable
        console.log('No session found, cannot check out:', checkoutFetchErr);
        setStatusMessage('No active session found for this camera. Cannot check out.');
        setStatusType('error');
        setLoading(false);
        return;
      }

      // Get the connection
      const connection = await primaryWallet.getConnection();
        
      try {
        // MATCH THE EXACT APPROACH FROM THE JAVASCRIPT TEST SCRIPTS
        // Create a method call like in the test script
        const methodCall = program.methods.checkOut();
        
        // Define the accounts like in the test script
        const accountsObj: Record<string, PublicKey> = {
          closer: userPublicKey,
          camera: cameraPublicKey,
          session: sessionPda,
          sessionUser: userPublicKey,
          rentDestination: userPublicKey, // Rent goes back to user
        };
        
        console.log('Check-out accounts:', accountsObj);
        
        // Directly send the transaction using the RPC method like in the test scripts
        const tx = await methodCall
          .accounts(accountsObj)
        .rpc();
      
      console.log('Check-out transaction sent with signature:', tx);
      
        // Use improved confirmation handling
        try {
          await confirmTransactionWithRetry(connection, tx);
      console.log('Check-out successful, signature:', tx);
      setTxResult(tx);
      setStatusMessage(`Successfully checked out! Transaction: ${tx}`);
      setStatusType('success');
        } catch (confirmError) {
          console.error('Transaction confirmation error:', confirmError);
          setStatusMessage(`Check-out sent but confirmation timed out. It may still succeed. Check the explorer: ${tx}`);
          setStatusType('info');
        }

        // Update session status after sending transaction
        setUserSessions(prev => ({
          ...prev,
          [camera.publicKey]: false
        }));
        
        // Update camera in the UI to show checked out
        setRegisteredCameras(cameras => 
          cameras.map(c => c.publicKey === camera.publicKey ? 
            {...c, userCheckedIn: false} : c
          )
        );

      // Refresh cameras to see updated session count
      await fetchRegisteredCameras();
      
      // Refresh active users analytics
      await fetchActiveUsersPerCamera();
      } catch (txErr) {
        console.error('Transaction error:', txErr);
        throw txErr;
      }
    } catch (err) {
      console.error('Error checking out:', err);

      let errorMessage = 'Failed to check out';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;

        // Special error messages
        if (err.message.includes('SessionNotFound')) {
          errorMessage = 'No active session found for this camera.';
        } else if (err.message.includes('AccountNotInitialized')) {
          errorMessage = 'Session account not initialized. You may need to check in first.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Function to fetch and count active users per camera
  const fetchActiveUsersPerCamera = async () => {
    if (!program || !connection) {
      console.log('Program or connection not available for user session analytics');
      return;
    }

    try {
      setLoadingAnalytics(true);
      console.log('Fetching user sessions for analytics...');
      
      // Fetch all user sessions
      const userSessionAccounts = await program.account.userSession.all();
      console.log('Found user session accounts:', userSessionAccounts.length);
      
      // Count active users per camera
      const activeUsersCount: {[key: string]: number} = {};
      
      for (const sessionInfo of userSessionAccounts) {
        try {
           
          const session = sessionInfo.account as any;
          const cameraKey = session.camera.toString();
          
          // Increment count for this camera
          activeUsersCount[cameraKey] = (activeUsersCount[cameraKey] || 0) + 1;
          
          console.log(`Found active session for camera ${cameraKey}`);
        } catch (error) {
          console.error('Error parsing user session:', error);
        }
      }
      
      setActiveUsersPerCamera(activeUsersCount);
      console.log('Active users per camera:', activeUsersCount);
    } catch (error) {
      console.error('Error fetching user sessions for analytics:', error);
      // Don't reset the analytics on error - keep showing last known state
    } finally {
      setLoadingAnalytics(false);
    }
  };

  // Add a function to fetch registered cameras
  const fetchRegisteredCameras = async () => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    try {
      setLoadingCameras(true);
      setError(null);
      setStatusMessage('Fetching registered cameras...');
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      // Find the registry PDA
      const [registryAddress] = await PublicKey.findProgramAddress(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      console.log('Registry address:', registryAddress.toString());

      // Try to fetch the registry account, but don't fail if it doesn't exist
       
      let registryAccount: any | undefined;
      try {
        // Explicitly use the cameraRegistry type
        registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
        console.log('Registry account successfully fetched:', registryAccount);
      } catch (registryFetchErr) { // Use the error variable
        console.warn('Could not fetch registry account, it may not be initialized yet:', registryFetchErr);
      }

      // Try to fetch raw camera accounts without relying on specific structure
      console.log('Attempting to fetch camera accounts from program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());

      // Get all program accounts without filtering by type - this way we can see what's available
      const allAccounts = await connection.getProgramAccounts(CAMERA_ACTIVATION_PROGRAM_ID);
      console.log('Found', allAccounts.length, 'program accounts of any type');

      // Try to properly parse the camera accounts using the IDL
      const cameras: CameraData[] = [];

      try {
        // Use the correct account type from program.account
        const cameraAccounts = await program.account.cameraAccount.all();
        console.log('Found camera accounts:', cameraAccounts.length);

        for (const accountInfo of cameraAccounts) {
          try {
             
            const account = accountInfo.account as any;
            
            // Access metadata properly through the account structure
            const camera: CameraData = {
              publicKey: accountInfo.publicKey.toString(),
              owner: account.owner.toString(),
              isActive: !!account.isActive,
              name: account.metadata?.name || 'Unnamed Camera',
              model: account.metadata?.model || 'Unknown Model',
              location: null, // Explicitly set to null for now until we handle i64 array conversions
              activeSessions: account.accessCount?.toNumber() || 0,
              totalSessions: account.accessCount?.toNumber() || 0,
              registrationDate: account.metadata?.registrationDate?.toNumber() || 0,
              lastActivity: account.lastActivityAt?.toNumber() || 0,
              // Set the userCheckedIn flag based on the stored sessions
              userCheckedIn: userSessions[accountInfo.publicKey.toString()] || false,
              // Extract device pubkey if available (Optional field for upgrade compatibility)
              devicePubkey: account.devicePubkey ? account.devicePubkey.toString() : undefined
            };

            cameras.push(camera);
            console.log('Successfully parsed camera account:', camera);
          } catch (error) {
            console.error('Error parsing camera account:', error);
          }
        }

        // Log each camera's PDA derivation for validation
        for (const camera of cameras) {
          try {
            const ownerPublicKey = new PublicKey(camera.owner);
            const [expectedPda] = PublicKey.findProgramAddressSync(
              [
                Buffer.from('camera'),
                Buffer.from(camera.name),
                ownerPublicKey.toBuffer()
              ],
              CAMERA_ACTIVATION_PROGRAM_ID
            );
            console.log(`Camera "${camera.name}" PDA validation:`, {
              stored: camera.publicKey,
              computed: expectedPda.toString(),
              match: camera.publicKey === expectedPda.toString()
            });
          } catch (err) {
            console.error('Error validating camera PDA:', err);
          }
        }
      } catch (error) {
        console.error('Error fetching camera accounts using IDL:', error);

        // Fallback: try to manually fetch and parse accounts
        console.log('Using fallback approach to fetch cameras');

        // Look for accounts that might be cameras based on buffer size and pattern
        for (const account of allAccounts) {
          try {
            // Don't try to process the registry account
            if (account.pubkey.equals(registryAddress)) {
              continue;
            }

            // Display account info for debugging
            console.log('Account data for potential camera:', {
              pubkey: account.pubkey.toString(),
              owner: account.account.owner.toString(),
              dataLength: account.account.data.length,
              executable: account.account.executable,
              lamports: account.account.lamports
            });

            try {
              // Try to decode using program
               
              const decodedAccount = await program.coder.accounts.decode(
                'cameraAccount',
                account.account.data
              ) as any;

              // If successful, we found a camera account
              console.log('Successfully decoded as camera account:', decodedAccount);

              // Create a CameraData object from the decoded account using the new structure
              const camera: CameraData = {
                publicKey: account.pubkey.toString(),
                owner: decodedAccount.owner.toString(),
                isActive: !!decodedAccount.isActive,
                name: decodedAccount.metadata?.name || 'Unnamed Camera',
                model: decodedAccount.metadata?.model || 'Unknown Model',
                location: null, // Explicitly set to null for now until we handle i64 array conversions
                activeSessions: decodedAccount.accessCount?.toNumber() || 0,
                totalSessions: decodedAccount.accessCount?.toNumber() || 0,
                registrationDate: decodedAccount.metadata?.registrationDate?.toNumber() || 0,
                lastActivity: decodedAccount.lastActivityAt?.toNumber() || 0,
                // Set the userCheckedIn flag based on the stored sessions
                userCheckedIn: userSessions[account.pubkey.toString()] || false,
                // Extract device pubkey if available (Optional field for upgrade compatibility)
                devicePubkey: decodedAccount.devicePubkey ? decodedAccount.devicePubkey.toString() : undefined
              };

              cameras.push(camera);
              continue;
            } catch (error) {
              const decodeError = error as Error;
              console.log('Not a camera account or format mismatch:', decodeError.message);
            }

            // If we can detect something that looks like a camera account
            // based on size - cameras are likely larger accounts
            const data = account.account.data;
            if (data.length > 100) {
              console.log('Found potential camera account by size:', account.pubkey.toString());

              // Create a minimal camera representation for unknown format
              cameras.push({
                publicKey: account.pubkey.toString(),
                owner: 'unknown (data format changed)',
                isActive: false,
                name: 'Camera (data format changed)',
                model: 'Unknown Model',
                location: null,
                activeSessions: 0,
                totalSessions: 0,
                registrationDate: 0,
                lastActivity: 0,
                userCheckedIn: false
              });
            }
          } catch (err) {
            console.error('Error processing account:', err);
          }
        }
      }

      console.log(`Found ${cameras.length} cameras:`, cameras);

      // No longer filtering by owner - show all cameras
      setRegisteredCameras(cameras);
      setStatusMessage(`Found ${cameras.length} registered cameras`);
      setStatusType('success');
      
      // Fetch active users analytics after loading cameras
      await fetchActiveUsersPerCamera();
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch cameras');
      setStatusMessage('Failed to fetch cameras: ' + (err instanceof Error ? err.message : 'Unknown error'));
      setStatusType('error');
    } finally {
      setLoadingCameras(false);
    }
  };

  // Add useEffect to fetch cameras when program is initialized
  useEffect(() => {
    if (initialized && program) {
      fetchRegisteredCameras();
    }
   
  }, [initialized, program]);

  // Subscribe to global camera list refresh events
  useEffect(() => {
    const unsubscribe = onCameraListRefresh(() => {
      console.log('[SolDevNetDebug] Received camera list refresh event');
      if (initialized && program) {
        fetchRegisteredCameras();
      }
    });
    return unsubscribe;
  }, [onCameraListRefresh, initialized, program]);

  // Add a helper function to get camera status display - Removed as unused
  // const getCameraStatusDisplay = (camera: CameraData) => {
  //   const isOldFormat = camera.owner === 'unknown (data format changed)';
  //
  //   // If it's an old format, we don't know the status
  //   if (isOldFormat) {
  //     return (
  //       <span className="text-yellow-600 bg-yellow-100 px-2 py-0.5 rounded text-xs">
  //         Unknown Format
  //       </span>
  //     );
  //   }
  //
  //   return (
  //     <span className={`text-${camera.isActive ? 'green' : 'red'}-600 bg-${camera.isActive ? 'green' : 'red'}-100 px-2 py-0.5 rounded text-xs`}>
  //       {camera.isActive ? 'Active' : 'Inactive'}
  //     </span>
  //   );
  // };

  // Add a function to deregister a camera
  const deregisterCamera = async (camera: CameraData) => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      setStatusMessage(`Deregistering camera ${camera.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const ownerPublicKey = new PublicKey(primaryWallet.address);

      // Get the camera public key
      const cameraPublicKey = new PublicKey(camera.publicKey);

      // Find the registry PDA
      const [registryAddress] = PublicKey.findProgramAddressSync(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );
     
      // Use Dynamic's signer to create and send the transaction directly
      const signer = await primaryWallet.getSigner();
      const connection = await primaryWallet.getConnection();

      // Set up transaction manually with proper signing
      const transaction = new Transaction();
      
      // Create the instruction with the correct accounts
      const ix = await program.methods
        .deregisterCamera()
        .accounts({
          owner: ownerPublicKey,
          camera: cameraPublicKey,
          cameraRegistry: registryAddress,
          systemProgram: SystemProgram.programId,
        })
        .instruction();
      
      // Add the instruction to the transaction
      transaction.add(ix);
      
      // Get a recent blockhash for the transaction
      const { blockhash } = await connection.getLatestBlockhash();
      transaction.recentBlockhash = blockhash;
      transaction.feePayer = ownerPublicKey;
      
      // Sign and send the transaction
      const signedTx = await signer.signTransaction(transaction);
      const txid = await connection.sendRawTransaction(signedTx.serialize());
      
      console.log('Deregister transaction sent with signature:', txid);
      
      // Wait for confirmation
      await connection.confirmTransaction(txid, 'confirmed');

      setTxResult(txid);
      setStatusMessage(`Camera deregistered! Transaction: ${txid}`);
      setStatusType('success');

      // Refresh the camera list
      await fetchRegisteredCameras();
    } catch (err) {
      console.error('Error deregistering camera:', err);

      let errorMessage = 'Failed to deregister camera';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;

        if (err.message.includes('Unauthorized')) {
          errorMessage += '. Only the camera owner can deregister it.';
        } else if (err.message.includes('blockhash')) {
          errorMessage = 'Network error: Could not get recent blockhash. Please try again.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Toggle camera active state
  const toggleCameraActive = async (camera: CameraData) => {
    if (!primaryWallet?.address || !program) {
      setStatusMessage('Wallet or program not available');
      setStatusType('error');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setTxResult(null);
      const newActiveState = !camera.isActive;
      setStatusMessage(`${newActiveState ? 'Activating' : 'Deactivating'} camera ${camera.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const ownerPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.publicKey);

      // Use Dynamic's signer to create and send the transaction directly
      const signer = await primaryWallet.getSigner();
      const connection = await primaryWallet.getConnection();

      // Set up transaction manually with proper signing
      const transaction = new Transaction();
      
      // Create the instruction with the correct accounts
      const ix = await program.methods
        .setCameraActive(newActiveState)
        .accounts({
          owner: ownerPublicKey,
          camera: cameraPublicKey,
        })
        .instruction();
      
      // Add the instruction to the transaction
      transaction.add(ix);
      
      // Get a recent blockhash for the transaction
      const { blockhash } = await connection.getLatestBlockhash();
      transaction.recentBlockhash = blockhash;
      transaction.feePayer = ownerPublicKey;
      
      // Sign and send the transaction
      const signedTx = await signer.signTransaction(transaction);
      const txid = await connection.sendRawTransaction(signedTx.serialize());
      
      console.log('Toggle active transaction sent with signature:', txid);
      
      // Wait for confirmation
      await connection.confirmTransaction(txid, 'confirmed');

      console.log(`Camera ${newActiveState ? 'activated' : 'deactivated'} successfully, signature:`, txid);
      setTxResult(txid);
      setStatusMessage(`Camera ${newActiveState ? 'activated' : 'deactivated'} successfully! Transaction: ${txid}`);
      setStatusType('success');

      // Refresh cameras to see updated state
      await fetchRegisteredCameras();
    } catch (err) {
      console.error('Error toggling camera active state:', err);

      let errorMessage = 'Failed to update camera state';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;
        
        if (err.message.includes('blockhash')) {
          errorMessage = 'Network error: Could not get recent blockhash. Please try again.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // --- Face Enrollment Functions ---
  const captureFace = async () => {
    if (!primaryWallet?.address) {
      setEnrollmentError('Wallet not connected');
      return;
    }
    setEnrollmentLoading(true);
    setEnrollmentStatus('Requesting face capture from Jetson...');
    setEnrollmentError(null);
    setCapturedEmbedding(null); // Clear previous embedding

    try {
      // --- TODO: Replace with your actual Jetson endpoint ---
      const JETSON_CAPTURE_ENDPOINT = 'http://JETSON_IP_ADDRESS:PORT/capture_face'; 
      
      console.log(`Sending request to Jetson: ${JETSON_CAPTURE_ENDPOINT}`);
      
      // Example fetch request (adjust headers/method as needed)
      const response = await fetch(JETSON_CAPTURE_ENDPOINT, {
        method: 'POST', // or 'GET' depending on your Jetson API
      });

      if (!response.ok) {
        throw new Error(`Jetson API error: ${response.statusText}`);
      }

      // --- TODO: Adjust how you process the response based on what Jetson sends ---
      // Assuming Jetson sends embedding as base64 string in JSON: { embedding: "base64data..." }
      const data = await response.json();
      if (!data.embedding) {
          throw new Error('No embedding data received from Jetson');
      }
      
      // Convert base64 embedding to Uint8Array
      const embeddingBytes = Buffer.from(data.embedding, 'base64'); 
      console.log(`Received embedding (${embeddingBytes.length} bytes)`);

      // --- TODO: Validate embedding size/format if necessary ---
      // Example: Check if it matches the expected size (e.g., 128 bytes)
      if (embeddingBytes.length === 0) { // Example simple validation
         throw new Error(`Invalid embedding size: ${embeddingBytes.length}`);
      }

      setCapturedEmbedding(embeddingBytes);
      setEnrollmentStatus('Face embedding captured successfully!');
      
    } catch (err) {
      console.error('Error capturing face:', err);
      const message = err instanceof Error ? err.message : 'Unknown error during face capture';
      setEnrollmentError(`Failed to capture face: ${message}`);
      setEnrollmentStatus('');
    } finally {
      setEnrollmentLoading(false);
    }
  };

  const mintFaceNFT = async () => {
      if (!primaryWallet?.address || !program || !connection) {
          setEnrollmentError('Wallet, program, or connection not available');
          return;
      }
      if (!capturedEmbedding) {
          setEnrollmentError('No face embedding captured. Please capture first.');
          return;
      }

      setEnrollmentLoading(true);
      setEnrollmentStatus('Minting Face NFT...');
      setEnrollmentError(null);
      setTxResult(null); // Clear previous general tx results

      try {
          if (!isSolanaWallet(primaryWallet)) {
              throw new Error('Not a Solana wallet');
          }
          
          const userPublicKey = new PublicKey(primaryWallet.address);

          // Find the recognition token PDA (matches upsert_recognition_token.rs)
          const [faceDataPda] = PublicKey.findProgramAddressSync(
              [
                  Buffer.from('recognition-token'),
                  userPublicKey.toBuffer()
              ],
              CAMERA_ACTIVATION_PROGRAM_ID
          );
          console.log('Target Face Data PDA:', faceDataPda.toString());
          console.log('User:', userPublicKey.toString());
          console.log('Embedding size:', capturedEmbedding.length);

          // Get the connection
          const connection = await primaryWallet.getConnection();

          // Use a better generic type for accounts
          const accounts: Record<string, PublicKey> = {
              user: userPublicKey,
              faceNft: faceDataPda,
              systemProgram: SystemProgram.programId,
          };

          // Call the upsertRecognitionToken instruction
          const tx = await program.methods
              .upsertRecognitionToken(
                Buffer.from(capturedEmbedding), // Pass embedding as Buffer
                "Debug Token", // display_name
                0  // source: phone_selfie
              )
              .accounts(accounts)
              .rpc();
          
          console.log('Enroll Face transaction sent:', tx);

          // Use confirmation helper
          await confirmTransactionWithRetry(connection, tx);

          console.log('Face NFT minted successfully, signature:', tx);
          setTxResult(tx); // Show general success message too
          setEnrollmentStatus(`Face NFT minted successfully! Check general status below. Tx: ${tx}`);
          setCapturedEmbedding(null); // Clear embedding after successful mint

      } catch (err) {
          console.error('Error minting Face NFT:', err);
          const message = err instanceof Error ? err.message : 'Unknown error during minting';
          // Check for specific program errors if possible from the message
          if (message.includes('InvalidFaceData')) {
             setEnrollmentError('Failed to mint: Invalid face data provided by Jetson.');
          } else {
             setEnrollmentError(`Failed to mint Face NFT: ${message}`);
          }
          setEnrollmentStatus('');
      } finally {
          setEnrollmentLoading(false);
      }
  };
  // ---------------------------------

  // Add a component for navigating to the camera view
  const CameraNavigationButton = ({ camera }: { camera: CameraData }) => {
    const navigateToCamera = () => {
      if (camera && camera.publicKey) {
        // Navigate directly to the camera view instead of quickstart
        navigate(`/app/camera/${encodeURIComponent(camera.publicKey)}`);
      }
    };

    return (
      <button
        onClick={navigateToCamera}
        className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm mr-2"
      >
        View Camera
      </button>
    );
  };

  // Render the camera list
  // Test PDA-based URL system
  const testPdaUrls = async () => {
    setLoading(true);
    setStatusMessage('Testing PDA-based URL system...');
    setStatusType('info');
    
    try {
      const { CONFIG } = await import('../../core/config');
      const { unifiedCameraService } = await import('../../camera/unified-camera-service');
      
      // Test known cameras
      const knownCameras = [
        { pda: CONFIG.JETSON_CAMERA_PDA, name: 'Jetson Orin Nano' },
        { pda: CONFIG.CAMERA_PDA, name: 'Raspberry Pi 5' }
      ];
      
      const results = [];
      
      for (const camera of knownCameras) {
        const pdaUrl = CONFIG.getCameraApiUrlByPda(camera.pda);
        console.log(`Testing ${camera.name} at ${pdaUrl}`);
        
        try {
          // Test connection
          const connectionResult = await unifiedCameraService.testConnection(camera.pda);
          
          // Test camera info endpoint
          const response = await fetch(`${pdaUrl}/api/camera/info`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            mode: 'cors',
            credentials: 'omit'
          });
          
          const infoResult = response.ok ? await response.json() : null;
          
          results.push({
            camera: camera.name,
            pda: camera.pda,
            url: pdaUrl,
            connectionTest: connectionResult.success,
            infoEndpoint: response.ok,
            info: infoResult
          });
          
        } catch (error) {
          results.push({
            camera: camera.name,
            pda: camera.pda,
            url: pdaUrl,
            connectionTest: false,
            infoEndpoint: false,
            error: error instanceof Error ? error.message : 'Unknown error'
          });
        }
      }
      
      console.log('PDA URL Test Results:', results);
      setStatusMessage(`PDA URL testing completed. Check console for details.`);
      setStatusType('success');
      
    } catch (error) {
      console.error('Error testing PDA URLs:', error);
      setStatusMessage(`PDA URL testing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  const renderCameraList = () => {
    const noCamera = registeredCameras.length === 0;

    if (loadingCameras) {
      return (
        <div className="flex justify-center items-center py-4">
          <div className="spinner mr-2"></div>
          <span>Loading cameras...</span>
        </div>
      );
    }

    if (noCamera) {
      return (
        <div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500">
          No cameras registered. Register a camera to get started.
        </div>
      );
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {registeredCameras.map((camera) => {
          const isOldFormat = !camera.name || camera.name === 'Camera (data format changed)';
          const isOwner = camera.owner === primaryWallet?.address;
          const isActive = camera.isActive;
          
          // Initialize face recognition toggle for this camera if not set
          if (useFaceRecognition[camera.publicKey] === undefined) {
            setUseFaceRecognition(prev => ({
              ...prev,
              [camera.publicKey]: false
            }));
          }
          
          return (
            <div 
              key={camera.publicKey} 
              className={`bg-white rounded-lg shadow-sm border p-4 ${
                camera.userCheckedIn ? 'border-blue-400' : 
                isActive ? 'border-green-200' : 'border-gray-200'
              }`}
            >
              <div className="flex flex-col h-full">
                <div>
                  <div className="flex items-center">
                    <h3 className="font-bold">{camera.name || 'Unknown Camera'}</h3>
                    {isOldFormat && (
                      <span className="ml-2 bg-yellow-100 text-yellow-800 text-xs px-2 py-0.5 rounded">Old Format</span>
                    )}
                    {camera.userCheckedIn && (
                      <span className="ml-2 bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded">
                        You're Checked In
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">Model: {camera.model || 'Unknown'}</p>
                  <p className="text-sm text-gray-600">Public Key: <span className="font-mono">{camera.publicKey}</span></p>
                  <p className="text-sm text-gray-600 mt-1">
                    Status: <span className={`font-medium ${isActive ? 'text-green-600' : 'text-red-600'}`}>
                      {isActive ? 'Active' : 'Inactive'}
                    </span>
                  </p>
                  
                  <p className="text-sm text-gray-600">
                    DePIN Signing: <span className={`font-medium ${camera.devicePubkey ? 'text-green-600' : 'text-gray-400'}`}>
                      {camera.devicePubkey ? 'Enabled' : 'Not Available'}
                    </span>
                    {camera.devicePubkey && (
                      <span className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">
                        Device Authenticated
                      </span>
                    )}
                  </p>

                  {!isOldFormat && camera.activeSessions !== undefined && (
                    <p className="text-sm text-gray-600">
                      Active Sessions: <span className="font-medium">{camera.activeSessions}</span>
                    </p>
                  )}

                  {!isOldFormat && camera.totalSessions !== undefined && (
                    <p className="text-sm text-gray-600">
                      Total Sessions: <span className="font-medium">{camera.totalSessions}</span>
                    </p>
                  )}

                  {/* Analytics: Active Users Currently Checked In */}
                  <div className="bg-gray-50 rounded-md p-2 mt-2">
                    <p className="text-sm text-gray-700 font-medium">
                      <span className="inline-flex items-center">
                        <span className={`w-2 h-2 rounded-full mr-2 ${
                          (activeUsersPerCamera[camera.publicKey] || 0) > 0 ? 'bg-green-500' : 'bg-gray-400'
                        }`}></span>
                        Active Users: <span className={`font-bold ml-1 ${
                          (activeUsersPerCamera[camera.publicKey] || 0) > 0 ? 'text-green-600' : 'text-gray-500'
                        }`}>
                          {activeUsersPerCamera[camera.publicKey] || 0}
                        </span>
                        <span className="text-xs text-gray-500 ml-1">
                          {(activeUsersPerCamera[camera.publicKey] || 0) === 1 ? 'user' : 'users'} checked in
                        </span>
                      </span>
                    </p>
                  </div>

                  <p className="text-sm text-gray-600">
                    Owner: <span className="font-mono text-xs">{camera.owner.substring(0, 8)}...</span>
                    {isOwner && <span className="ml-1 text-blue-600 text-xs">(you)</span>}
                  </p>

                  {camera.registrationDate && (
                    <p className="text-sm text-gray-600">
                      Registered: <span>{new Date(camera.registrationDate * 1000).toLocaleDateString()}</span>
                    </p>
                  )}
                </div>

                <div className="flex flex-col mt-4 gap-2">
                  {isActive && (
                    <div className="flex flex-wrap gap-2 mb-2">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id={`face-recognition-${camera.publicKey}`}
                          checked={useFaceRecognition[camera.publicKey] || false}
                          onChange={() => {
                            setUseFaceRecognition(prev => ({
                              ...prev,
                              [camera.publicKey]: !prev[camera.publicKey]
                            }));
                          }}
                          className="mr-2 h-4 w-4 text-blue-600"
                        />
                        <label htmlFor={`face-recognition-${camera.publicKey}`} className="text-sm text-gray-700">
                          Use Face Recognition
                        </label>
                      </div>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    {isActive && !camera.userCheckedIn && (
                    <button
                      className="btn-primary text-sm py-1 px-3"
                        onClick={() => takePicture(camera, useFaceRecognition[camera.publicKey] || false)}
                      disabled={loading}
                    >
                      Check In
                    </button>
                  )}

                    {/* Only show check out if user is checked in */}
                    {isActive && camera.userCheckedIn && (
                    <button
                      className="bg-orange-500 hover:bg-orange-600 text-white text-sm py-1 px-3 rounded"
                      onClick={() => checkOutCamera(camera)}
                      disabled={loading}
                    >
                      Check Out
                    </button>
                  )}

                  {isOwner && (
                    <button
                      className={`text-sm py-1 px-3 rounded ${
                        isActive
                          ? 'bg-red-100 text-red-700 hover:bg-red-200'
                          : 'bg-green-100 text-green-700 hover:bg-green-200'
                      }`}
                      onClick={() => toggleCameraActive(camera)}
                      disabled={loading}
                    >
                      {isActive ? 'Deactivate' : 'Activate'}
                    </button>
                  )}

                  {isOwner && (
                    <button
                      className="bg-red-500 hover:bg-red-600 text-white text-sm py-1 px-3 rounded"
                      onClick={() => deregisterCamera(camera)}
                      disabled={loading}
                    >
                      Deregister
                    </button>
                  )}

                    <CameraNavigationButton camera={camera} />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Modify the render function to remove Activity Management and update NFC URL Generator
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Solana DevNet Debug</h1>
      <p className="mb-6">Camera Activation Program Dashboard</p>

      {/* Connection Information */}
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-2">Connection Information</h2>
        <div className="bg-blue-50 p-4 rounded mb-4">
          <p>This dashboard helps you manage camera devices on Solana. Use it to register new cameras and generate NFC URLs.</p>
        </div>

        {statusMessage && (
          <div className={`p-4 rounded mb-4 ${statusType === 'success' ? 'bg-green-100 text-green-800' :
              statusType === 'error' ? 'bg-red-100 text-red-800' :
                'bg-blue-100 text-blue-800'
            }`}>
            {statusMessage}
          </div>
        )}

        {error && (
          <div className="bg-red-100 text-red-800 p-4 rounded mb-4">
            Error: {error}
          </div>
        )}

        {txResult && (
          <div className="bg-green-100 text-green-800 p-4 rounded mb-4">
            <div className="flex flex-col">
              <div className="mb-2">Transaction sent: {txResult}</div>
              <div className="flex">
                <a 
                  href={`https://explorer.solana.com/tx/${txResult}?cluster=devnet`} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="bg-green-700 hover:bg-green-800 text-white px-3 py-1 rounded text-sm mr-2"
                >
                  View in Explorer
                </a>
                <button
                  onClick={() => navigator.clipboard.writeText(txResult)}
                  className="bg-gray-200 text-gray-700 px-3 py-1 rounded text-sm hover:bg-gray-300"
                >
                  Copy Signature
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Connection Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div>
          <h2 className="text-xl font-bold mb-2">Connection Status</h2>
          <div className="bg-white p-4 rounded shadow">
            <div className="mb-4">
              <h3 className="font-bold">Wallet</h3>
              <div className="flex items-center mt-2">
                <div className={`w-3 h-3 rounded-full ${primaryWallet ? 'bg-green-500' : 'bg-red-500'} mr-2`}></div>
                <span>{primaryWallet ? `Connected: ${primaryWallet.address.slice(0, 8)}...${primaryWallet.address.slice(-5)}` : 'Not connected'}</span>
              </div>
            </div>

            <div>
              <h3 className="font-bold">Program</h3>
              <div className="flex items-center mt-2">
                <div className={`w-3 h-3 rounded-full ${initialized ? 'bg-green-500' : 'bg-yellow-500'} mr-2`}></div>
                <span>{initialized ? 'Initialized' : 'Not initialized'}</span>
              </div>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-xl font-bold mb-2">Registered Cameras</h2>
          <div className="bg-white p-4 rounded shadow">
            <div className="bg-green-100 text-green-800 p-2 rounded mb-4">
              Found {registeredCameras.length} registered cameras
            </div>
            
            {/* Analytics Summary */}
            <div className="bg-blue-50 p-3 rounded mb-4">
              <h3 className="font-semibold text-blue-800 mb-2 flex items-center">
                Live Analytics
                {loadingAnalytics && (
                  <div className="ml-2 w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                )}
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-blue-600 font-medium">Total Active Users:</span>{' '}
                  <span className="font-bold text-blue-800">
                    {Object.values(activeUsersPerCamera).reduce((sum, count) => sum + count, 0)}
                  </span>
                </div>
                <div>
                  <span className="text-blue-600 font-medium">Cameras with Users:</span>{' '}
                  <span className="font-bold text-blue-800">
                    {Object.values(activeUsersPerCamera).filter(count => count > 0).length} / {registeredCameras.length}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex space-x-2 mb-4">
              {!initialized && (
                <button
                  onClick={initializeRegistry}
                  className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 font-semibold"
                  disabled={loading}
                >
                  {loading ? 'Initializing...' : 'Initialize Registry'}
                </button>
              )}
              
              <button
                onClick={async () => {
                  await fetchRegisteredCameras();
                  await fetchActiveUsersPerCamera();
                }}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                disabled={loading || loadingAnalytics || !initialized}
              >
                {(loadingCameras || loadingAnalytics) ? 'Loading...' : 'Refresh cameras & analytics'}
              </button>
              
              <button
                onClick={testPdaUrls}
                disabled={loading}
                className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                Test PDA URLs
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Camera Management & Face Enrollment */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {/* Camera Management */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="font-semibold mb-4">Camera Management</h2>
          <div className="bg-blue-50 p-3 mb-4 rounded-md text-sm">
            <p className="font-semibold">How to register a new camera:</p>
            <ol className="list-decimal pl-5 mt-1 space-y-1">
              <li>Enter a unique camera name</li>
              <li>Enter a camera model</li>
              <li>Click "Register Camera"</li>
              <li>Approve the transaction in your wallet</li>
            </ol>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Camera Name</label>
              <input
                type="text"
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                placeholder="e.g., Living Room Camera"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Camera Model</label>
              <input
                type="text"
                value={cameraModel}
                onChange={(e) => setCameraModel(e.target.value)}
                placeholder="e.g., Raspberry Pi Camera"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            <button
              onClick={registerCamera}
              disabled={loading || !primaryWallet?.address || !initialized}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : 'Register Camera'}
            </button>
          </div>
        </div>

        {/* --- Face Enrollment Section --- */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="font-semibold mb-4">Face Enrollment</h2>
          <div className="bg-blue-50 p-3 mb-4 rounded-md text-sm">
            <p className="font-semibold">Enroll your face for recognition:</p>
            <ol className="list-decimal pl-5 mt-1 space-y-1">
              <li>Ensure your Jetson is running and accessible.</li>
              <li>Position your face clearly in the camera view below.</li>
              <li>Click "Capture Face" to get the embedding from the Jetson.</li>
              <li>Click "Mint Face NFT" to store it on-chain with your wallet.</li>
            </ol>
          </div>

          {/* TODO: Live Camera Feed */}
          <div className="mb-4 p-2 border border-gray-300 rounded bg-gray-100 h-48 flex items-center justify-center text-gray-500">
            {jetsonStreamUrl ? (
              <img src={jetsonStreamUrl} alt="Jetson Camera Feed" className="max-h-full max-w-full" />
            ) : (
              <span>Camera Feed Placeholder (Set Jetson Stream URL in code)</span>
            )}
          </div>

          {/* Enrollment Status/Error Display */}
          {enrollmentStatus && (
            <div className={`p-2 rounded mb-2 text-sm ${enrollmentError ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'}`}>
              {enrollmentStatus}
            </div>
          )}
          {enrollmentError && (
            <div className="p-2 rounded mb-2 text-sm bg-red-100 text-red-700">
              Error: {enrollmentError}
            </div>
          )}

          <div className="space-y-3">
             <p className="text-sm text-gray-600">
                Enrolling for Wallet: <span className="font-mono">{primaryWallet?.address ? `${primaryWallet.address.slice(0, 8)}...` : 'N/A'}</span>
             </p>
            
            <button
              onClick={captureFace}
              disabled={enrollmentLoading || !primaryWallet?.address}
              className="w-full bg-teal-600 text-white py-2 px-4 rounded-md hover:bg-teal-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {enrollmentLoading && enrollmentStatus.includes('Jetson') ? 'Capturing...' : '1. Capture Face from Jetson'}
            </button>

            {capturedEmbedding && (
                 <div className="text-sm text-green-700">
                    Embedding captured ({capturedEmbedding.length} bytes). Ready to mint.
                 </div>
            )}

            <button
              onClick={mintFaceNFT}
              disabled={enrollmentLoading || !primaryWallet?.address || !capturedEmbedding}
              className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {enrollmentLoading && enrollmentStatus.includes('Minting') ? 'Minting...' : '2. Mint Face NFT'}
            </button>
          </div>
        </div>
        {/* ----------------------------- */}

        {/* NFC URL Generator */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="font-semibold mb-4">NFC URL Generator</h2>
          <div className="bg-blue-50 p-3 mb-4 rounded-md text-sm">
            <p className="font-semibold">Generate a URL to connect to a camera:</p>
            <ol className="list-decimal pl-5 mt-1 space-y-1">
              <li>Select a camera from the list below</li>
              <li>Click "Generate NFC URL"</li>
              <li>The link will direct users to the camera quickstart page</li>
            </ol>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Select Camera</label>
            <select
              id="nfc-camera-select"
              className="w-full p-2 border border-gray-300 rounded-md mb-4"
              disabled={registeredCameras.length === 0}
            >
              {registeredCameras.length === 0 ? (
                <option>No cameras available</option>
              ) : (
                registeredCameras.map((camera, index) => (
                  <option key={index} value={camera.publicKey}>
                    {camera.name || 'Unnamed Camera'} ({camera.publicKey.slice(0, 8)}...)
                  </option>
                ))
              )}
            </select>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Generated URL</label>
              <div className="flex">
                <input
                  id="nfc-url-output"
                  type="text"
                  readOnly
                  className="w-full p-2 border border-gray-300 rounded-l-md bg-gray-50"
                  placeholder="Click 'Generate NFC URL' to create a link"
                />
                <button
                  id="copy-url-button"
                  className="bg-gray-200 text-gray-700 px-4 py-2 rounded-r-md hover:bg-gray-300"
                  onClick={() => {
                    const urlInput = document.getElementById('nfc-url-output') as HTMLInputElement;
                    if (urlInput && urlInput.value) {
                      navigator.clipboard.writeText(urlInput.value);
                      setStatusMessage('URL copied to clipboard');
                      setStatusType('success');
                    }
                  }}
                >
                  Copy
                </button>
              </div>
            </div>

            <button
              onClick={() => {
                const select = document.getElementById('nfc-camera-select') as HTMLSelectElement;
                const output = document.getElementById('nfc-url-output') as HTMLInputElement;

                if (select && output && select.value) {
                  // Generate the URL for the app with the camera ID
                  const cameraId = select.value;
                  const baseUrl = window.location.origin; // e.g., "https://example.com"

                  // Direct link to camera control page instead of quickstart
                  const appUrl = `${baseUrl}/app/camera/${cameraId}`;

                  // Set the URL in the input field
                  output.value = appUrl;

                  // Update status
                  setStatusMessage('NFC URL generated successfully');
                  setStatusType('success');
                }
              }}
              disabled={registeredCameras.length === 0}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Generate NFC URL
            </button>

            <button
              onClick={() => {
                const select = document.getElementById('nfc-camera-select') as HTMLSelectElement;
                if (select && select.value) {
                  const cameraId = select.value;
                  const baseUrl = window.location.origin;
                  const appUrl = `${baseUrl}/app/camera/${cameraId}`;
                  window.open(appUrl, '_blank');
                }
              }}
              disabled={registeredCameras.length === 0}
              className="w-full mt-2 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Test Camera Link
            </button>
          </div>
        </div>
      </div>

      {/* Debug Information */}
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-2">Debug Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-600 mb-1">Program ID</p>
            <p className="font-mono break-all">{CAMERA_ACTIVATION_PROGRAM_ID.toString()}</p>
          </div>
          <div>
            <p className="text-gray-600 mb-1">Connection Endpoint</p>
            <p className="font-mono">{connection?.rpcEndpoint || 'Not connected'}</p>
          </div>
          <div>
            <p className="text-gray-600 mb-1">Wallet Status</p>
            <p>Connected: {primaryWallet ? 'Yes' : 'No'}</p>
            {primaryWallet && <p className="font-mono break-all">Address: {primaryWallet.address}</p>}
          </div>
          <div>
            <p className="text-gray-600 mb-1">Program Status</p>
            <p>Registry Initialized: {initialized ? 'Yes' : 'No'}</p>
          </div>
        </div>
      </div>

      {/* Camera List */}
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-2">All Registered Cameras</h2>
        {renderCameraList()}
      </div>
    </div>
  );
} 