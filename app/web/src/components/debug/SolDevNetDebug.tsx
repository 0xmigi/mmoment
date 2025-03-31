import { useState, useEffect } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, SystemProgram, Transaction, Connection } from '@solana/web3.js';
import { Program, AnchorProvider, BN } from '@coral-xyz/anchor';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../../anchor/setup';
import { IDL, MySolanaProject } from '../../anchor/idl';
import { isSolanaWallet } from '@dynamic-labs/solana';

// Simple cache for account info to reduce RPC calls
const accountInfoCache = new Map<string, { data: any, timestamp: number }>();
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
interface ActivityType {
  photoCapture?: {};
  videoRecord?: {};
  liveStream?: {};
}

interface CameraData {
  publicKey: string;
  owner: string;
  isActive: boolean;
  activityCounter?: number;
  lastActivityType?: ActivityType;
  metadata: {
    name: string;
    model: string;
    registrationDate: number;
    lastActivity: number;
    location?: [number, number] | null;
  };
}

export function SolDevNetDebug() {
  const dynamicContext = useDynamicContext();
  const { primaryWallet } = dynamicContext;
  const { connection } = useConnection();

  // State variables
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txResult, setTxResult] = useState<string | null>(null);
  const [program, setProgram] = useState<Program<MySolanaProject> | null>(null);

  // Form state
  const [cameraName, setCameraName] = useState('');
  const [cameraModel, setCameraModel] = useState('');

  // Status messages
  const [statusMessage, setStatusMessage] = useState('');
  const [statusType, setStatusType] = useState<'info' | 'success' | 'error'>('info');

  // Add state for registered cameras
  const [registeredCameras, setRegisteredCameras] = useState<CameraData[]>([]);
  const [loadingCameras, setLoadingCameras] = useState(false);

  // Initialize program when wallet is connected
  useEffect(() => {
    if (!primaryWallet?.address || !connection) return;

    try {
      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          // We don't need these methods as we'll use the wallet directly
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program
      const prog = new Program<MySolanaProject>(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
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
          const registryAccountInfo = await getCachedAccountInfo(connection, registryAddress);
          const isInitialized = !!registryAccountInfo && registryAccountInfo.data.length > 0;

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

      // Get the connection from the wallet
      const walletConnection = await primaryWallet.getConnection();

      // Get the signer
      const signer = await primaryWallet.getSigner();

      const ownerPublicKey = new PublicKey(primaryWallet.address);

      // Find the camera PDA using the camera name for derivation
      const cameraNameBuffer = Buffer.from(cameraName.trim());
      console.log('Using camera name for PDA:', cameraName);
      console.log('Camera name buffer length:', cameraNameBuffer.length);
      console.log('Camera name bytes:', [...cameraNameBuffer]);

      // Get the IDL structure for registerCamera
      const registerCameraInstruction = program.idl.instructions.find(i => i.name === 'registerCamera');
      if (registerCameraInstruction) {
        console.log('Register Camera Instruction:', {
          accounts: registerCameraInstruction.accounts.map(a => ({
            name: a.name,
            isMut: a.isMut,
            isSigner: a.isSigner
          })),
          args: registerCameraInstruction.args
        });
      }

      const [cameraAddress] = await PublicKey.findProgramAddress(
        [
          Buffer.from('camera'),
          cameraNameBuffer,
          ownerPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      console.log('Derived camera address:', cameraAddress.toString());

      // Find the registry PDA
      const [registryAddress] = await PublicKey.findProgramAddress(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Create the args object matching the Rust RegisterCameraArgs struct
      const args = {
        name: cameraName,
        model: cameraModel,
        location: null, // No location
        fee: new BN(100) // Minimum fee required by the program
      };

      console.log('Register camera args:', JSON.stringify(args, null, 2));

      // Build the transaction instruction
      const ix = await program.methods
        .registerCamera(args)
        .accounts({
          owner: ownerPublicKey,
          camera: cameraAddress,
          registry: registryAddress,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Create a new transaction and add the instruction
      const transaction = new Transaction().add(ix);

      // Set recent blockhash and fee payer
      transaction.recentBlockhash = (await walletConnection.getLatestBlockhash()).blockhash;
      transaction.feePayer = ownerPublicKey;

      // Sign and send the transaction using the signer
      console.log('Sending transaction to register camera...');
      const result = await signer.signAndSendTransaction(transaction);
      const signature = result.signature;
      console.log('Transaction sent, signature:', signature);

      setTxResult(signature);
      setStatusMessage(`Camera registered! Transaction: ${signature}`);
      setStatusType('success');

      // After successful registration, immediately fetch cameras
      await fetchRegisteredCameras();

      // Clear form
      setCameraName('');
      setCameraModel('');
    } catch (err) {
      console.error('Error registering camera:', err);

      // Provide more detailed error message
      let errorMessage = 'Failed to register camera';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;

        // Check for common errors
        if (err.message.includes('This transaction has already been processed')) {
          errorMessage += '. You may have already registered a camera with this name.';
        } else if (err.message.includes('insufficient funds')) {
          errorMessage += '. Please make sure you have enough SOL to cover the transaction.';
        } else if (err.message.includes('blockhash')) {
          errorMessage += '. Network might be congested, please try again.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Record activity

  // Add a specialized function for taking pictures without using activity keypair
  const takePicture = async (camera: CameraData) => {
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
      setStatusMessage(`Taking picture with camera ${camera.metadata.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      const ownerPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.publicKey);

      // Create metadata with timestamp
      const metadata = JSON.stringify({
        timestamp: new Date().toISOString(),
        action: 'photo_capture',
        userAddress: primaryWallet.address
      });

      console.log('Using camera:', cameraPublicKey.toString());
      console.log('Using metadata:', metadata);

      // Get the connection and signer
      const connection = await primaryWallet.getConnection();
      const signer = await primaryWallet.getSigner();

      // Create instruction
      const ix = await program.methods
        .recordActivity({
          activityType: { photoCapture: {} },
          metadata
        })
        .accounts({
          owner: ownerPublicKey,
          camera: cameraPublicKey,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Create and send transaction
      const transaction = new Transaction().add(ix);
      transaction.recentBlockhash = (await connection.getLatestBlockhash()).blockhash;
      transaction.feePayer = ownerPublicKey;

      const result = await signer.signAndSendTransaction(transaction);
      const signature = result.signature;

      console.log('Picture taken successfully, signature:', signature);
      setTxResult(signature);
      setStatusMessage(`Picture taken successfully! Transaction: ${signature}`);
      setStatusType('success');

      // Refresh cameras to see updated activity count
      await fetchRegisteredCameras();
    } catch (err) {
      console.error('Error taking picture:', err);

      let errorMessage = 'Failed to take picture';
      if (err instanceof Error) {
        errorMessage += ': ' + err.message;

        // Special error messages
        if (err.message.includes('CameraInactive')) {
          errorMessage = 'Camera is currently inactive. Please activate it first.';
        } else if (err.message.includes('Unauthorized')) {
          errorMessage = 'You are not authorized to use this camera.';
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
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
      let registryAccount;
      try {
        registryAccount = await program.account.cameraRegistry.fetch(registryAddress);
        console.log('Registry account successfully fetched:', registryAccount);
      } catch (err) {
        console.warn('Could not fetch registry account, it may not be initialized yet:', err);
      }

      // Try to fetch raw camera accounts without relying on specific structure
      console.log('Attempting to fetch camera accounts from program ID:', CAMERA_ACTIVATION_PROGRAM_ID.toString());

      // Get all program accounts without filtering by type - this way we can see what's available
      const allAccounts = await connection.getProgramAccounts(CAMERA_ACTIVATION_PROGRAM_ID);
      console.log('Found', allAccounts.length, 'program accounts of any type');

      // Try to properly parse the camera accounts using the IDL
      const cameras: CameraData[] = [];

      // Log the IDL structure to debug
      console.log('IDL structure for cameraAccount:',
        program.idl.accounts.find(a => a.name === 'cameraAccount')
      );

      try {
        console.log('Fetching all camera accounts directly with program...');
        const cameraAccounts = await program.account.cameraAccount.all();
        console.log('Camera accounts successfully fetched:', cameraAccounts.length);

        if (cameraAccounts.length === 0) {
          console.log('No camera accounts found with Anchor, trying manual approach');
          throw new Error('No camera accounts found with Anchor');
        }

        // Format each camera account data, handling potential format differences
        for (const account of cameraAccounts) {
          try {
            const data = account.account;
            console.log('Raw camera account data:', {
              publicKey: account.publicKey.toString(),
              data: JSON.stringify(data, (_, value) => {
                if (typeof value === 'object' && value !== null && 'toNumber' in value) {
                  return value.toNumber();
                }
                return value;
              }, 2)
            });

            const camera: CameraData = {
              publicKey: account.publicKey.toString(),
              owner: data.owner?.toString() || 'unknown',
              isActive: !!data.isActive, // Convert to boolean explicitly
              metadata: {
                name: data.metadata?.name || 'Unnamed Camera',
                model: data.metadata?.model || 'Unknown Model',
                registrationDate: data.metadata?.registrationDate?.toNumber() || 0,
                lastActivity: data.metadata?.lastActivity?.toNumber() || 0,
                location: null
              }
            };

            // Handle optional new fields that might not be in old accounts
            if (data.activityCounter !== undefined) {
              try {
                camera.activityCounter = data.activityCounter.toNumber();
              } catch (e) {
                console.warn('Could not convert activityCounter to number:', e);
              }
            }

            if (data.lastActivityType) {
              camera.lastActivityType = data.lastActivityType;
            }

            cameras.push(camera);
            console.log('Successfully parsed camera account:', camera.publicKey);
          } catch (error) {
            console.error('Error parsing camera account:', error);
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
              );

              // If successful, we found a camera account
              console.log('Successfully decoded as camera account:', decodedAccount);

              // Create a CameraData object from the decoded account
              const camera: CameraData = {
                publicKey: account.pubkey.toString(),
                owner: decodedAccount.owner.toString(),
                isActive: !!decodedAccount.isActive,
                metadata: {
                  name: decodedAccount.metadata.name,
                  model: decodedAccount.metadata.model,
                  registrationDate: decodedAccount.metadata.registrationDate.toNumber(),
                  lastActivity: decodedAccount.metadata.lastActivity.toNumber(),
                  location: null
                }
              };

              if (decodedAccount.activityCounter) {
                camera.activityCounter = decodedAccount.activityCounter.toNumber();
              }

              if (decodedAccount.lastActivityType) {
                camera.lastActivityType = decodedAccount.lastActivityType;
              }

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
                metadata: {
                  name: 'Camera (data format changed)',
                  model: 'Unknown Model',
                  registrationDate: 0,
                  lastActivity: 0,
                  location: null
                }
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

  // Add a helper function to get camera status display
  const getCameraStatusDisplay = (camera: CameraData) => {
    const isOldFormat = camera.owner === 'unknown (data format changed)';

    // If it's an old format, we don't know the status
    if (isOldFormat) {
      return (
        <span className="text-yellow-600 bg-yellow-100 px-2 py-0.5 rounded text-xs">
          Unknown Format
        </span>
      );
    }

    return (
      <span className={`text-${camera.isActive ? 'green' : 'red'}-600 bg-${camera.isActive ? 'green' : 'red'}-100 px-2 py-0.5 rounded text-xs`}>
        {camera.isActive ? 'Active' : 'Inactive'}
      </span>
    );
  };

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
      setStatusMessage(`Deregistering camera ${camera.metadata.name}...`);
      setStatusType('info');

      // Check if it's a Solana wallet
      if (!isSolanaWallet(primaryWallet)) {
        throw new Error('This is not a Solana wallet');
      }

      // Get the connection from the wallet
      const walletConnection = await primaryWallet.getConnection();

      // Get the signer
      const signer = await primaryWallet.getSigner();

      const ownerPublicKey = new PublicKey(primaryWallet.address);

      // Get the camera public key
      const cameraPublicKey = new PublicKey(camera.publicKey);

      // Find the registry PDA
      const [registryAddress] = await PublicKey.findProgramAddress(
        [Buffer.from('camera-registry')],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Build the transaction instruction
      const ix = await program.methods
        .deregisterCamera()
        .accounts({
          owner: ownerPublicKey,
          camera: cameraPublicKey,
          registry: registryAddress,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Create a new transaction and add the instruction
      const transaction = new Transaction().add(ix);

      // Set recent blockhash and fee payer
      transaction.recentBlockhash = (await walletConnection.getLatestBlockhash()).blockhash;
      transaction.feePayer = ownerPublicKey;

      // Sign and send the transaction using the signer
      const result = await signer.signAndSendTransaction(transaction);
      const signature = result.signature;

      setTxResult(signature);
      setStatusMessage(`Camera deregistered! Transaction: ${signature}`);
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
        }
      }

      setError(errorMessage);
      setStatusMessage(errorMessage);
      setStatusType('error');
    } finally {
      setLoading(false);
    }
  };

  // Render the camera list
  const renderCameraList = () => {
    if (loadingCameras) {
      return <div className="text-center py-4">Loading cameras...</div>;
    }

    if (registeredCameras.length === 0) {
      return <div className="text-center py-4">No cameras registered yet</div>;
    }

    return (
      <div className="space-y-4">
        {registeredCameras.map((camera, index) => {
          const isOldFormat = camera.owner === 'unknown (data format changed)';
          const isOwner = camera.owner === primaryWallet?.address;

          return (
            <div key={index} className="bg-white p-4 rounded-lg shadow">
              <div className="flex justify-between items-start">
                <div>
                  <div className="flex items-center">
                    <h3 className="font-bold">{camera.metadata.name || 'Unknown Camera'}</h3>
                    {isOldFormat && (
                      <span className="ml-2 bg-yellow-100 text-yellow-800 text-xs px-2 py-0.5 rounded">Old Format</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">Model: {camera.metadata.model || 'Unknown'}</p>
                  <p className="text-sm text-gray-600">Public Key: <span className="font-mono">{camera.publicKey}</span></p>
                  <p className="text-sm text-gray-600 mt-1">
                    Status: {getCameraStatusDisplay(camera)}
                  </p>
                  <p className="text-sm text-gray-600">Owner: {
                    camera.owner === 'unknown (data format changed)'
                      ? 'Unknown (old format)'
                      : `${camera.owner.slice(0, 8)}...${camera.owner.slice(-8)}`
                  }</p>

                  {!isOldFormat && camera.activityCounter !== undefined && (
                    <p className="text-sm text-gray-600">
                      Activity Counter: <span className="font-medium">{camera.activityCounter}</span>
                    </p>
                  )}

                  {!isOldFormat && camera.lastActivityType && (
                    <p className="text-sm text-gray-600">
                      Last Activity: <span className="font-medium">{Object.keys(camera.lastActivityType)[0]}</span>
                    </p>
                  )}
                </div>

                <div className="flex flex-col items-end space-y-2">
                  <div className={`w-3 h-3 rounded-full ${isOldFormat
                      ? 'bg-yellow-500'
                      : (camera.isActive ? 'bg-green-500' : 'bg-red-500')
                    }`} />

                  {/* Owner actions */}
                  {isOwner && !isOldFormat && (
                    <div className="flex flex-col space-y-2">
                      <button
                        onClick={() => deregisterCamera(camera)}
                        disabled={loading}
                        className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-sm"
                      >
                        {loading ? 'Processing...' : 'Deregister Camera'}
                      </button>
                    </div>
                  )}

                  {/* Camera control buttons - available to all users */}
                  {!isOldFormat && camera.isActive && (
                    <button
                      onClick={() => takePicture(camera)}
                      disabled={loading}
                      className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-sm flex items-center gap-1"
                    >
                      {loading ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing...
                        </>
                      ) : (
                        <>
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                          Take Picture
                        </>
                      )}
                    </button>
                  )}

                  <button
                    onClick={() => {
                      // Set the selected camera in the dropdown for NFC URL generation
                      const dropdown = document.getElementById('nfc-camera-select') as HTMLSelectElement;
                      if (dropdown) {
                        dropdown.value = camera.publicKey;

                        // Trigger the NFC URL generation
                        const generateButton = document.querySelector('button[disabled]') as HTMLButtonElement;
                        if (generateButton && !generateButton.disabled) {
                          generateButton.click();
                        }
                      }
                    }}
                    className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                  >
                    Use for NFC
                  </button>

                  <div className="flex flex-col space-y-2 mb-4">
                    {/* Direct path to camera view - no redirection */}
                    <a
                      href={`/app/camera/${camera.publicKey}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                    >
                      Open Camera
                    </a>
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
            Transaction successful: {txResult}
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

            <button
              onClick={fetchRegisteredCameras}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 mb-4"
              disabled={loading || !initialized}
            >
              {loadingCameras ? 'Loading...' : 'Refresh camera list'}
            </button>
          </div>
        </div>
      </div>

      {/* Camera Management */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
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
                    {camera.metadata.name || 'Unnamed Camera'} ({camera.publicKey.slice(0, 8)}...)
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