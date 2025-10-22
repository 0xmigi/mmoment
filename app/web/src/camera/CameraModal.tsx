/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, Transaction, SystemProgram } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../anchor/idl';
import { CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { timelineService } from '../timeline/timeline-service';
import { unifiedCameraService } from './unified-camera-service';
import { useSocialProfile } from '../auth/social/useSocialProfile';
import { CONFIG } from '../core/config';
import { buildAndSponsorTransaction } from '../services/gas-sponsorship';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCheckStatusChange?: (isCheckedIn: boolean) => void;
  camera: {
    id: string;
    owner: string;
    ownerDisplayName?: string;
    ownerPfpUrl?: string;
    isLive: boolean;
    isStreaming: boolean;
    status: 'ok' | 'error' | 'offline';
    lastSeen?: number;
    // activityCounter?: number; // Replaced with live active users analytics
    model?: string;
    // New properties for development info
    showDevInfo?: boolean;
    defaultDevCamera?: string;
  };
}

export function CameraModal({ isOpen, onClose, onCheckStatusChange, camera }: CameraModalProps) {
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const { primaryProfile } = useSocialProfile();
  const [isCheckedIn, setIsCheckedIn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Configuration states for Jetson camera features
  const [gestureVisualization, setGestureVisualization] = useState(false);
  const [faceVisualization, setFaceVisualization] = useState(false);
  const [gestureControls, setGestureControls] = useState(false);
  const [configLoading, setConfigLoading] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<{ gesture: string; confidence: number } | null>(null);

  // State for active users analytics
  const [activeUsersCount, setActiveUsersCount] = useState<number>(0);
  const [loadingActiveUsers, setLoadingActiveUsers] = useState(false);

  // Check if current user is the owner
  const isOwner = primaryWallet?.address === camera.owner;

  // Check if this is a Jetson camera (has advanced features)
  const isJetsonCamera = camera.id === CONFIG.JETSON_CAMERA_PDA || camera.model === 'jetson' || camera.model === 'jetson_orin_nano';

  // Add more frequent status updates to the parent component
  useEffect(() => {
    if (onCheckStatusChange) {
      console.log("[CameraModal] Notifying parent of check-in status:", isCheckedIn);
      onCheckStatusChange(isCheckedIn);
    }
  }, [isCheckedIn, onCheckStatusChange]);

  // Add a more frequent check for session status and active users
  useEffect(() => {
    if (!isOpen || !camera.id) return;

    console.log("[CameraModal] Checking session status and active users on open");

    // Only check session status if wallet is connected
    if (primaryWallet?.address) {
      checkSessionStatus();
    }

    // Always fetch active users (doesn't need wallet)
    fetchActiveUsersForCamera();

    // Also set up a periodic check while the modal is open
    const intervalId = setInterval(() => {
      console.log("[CameraModal] Periodic session status and active users check");
      if (primaryWallet?.address) {
        checkSessionStatus();
      }
      fetchActiveUsersForCamera();
    }, 3000); // Check every 3 seconds

    return () => {
      console.log("[CameraModal] Cleaning up session status and active users check");
      clearInterval(intervalId);
    };
  }, [isOpen, camera.id, primaryWallet]);

  // Clear errors and load configuration when modal opens
  useEffect(() => {
    const loadConfiguration = async () => {
      if (isOpen) {
        setError(null);
        
        // Load current configuration for Jetson cameras
        if (isJetsonCamera && camera.id) {
          try {
            console.log('[CameraModal] Loading current computer vision state...');
            
            // Load gesture controls state from unified service
            const gestureControlsEnabled = await unifiedCameraService.getGestureControlsStatus(camera.id);
            setGestureControls(gestureControlsEnabled);
            console.log('[CameraModal] Gesture controls state:', gestureControlsEnabled);
            
            // Load visualization states from localStorage (persist across modal opens)
            const storedGestureViz = localStorage.getItem(`jetson_gesture_viz_${camera.id}`) === 'true';
            const storedFaceViz = localStorage.getItem(`jetson_face_viz_${camera.id}`) === 'true';
            
            setGestureVisualization(storedGestureViz);
            setFaceVisualization(storedFaceViz);
            
            console.log('[CameraModal] Loaded visualization states - Gesture:', storedGestureViz, 'Face:', storedFaceViz);
            
            console.log('[CameraModal] Computer vision configuration loaded successfully');
          } catch (error) {
            console.error('Error loading computer vision configuration:', error);
            // Set defaults on error
            setGestureVisualization(false);
            setFaceVisualization(false);
            setGestureControls(false);
          }
        } else {
          // Reset states for non-Jetson cameras
          setGestureVisualization(false);
          setFaceVisualization(false);
          setGestureControls(false);
        }
      }
    };

    loadConfiguration();
  }, [isOpen, isJetsonCamera, camera.id]);

  // Expose test function to window for debugging
  useEffect(() => {
    if (isJetsonCamera && camera.id) {
      (window as any).testVisualizationEndpoints = testVisualizationEndpoints;
    }
    
    return () => {
      delete (window as any).testVisualizationEndpoints;
    };
  }, [isJetsonCamera, camera.id]);

  // Poll current gesture when modal is open and gesture visualization is enabled
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (isOpen && isJetsonCamera && camera.id && (gestureVisualization || gestureControls)) {
      const pollGesture = async () => {
        try {
          const result = await unifiedCameraService.getCurrentGesture(camera.id);
          if (result.success && result.data) {
            setCurrentGesture({
              gesture: result.data.gesture || 'none',
              confidence: result.data.confidence || 0
            });
          } else {
            setCurrentGesture(null);
          }
        } catch (error) {
          console.error('Error polling gesture:', error);
          setCurrentGesture(null);
        }
      };

      // Poll immediately and then every 2 seconds
      pollGesture();
      intervalId = setInterval(pollGesture, 2000);
    } else {
      setCurrentGesture(null);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isOpen, isJetsonCamera, camera.id, gestureVisualization, gestureControls]);

  const checkSessionStatus = async () => {
    if (!camera.id || !primaryWallet?.address || !connection) return;
    
    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Check if session account exists (use getAccountInfo to avoid decode errors from old sessions)
      try {
        const sessionAccountInfo = await connection.getAccountInfo(sessionPda);
        if (sessionAccountInfo && sessionAccountInfo.data.length === 102) {
          // New session structure (102 bytes) - try to decode
          try {
            await program.account.userSession.fetch(sessionPda);
            console.log("[CameraModal] Session exists, setting checked-in: true");
            setIsCheckedIn(true);
          } catch (decodeErr) {
            console.log("[CameraModal] Session exists but can't decode, setting checked-in: false");
            setIsCheckedIn(false);
          }
        } else if (sessionAccountInfo) {
          // Old session structure - consider not checked in
          console.log("[CameraModal] Old session structure detected, setting checked-in: false");
          setIsCheckedIn(false);
        } else {
          console.log("[CameraModal] No session found, setting checked-in: false");
          setIsCheckedIn(false);
        }
      } catch (err) {
        console.log("[CameraModal] Error checking session, setting checked-in: false");
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error('[CameraModal] Error checking session status:', err);
      setIsCheckedIn(false);
    }
  };

  // Function to fetch active users count for this specific camera
  const fetchActiveUsersForCamera = async () => {
    if (!camera.id || !connection) return;

    try {
      setLoadingActiveUsers(true);
      console.log('[CameraModal] Fetching active users for camera:', camera.id);

      // Create a read-only program instance (no wallet needed for reading data)
      const provider = new AnchorProvider(
        connection,
        {} as any, // No wallet needed for read-only operations
        { commitment: 'confirmed' }
      );
      const program = new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);

      // Fetch only NEW session accounts (102 bytes) to avoid decode errors from old sessions (94 bytes)
      const sessionAccounts = await connection.getProgramAccounts(program.programId, {
        filters: [
          { dataSize: 102 }  // New session structure size
        ]
      });

      console.log('[CameraModal] Found', sessionAccounts.length, 'new session accounts (102 bytes)');

      // Count sessions for this specific camera
      let count = 0;
      for (const accountInfo of sessionAccounts) {
        try {
          const session = program.coder.accounts.decode('userSession', accountInfo.account.data);
          const sessionCameraKey = session.camera.toString();

          console.log('[CameraModal] Session decoded - Camera:', sessionCameraKey, 'Looking for:', camera.id);

          if (sessionCameraKey === camera.id) {
            count++;
            console.log('[CameraModal] ✅ Found active session for this camera');
          }
        } catch (error) {
          console.error('[CameraModal] ❌ Failed to decode session:', error);
          continue;
        }
      }
      
      setActiveUsersCount(count);
      console.log('[CameraModal] Active users count for this camera:', count);
    } catch (error) {
      console.error('[CameraModal] Error fetching active users:', error);
      // Don't reset count on error - keep showing last known state
    } finally {
      setLoadingActiveUsers(false);
    }
  };

  // Initialize program when needed
  const initializeProgram = async () => {
    if (!primaryWallet?.address || !connection) {
      throw new Error('Wallet or connection not available');
    }

    // Check if it's a Solana wallet
    if (!isSolanaWallet(primaryWallet)) {
      throw new Error('This is not a Solana wallet');
    }

    try {
      // Create a provider using the connection and wallet
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          // We'll handle signing separately
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );

      // Create the program
      return new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);
    } catch (error) {
      console.error('Error initializing program:', error);
      throw new Error('Failed to initialize Solana program');
    }
  };

  const handleCameraExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.id}?cluster=devnet`, '_blank');
  };

  const handleOwnerExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.owner}?cluster=devnet`, '_blank');
  };

  // Format address for display
  const formatAddress = (address: string, start = 6, end = 6) => {
    if (!address) return '';
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  };

  const handleDevCameraClick = () => {
    if (camera.defaultDevCamera) {
      // Redirect to the correct camera page URL
      const baseUrl = window.location.origin;
      window.location.href = `${baseUrl}/app/camera/${camera.defaultDevCamera}`;
    }
  };

  // Add check-in event to timeline
  const addCheckInEvent = (transactionId: string) => {
    if (primaryWallet?.address) {
      timelineService.emitEvent({
        type: 'check_in',
        user: {
          address: primaryWallet.address,
          displayName: primaryProfile?.displayName,
          username: primaryProfile?.username
        },
        timestamp: Date.now(),
        transactionId: transactionId,
        cameraId: camera.id
      });
    }
  };

  // Add check-out event to timeline
  const addCheckOutEvent = (transactionId: string) => {
    if (primaryWallet?.address) {
      timelineService.emitEvent({
        type: 'check_out',
        user: {
          address: primaryWallet.address
        },
        timestamp: Date.now(),
        transactionId: transactionId,
        cameraId: camera.id
      });
    }
  };

  // Test visualization endpoints directly
  const testVisualizationEndpoints = async () => {
    if (!isJetsonCamera || !camera.id) return;
    
    console.log('[CameraModal] Testing visualization endpoints...');
    
    try {
      // Test both visualization endpoints through unified camera service
      const currentCameraId = camera.id;
      
      console.log('Testing face visualization endpoint...');
      const faceResult = await unifiedCameraService.toggleFaceVisualization?.(currentCameraId, true);
      console.log('Face viz result:', faceResult);
      
      console.log('Testing gesture visualization endpoint...');
      const gestureResult = await unifiedCameraService.toggleGestureVisualization?.(currentCameraId, true);
      console.log('Gesture viz result:', gestureResult);
      

      
    } catch (error) {
      console.error('Error testing visualization endpoints:', error);
    }
  };

  // Handle gesture visualization toggle
  const handleGestureVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !gestureVisualization;
      console.log('[CameraModal] Toggling gesture visualization to:', newState);
      
      const result = await unifiedCameraService.toggleGestureVisualization(camera.id, newState);
      
      if (result.success) {
        setGestureVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_gesture_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Gesture visualization toggled successfully to:', newState);
        
        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle gesture visualization:', result.error);
        setError(result.error || 'Failed to toggle gesture visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling gesture visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle gesture visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle face visualization toggle
  const handleFaceVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !faceVisualization;
      console.log('[CameraModal] Toggling face visualization to:', newState);
      
      const result = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      
      if (result.success) {
        setFaceVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_face_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Face visualization toggled successfully to:', newState);
        
        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle face visualization:', result.error);
        setError(result.error || 'Failed to toggle face visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling face visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle face visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle gesture controls toggle
  const handleGestureControlsToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !gestureControls;
      const result = await unifiedCameraService.toggleGestureControls(camera.id, newState);
      
      if (result.success) {
        setGestureControls(newState);
      } else {
        setError(result.error || 'Failed to toggle gesture controls');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to toggle gesture controls');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle check-in
  const handleCheckIn = async () => {
    if (!camera.id || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Check if user already has an active session - if so, check out first
      try {
        const sessionAccountInfo = await connection.getAccountInfo(sessionPda);
        if (sessionAccountInfo) {
          console.log('[CameraModal] Existing session found, checking out first');

        // Check out existing session
        const checkOutIx = await program.methods
          .checkOut()
          .accounts({
            closer: userPublicKey,
            camera: cameraPublicKey,
            session: sessionPda,
            sessionUser: userPublicKey,
            rentDestination: userPublicKey, // Rent goes back to user
          })
          .instruction();

        const checkOutTx = new Transaction().add(checkOutIx);
        const { blockhash: checkOutBlockhash } = await connection.getLatestBlockhash();
        checkOutTx.recentBlockhash = checkOutBlockhash;
        checkOutTx.feePayer = userPublicKey;

        // Sign and send checkout transaction
        let signedCheckOutTx;
        if (typeof (primaryWallet as any).getSigner === 'function') {
          const signer = await (primaryWallet as any).getSigner();
          signedCheckOutTx = await signer.signTransaction(checkOutTx);
        } else {
          signedCheckOutTx = await (primaryWallet as any).signTransaction(checkOutTx);
        }

          const checkOutSig = await connection.sendRawTransaction(signedCheckOutTx.serialize());
          await connection.confirmTransaction(checkOutSig, 'confirmed');
          console.log('[CameraModal] Checked out existing session:', checkOutSig);
        }
      } catch (err) {
        // No existing session - this is fine, continue with check-in
        console.log('[CameraModal] No existing session found, proceeding with check-in');
      }

      // Derive recognition token PDA and check if it exists
      const [recognitionTokenPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('recognition-token'), userPublicKey.toBuffer()],
        program.programId
      );

      // Check if recognition token account exists
      let hasRecognitionToken = false;
      try {
        await program.account.recognitionToken.fetch(recognitionTokenPda);
        hasRecognitionToken = true;
        console.log('[CameraModal] Recognition token found for user');
      } catch (err) {
        console.log('[CameraModal] No recognition token found for user');
      }

      // Create the accounts object for check-in
      // NOTE: payer will be set to the fee payer (gas sponsor) in the transaction
      const accounts: any = {
        user: userPublicKey,
        payer: userPublicKey, // This will be overridden by the fee payer in gas sponsorship
        camera: cameraPublicKey,
        recognitionToken: hasRecognitionToken ? recognitionTokenPda : null, // Pass null for optional accounts that don't exist
        session: sessionPda,
        systemProgram: SystemProgram.programId
      };

      // Build check-in transaction function
      const buildCheckInTx = async () => {
        const ix = await program.methods
          .checkIn(false) // false = don't require face recognition
          .accounts(accounts)
          .instruction();

        return new Transaction().add(ix);
      };

      // Check user's SOL balance to decide whether to sponsor
      const balance = await connection.getBalance(userPublicKey);
      const solBalance = balance / 1e9;
      const MIN_SOL_FOR_SELF_PAY = 0.01; // 0.01 SOL minimum to pay own fees

      let signature: string;

      if (solBalance >= MIN_SOL_FOR_SELF_PAY) {
        // User has enough SOL - use regular transaction
        console.log(`💰 [CameraModal] User has ${solBalance.toFixed(4)} SOL, using regular check-in...`);
        const checkInTx = await buildCheckInTx();
        const { blockhash } = await connection.getLatestBlockhash();
        checkInTx.recentBlockhash = blockhash;
        checkInTx.feePayer = userPublicKey;

        const signer = await (primaryWallet as any).getSigner();
        const signedTx = await signer.signTransaction(checkInTx);
        signature = await connection.sendRawTransaction(signedTx.serialize());
        await connection.confirmTransaction(signature, 'confirmed');
        console.log('✅ [CameraModal] Regular check-in successful!', signature);
      } else {
        // Low balance - try gas sponsorship
        console.log(`🎉 [CameraModal] Low balance (${solBalance.toFixed(4)} SOL), attempting gas-sponsored check-in...`);
        const signer = await (primaryWallet as any).getSigner();
        const sponsorResult = await buildAndSponsorTransaction(
          userPublicKey,
          signer,
          buildCheckInTx,
          'check_in',
          connection
        );

        if (!sponsorResult.success) {
          if (sponsorResult.requiresUserPayment) {
            throw new Error('You\'ve used all 10 free interactions! Please add SOL to your wallet to continue.');
          } else {
            throw new Error(sponsorResult.error || 'Failed to sponsor transaction');
          }
        }

        signature = sponsorResult.signature!;
        console.log('✅ [CameraModal] Gas-sponsored check-in successful!', signature);
      }

      console.log('Check-in transaction confirmed successfully');

      // 🎉 NEW: Use unified check-in endpoint - no more race conditions!
      // This triggers immediate blockchain sync and recognition token loading
      console.log('🚀 [CameraModal] Calling unified check-in endpoint...');
      try {
        const checkinResult = await unifiedCameraService.checkin(camera.id, {
          wallet_address: primaryWallet.address,
          display_name: primaryProfile?.displayName,
          username: primaryProfile?.username,
          transaction_signature: signature
        });

        if (checkinResult.success) {
          console.log('✅ [CameraModal] Unified check-in successful!', checkinResult.data);
          console.log(`   Display name: ${checkinResult.data?.display_name}`);
          console.log(`   Session ID: ${checkinResult.data?.session_id}`);
        } else {
          console.warn('⚠️  [CameraModal] Unified check-in failed:', checkinResult.error);
          // Fall back to old method if unified check-in fails
          console.log('📤 [CameraModal] Falling back to separate profile send...');
          await unifiedCameraService.sendUserProfile(camera.id, {
            wallet_address: primaryWallet.address,
            display_name: primaryProfile?.displayName,
            username: primaryProfile?.username
          });
        }
      } catch (err) {
        console.error('❌ [CameraModal] Unified check-in error:', err);
        // Don't fail the overall check-in if this fails
      }

      setIsCheckedIn(true);

      // Add to timeline
      addCheckInEvent(signature);

      // Refresh the timeline
      timelineService.refreshEvents();

      // Refresh active users count
      await fetchActiveUsersForCamera();

      // Notify parent component
      if (onCheckStatusChange) {
        onCheckStatusChange(true);
      }
      
    } catch (error) {
      console.error('Check-in error:', error);
      
      if (error instanceof Error) {
        let errorMsg = error.message;
        
        // Check for common error messages and provide more user-friendly versions
        if (errorMsg.includes('custom program error: 0x64')) {
          errorMsg = 'Program error: The camera may not be configured correctly.';
        } else if (errorMsg.includes('insufficient funds')) {
          errorMsg = 'Insufficient SOL in your wallet. Please add more SOL and try again.';
        } else if (errorMsg.includes('already in use')) {
          errorMsg = 'You are already checked in to this camera.';
          setIsCheckedIn(true);
          return;
        } else if (errorMsg.length > 150) {
          // If the error message is too long, provide a shorter, more general message
          errorMsg = 'An error occurred during check-in. Please check the console for details.';
        }
        
        setError(errorMsg);
      } else {
        setError('Unknown error during check-in');
      }
    } finally {
      setLoading(false);
    }
  };

  // Handle check-out
  const handleCheckOut = async () => {
    if (!camera.id || !primaryWallet?.address) {
      setError('No camera or wallet available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const program = await initializeProgram();
      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(camera.id);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      // Fetch the session account to get the actual user
      const sessionAccount = await program.account.userSession.fetch(sessionPda) as any;
      console.log('[CameraModal] Session account data:', {
        user: sessionAccount.user.toString(),
        camera: sessionAccount.camera.toString(),
        checkInTime: new Date(sessionAccount.checkInTime.toNumber() * 1000).toISOString(),
      });

      // Build check-out transaction function
      const buildCheckOutTx = async () => {
        const ix = await program.methods
          .checkOut()
          .accounts({
            closer: userPublicKey,
            camera: cameraPublicKey,
            session: sessionPda,
            sessionUser: sessionAccount.user as PublicKey,  // Use actual session user
            rentDestination: userPublicKey, // Rent goes back to user on self-checkout
          })
          .instruction();

        return new Transaction().add(ix);
      };

      // Check user's SOL balance to decide whether to sponsor
      const balance = await connection.getBalance(userPublicKey);
      const solBalance = balance / 1e9;
      const MIN_SOL_FOR_SELF_PAY = 0.01; // 0.01 SOL minimum to pay own fees

      let signature: string;

      if (solBalance >= MIN_SOL_FOR_SELF_PAY) {
        // User has enough SOL - use regular transaction
        console.log(`💰 [CameraModal] User has ${solBalance.toFixed(4)} SOL, using regular check-out...`);
        const checkOutTx = await buildCheckOutTx();
        const { blockhash } = await connection.getLatestBlockhash();
        checkOutTx.recentBlockhash = blockhash;
        checkOutTx.feePayer = userPublicKey;

        const signer = await (primaryWallet as any).getSigner();
        const signedTx = await signer.signTransaction(checkOutTx);
        signature = await connection.sendRawTransaction(signedTx.serialize());
        await connection.confirmTransaction(signature, 'confirmed');
        console.log('✅ [CameraModal] Regular check-out successful!', signature);
      } else {
        // Low balance - try gas sponsorship
        console.log(`🎉 [CameraModal] Low balance (${solBalance.toFixed(4)} SOL), attempting gas-sponsored check-out...`);
        const signer = await (primaryWallet as any).getSigner();
        const sponsorResult = await buildAndSponsorTransaction(
          userPublicKey,
          signer,
          buildCheckOutTx,
          'check_out',
          connection
        );

        if (!sponsorResult.success) {
          if (sponsorResult.requiresUserPayment) {
            throw new Error('You\'ve used all 10 free interactions! Please add SOL to your wallet to continue.');
          } else {
            throw new Error(sponsorResult.error || 'Failed to sponsor transaction');
          }
        }

        signature = sponsorResult.signature!;
        console.log('✅ [CameraModal] Gas-sponsored check-out successful!', signature);
      }

      console.log('Check-out transaction confirmed successfully');

      // Remove user profile from camera after successful check-out
      try {
        const removeResult = await unifiedCameraService.removeUserProfile(camera.id, primaryWallet.address);
        if (removeResult.success) {
          console.log('[CameraModal] User profile removed successfully from camera');
        } else {
          console.warn('[CameraModal] Failed to remove user profile from camera:', removeResult.error);
        }
      } catch (err) {
        console.warn('[CameraModal] Failed to remove user profile from camera:', err);
        // Don't fail the check-out if this fails
      }

      setIsCheckedIn(false);

      // Add to timeline
      addCheckOutEvent(signature);

      // Refresh the timeline
      timelineService.refreshEvents();

      // Refresh active users count
      await fetchActiveUsersForCamera();

      // Notify parent component
      if (onCheckStatusChange) {
        onCheckStatusChange(false);
      }
      
    } catch (error) {
      console.error('Check-out error:', error);
      
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('Unknown error during check-out');
      }
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Camera Details
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Camera Content */}
          <div className="p-4">
            {camera.showDevInfo ? (
              // Development section when no camera is connected
              <div className="space-y-4">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
                  <h3 className="text-sm font-medium text-yellow-800 mb-1">Development Mode</h3>
                  <p className="text-xs text-yellow-700 mb-3">
                    No camera is currently connected. Connect to a physical camera below:
                  </p>
                  <div className="space-y-2">
                    <button
                      onClick={handleDevCameraClick}
                      className="text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Pi5 <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                    <button
                      onClick={() => {
                        const baseUrl = window.location.origin;
                        window.location.href = `${baseUrl}/app/camera/${CONFIG.JETSON_CAMERA_PDA}`;
                      }}
                      className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Orin Nano <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              // Original camera details layout
              <>
                {/* Camera PDA */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-50 flex-shrink-0 flex items-center justify-center overflow-hidden">
                    <Camera className="w-5 h-5 text-blue-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">{formatAddress(camera.id)}</div>
                  </div>
                  <button
                    onClick={handleCameraExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Owner */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-gray-100 flex-shrink-0 flex items-center justify-center overflow-hidden ml-0">
                    <User className="w-5 h-5 text-gray-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700 flex items-center">
                      Owner
                      {isOwner && <span className="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">you</span>}
                    </div>
                    <div className="text-sm font-medium">
                      {formatAddress(camera.owner, 9, 5)}
                    </div>
                  </div>
                  <button
                    onClick={handleOwnerExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Camera Name */}
                {camera.ownerDisplayName && (
                  <div className="mb-4">
                    <div className="text-sm text-gray-700">Camera Name</div>
                    <div className="text-sm">{camera.ownerDisplayName}</div>
                  </div>
                )}

                {/* Type */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Type</div>
                  <div className="text-sm">{camera.model || "pi5"}</div>
                </div>

                {/* Active Users Analytics */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700 flex items-center">
                    Users checked in
                    {loadingActiveUsers && (
                      <div className="ml-2 w-3 h-3 border border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    )}
                  </div>
                  <div className="flex items-center mt-1">
                    <span className={`w-2 h-2 rounded-full mr-2 ${
                      activeUsersCount > 0 ? 'bg-green-500' : 'bg-gray-400'
                    }`}></span>
                    <span className={`text-sm font-medium ${
                      activeUsersCount > 0 ? 'text-green-600' : 'text-gray-500'
                    }`}>
                      {activeUsersCount} {activeUsersCount === 1 ? 'user' : 'users'} currently active
                    </span>
                  </div>
                </div>

                {/* Computer Vision Controls - Only for Jetson cameras */}
                {isJetsonCamera && (
                  <div className="mb-4 pt-4 border-t border-gray-200">
                    <div className="space-y-2">
                      {/* Face Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Face Detection Overlay</div>
                          <div className="text-xs text-gray-500">Shows face detection boxes</div>
                        </div>
                        <button
                          onClick={handleFaceVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            faceVisualization 
                              ? 'bg-blue-600 hover:bg-blue-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              faceVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Gesture Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Gesture Detection Overlay</div>
                          <div className="text-xs text-gray-500">Shows hand gesture tracking</div>
                        </div>
                        <button
                          onClick={handleGestureVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            gestureVisualization 
                              ? 'bg-blue-600 hover:bg-blue-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              gestureVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Gesture Controls Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Gesture Photo/Video Capture</div>
                          <div className="text-xs text-gray-500">Peace sign = photo, thumbs up = video</div>
                        </div>
                        <button
                          onClick={handleGestureControlsToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 ${
                            gestureControls 
                              ? 'bg-green-600 hover:bg-green-700' 
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              gestureControls ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Current Gesture Status - Only when gesture features are enabled */}
                      {(gestureVisualization || gestureControls) && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <div className="text-xs text-gray-500 mb-1">Current Gesture Detected</div>
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-medium">
                              {currentGesture ? (
                                <span className="capitalize">
                                  {currentGesture.gesture === 'none' ? 'No gesture detected' : currentGesture.gesture.replace('_', ' ')}
                                </span>
                              ) : (
                                <span className="text-gray-400">Loading...</span>
                              )}
                            </div>
                            {currentGesture && currentGesture.gesture !== 'none' && (
                              <div className="text-xs text-gray-500">
                                {Math.round(currentGesture.confidence * 100)}% confidence
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <p className="text-xs text-red-700 mt-1">{error}</p>
                  </div>
                )}

                {/* Check In/Out Button */}
                {isCheckedIn ? (
                  <button
                    onClick={handleCheckOut}
                    disabled={loading}
                    className="w-full bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {loading ? 'Processing...' : 'Check Out'}
                  </button>
                ) : (
                  <button
                    onClick={handleCheckIn}
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {loading ? 'Processing...' : 'Check In'}
                  </button>
                )}
              </>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}