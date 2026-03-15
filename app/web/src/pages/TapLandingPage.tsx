import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { X, Camera, ExternalLink, User, MapPin } from 'lucide-react';
import Logo from '../ui/common/Logo';
import { CONFIG } from '../core/config';
import { useCamera, fetchCameraByPublicKey } from '../camera/CameraProvider';
import { useConnection } from '@solana/wallet-adapter-react';
import { WebRTCStreamPlayer } from '../media/WebRTCStreamPlayer';
import { Timeline } from '../timeline/Timeline';

// ------------------------------------------------------------------ //
//  Location name overrides for demo / named venues                    //
// ------------------------------------------------------------------ //

const CAMERA_LOCATION_NAMES: Record<string, string> = {
  'ArQxL9kzhZ8QhJtNodnuMvkd3HGdkwSsTzbD4qD9QqKv': 'Foundry 3.3',
};

// ------------------------------------------------------------------ //
//  Types                                                               //
// ------------------------------------------------------------------ //

type TapState =
  | 'verifying'       // calling /api/nfc/verify-tap
  | 'verified'        // valid tap, stream loading
  | 'already_valid'   // had a stored presence token that's still good
  | 'failed'          // invalid tap / network error
  | 'no_nfc_params';  // arrived without picc_data & cmac (direct URL access, not from a tap)

// ------------------------------------------------------------------ //
//  Presence token helpers (sessionStorage, scoped per camera)          //
// ------------------------------------------------------------------ //

const STORAGE_KEY = (cameraPda: string) => `mmoment_presence_${cameraPda}`;

function storePresenceToken(cameraPda: string, token: string, expiresInSeconds: number) {
  const expires = Date.now() + expiresInSeconds * 1000;
  sessionStorage.setItem(STORAGE_KEY(cameraPda), JSON.stringify({ token, expires }));
}

function loadPresenceToken(cameraPda: string): string | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY(cameraPda));
    if (!raw) return null;
    const { token, expires } = JSON.parse(raw);
    if (Date.now() > expires) {
      sessionStorage.removeItem(STORAGE_KEY(cameraPda));
      return null;
    }
    return token;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ //
//  Component                                                           //
// ------------------------------------------------------------------ //

export default function TapLandingPage() {
  const { cameraPda } = useParams<{ cameraPda: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { primaryWallet } = useDynamicContext();
  const { setSelectedCamera, isCheckedIn, selectedCamera } = useCamera();
  const { connection } = useConnection();

  const [tapState, setTapState] = useState<TapState>('verifying');
  const [presenceToken, setPresenceToken] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [showBottomSheet, setShowBottomSheet] = useState(false);
  const [showCameraInfo, setShowCameraInfo] = useState(false);
  const [activeCount, setActiveCount] = useState<number>(0);
  const activeCountInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // NFC SUN params from the tag URL (NFC Developer App uses picc_data/enc/cmac)
  const piccData = searchParams.get('picc_data') || '';
  const encFileData = searchParams.get('enc') || '';
  const cmac = searchParams.get('cmac') || '';

  const cameraApiUrl = cameraPda ? CONFIG.getCameraApiUrlByPda(cameraPda) : null;

  // Derive a display name — use hardcoded venue name for demo, fallback to on-chain name
  const locationName = (cameraPda && CAMERA_LOCATION_NAMES[cameraPda]) || selectedCamera?.metadata?.name || null;

  // ---------------------------------------------------------------- //
  //  Load camera into context so WebRTCStreamPlayer works             //
  //  Also store directCameraId in localStorage — WebRTCStreamPlayer   //
  //  uses this as a fallback when the route param isn't "cameraId"    //
  //  (tap route uses "cameraPda", not "cameraId")                     //
  // ---------------------------------------------------------------- //
  useEffect(() => {
    if (!cameraPda) return;
    // Critical: WebRTCStreamPlayer reads this as fallback for currentCameraId
    localStorage.setItem('directCameraId', cameraPda);
    // Use fetchCameraByPublicKey directly — it creates its own dummy wallet
    // so it works for anonymous users (fetchCameraById requires program/auth)
    fetchCameraByPublicKey(cameraPda, connection).then((camera) => {
      if (camera) setSelectedCamera(camera);
    });
  }, [cameraPda, connection]);

  // ---------------------------------------------------------------- //
  //  Fetch active session count from Jetson (public, no auth needed)  //
  // ---------------------------------------------------------------- //
  useEffect(() => {
    if (!cameraApiUrl) return;

    const fetchCount = async () => {
      try {
        const res = await fetch(`${cameraApiUrl}/api/status?_t=${Date.now()}`);
        if (res.ok) {
          const data = await res.json();
          // Jetson nests under data.data or at top level — match jetson-camera.ts parsing
          const count = data.data?.activeSessionCount ?? data.activeSessionCount ?? data.active_sessions ?? 0;
          setActiveCount(count);
        }
      } catch { /* camera unreachable — leave count as-is */ }
    };

    fetchCount();
    activeCountInterval.current = setInterval(fetchCount, 15000);
    return () => {
      if (activeCountInterval.current) clearInterval(activeCountInterval.current);
    };
  }, [cameraApiUrl]);

  // ---------------------------------------------------------------- //
  //  NFC SUN verification                                             //
  // ---------------------------------------------------------------- //
  const isDevBypass = searchParams.get('dev') === 'true' && window.location.hostname === 'localhost';

  const verifyTap = useCallback(async () => {
    if (!cameraPda || !cameraApiUrl) return;

    // Dev bypass — skip NFC verification on localhost with ?dev=true
    if (isDevBypass) {
      setPresenceToken('dev-bypass');
      setTapState('verified');
      setShowBottomSheet(true);
      return;
    }

    // 1. Check for a still-valid stored presence token
    const stored = loadPresenceToken(cameraPda);
    if (stored) {
      setPresenceToken(stored);
      setTapState('already_valid');
      setShowBottomSheet(true);
      return;
    }

    // 2. No stored token — check we have NFC params
    if (!piccData || !cmac) {
      setTapState('no_nfc_params');
      return;
    }

    // 3. Call the camera service verify-tap endpoint
    setTapState('verifying');
    try {
      const res = await fetch(`${cameraApiUrl}/api/nfc/verify-tap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ picc_data: piccData, enc: encFileData, cmac }),
      });

      const data = await res.json();

      if (res.ok && data.valid) {
        storePresenceToken(cameraPda, data.presence_token, data.expires_in ?? 28800);
        setPresenceToken(data.presence_token);
        setTapState('verified');
        setTimeout(() => setShowBottomSheet(true), 3000);
      } else {
        setErrorMsg(data.error || 'Tap verification failed.');
        setTapState('failed');
      }
    } catch (err) {
      setErrorMsg('Could not reach the camera. Are you near the device?');
      setTapState('failed');
    }
  }, [cameraPda, cameraApiUrl, piccData, cmac]);

  useEffect(() => {
    verifyTap();
  }, [verifyTap]);

  // ---------------------------------------------------------------- //
  //  Auto-redirect: logged-in user with verified tap → camera page    //
  //  This fires when returning from /login after creating an account  //
  // ---------------------------------------------------------------- //
  useEffect(() => {
    if (!cameraPda || !primaryWallet) return;
    const tapVerified = tapState === 'verified' || tapState === 'already_valid';
    if (!tapVerified) return;

    // Auto-navigate to camera page with fromTap flag for auto check-in
    navigate(`/app/camera/${cameraPda}`, {
      state: { presenceToken, fromTap: true },
      replace: true, // replace tap page in history so back doesn't loop
    });
  }, [cameraPda, primaryWallet, tapState]);

  // ---------------------------------------------------------------- //
  //  Route to check-in if user is logged in and already verified       //
  // ---------------------------------------------------------------- //
  const handleCheckIn = () => {
    if (!cameraPda) return;
    navigate(`/app/camera/${cameraPda}`, { state: { presenceToken, fromTap: true } });
  };

  // ---------------------------------------------------------------- //
  //  Render — blocked states                                          //
  // ---------------------------------------------------------------- //
  if (!cameraPda) {
    return <div className="flex items-center justify-center h-screen text-white bg-black">Invalid tap link.</div>;
  }

  // Blocked — arrived without a tap
  if (tapState === 'no_nfc_params') {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-black text-white gap-6 px-6">
        <Logo width={48} height={33} className="text-white" />
        <h1 className="text-xl font-semibold text-center">Physical tap required</h1>
        <p className="text-sm text-gray-400 text-center max-w-xs">
          This camera is only accessible by tapping the NFC tag at the physical location.
        </p>
      </div>
    );
  }

  // Failed verification
  if (tapState === 'failed') {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-black text-white gap-6 px-6">
        <Logo width={48} height={33} className="text-white" />
        <h1 className="text-xl font-semibold text-center">Tap not recognised</h1>
        <p className="text-sm text-gray-400 text-center max-w-xs">{errorMsg}</p>
        <button
          onClick={verifyTap}
          className="px-5 py-2.5 rounded-lg bg-white text-black text-sm font-medium hover:bg-gray-100 transition-colors"
        >
          Try again
        </button>
      </div>
    );
  }

  const streamLoaded = tapState === 'verified' || tapState === 'already_valid';

  // ---------------------------------------------------------------- //
  //  Render — main tap experience                                     //
  // ---------------------------------------------------------------- //
  return (
    <div className="relative w-full h-[100dvh] bg-black overflow-hidden">

      {/* ---- Live stream (full-screen background) ---- */}
      {/* Override WebRTCStreamPlayer's inner px-2, aspect-ratio, rounded-lg, and object-contain
          so the video fills the entire viewport edge-to-edge */}
      <style>{`
        .tap-stream-fullscreen > div { padding: 0 !important; }
        .tap-stream-fullscreen > div > div {
          aspect-ratio: unset !important;
          border-radius: 0 !important;
          position: absolute !important;
          inset: 0 !important;
          width: 100% !important;
          height: 100% !important;
        }
        .tap-stream-fullscreen video {
          object-fit: cover !important;
        }
      `}</style>
      {streamLoaded ? (
        // Show live stream for ALL users (stream is not behind auth)
        <div className="absolute inset-0 tap-stream-fullscreen">
          <WebRTCStreamPlayer streamType="clean" />
        </div>
      ) : (
        // Still verifying
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* ---- Top bar — Logo left, camera status right ---- */}
      {streamLoaded && (
        <div className="absolute top-0 left-0 right-0 z-50">
          <div className="px-4 h-16 flex items-center justify-between">
            {/* Left — Logo + brand */}
            <div
              onClick={() => navigate('/')}
              className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            >
              <Logo width={30} height={21} className="text-white" />
              <h1 className="text-2xl text-white font-bold">Moment</h1>
            </div>

            {/* Right — Camera status + location */}
            <button
              type="button"
              onClick={() => setShowCameraInfo(true)}
              className="flex items-center gap-2 cursor-pointer"
            >
              <div className="flex items-center bg-black/70 backdrop-blur-sm rounded overflow-hidden">
                <div className="bg-green-500 text-white text-xs font-bold px-1.5 py-0.5">
                  ONLINE
                </div>
                {locationName && (
                  <div className="text-white text-xs px-1.5 py-0.5 border-l border-white/20 flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    {locationName}
                  </div>
                )}
              </div>
            </button>
          </div>

          {/* People count — right-aligned, tight below the top bar */}
          {activeCount > 0 && (
            <div className="px-4 -mt-2 flex justify-end">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 bg-green-400 rounded-full" />
                <span className="text-white text-sm">
                  {activeCount} {activeCount === 1 ? 'person' : 'people'} here
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---- Timeline overlay (left side, below status badge) ---- */}
      {/* Only show timeline to authenticated users — activity data is private */}
      {streamLoaded && primaryWallet && (
        <div className="absolute top-[104px] left-2 z-30 px-2">
          <Timeline
            variant="camera"
            cameraId={cameraPda}
            mobileOverlay={true}
          />
        </div>
      )}

      {/* ---- Bottom action sheet ---- */}
      {streamLoaded && showBottomSheet && (
        <div className="absolute bottom-0 left-0 right-0 z-40 px-4 pb-8 pt-16 bg-gradient-to-t from-black/90 via-black/50 to-transparent">
          {primaryWallet ? (
            // User is logged in — offer full check-in
            isCheckedIn ? (
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-green-400 rounded-full" />
                  <span className="text-white/80 text-sm">Checked in</span>
                </div>
                <button
                  onClick={handleCheckIn}
                  className="w-full py-3 rounded-xl bg-white text-black font-semibold text-sm hover:bg-gray-100 transition-colors"
                >
                  Open camera
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <p className="text-white/80 text-sm text-center">
                  You're watching live. Check in to be recognised in the feed.
                </p>
                <button
                  onClick={handleCheckIn}
                  className="w-full py-3 rounded-xl bg-white text-black font-semibold text-sm hover:bg-gray-100 transition-colors"
                >
                  Check in
                </button>
              </div>
            )
          ) : (
            // Anonymous viewer — sign-up prompt
            <div className="flex flex-col gap-3">
              <p className="text-white/80 text-sm text-center">
                Create an account to be recognised, save captures, and join competitions.
              </p>
              <button
                onClick={() => navigate(`/login?redirect=${encodeURIComponent(`/app/camera/${cameraPda}?fromTap=true`)}`)}
                className="w-full py-3 rounded-xl bg-white text-black font-semibold text-sm hover:bg-gray-100 transition-colors"
              >
                Create account
              </button>
            </div>
          )}
        </div>
      )}

      {/* ---- Camera info modal (read-only, no auth needed) ---- */}
      {showCameraInfo && (
        <div className="fixed inset-0 z-[90]">
          <div
            className="fixed inset-0 bg-black/50"
            onClick={() => setShowCameraInfo(false)}
          />
          <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
            <div className="relative w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
              {/* Header */}
              <div className="flex items-center justify-between p-3 border-b border-gray-100">
                <h2 className="text-base font-medium">Camera Details</h2>
                <button
                  type="button"
                  onClick={() => setShowCameraInfo(false)}
                  className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              </div>

              <div className="p-4">
                {/* Camera PDA */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-primary-light flex-shrink-0 flex items-center justify-center">
                    <Camera className="w-5 h-5 text-primary" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">
                      {cameraPda ? `${cameraPda.slice(0, 6)}...${cameraPda.slice(-6)}` : '—'}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => window.open(`https://solscan.io/account/${cameraPda}?cluster=devnet`, '_blank')}
                    className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Owner */}
                {selectedCamera?.owner && (
                  <div className="flex items-center mb-4">
                    <div className="w-10 h-10 rounded-full bg-gray-100 flex-shrink-0 flex items-center justify-center">
                      <User className="w-5 h-5 text-gray-500" />
                    </div>
                    <div className="ml-3 flex-1">
                      <div className="text-sm text-gray-700">Owner</div>
                      <div className="text-sm font-medium">
                        {`${selectedCamera.owner.slice(0, 9)}...${selectedCamera.owner.slice(-5)}`}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => window.open(`https://solscan.io/account/${selectedCamera.owner}?cluster=devnet`, '_blank')}
                      className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center"
                    >
                      View <ExternalLink className="w-3 h-3 ml-1" />
                    </button>
                  </div>
                )}

                {/* Type */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Type</div>
                  <div className="text-sm">{selectedCamera?.metadata?.model || 'camera'}</div>
                </div>

                {/* Users checked in */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Users checked in</div>
                  <div className="flex items-center mt-1">
                    <span className={`w-2 h-2 rounded-full mr-2 ${activeCount > 0 ? 'bg-green-500' : 'bg-gray-400'}`} />
                    <span className={`text-sm font-medium ${activeCount > 0 ? 'text-green-600' : 'text-gray-500'}`}>
                      {activeCount} {activeCount === 1 ? 'user' : 'users'} currently active
                    </span>
                  </div>
                </div>

                {/* Capabilities */}
                <div className="pt-3 border-t border-gray-100">
                  <div className="text-sm text-gray-700 mb-2">Capabilities</div>
                  <div className="flex flex-wrap gap-1.5">
                    {['Face Recognition', 'Photo Capture', 'Pose Tracking', 'Live Streaming', 'On-chain Identity'].map((cap) => (
                      <span
                        key={cap}
                        className="text-xs bg-gray-100 text-gray-600 rounded-full px-2.5 py-1"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
