import { useEffect, useState, useCallback } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { CONFIG } from '../core/config';
import { useCamera } from '../camera/CameraProvider';
import { WebRTCStreamPlayer } from '../media/WebRTCStreamPlayer';

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
  const { setSelectedCamera, fetchCameraById, isCheckedIn } = useCamera();

  const [tapState, setTapState] = useState<TapState>('verifying');
  const [presenceToken, setPresenceToken] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [showSignUpPrompt, setShowSignUpPrompt] = useState(false);

  // NFC SUN params from the tag URL (NFC Developer App uses picc_data/enc/cmac)
  const piccData = searchParams.get('picc_data') || '';
  const encFileData = searchParams.get('enc') || '';
  const cmac = searchParams.get('cmac') || '';

  const cameraApiUrl = cameraPda ? CONFIG.getCameraApiUrlByPda(cameraPda) : null;

  // ---------------------------------------------------------------- //
  //  Load camera into context so WebRTCStreamPlayer works             //
  // ---------------------------------------------------------------- //
  useEffect(() => {
    if (!cameraPda) return;
    fetchCameraById(cameraPda).then((camera) => {
      if (camera) setSelectedCamera(camera);
    });
  }, [cameraPda]);

  // ---------------------------------------------------------------- //
  //  NFC SUN verification                                             //
  // ---------------------------------------------------------------- //
  const verifyTap = useCallback(async () => {
    if (!cameraPda || !cameraApiUrl) return;

    // 1. Check for a still-valid stored presence token
    const stored = loadPresenceToken(cameraPda);
    if (stored) {
      setPresenceToken(stored);
      setTapState('already_valid');
      setShowSignUpPrompt(true);
      return;
    }

    // 2. No stored token — check we have NFC params
    if (!piccData || !cmac) {
      // Direct URL access without a tap — don't show the stream
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
        // Show sign-up prompt after a short delay so the stream can settle
        setTimeout(() => setShowSignUpPrompt(true), 4000);
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
  //  Route to check-in if user is logged in and already verified       //
  // ---------------------------------------------------------------- //
  const handleCheckIn = () => {
    if (!cameraPda) return;
    // Navigate to the standard camera view with the presence token in state
    navigate(`/app/camera/${cameraPda}`, { state: { presenceToken, fromTap: true } });
  };

  // ---------------------------------------------------------------- //
  //  Render                                                            //
  // ---------------------------------------------------------------- //
  if (!cameraPda) {
    return <div className="flex items-center justify-center h-screen text-white bg-black">Invalid tap link.</div>;
  }

  // Blocked — arrived without a tap
  if (tapState === 'no_nfc_params') {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-black text-white gap-4 px-6">
        <div className="text-4xl">📵</div>
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
      <div className="flex flex-col items-center justify-center h-screen bg-black text-white gap-4 px-6">
        <div className="text-4xl">⚠️</div>
        <h1 className="text-xl font-semibold text-center">Tap not recognised</h1>
        <p className="text-sm text-gray-400 text-center max-w-xs">{errorMsg}</p>
        <button
          onClick={verifyTap}
          className="mt-2 px-5 py-2 rounded-full bg-white text-black text-sm font-medium"
        >
          Try again
        </button>
      </div>
    );
  }

  const streamLoaded = tapState === 'verified' || tapState === 'already_valid';

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">

      {/* ---- Live stream ---- */}
      {streamLoaded ? (
        <div className="absolute inset-0">
          <WebRTCStreamPlayer streamType="clean" />
        </div>
      ) : (
        // Verifying — show a minimal loading state without blocking anything
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* ---- Top bar ---- */}
      {streamLoaded && (
        <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-gradient-to-b from-black/60 to-transparent">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-white text-sm font-medium">LIVE</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 bg-green-400 rounded-full" />
            <span className="text-white/80 text-xs">Tap verified</span>
          </div>
        </div>
      )}

      {/* ---- Bottom action sheet ---- */}
      {streamLoaded && showSignUpPrompt && (
        <div className="absolute bottom-0 left-0 right-0 px-4 pb-8 pt-6 bg-gradient-to-t from-black/80 to-transparent">
          {primaryWallet ? (
            // User is logged in — offer full check-in
            isCheckedIn ? (
              <div className="text-center text-white/70 text-sm">
                You're checked in. Face recognition active.
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <p className="text-white/80 text-sm text-center">
                  You're watching live. Check in to be recognised in the feed.
                </p>
                <button
                  onClick={handleCheckIn}
                  className="w-full py-3 rounded-2xl bg-white text-black font-semibold text-sm"
                >
                  Check in
                </button>
              </div>
            )
          ) : (
            // Anonymous viewer — soft sign-up prompt
            <div className="flex flex-col gap-3">
              <p className="text-white/80 text-sm text-center">
                Create an account to be recognised, save captures, and join competitions.
              </p>
              <button
                onClick={() => navigate('/login', { state: { returnTo: `/tap/${cameraPda}?picc_data=${piccData}&cmac=${cmac}` } })}
                className="w-full py-3 rounded-2xl bg-white text-black font-semibold text-sm"
              >
                Create account
              </button>
              <button
                onClick={() => setShowSignUpPrompt(false)}
                className="text-white/40 text-xs text-center"
              >
                Continue watching anonymously
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
