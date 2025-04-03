import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { MainLayout, CameraView, GalleryView, ActivitiesView } from '../ui';
import { LandingPage, ProductPage, QuickStartView } from '../pages';
import { Settings } from '../ui/settings/Settings';
import { AccountPage } from '../ui/account/AccountPage';
import { SolDevNetDebug } from '../ui/debug/SolDevNetDebug';
import { DebugPage } from '../ui/debug/DebugPage';
import { useCamera, fetchCameraByPublicKey } from '../camera/CameraProvider';
import { useConnection } from '@solana/wallet-adapter-react';

function Router() {
  const [activeTab, setActiveTab] = useState<'camera' | 'gallery' | 'activities' | 'account'>('camera');
  const { fetchCameraById, setSelectedCamera } = useCamera();
  const location = useLocation();
  const navigate = useNavigate();
  const { connection } = useConnection();

  // Extract camera ID from URL if present
  useEffect(() => {
    const loadCameraFromUrl = async () => {
      const match = location.pathname.match(/\/app\/camera\/([^\/]+)/);
      
      // Only load camera if we're on a camera-specific route
      if (match && match[1]) {
        const cameraPublicKey = decodeURIComponent(match[1]);
        
        // Check for expiration time in URL params
        const urlParams = new URLSearchParams(location.search);
        const expirationTime = urlParams.get('expires');
        
        // Validate expiration time if present
        if (expirationTime) {
          const expireTimestamp = parseInt(expirationTime, 10);
          const currentTime = Date.now();
          
          // If URL has expired, show error and redirect
          if (isNaN(expireTimestamp) || currentTime > expireTimestamp) {
            console.error('NFC URL has expired or is invalid');
            alert('This camera access link has expired. Please scan the NFC tag again for a new link.');
            navigate('/app');
            return;
          }
          
          console.log(`Valid NFC URL access, expires at ${new Date(expireTimestamp).toLocaleString()}`);
        }
        
        try {
          console.log(`[Router] Loading camera with public key: ${cameraPublicKey}`);
          
          // IMPORTANT - Force this to be displayed in the UI
          // This is critical for showing the camera ID even if the useProgram hook isn't available yet
          localStorage.setItem('directCameraId', cameraPublicKey);
          
          // First try using the direct method if connection is available
          if (connection) {
            console.log(`[Router] Trying direct camera loading with connection...`);
            const camera = await fetchCameraByPublicKey(cameraPublicKey, connection);
            if (camera) {
              setSelectedCamera(camera);
              console.log(`[Router] Loaded camera directly: ${camera.metadata.name} (${camera.publicKey})`);
              return;
            } else {
              console.log(`[Router] Direct camera loading failed, falling back to provider method`);
            }
          }
          
          // Fall back to provider method
          const camera = await fetchCameraById(cameraPublicKey);
          if (camera) {
            setSelectedCamera(camera);
            console.log(`[Router] Loaded camera from URL: ${camera.metadata.name} (${camera.publicKey})`);
          } else {
            console.warn(`[Router] Camera with public key ${cameraPublicKey} not found, but we'll keep showing the ID`);
            // Don't navigate away, let the CameraView handle the error
          }
        } catch (error) {
          console.error('[Router] Error loading camera from URL:', error);
        }
      } else if (location.pathname === '/app' || location.pathname === '/app/') {
        // If we're on the default /app route (with or without trailing slash), clear the selected camera and localStorage data
        setSelectedCamera(null);
        localStorage.removeItem('directCameraId');
        
        // Make sure we're also clearing any camera account references in CameraView
        // This ensures a complete disconnect when going to the default route
        console.log('[Router] On default route - ensuring all camera references are cleared');
      }
    };

    loadCameraFromUrl();
  }, [location.pathname, location.search, fetchCameraById, setSelectedCamera, navigate, connection]);

  return (
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/product" element={<ProductPage />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/soldevnetdebug" element={<SolDevNetDebug />} />
        <Route path="/debug" element={<DebugPage />} />
        <Route
          path="/account"
          element={
            <MainLayout activeTab="account" onTabChange={setActiveTab}>
              <AccountPage />
            </MainLayout>
          }
        />
        <Route
          path="/app"
          element={
            <MainLayout activeTab={activeTab} onTabChange={setActiveTab}>
              {activeTab === 'camera' ? <CameraView /> :
                activeTab === 'gallery' ? <GalleryView /> :
                  <ActivitiesView />}
            </MainLayout>
          }
        />
        <Route
          path="/app/camera/:cameraId"
          element={
            <MainLayout activeTab="camera" onTabChange={setActiveTab}>
              <CameraView />
            </MainLayout>
          }
        />
        <Route
          path="/app/gallery/:cameraId"
          element={
            <MainLayout activeTab="gallery" onTabChange={setActiveTab}>
              <GalleryView />
            </MainLayout>
          }
        />
        <Route
          path="/app/activities/:cameraId"
          element={
            <MainLayout activeTab="activities" onTabChange={setActiveTab}>
              <ActivitiesView />
            </MainLayout>
          }
        />
        <Route
          path="/quickstart/:cameraId/:sessionId/:timestamp"
          element={<QuickStartView />}
        />
      </Routes>
  );
}

export default Router;

