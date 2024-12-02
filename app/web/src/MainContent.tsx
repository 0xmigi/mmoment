import { Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { CameraView } from './components/layout/CameraView';
import { GalleryView } from './components/layout/GalleryView';
import LandingPage from './components/LandingPage';
import QuickStartView from './components/QuickStartView';

function MainContent() {
  const [activeTab, setActiveTab] = useState<'camera' | 'gallery'>('camera');

  return (
    <Routes>
      {/* Make landing page the default route */}
      <Route path="/" element={<LandingPage />} />
      
      {/* Move app interface to /app route */}
      <Route
        path="/app"
        element={
          <MainLayout activeTab={activeTab} onTabChange={setActiveTab}>
            {activeTab === 'camera' ? <CameraView /> : <GalleryView />}
          </MainLayout>
        }
      />
      
      {/* Update quickstart route for NFC taps */}
      <Route 
        path="/quickstart/:cameraId/:sessionId/:timestamp" 
        element={<QuickStartView />} 
      />
    </Routes>
  );
}

export default MainContent;