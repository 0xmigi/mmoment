import { Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { CameraView } from './components/layout/CameraView';
import { GalleryView } from './components/layout/GalleryView';
import LandingPage from './components/LandingPage';
import ProductPage from './components/ProductPage';
import QuickStartView from './components/QuickStartView';
import { ActivitiesView } from './components/layout/ActivitiesView';

function MainContent() {
  const [activeTab, setActiveTab] = useState<'camera' | 'gallery' | 'activities'>('camera');

  return (
    <Routes>
      {/* Make landing page the default route */}
      <Route path="/" element={<LandingPage />} />
      
      {/* Add product route */}
      <Route path="/product" element={<ProductPage />} />

      {/* Move app interface to /app route */}
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

      {/* Update quickstart route for NFC taps */}
      <Route
        path="/quickstart/:cameraId/:sessionId/:timestamp"
        element={<QuickStartView />}
      />
    </Routes>
  );
}

export default MainContent;