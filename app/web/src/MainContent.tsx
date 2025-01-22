import { Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import { MainLayout } from './components/layout/MainLayout';
import { CameraView } from './components/layout/CameraView';
import { GalleryView } from './components/layout/GalleryView';
import LandingPage from './components/LandingPage';
import ProductPage from './components/ProductPage';
import QuickStartView from './components/QuickStartView';
import { ActivitiesView } from './components/layout/ActivitiesView';
import { TestPage } from './components/headless/TestPage';

function MainContent() {
  const [activeTab, setActiveTab] = useState<'camera' | 'gallery' | 'activities'>('camera');

  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/product" element={<ProductPage />} />
      <Route path="/test-headless" element={<TestPage />} />
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
        path="/quickstart/:cameraId/:sessionId/:timestamp"
        element={<QuickStartView />}
      />
    </Routes>
  );
}

export default MainContent;