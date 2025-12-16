import { BrowserRouter } from 'react-router-dom';
import Router from './Router';
import { NotificationProvider } from '../ui/feedback/NotificationProvider';
import { CameraProvider } from '../camera/CameraProvider';
import { DeveloperSettings } from '../storage';
import { CONFIG } from './config';
import { useProfileSync } from '../auth/useProfileSync';

function App() {
  // Auto-sync user profile to backend when wallet connects
  useProfileSync();

  // Only show DeveloperSettings in development mode
  const isDevelopment = !CONFIG.isProduction;

  return (
    <NotificationProvider>
      <CameraProvider>
        <BrowserRouter>
          <Router />
        </BrowserRouter>
        {isDevelopment && <DeveloperSettings />}
      </CameraProvider>
    </NotificationProvider>
  );
}

export default App;
