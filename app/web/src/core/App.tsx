import { BrowserRouter } from 'react-router-dom';
import Router from './Router';
import { NotificationProvider } from '../ui/feedback/NotificationProvider';
import { CameraProvider } from '../camera/CameraProvider';
import { PinataSettings } from '../storage';
import { CONFIG } from './config';

function App() {
  // Only show PinataSettings in development mode
  const isDevelopment = !CONFIG.isProduction;

  return (
    <NotificationProvider>
      <CameraProvider>
        <BrowserRouter>
          <Router />
        </BrowserRouter>
        {isDevelopment && <PinataSettings />}
      </CameraProvider>
    </NotificationProvider>
  );
}

export default App;
