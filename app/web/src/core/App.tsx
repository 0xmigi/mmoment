import { BrowserRouter } from 'react-router-dom';
import Router from './Router';
import { NotificationProvider } from '../ui/feedback/NotificationProvider';
import { CameraProvider } from '../camera/CameraProvider';

function App() {
  return (
    <NotificationProvider>
      <CameraProvider>
        <BrowserRouter>
          <Router />
        </BrowserRouter>
      </CameraProvider>
    </NotificationProvider>
  );
}

export default App;
