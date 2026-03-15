import { useState, useEffect } from 'react';
import { CONFIG } from '../../core/config';
import { Link } from 'react-router-dom';

export function DebugPage() {
  const [info] = useState({
    userAgent: navigator.userAgent,
    timestamp: new Date().toISOString(),
    screen: {
      width: window.screen.width,
      height: window.screen.height,
      orientation: window.screen.orientation?.type || 'unknown',
      pixelRatio: window.devicePixelRatio
    },
    network: {
      type: (navigator as any).connection?.type || 'unknown',
      effectiveType: (navigator as any).connection?.effectiveType || 'unknown',
      downlink: (navigator as any).connection?.downlink || 'unknown',
      rtt: (navigator as any).connection?.rtt || 'unknown'
    },
    config: {
      CAMERA_API_URL: CONFIG.CAMERA_API_URL,
      BACKEND_URL: CONFIG.BACKEND_URL,
      TIMELINE_WS_URL: CONFIG.TIMELINE_WS_URL,
      isMobile: CONFIG.isMobileBrowser
    }
  });

  const checkCameraConnection = async () => {
    try {
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/health`, {
        headers: { 'Cache-Control': 'no-cache' }
      });
      const data = await response.json();
      return { success: response.ok, status: response.status, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to connect' 
      };
    }
  };

  const checkTimelineConnection = async () => {
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/health`, {
        headers: { 'Cache-Control': 'no-cache' }
      });
      const data = await response.json();
      return { success: response.ok, status: response.status, data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to connect' 
      };
    }
  };

  // Fix the type definitions
  type ConnectionResult = ReturnType<typeof checkCameraConnection> extends Promise<infer T> ? T : never;
  
  const [cameraStatus, setCameraStatus] = useState<{ 
    loading: boolean; 
    result: ConnectionResult | null;
  }>({ loading: true, result: null });
  
  const [timelineStatus, setTimelineStatus] = useState<{
    loading: boolean; 
    result: ConnectionResult | null;
  }>({ loading: true, result: null });

  useEffect(() => {
    const runChecks = async () => {
      setCameraStatus({ loading: true, result: null });
      setTimelineStatus({ loading: true, result: null });
      
      const camera = await checkCameraConnection();
      setCameraStatus({ loading: false, result: camera });
      
      const timeline = await checkTimelineConnection();
      setTimelineStatus({ loading: false, result: timeline });
    };
    
    runChecks();
  }, []);

  return (
    <div className="max-w-lg mx-auto p-4">
      <h1 className="text-xl font-bold mb-4">Mobile Debug</h1>
      
      <div className="mb-6">
        <h2 className="font-semibold">Device Information</h2>
        <pre className="bg-gray-100 p-2 overflow-auto text-xs mt-2 rounded">
          {JSON.stringify(info, null, 2)}
        </pre>
      </div>
      
      <div className="mb-6">
        <h2 className="font-semibold">Camera Connection</h2>
        {cameraStatus.loading ? (
          <p>Checking...</p>
        ) : (
          <pre className="bg-gray-100 p-2 overflow-auto text-xs mt-2 rounded">
            {JSON.stringify(cameraStatus.result, null, 2)}
          </pre>
        )}
        <button 
          className="mt-2 px-3 py-1 bg-primary text-white rounded"
          onClick={async () => {
            setCameraStatus({ loading: true, result: null });
            const result = await checkCameraConnection();
            setCameraStatus({ loading: false, result });
          }}
        >
          Retry
        </button>
      </div>
      
      <div className="mb-6">
        <h2 className="font-semibold">Timeline Connection</h2>
        {timelineStatus.loading ? (
          <p>Checking...</p>
        ) : (
          <pre className="bg-gray-100 p-2 overflow-auto text-xs mt-2 rounded">
            {JSON.stringify(timelineStatus.result, null, 2)}
          </pre>
        )}
        <button 
          className="mt-2 px-3 py-1 bg-primary text-white rounded"
          onClick={async () => {
            setTimelineStatus({ loading: true, result: null });
            const result = await checkTimelineConnection();
            setTimelineStatus({ loading: false, result });
          }}
        >
          Retry
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
        <div className="p-4 border rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Walrus Storage</h3>
          <p className="mb-4">Test the Walrus decentralized storage integration</p>
          <Link 
            to="/debug/walrus" 
            className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
          >
            Test Walrus Storage
          </Link>
        </div>
      </div>
    </div>
  );
} 