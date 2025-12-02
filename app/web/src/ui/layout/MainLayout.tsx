import { HeadlessAuthButton } from '../../auth';
import { useIsLoggedIn, useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Menu, X, LogIn } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import Logo from '../common/Logo';
import { AuthModal } from '../../auth/components/AuthModal';

interface MainLayoutProps {
  children: React.ReactNode;
  activeTab: 'camera' | 'gallery' | 'activities' | 'account';
  onTabChange: (tab: 'camera' | 'gallery' | 'activities' | 'account') => void;
}

export function MainLayout({ children, activeTab, onTabChange }: MainLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isLoggedIn = useIsLoggedIn();
  useDynamicContext();
  const navigate = useNavigate();
  const { cameraId } = useParams<{ cameraId?: string }>();
  const location = useLocation();
  const [showAuthModal, setShowAuthModal] = useState(false);
  
  // Debug logging for navigation
  useEffect(() => {
    const pathMatch = location.pathname.match(/\/app\/(camera|gallery|activities)\/([^\/]+)/);
    const localStorageCameraId = localStorage.getItem('directCameraId');
    
    console.log('Navigation Debug:', {
      currentPath: location.pathname,
      activeTab,
      cameraIdFromParams: cameraId,
      cameraIdFromPath: pathMatch ? pathMatch[2] : null,
      cameraIdFromLocalStorage: localStorageCameraId
    });
  }, [location.pathname, activeTab, cameraId]);
  
  // Handle tab navigation
  // Camera tab preserves cameraId in URL, Gallery/Activities use general routes
  const handleTabChange = (tab: 'camera' | 'gallery' | 'activities' | 'account') => {
    onTabChange(tab);

    // Extract camera ID from current path if available
    const matchPath = location.pathname.match(/\/app\/(camera|gallery|activities)\/([^\/]+)/);
    const currentCameraId = cameraId || (matchPath ? matchPath[2] : localStorage.getItem('directCameraId'));

    if (tab === 'camera' && currentCameraId) {
      // Camera tab: preserve cameraId in URL
      console.log(`Navigating to camera with ID: ${currentCameraId}`);
      navigate(`/app/camera/${currentCameraId}`);
    } else if (tab === 'gallery') {
      // Gallery: always use general route (filters available in view)
      console.log('Navigating to general gallery view');
      navigate('/app/gallery');
    } else if (tab === 'activities') {
      // Activities: always use general route (filters available in view)
      console.log('Navigating to general activities view');
      navigate('/app/activities');
    } else {
      navigate('/app');
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-white">
      {/* Top Bar */}
      <div className="flex-none z-[40]">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          {/* Left side - Logo and Desktop Nav */}
          <div className="flex items-center gap-8">
            <div
              onClick={() => navigate('/')}
              className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            >
              <Logo width={30} height={21} className="text-black" />
              <h1 className="text-2xl text-black font-bold">
                Moment
              </h1>
            </div>
            {isLoggedIn && activeTab !== 'account' && (
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  type='button'
                  onClick={() => handleTabChange('camera')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'camera'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => handleTabChange('gallery')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'gallery'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Gallery
                </button>
                <button
                  type='button'
                  onClick={() => handleTabChange('activities')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'activities'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Activities
                </button>
                {/* <button
                  type='button'
                  onClick={() => navigate('/soldevnetdebug')}
                  className="px-4 py-2 rounded-lg flex items-center gap-2 bg-white text-stone-400 hover:text-stone-800"
                >
                  DevNet Debug
                </button> */}
              </div>
            )}
          </div>

          {/* Right side - Auth button and mobile menu */}
          <div className="flex items-center gap-2">
            {activeTab === 'account' ? (
              <button
                type='button'
                onClick={() => navigate('/app')}
                className="p-2 rounded-lg bg-[#999999] hover:bg-[#d1dfff] transition-colors"
              >
                <X className="w-5 h-5 text-gray-50" />
              </button>
            ) : (
              <>
                <HeadlessAuthButton />
                {isLoggedIn && (
                  <button
                    type='button'
                    onClick={() => setIsOpen(true)}
                    className="sm:hidden p-2 rounded-lg bg-[#999999] hover:bg-[#d1dfff] transition-colors"
                  >
                    <Menu className="w-5 h-5 text-gray-50" />
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu - Don't show on account page */}
      {isLoggedIn && isOpen && activeTab !== 'account' && (
        <div className="fixed inset-x-0 z-[80] sm:hidden">
          {/* Semi-transparent overlay */}
          <div
            className="fixed inset-0 bg-black/50"
            onClick={() => setIsOpen(false)}
          />

          <div className="relative">
            {/* Keep header visible when menu is open */}
            <div className="bg-white px-4 h-16 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Logo width={30} height={21} className="text-black" />
                <h1 className="text-2xl text-black font-bold">Moment</h1>
              </div>
              <button
                type='button'
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              >
                <X className="w-5 h-5 text-gray-700" />
              </button>
            </div>

            {/* Navigation Card */}
            <div className="w-full bg-white shadow-lg rounded-b-xl">
              {/* Navigation Links */}
              <div className="p-6 space-y-4">
                <button
                  type='button'
                  onClick={() => {
                    handleTabChange('camera');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'camera'
                    ? 'bg-gray-100 text-black'
                    : 'text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  Camera
                </button>

                <button
                  type='button'
                  onClick={() => {
                    handleTabChange('gallery');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'gallery'
                    ? 'bg-gray-100 text-black'
                    : 'text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  Gallery
                </button>

                <button
                  type='button'
                  onClick={() => {
                    handleTabChange('activities');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'activities'
                    ? 'bg-gray-100 text-black'
                    : 'text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  Activities
                </button>

                {/* <button
                  type='button'
                  onClick={() => {
                    navigate('/settings');
                    setIsOpen(false);
                  }}
                  className="w-full text-left px-4 py-3 rounded-lg transition-colors text-gray-600 hover:bg-gray-50"
                >
                  Settings
                </button>

                <button
                  type='button'
                  onClick={() => {
                    navigate('/soldevnetdebug');
                    setIsOpen(false);
                  }}
                  className="w-full text-left px-4 py-3 rounded-lg transition-colors text-gray-600 hover:bg-gray-50"
                >
                  DevNet Debug
                </button> */}

                {/* Divider */}
                <div className="my-4 border-t border-gray-200" />

                {/* Footer Info */}
                <div className="px-4 flex justify-center py-2">
                  <p className="text-sm text-gray-500">
                    social memory and compute for IRL
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Main Content */}
      <div className="flex-1 relative">
      {isLoggedIn ? (
        <div>
          {children}
        </div>
      ) : (
        <div className="h-full flex flex-col items-center justify-center p-4">
          <div className="max-w-md w-full bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <div className="text-center mb-6">
              <Logo width={40} height={32} className="mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900">Welcome to Moment</h2>
              <p className="text-gray-600 mt-2">
                Please sign in to access the app
              </p>
            </div>
            
            <button
              onClick={() => setShowAuthModal(true)}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors"
            >
              <LogIn className="w-5 h-5" />
              <span>Sign In</span>
            </button>
            
            <p className="text-sm text-center mt-4 text-gray-500">
              Capture moments and their context instantly
            </p>
          </div>
          
          {/* Auth Modal */}
          <AuthModal 
            isOpen={showAuthModal} 
            onClose={() => setShowAuthModal(false)} 
          />
        </div>
      )}
      </div>
    </div>
  );
}