import { useIsLoggedIn, useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { X, User } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import { AuthModal } from '../../auth/components/AuthModal';
import Logo from '../common/Logo';

interface MainLayoutProps {
  children: React.ReactNode;
  activeTab: 'camera' | 'gallery' | 'activities' | 'account';
  onTabChange: (tab: 'camera' | 'gallery' | 'activities' | 'account') => void;
}

interface SocialCredential {
  oauthProvider: string;
  oauthUsername: string;
  oauthDisplayName: string;
  oauthAccountPhotos: string[];
}

export function MainLayout({ children, activeTab, onTabChange }: MainLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const isLoggedIn = useIsLoggedIn();
  const { sdkHasLoaded, primaryWallet, user } = useDynamicContext();
  const { connection } = useConnection();
  const navigate = useNavigate();
  const { cameraId } = useParams<{ cameraId?: string }>();
  const location = useLocation();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [solBalance, setSolBalance] = useState<number | null>(null);

  // Redirect to login if not authenticated (wait for SDK to hydrate first)
  useEffect(() => {
    if (!sdkHasLoaded) return;
    if (!isLoggedIn) {
      const currentPath = location.pathname + location.search;
      navigate(`/login?redirect=${encodeURIComponent(currentPath)}`);
    }
  }, [sdkHasLoaded, isLoggedIn, navigate, location.pathname, location.search]);

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

  // Fetch SOL balance
  useEffect(() => {
    const fetchBalance = async () => {
      if (!primaryWallet?.address || !connection) return;
      try {
        const publicKey = new PublicKey(primaryWallet.address);
        const balance = await connection.getBalance(publicKey);
        setSolBalance(balance / LAMPORTS_PER_SOL);
      } catch (error) {
        console.error('Error fetching balance:', error);
      }
    };
    fetchBalance();
    const interval = setInterval(fetchBalance, 30000);
    return () => clearInterval(interval);
  }, [primaryWallet?.address, connection]);

  // Handle tab navigation
  const handleTabChange = (tab: 'camera' | 'gallery' | 'activities' | 'account') => {
    onTabChange(tab);

    const matchPath = location.pathname.match(/\/app\/(camera|gallery|activities)\/([^\/]+)/);
    const currentCameraId = cameraId || (matchPath ? matchPath[2] : localStorage.getItem('directCameraId'));

    if (tab === 'camera' && currentCameraId) {
      navigate(`/app/camera/${currentCameraId}`);
    } else if (tab === 'gallery') {
      navigate('/app/gallery');
    } else if (tab === 'activities') {
      navigate('/app/activities');
    } else {
      navigate('/app');
    }
  };

  // Resolve user display info
  const farcasterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is SocialCredential =>
      cred?.oauthProvider?.toLowerCase() === 'farcaster'
  );
  const googleCred = user?.verifiedCredentials?.find(
    (cred: any): cred is SocialCredential =>
      cred?.oauthProvider?.toLowerCase() === 'google'
  );
  const twitterCred = user?.verifiedCredentials?.find(
    (cred: any): cred is SocialCredential =>
      cred?.oauthProvider?.toLowerCase() === 'twitter'
  );
  const socialCred = farcasterCred || googleCred || twitterCred;

  const providerLabels: Record<string, string> = {
    farcaster: 'Farcaster',
    google: 'Google',
    twitter: 'X / Twitter',
  };
  const displayName = socialCred?.oauthDisplayName
    || user?.email
    || primaryWallet?.address?.slice(0, 6) + '...' + primaryWallet?.address?.slice(-4)
    || 'Account';
  const socialProvider = socialCred?.oauthProvider
    ? (providerLabels[socialCred.oauthProvider] || socialCred.oauthProvider)
    : (user?.email ? 'Email' : 'Wallet');
  const profilePhoto = socialCred?.oauthAccountPhotos?.[0];

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
              <Logo width={30} height={21} className="text-neutral-900" />
              <h1 className="text-2xl text-neutral-900 font-bold">
                Moment
              </h1>
            </div>
            {isLoggedIn && activeTab !== 'account' && (
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  type='button'
                  onClick={() => handleTabChange('camera')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'camera'
                    ? 'bg-white text-neutral-900'
                    : 'bg-white text-neutral-400 hover:text-neutral-900'
                    }`}
                >
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => handleTabChange('gallery')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'gallery'
                    ? 'bg-white text-neutral-900'
                    : 'bg-white text-neutral-400 hover:text-neutral-900'
                    }`}
                >
                  Gallery
                </button>
                <button
                  type='button'
                  onClick={() => handleTabChange('activities')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'activities'
                    ? 'bg-white text-neutral-900'
                    : 'bg-white text-neutral-400 hover:text-neutral-900'
                    }`}
                >
                  Activities
                </button>
              </div>
            )}
          </div>

          {/* Right side - Profile pill or close button */}
          <div className="flex items-center gap-2">
            {activeTab === 'account' ? (
              <button
                type='button'
                onClick={() => navigate('/app')}
                className="p-2 rounded-lg bg-neutral-100 hover:bg-neutral-200 transition-colors"
              >
                <X className="w-5 h-5 text-neutral-600" />
              </button>
            ) : !primaryWallet?.address ? (
              <>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="px-4 py-2 bg-neutral-100 text-neutral-900 rounded-lg hover:bg-neutral-200 transition-colors font-medium text-sm"
                >
                  Log in
                </button>
                <AuthModal
                  isOpen={showAuthModal}
                  onClose={() => setShowAuthModal(false)}
                />
              </>
            ) : (
              <button
                type='button'
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-1.5 bg-neutral-100 rounded-lg hover:bg-neutral-200 transition-colors"
              >
                {profilePhoto ? (
                  <img src={profilePhoto} alt="" referrerPolicy="no-referrer" className="w-6 h-6 rounded-full" />
                ) : (
                  <div className="w-6 h-6 rounded-full bg-neutral-200 flex items-center justify-center">
                    <User className="w-3.5 h-3.5 text-neutral-400" />
                  </div>
                )}
                <span className="text-sm font-medium text-neutral-900">{displayName}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Unified Dropdown Menu */}
      {isLoggedIn && isOpen && activeTab !== 'account' && (
        <div className="fixed inset-0 z-[80]" ref={dropdownRef}>
          {/* Scrim overlay */}
          <div
            className="fixed inset-0 bg-black/30"
            onClick={() => setIsOpen(false)}
          />

          <div className="relative">
            {/* Header with logo + close */}
            <div className="bg-white px-4 h-16 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Logo width={30} height={21} className="text-neutral-900" />
                <h1 className="text-2xl text-neutral-900 font-bold">Moment</h1>
              </div>
              <button
                type='button'
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-lg bg-neutral-100 hover:bg-neutral-200 transition-colors"
              >
                <X className="w-5 h-5 text-neutral-600" />
              </button>
            </div>

            {/* Dropdown panel */}
            <div className="bg-white shadow-lg rounded-b-xl px-6 pb-6 pt-2 space-y-4">
              {/* Account Card */}
              <button
                type='button'
                onClick={() => {
                  setIsOpen(false);
                  navigate('/account');
                }}
                className="w-full bg-neutral-100 rounded-xl p-4 text-left hover:bg-neutral-200 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {profilePhoto ? (
                    <img src={profilePhoto} alt="" referrerPolicy="no-referrer" className="w-10 h-10 rounded-full" />
                  ) : (
                    <div className="w-10 h-10 rounded-full bg-neutral-200 flex items-center justify-center">
                      <User className="w-5 h-5 text-neutral-400" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-[15px] font-semibold text-neutral-900 truncate">{displayName}</p>
                    <p className="text-xs text-neutral-400">{socialProvider}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-[15px] font-semibold text-neutral-900">
                      {solBalance !== null ? `${solBalance.toFixed(2)} SOL` : '—'}
                    </p>
                  </div>
                </div>
                {/* Storage indicator */}
                <div className="mt-3 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-neutral-600">Storage</span>
                    <span className="text-xs text-neutral-400">23.5 MB / 100 MB</span>
                  </div>
                  <div className="w-full h-1 bg-neutral-200 rounded-sm">
                    <div className="h-1 bg-accent rounded-sm" style={{ width: '24%' }} />
                  </div>
                </div>
              </button>

              {/* Nav links */}
              <div className="space-y-1">
                {(['camera', 'gallery', 'activities'] as const).map((tab) => (
                  <button
                    key={tab}
                    type='button'
                    onClick={() => {
                      handleTabChange(tab);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left px-4 py-3 rounded-lg text-[15px] transition-colors ${
                      activeTab === tab
                        ? 'bg-neutral-100 text-neutral-900 font-medium'
                        : 'text-neutral-400 hover:bg-neutral-50'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>

              {/* Tagline */}
              <div className="flex justify-center pt-1">
                <p className="text-xs text-neutral-400">
                  social memory and compute for IRL
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 relative">
        {isLoggedIn && <div>{children}</div>}
      </div>
    </div>
  );
}