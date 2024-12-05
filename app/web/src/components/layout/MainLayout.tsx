import { DynamicAuthButton } from '../DynamicAuthButton';
import { useIsLoggedIn } from '@dynamic-labs/sdk-react-core';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../Logo';

interface MainLayoutProps {
  children: React.ReactNode;
  activeTab: 'camera' | 'gallery';
  onTabChange: (tab: 'camera' | 'gallery') => void;
}

export function MainLayout({ children, activeTab, onTabChange }: MainLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isLoggedIn = useIsLoggedIn();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white w-full overflow-y-auto">
      {/* Top Bar */}
      <div className="w-full flex-none">
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
            {isLoggedIn && (
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  type='button'
                  onClick={() => onTabChange('camera')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                    activeTab === 'camera'
                      ? 'bg-white text-stone-500 hover:text-stone-800'
                      : 'bg-white text-stone-200 hover:text-stone-500'
                  }`}
                >
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => onTabChange('gallery')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                    activeTab === 'gallery'
                      ? 'bg-white text-stone-500 hover:text-stone-800'
                      : 'bg-white text-stone-200 hover:text-stone-500'
                  }`}
                >
                  Gallery
                </button>
              </div>
            )}
          </div>

          {/* Right side - Auth button and mobile menu */}
          <div className="flex items-center gap-2">
            <DynamicAuthButton />
            {isLoggedIn && (
              <button
                type='button'
                onClick={() => setIsOpen(true)}
                className="sm:hidden p-2 rounded-s bg-[#e7eeff]"
              >
                <Menu className="w-5 h-5 text-gray-700" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isLoggedIn && isOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 sm:hidden">
          <div className="absolute right-0 top-0 h-full w-64 bg-[#e7eeff] p-4">
            <button
              type='button'
              onClick={() => setIsOpen(false)}
              className="absolute top-4 right-4"
            >
              <X className="w-6 h-6 text-gray-700" />
            </button>

            <div className="pt-16 flex flex-col gap-4">
              <div className="border-t pt-4">
                <button
                  type='button'
                  onClick={() => {
                    onTabChange('camera');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-2 rounded-lg ${
                    activeTab === 'camera'
                      ? 'bg-purple-100 text-purple-700'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => {
                    onTabChange('gallery');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-2 rounded-lg ${
                    activeTab === 'gallery'
                      ? 'bg-purple-100 text-purple-700'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Gallery
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      {isLoggedIn ? (
        <div className="w-full mx-auto px-4">
          {children}
        </div>
      ) : (
        <div className="flex items-center justify-center min-h-[calc(100vh-4rem)]">
          <p className="text-gray-600">Please connect your wallet to continue</p>
        </div>
      )}
    </div>
  );
}