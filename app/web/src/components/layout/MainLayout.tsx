import { DynamicAuthButton } from '../DynamicAuthButton';
import { useIsLoggedIn } from '@dynamic-labs/sdk-react-core';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Logo from '../Logo';

interface MainLayoutProps {
  children: React.ReactNode;
  activeTab: 'camera' | 'gallery' | 'activities';  // Add activities
  onTabChange: (tab: 'camera' | 'gallery' | 'activities') => void;
}

export function MainLayout({ children, activeTab, onTabChange }: MainLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const isLoggedIn = useIsLoggedIn();
  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-screen bg-white">
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
            {isLoggedIn && (
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  type='button'
                  onClick={() => onTabChange('camera')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'camera'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => onTabChange('gallery')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'gallery'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Gallery
                </button>
                <button
                  type='button'
                  onClick={() => onTabChange('activities')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'activities'
                    ? 'bg-white text-stone-800'
                    : 'bg-white text-stone-400 hover:text-stone-800'
                    }`}
                >
                  Activities
                </button>
              </div>
            )}
          </div>

          {/* Right side - Auth button and mobile menu */}
          <div className="flex items-center gap-2">
            {/* {isLoggedIn && (
              <button
                onClick={() => navigate('/settings')}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
                aria-label="Settings"
              >
                <SettingsIcon className="w-5 h-5 text-gray-600" />
              </button>
            )} */}
            <DynamicAuthButton />
            {/* Update the mobile menu trigger button style */}
            {isLoggedIn && (
              <button
                type='button'
                onClick={() => setIsOpen(true)}
                className="sm:hidden p-2 rounded-lg bg-[#e7eeff] hover:bg-[#d1dfff] transition-colors"
              >
                <Menu className="w-5 h-5 text-gray-700" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isLoggedIn && isOpen && (
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
                className="p-2 rounded-lg bg-[#e7eeff] hover:bg-[#d1dfff] transition-colors"
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
                    onTabChange('camera');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'camera'
                    ? 'bg-[#e7eeff] text-black'
                    : 'text-gray-600 hover:bg-gray-50'
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
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'gallery'
                    ? 'bg-[#e7eeff] text-black'
                    : 'text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  Gallery
                </button>

                <button
                  type='button'
                  onClick={() => {
                    onTabChange('activities');
                    setIsOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${activeTab === 'activities'
                    ? 'bg-[#e7eeff] text-black'
                    : 'text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  Activities
                </button>

                <button
                  type='button'
                  onClick={() => {
                    navigate('/settings');
                    setIsOpen(false);
                  }}
                  className="w-full text-left px-4 py-3 rounded-lg transition-colors text-gray-600 hover:bg-gray-50"
                >
                  Settings
                </button>

                {/* Divider */}
                <div className="my-4 border-t border-gray-200" />

                {/* Footer Info */}
                <div className="px-4 flex justify-center py-2">
                  <p className="text-sm text-gray-500">
                    Moment is a secure content network
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Main Content */}
      <div className="flex-1 overflow-y-auto relative">
      {isLoggedIn ? (
        <div className="h-full">
          {children}
        </div>
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-600">Please connect your wallet to continue</p>
        </div>
      )}
      </div>
    </div>
  );
}