import { WalletMultiButton } from '@solana/wallet-adapter-react-ui';
import { Camera, Image } from 'lucide-react';
import type React from 'react';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useNavigate } from 'react-router-dom';
import Logo from '../Logo';

interface MainLayoutProps {
  children: React.ReactNode;
  activeTab: 'camera' | 'gallery';
  onTabChange: (tab: 'camera' | 'gallery') => void;
}

export function MainLayout({ children, activeTab, onTabChange }: MainLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { } = useWallet();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white w-full overflow-y-auto">
      {/* Main Content - Centered and follows wireframe */}
      <main className="flex-1 min-h-[calc(100vh-4rem)] items-center px-4">
        {/* Top Bar */}
        <div className="w-full flex-none">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
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
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  type='button'
                  onClick={() => onTabChange('camera')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'camera'
                    ? 'bg-white text-stone-500 hover:text-stone-800'
                    : 'bg-white text-stone-200 hover:text-stone-500'
                    }`}
                >
                  <Camera className="w-4 h-4" />
                  Camera
                </button>
                <button
                  type='button'
                  onClick={() => onTabChange('gallery')}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${activeTab === 'gallery'
                    ? 'bg-white text-stone-500 hover:text-stone-800'
                    : 'bg-white text-stone-200 hover:text-stone-500'
                    }`}
                >
                  <Image className="w-4 h-4" />
                  Gallery
                </button>
              </div>
            </div>
            <WalletMultiButton className="!bg-purple-600" />
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="sm:hidden">
          <button
            type='button'
            onClick={() => setIsOpen(true)}
            className="sm:hidden fixed top-4 right-4 z-50 p-2 rounded-lg bg-white shadow-md"
          >
            <Menu className="w-6 h-6 text-gray-700" />
          </button>

          {/* Slide-out Menu */}
          {isOpen && (
            <div className="fixed inset-0 bg-black/50 z-50">
              <div className="absolute right-0 top-0 h-full w-64 bg-white shadow-lg p-4">
                {/* Close Button */}
                <button
                  type='button'
                  onClick={() => setIsOpen(false)}
                  className="absolute top-4 right-4"
                >
                  <X className="w-6 h-6 text-gray-700" />
                </button>

                {/* Menu Content */}
                <div className="pt-16 flex flex-col gap-4">
                  <WalletMultiButton className="!bg-purple-600 !w-full" />

                  <div className="border-t pt-4">
                    <button
                      type='button'
                      onClick={() => {
                        onTabChange('camera');
                        setIsOpen(false);
                      }}
                      className={`w-full text-left px-4 py-2 rounded-lg ${activeTab === 'camera'
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
                      className={`w-full text-left px-4 py-2 rounded-lg ${activeTab === 'gallery'
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
        </div>
        <div className="w-full mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}