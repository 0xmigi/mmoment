import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { Menu, X } from 'lucide-react';
import Logo from '../common/Logo';

export default function Navbar() {
    const { primaryWallet } = useDynamicContext();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const navigate = useNavigate();

    const handleLaunchApp = () => {
        if (primaryWallet?.address) {
            navigate('/app');
        } else {
            navigate('/login');
        }
    };

    return (
        <>
            <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-sm">
                <div className="max-w-7xl mx-auto px-4">
                    <div className="flex items-center justify-between h-16">
                        {/* Left: Logo */}
                        <a href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                            <Logo width={30} height={21} className="text-black" />
                            <span className="text-2xl font-bold text-black">Moment</span>
                        </a>

                        {/* Right: Desktop Nav Items */}
                        <div className="hidden sm:flex items-center gap-6">
                            {/* About Link */}
                            <a
                                href="/about"
                                className="text-gray-600 hover:text-gray-900 transition-colors text-sm font-medium"
                            >
                                About
                            </a>

                            {/* Launch App Button */}
                            <button
                                onClick={handleLaunchApp}
                                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
                            >
                                Launch App
                            </button>
                        </div>

                        {/* Mobile: Hamburger Button */}
                        <button
                            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                            className="sm:hidden p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
                        >
                            {mobileMenuOpen ? (
                                <X className="w-5 h-5 text-gray-700" />
                            ) : (
                                <Menu className="w-5 h-5 text-gray-700" />
                            )}
                        </button>
                    </div>
                </div>
            </nav>

            {/* Mobile Navigation Menu */}
            {mobileMenuOpen && (
                <div className="fixed inset-0 z-40 sm:hidden">
                    {/* Overlay */}
                    <div
                        className="fixed inset-0 bg-black/50"
                        onClick={() => setMobileMenuOpen(false)}
                    />

                    {/* Menu Panel */}
                    <div className="fixed top-0 left-0 right-0 bg-white shadow-lg">
                        {/* Header */}
                        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
                            <a href="/" className="flex items-center gap-2">
                                <Logo width={30} height={21} className="text-black" />
                                <span className="text-2xl font-bold text-black">Moment</span>
                            </a>
                            <button
                                onClick={() => setMobileMenuOpen(false)}
                                className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
                            >
                                <X className="w-5 h-5 text-gray-700" />
                            </button>
                        </div>

                        {/* Menu Items */}
                        <div className="px-4 pb-6 space-y-2">
                            {/* About */}
                            <a
                                href="/about"
                                onClick={() => setMobileMenuOpen(false)}
                                className="block px-3 py-3 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium text-gray-900"
                            >
                                About
                            </a>

                            {/* Launch App Button */}
                            <button
                                onClick={() => {
                                    setMobileMenuOpen(false);
                                    handleLaunchApp();
                                }}
                                className="w-full px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
                            >
                                Launch App
                            </button>
                        </div>
                    </div>
                </div>
            )}

        </>
    );
}
