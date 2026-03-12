import { useNavigate } from 'react-router-dom';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import Logo from '../common/Logo';

export default function Navbar() {
    const { primaryWallet } = useDynamicContext();
    const navigate = useNavigate();

    const handleGetStarted = () => {
        if (primaryWallet?.address) {
            navigate('/app');
        } else {
            navigate('/login');
        }
    };

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    {/* Left: Logo */}
                    <a href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                        <Logo width={30} height={21} className="text-neutral-900" />
                        <span className="text-2xl font-bold text-neutral-900">Moment</span>
                    </a>

                    {/* Right: Get Started button */}
                    <button
                        onClick={handleGetStarted}
                        className="px-4 py-2 bg-neutral-100 text-neutral-600 rounded-lg hover:bg-neutral-200 transition-colors text-sm font-medium"
                    >
                        Get Started
                    </button>
                </div>
            </div>
        </nav>
    );
}
