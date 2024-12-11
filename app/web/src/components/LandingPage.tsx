import { useNavigate } from 'react-router-dom';
import Logo from './Logo'
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

const LandingPage = () => {
    const navigate = useNavigate();
    const { setShowAuthFlow } = useDynamicContext();

    return (
        <div className="min-h-screen bg-white">
            {/* Top Navigation */}
            <div className="fixed top-0 right-0 p-4">
                <button
                    onClick={() => {
                        setShowAuthFlow(true);
                        console.log("Attempting to show auth flow");
                        navigate('/app')
                      }}
                    className="px-6 py-2 bg-[#e7eeff] text-black rounded-lg hover:bg-[#a5bafc] transition-colors"
                >
                    Open App
                </button>
            </div>

            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Container for content */}
                <div className="min-h-screen flex flex-col lg:flex-row items-center justify-center gap-8 lg:gap-16">
                    {/* Left side - Image */}
                    <div className="w-full lg:w-1/2 order-2 lg:order-1">
                        <div className="aspect-video bg-gray-100 rounded-2xl overflow-hidden">
                            <img
                                // Replace this URL with your actual image path
                                src="/cyberchunk.png"
                                alt="Moment Camera System"
                                className="w-full h-full object-cover"
                            />
                        </div>
                    </div>

                    {/* Right side - Content */}
                    <div className="w-full lg:w-1/2 order-1 lg:order-2 text-center lg:text-left">
                        <div className="flex items-center justify-center lg:justify-start gap-4 mb-6">
                            <Logo width={50} height={42} className="text-gray-900" />
                            <h1 className="text-5xl font-bold text-gray-900">
                                Moment
                            </h1>
                        </div>
                        <p className="text-l text-gray-600 mb-8 max-w-md mx-auto lg:mx-0">
                            Capture moments and context instantly with a NFC-enabled camera system. Just tap your phone and record.
                        </p>
                        <div className="space-y-4">
                            <div className="flex items-center justify-center lg:justify-start gap-2 text-gray-600">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                <span>Live content hubs</span>
                            </div>
                            <div className="flex items-center justify-center lg:justify-start gap-2 text-gray-600">
                                <div className="w-2 h-2 bg-pink-300 rounded-full" />
                                <span>Secured by Solana</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LandingPage;