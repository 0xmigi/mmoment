import { useNavigate } from 'react-router-dom';
import Logo from '../ui/common/Logo';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { useEffect } from 'react';
import Navbar from '../ui/layout/Navbar';
import Footer from '../ui/layout/Footer';

export default function LandingPage() {
    const { primaryWallet } = useDynamicContext();
    const navigate = useNavigate();

    useEffect(() => {
        if (primaryWallet?.address) {
            navigate('/app');
        }
    }, [primaryWallet, navigate]);

    return (
        <div className="bg-white min-h-screen overflow-auto">
            {/* Navigation */}
            <Navbar />

            {/* Main Content */}
            <main>
                {/* Top Section */}
                <section className="h-[90vh] flex items-center relative pt-20 sm:pt-0">
                    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-12">
                            {/* Left side - Image */}
                            <div className="w-full lg:w-1/2 mt-4 sm:mt-0">
                                <div className="aspect-video bg-gray-100 rounded-2xl overflow-hidden">
                                    <img
                                        src="/cyberchunk.png"
                                        alt="Moment Camera System"
                                        className="w-full h-full object-cover"
                                    />
                                </div>
                            </div>

                            {/* Right side - Content */}
                            <div className="w-full lg:w-1/2 text-center lg:text-left">
                                <div className="flex items-center justify-center lg:justify-start gap-4 mb-6">
                                    <Logo width={40} height={32} className="text-gray-900 sm:w-[50px] sm:h-[42px]" />
                                    <h1 className="text-3xl sm:text-5xl font-bold text-gray-900">
                                        Moment
                                    </h1>
                                </div>
                                <p className="text-base sm:text-xl text-gray-600 mb-6 max-w-md mx-auto lg:mx-0">
                                    Capture moments and their context instantly with a NFC-enabled camera system. Just tap your phone and record.
                                </p>
                                <div className="space-y-3">
                                    <div className="flex items-center justify-center lg:justify-start gap-2 text-sm sm:text-base text-gray-600">
                                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                        <span>Live content hubs</span>
                                    </div>
                                    <div className="flex items-center justify-center lg:justify-start gap-2 text-sm sm:text-base text-gray-600">
                                        <div className="w-2 h-2 bg-accent rounded-full" />
                                        <span>Secured by Solana</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Grid Pattern Section - fades to black via CSS */}
                <section className="relative mt-0 grid-pattern pb-24">
                    <div className="relative z-10 py-16 sm:py-24">
                        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                            <h3 className="text-2xl sm:text-4xl font-bold text-center mb-3">Learn more?</h3>
                            <p className="text-sm sm:text-base text-gray-600 text-center mb-8 sm:mb-12 max-w-2xl mx-auto">
                                Read the blog post that inspired our project
                            </p>
                            <div className="flex justify-center">
                                <button
                                    onClick={() => window.open('https://paragraph.xyz/@0xmigi.eth/more-physical-and-more-social', "_blank")}
                                    className="px-6 sm:px-8 py-2.5 sm:py-3 bg-gray-100 text-gray-700 border-2 border-accent text-sm sm:text-base rounded-lg hover:bg-gray-200 transition-colors"
                                >
                                    Read the post →
                                </button>
                            </div>
                        </div>
                    </div>
                </section>
            </main>

            {/* Footer - no top padding since it merges with section above */}
            <Footer />
        </div>
    );
}