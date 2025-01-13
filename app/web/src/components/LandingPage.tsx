import { useNavigate } from 'react-router-dom';
import Logo from './Logo'
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

const StepIcon = ({ children }: { children: React.ReactNode }) => (
    <div className="w-16 h-16 flex items-center justify-center rounded-full bg-[#e7eeff] mb-6 mx-auto">
        <div className="w-8 h-8 text-[#4b5563]">
            {children}
        </div>
    </div>
);

const LandingPage = () => {
    const navigate = useNavigate();
    const { setShowAuthFlow } = useDynamicContext();

    const steps = [
        {
            title: "Find a Moment Camera",
            description: "Locate a Moment camera at your favorite social spots - gyms, restaurants, parties, or bars. Our network of cameras is growing every day.",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
            )
        },
        {
            title: "Tap to Connect",
            description: "Simply tap your phone on the camera's NFC reader - just like using contactless payment. Quick, secure, and hassle-free.",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
            )
        },
        {
            title: "Capture Your Moment",
            description: "Get a live preview to frame your perfect shot or record a message. Our high-quality cameras ensure you look your best.",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
            )
        },
        {
            title: "Share & Download",
            description: "Instantly access your captures - download them directly to your phone or share them on your favorite social platforms.",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
            )
        }
    ];

    return (
        <div className="bg-white min-h-screen overflow-auto">
            {/* Navigation */}
            <div className="fixed top-0 right-0 p-4 z-50">
                <div className="flex items-center gap-4">
                    {/* <button
                        onClick={() => navigate('/product')}
                        className="px-2 text-sm py-2 text-gray-600 hover:text-gray-900"
                    >
                        Product
                    </button> */}
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
            </div>

            {/* Main Content */}
            <main>
                {/* Top Section */}
                <section className="h-[85vh] flex items-center relative">
                    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div className="flex flex-col lg:flex-row items-center justify-between gap-12">
                            {/* Left side - Image */}
                            <div className="w-full lg:w-1/2">
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
                                <div className="flex items-center justify-center lg:justify-start gap-4 mb-8">
                                    <Logo width={50} height={42} className="text-gray-900" />
                                    <h1 className="text-5xl font-bold text-gray-900">
                                        Moment
                                    </h1>
                                </div>
                                <p className="text-xl text-gray-600 mb-8 max-w-md mx-auto lg:mx-0">
                                    Capture moments and their context instantly with a NFC-enabled camera system. Just tap your phone and record.
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
                    {/* Scroll indicator - only visible on desktop */}
                    <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 hidden lg:flex flex-col items-center text-gray-400">
                        <svg className="w-6 h-6 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                        </svg>
                    </div>
                </section>

                {/* Grid Pattern Section */}
                <section className="bg-gray-50 py-32 relative mt-[-5vh] grid-pattern">
                    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                        <h3 className="text-4xl font-bold text-center mb-4">Learn more?</h3>
                        <p className="text-gray-600 text-center mb-16 max-w-2xl mx-auto">
                            Read the blog post that inspired our project
                        </p>
                        <div className="flex justify-center">
                            <button
                                onClick={() => window.open('https://paragraph.xyz/@0xmigi.eth/more-physical-and-more-social', "_blank")}
                                className="px-8 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors"
                            >
                                link to post →
                            </button>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
};

export default LandingPage;