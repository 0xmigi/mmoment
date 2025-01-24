import { useNavigate } from 'react-router-dom';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';

const StepIcon = ({ children }: { children: React.ReactNode }) => (
    <div className="w-16 h-16 flex items-center justify-center rounded-full bg-[#e7eeff] mb-6 mx-auto">
        <div className="w-8 h-8 text-[#4b5563]">
            {children}
        </div>
    </div>
);

const ProductPage = () => {
    const navigate = useNavigate();
    const { setShowAuthFlow, primaryWallet } = useDynamicContext();

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

    const handleGetStarted = () => {
        if (!primaryWallet) {
            setShowAuthFlow(true);
        } else {
            navigate('/app');
        }
    };

    return (
        <div className="bg-white min-h-screen overflow-auto">
            {/* Navigation */}
            <nav className="fixed top-0 right-0 p-4 z-50">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/')}
                        className="px-2 text-sm py-2 text-gray-600 hover:text-gray-900"
                    >
                        Home
                    </button>
                    <div className="px-2 text-sm py-2 text-gray-900 relative">
                        Product
                        <div className="absolute -bottom-1 left-1/2 w-1 h-1 bg-black rounded-full transform -translate-x-1/2" />
                    </div>
                    <button
                        onClick={handleGetStarted}
                        className="px-6 py-2 bg-[#e7eeff] text-black rounded-lg hover:bg-[#a5bafc] transition-colors"
                    >
                        Open App
                    </button>
                </div>
            </nav>

            {/* Steps Section */}
            <section className="py-32">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                    <h2 className="text-4xl font-bold text-center mb-4">How It Works</h2>
                    <p className="text-gray-600 text-center mb-16 max-w-2xl mx-auto">
                        Get started with Moment in four simple steps
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                        {steps.map((step, index) => (
                            <div key={index} className="relative">
                                {index < steps.length - 1 && (
                                    <div className="hidden lg:block absolute top-8 left-full w-full h-0.5 bg-gray-200 -z-10" />
                                )}
                                <div className="bg-white p-8 rounded-xl">
                                    <StepIcon>{step.icon}</StepIcon>
                                    <h3 className="text-xl font-semibold mb-4 text-center">{step.title}</h3>
                                    <p className="text-gray-600 text-center">{step.description}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
};

export default ProductPage; 