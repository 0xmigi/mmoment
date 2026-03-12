import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useConnectWithOtp, useEmbeddedWallet, useIsLoggedIn } from '@dynamic-labs/sdk-react-core';
import { HeadlessSocialLogin } from '../auth/components/HeadlessSocialLogin';
import Logo from '../ui/common/Logo';

export default function LoginPage() {
    const { connectWithEmail, verifyOneTimePassword } = useConnectWithOtp();
    const { createEmbeddedWallet, userHasEmbeddedWallet } = useEmbeddedWallet();
    const navigate = useNavigate();
    const location = useLocation();
    const isLoggedIn = useIsLoggedIn();

    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [showOtpInput, setShowOtpInput] = useState(false);
    const [otp, setOtp] = useState('');

    // Get the redirect destination from query params (survives OAuth redirects), default to /app
    const redirectTo =
      new URLSearchParams(location.search).get('redirect') ||
      (location.state as { returnTo?: string })?.returnTo ||
      '/app';

    // If already logged in, redirect
    useEffect(() => {
        if (isLoggedIn) {
            navigate(redirectTo);
        }
    }, [isLoggedIn, navigate, redirectTo]);

    const handleSuccessfulAuth = () => {
        navigate(redirectTo);
    };

    const handleEmailSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setIsLoading(true);
            setError('');
            await connectWithEmail(email);
            setShowOtpInput(true);
        } catch (error) {
            console.error('Failed to send verification code:', error);
            setError('Failed to send verification code. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleOtpSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            setIsLoading(true);
            setError('');
            await verifyOneTimePassword(otp);

            if (!userHasEmbeddedWallet) {
                await createEmbeddedWallet();
            }

            handleSuccessfulAuth();
        } catch (error) {
            console.error('Failed to verify code:', error);
            setError('Invalid verification code. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col bg-white">
            {/* Header */}
            <div className="flex-none">
                <div className="max-w-7xl mx-auto px-4 h-16 flex items-center">
                    <a href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                        <Logo width={30} height={21} className="text-black" />
                        <span className="text-2xl font-bold text-black">Moment</span>
                    </a>
                </div>
            </div>

            {/* Centered Content */}
            <div className="flex-1 flex items-center justify-center p-4">
                <div className="w-full max-w-[360px] space-y-6">
                    {/* Header */}
                    <h1 className="text-xl font-medium text-neutral-900">Log in or sign up</h1>

                    {error && (
                        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
                            {error}
                        </div>
                    )}

                    {showOtpInput ? (
                        <form onSubmit={handleOtpSubmit} className="space-y-4">
                            <div>
                                <input
                                    type="text"
                                    id="otp"
                                    value={otp}
                                    onChange={(e) => setOtp(e.target.value)}
                                    disabled={isLoading}
                                    className="w-full px-3 py-2 border border-neutral-200 rounded-lg shadow-sm focus:ring-primary focus:border-primary disabled:opacity-50 disabled:cursor-not-allowed"
                                    placeholder="Enter verification code"
                                    required
                                />
                                <p className="mt-2 text-xs text-neutral-400">
                                    We sent a code to your email address
                                </p>
                            </div>
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full bg-neutral-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-neutral-600/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? 'Verifying...' : 'Verify'}
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setShowOtpInput(false);
                                    setOtp('');
                                    setError('');
                                }}
                                className="w-full text-sm text-neutral-400 hover:text-neutral-600"
                            >
                                Back to log in
                            </button>
                        </form>
                    ) : (
                        <div className="space-y-3">
                            {/* Email at the top */}
                            <form onSubmit={handleEmailSubmit}>
                                <input
                                    type="email"
                                    id="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    disabled={isLoading}
                                    className="w-full px-3 py-2 border border-neutral-200 rounded-lg shadow-sm focus:ring-primary focus:border-primary disabled:opacity-50 disabled:cursor-not-allowed"
                                    placeholder="name@example.com"
                                    required
                                />
                                <button
                                    type="submit"
                                    disabled={isLoading}
                                    className="w-full mt-2 bg-neutral-600 text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-neutral-600/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isLoading ? 'Please wait...' : 'Continue with Email'}
                                </button>
                            </form>

                            <div className="relative my-6">
                                <div className="absolute inset-0 flex items-center">
                                    <div className="w-full border-t border-neutral-200"></div>
                                </div>
                                <div className="relative flex justify-center text-xs">
                                    <span className="px-2 bg-white text-neutral-400">or continue with</span>
                                </div>
                            </div>

                            {/* Social logins (Google, Farcaster, X) */}
                            <HeadlessSocialLogin onSuccess={handleSuccessfulAuth} />
                        </div>
                    )}

                    {/* Footer text */}
                    <p className="text-center text-sm text-neutral-400 mt-6">
                        Capture moments and their context instantly
                    </p>
                </div>
            </div>
        </div>
    );
}
