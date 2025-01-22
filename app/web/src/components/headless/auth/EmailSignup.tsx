import { FormEvent, useState } from 'react';
import { useConnectWithOtp, useDynamicContext } from '@dynamic-labs/sdk-react-core';

export const EmailSignup = () => {
  const { user } = useDynamicContext();
  const { connectWithEmail, verifyOneTimePassword } = useConnectWithOtp();
  const [showOtpInput, setShowOtpInput] = useState(false);
  const [email, setEmail] = useState('');

  const handleEmailSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      await connectWithEmail(email);
      setShowOtpInput(true);
    } catch (error) {
      console.error('Failed to send OTP:', error);
    }
  };

  const handleOtpSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const otp = (e.currentTarget.elements.namedItem('otp') as HTMLInputElement).value;
    try {
      await verifyOneTimePassword(otp);
    } catch (error) {
      console.error('Failed to verify OTP:', error);
    }
  };

  if (user) {
    return (
      <div className="p-4">
        <p className="text-sm text-gray-600">Signed in as: {user.email}</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      {!showOtpInput ? (
        <form onSubmit={handleEmailSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Enter your email"
              required
            />
          </div>
          <button
            type="submit"
            className="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            Continue with Email
          </button>
        </form>
      ) : (
        <form onSubmit={handleOtpSubmit} className="space-y-4">
          <div>
            <label htmlFor="otp" className="block text-sm font-medium text-gray-700">
              Enter verification code
            </label>
            <input
              type="text"
              id="otp"
              name="otp"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="Enter code"
              required
            />
          </div>
          <button
            type="submit"
            className="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            Verify Code
          </button>
        </form>
      )}
    </div>
  );
}; 