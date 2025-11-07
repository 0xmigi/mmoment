/**
 * Firestarter SDK Provider for MMOMENT
 * Wraps the app with Pipe storage functionality
 */

import { FirestarterProvider } from 'firestarter-sdk/react';
import type { PipeConfig } from 'firestarter-sdk/browser';

const config: PipeConfig = {
  baseUrl: 'https://us-west-01-firestarter.pipenetwork.com',
};

export function PipeProvider({ children }: { children: React.ReactNode }) {
  return <FirestarterProvider config={config}>{children}</FirestarterProvider>;
}
