/**
 * Device Setup Wizard for MMOMENT Cameras
 * 
 * Uses QR code approach - user shows QR to camera for setup
 * Designed for /app/register page - single location setup experience
 */

import { QrRegistrationWizard } from './QrRegistrationWizard';

interface SetupWizardProps {
  onComplete?: (cameraData: any) => void;
  onError?: (error: string) => void;
}

export function DeviceSetupWizard({ onComplete, onError }: SetupWizardProps) {
  return (
    <QrRegistrationWizard 
      onComplete={onComplete}
      onError={onError}
    />
  );
}