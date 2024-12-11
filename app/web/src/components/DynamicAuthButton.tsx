// src/components/DynamicAuthButton.tsx
import { DynamicWidget, useDynamicContext } from "@dynamic-labs/sdk-react-core";

export function DynamicAuthButton() {
  const { primaryWallet, user } = useDynamicContext();

  return (
    <div className="flex z-[60] items-center gap-2">
      {user && primaryWallet && (
        <span className="text-sm">
          {primaryWallet.address.slice(0, 4)}...{primaryWallet.address.slice(-4)}
        </span>
      )}
      <DynamicWidget />
    </div>
  );
}