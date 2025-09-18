// src/components/DynamicAuthButton.tsx
import { DynamicWidget } from "@dynamic-labs/sdk-react-core";

export function DynamicAuthButton() {
  return (
    <div className="flex z-[60] items-center gap-2">
      <DynamicWidget />
    </div>
  );
}