import { useState } from 'react';
import { Dialog } from '@headlessui/react';
import { X } from 'lucide-react';
import { QueueConfig } from '../hooks/useQueue';

interface QueueConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: QueueConfig;
  onSave: (config: Partial<{ enabled: boolean; maxSlotDuration: number; minSlotDuration: number; location: string | null }>) => Promise<void>;
}

const DURATION_PRESETS = [
  { label: '30 min', value: 1800 },
  { label: '1 hour', value: 3600 },
  { label: '1.5 hours', value: 5400 },
  { label: '2 hours', value: 7200 },
  { label: '3 hours', value: 10800 },
];

export function QueueConfigModal({ isOpen, onClose, config, onSave }: QueueConfigModalProps) {
  const [maxDuration, setMaxDuration] = useState(config.maxSlotDuration);
  const [minDuration, setMinDuration] = useState(config.minSlotDuration);
  const [enabled, setEnabled] = useState(config.enabled);
  const [location, setLocation] = useState(config.location || '');
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave({
        enabled,
        maxSlotDuration: maxDuration,
        minSlotDuration: minDuration,
        location: location.trim() || null,
      });
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-[100]">
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2">
        <Dialog.Panel className="mx-auto w-full sm:max-w-sm rounded-xl bg-white shadow-xl">
          <div className="p-5">
            <div className="flex items-center justify-between mb-5">
              <Dialog.Title className="text-lg font-semibold text-[#1A1A18]">
                Queue Settings
              </Dialog.Title>
              <button onClick={onClose} className="p-1 hover:bg-[#F3F3EF] rounded">
                <X className="w-4 h-4 text-[#8A8A82]" />
              </button>
            </div>

            {/* Location */}
            <div className="mb-6">
              <label className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-2 block">Location (address)</label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. 123 Main St, Brooklyn, NY"
                className="w-full px-3 py-2 text-sm text-[#1A1A18] bg-white border border-[#E8E8E3] rounded-lg outline-none focus:border-[#8A8A82] transition-colors"
              />
            </div>

            {/* Enable/Disable */}
            <div className="flex items-center justify-between mb-6">
              <span className="text-sm text-[#5C5C56]">Queue enabled</span>
              <button
                onClick={() => setEnabled(!enabled)}
                className={`relative w-10 h-5 rounded-full transition-colors ${enabled ? 'bg-[#1A1A18]' : 'bg-[#E8E8E3]'}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${enabled ? 'left-5' : 'left-0.5'}`} />
              </button>
            </div>

            {/* Max slot duration */}
            <div className="mb-5">
              <label className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-2 block">Max slot duration</label>
              <div className="flex flex-wrap gap-2">
                {DURATION_PRESETS.map(preset => (
                  <button
                    key={preset.value}
                    onClick={() => setMaxDuration(preset.value)}
                    className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                      maxDuration === preset.value
                        ? 'bg-[#1A1A18] text-white border-[#1A1A18]'
                        : 'bg-white text-[#5C5C56] border-[#E8E8E3] hover:border-[#8A8A82]'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Min slot duration */}
            <div className="mb-6">
              <label className="text-xs font-semibold tracking-[0.1em] uppercase text-[#8A8A82] mb-2 block">Min slot duration</label>
              <div className="flex flex-wrap gap-2">
                {DURATION_PRESETS.filter(p => p.value <= maxDuration).map(preset => (
                  <button
                    key={preset.value}
                    onClick={() => setMinDuration(preset.value)}
                    className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                      minDuration === preset.value
                        ? 'bg-[#1A1A18] text-white border-[#1A1A18]'
                        : 'bg-white text-[#5C5C56] border-[#E8E8E3] hover:border-[#8A8A82]'
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleSave}
              disabled={saving || minDuration > maxDuration}
              className="w-full py-2.5 bg-[#1A1A18] text-white text-sm font-medium rounded-xl hover:bg-[#2a2a28] disabled:opacity-40 transition-colors"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}
