// src/components/EventListener.tsx
// NEW PRIVACY ARCHITECTURE: Uses TimelineUpdated event which doesn't expose user info

import { useEffect, useState } from 'react';
import { useProgram } from '../anchor/setup';

interface Event {
  timestamp: string;
  signature: string;
  cameraAccount?: string;
  activityCount?: number;
}

export default function EventListener() {
  const { program } = useProgram();
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    if (!program) return;

    let eventListener: number | null = null;

    try {
      // Use TimelineUpdated event (privacy-preserving - no user info exposed)
      eventListener = program.addEventListener('TimelineUpdated', (event: any) => {
        const newEvent: Event = {
          timestamp: new Date().toISOString(),
          signature: event.signature,
          cameraAccount: event.camera?.toString(),
          activityCount: event.activityCount?.toNumber?.() || event.activityCount
        };
        setEvents(prev => [...prev, newEvent]);
      });

      // Return cleanup function
      return () => {
        try {
          if (eventListener !== null) {
            program.removeEventListener(eventListener);
          }
        } catch (e) {
          console.error("Failed to remove listener:", e);
        }
      };
    } catch (e) {
      console.error("Error setting up event listener:", e);
      return undefined;
    }
  }, [program]);

  if (events.length === 0) return null;

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      {events.map((event, index) => (
        <div key={index} className="mb-4 last:mb-0">
          <p className="text-gray-800">Time: {event.timestamp}</p>
          <p className="text-gray-800">Camera: {event.cameraAccount}</p>
          {event.activityCount && (
            <p className="text-gray-800">Activities: {event.activityCount}</p>
          )}
        </div>
      ))}
    </div>
  );
}