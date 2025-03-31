// src/components/EventListener.tsx

import { useEffect, useState } from 'react';
import { useProgram } from '../anchor/setup';

interface Event {
  timestamp: string;
  signature: string;
  user?: string;
  cameraAccount?: string;
}

export default function EventListener() {
  const { program } = useProgram();
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    if (!program) return;

    let eventListener: number | null = null;
    
    try {
      // Use ActivityRecorded event which is defined in the Solana program
      eventListener = program.addEventListener('ActivityRecorded', (event: any) => {
        const newEvent: Event = {
          timestamp: new Date().toISOString(),
          signature: event.signature,
          user: event.user?.toString(),
          cameraAccount: event.camera?.toString() // field is likely called 'camera' based on program
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
          <p className="text-gray-800">User: {event.user}</p>
          <p className="text-gray-800">Camera: {event.cameraAccount}</p>
        </div>
      ))}
    </div>
  );
}